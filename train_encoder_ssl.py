"""Train CausalContentEncoder with ce_768 anchoring plus WavLM auxiliary loss.

The deployed encoder output remains anchored to the teacher ce_768 space used
by the decoder. Frozen WavLM deep-layer features are used only through a
training-time projection head. Speaker separation is still encouraged because:
  1. WavLM layers 6+9 are content-rich / speaker-poor (known from SSL probing)
  2. FSQ bottleneck (5d, 12800 codes) further destroys speaker information
  3. The encoder never sees speaker identity

This is the same principle that makes MioCodec work, applied to causal mel input.

Requires: pre-extracted ssl_{index:05d}.npz files (run extract_wavlm_targets.py first).
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astrape.encoder import CausalContentEncoder, EncoderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# --- Dataset (from cache, no audio loading) ---


class SSLTargetDataset:
    """Loads mel, ce_768 content targets, and pre-extracted WavLM features."""

    def __init__(self, data_dir: str | Path, indices: np.ndarray):
        self.data_dir = Path(data_dir)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        sample_path = self.data_dir / f"s_{index:05d}.npz"
        ssl_path = self.data_dir / f"ssl_{index:05d}.npz"

        with np.load(sample_path) as data:
            if "logmel" in data:
                mel = torch.from_numpy(data["logmel"]).float()
            else:
                mel_path = self.data_dir / f"m_{index:05d}.npz"
                with np.load(mel_path) as mdata:
                    mel = torch.from_numpy(mdata["logmel"]).float()
            content_target = torch.from_numpy(data["ce_768"]).float()

        with np.load(ssl_path) as sdata:
            ssl_target = torch.from_numpy(sdata["wavlm_25hz"]).float()

        return {
            "mel": mel,
            "content_target": content_target,
            "ssl_target": ssl_target,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_mel_t = max(b["mel"].shape[1] for b in batch)
    max_content_t = max(b["content_target"].shape[0] for b in batch)
    max_ssl_t = max(b["ssl_target"].shape[0] for b in batch)

    if max_mel_t % 2:
        max_mel_t += 1

    mels, content_targets, ssl_targets = [], [], []
    mel_lens, content_lens, ssl_lens = [], [], []
    for b in batch:
        mel_len = b["mel"].shape[1]
        content_len = b["content_target"].shape[0]
        ssl_len = b["ssl_target"].shape[0]
        mels.append(F.pad(b["mel"], (0, max_mel_t - mel_len)))
        content_targets.append(
            F.pad(b["content_target"], (0, 0, 0, max_content_t - content_len))
        )
        ssl_targets.append(F.pad(b["ssl_target"], (0, 0, 0, max_ssl_t - ssl_len)))
        mel_lens.append(mel_len)
        content_lens.append(content_len)
        ssl_lens.append(ssl_len)

    return {
        "mel": torch.stack(mels),
        "target_content": torch.stack(content_targets),
        "ssl_target": torch.stack(ssl_targets),
        "mel_lens": torch.tensor(mel_lens, dtype=torch.long),
        "content_lens": torch.tensor(content_lens, dtype=torch.long),
        "ssl_lens": torch.tensor(ssl_lens, dtype=torch.long),
    }


# --- Loss ---


def make_frame_mask(
    mel_lens: torch.Tensor,
    target_lens: torch.Tensor,
    max_frames: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mask frames valid for both causal encoder output and a padded target."""
    device = device or target_lens.device
    if max_frames <= 0:
        return torch.zeros((target_lens.numel(), 0), dtype=torch.bool, device=device)
    mel_lens = mel_lens.to(device)
    target_lens = target_lens.to(device)
    valid_lens = torch.minimum(mel_lens // 2, target_lens).clamp(max=max_frames)
    positions = torch.arange(max_frames, device=device).unsqueeze(0)
    return positions < valid_lens.unsqueeze(1)


def feature_target_loss(
    pred_768: torch.Tensor,
    target_768: torch.Tensor,
    mask: torch.Tensor,
    l1_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cosine + L1 loss against a 768d target, masked.

    Args:
        pred_768: (B, 768, T_25hz)
        target_768: (B, T_25hz, 768)
        mask: (B, T_25hz) valid frames
    """
    pred = pred_768.transpose(1, 2)  # (B, T, 768)

    T = min(pred.shape[1], target_768.shape[1], mask.shape[1])
    pred = pred[:, :T]
    target = target_768[:, :T]
    mask = mask[:, :T]

    if T == 0 or not mask.any():
        zero = pred_768.sum() * 0.0
        return zero, {"cosine_sim": 0.0, "l1": 0.0, "frames": 0.0}

    pred_flat = pred[mask]
    target_flat = target[mask]

    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
    cosine_loss = 1.0 - cosine_sim
    l1 = F.l1_loss(pred_flat, target_flat)

    total = cosine_loss + l1_weight * l1
    return total, {
        "cosine_sim": cosine_sim.item(),
        "l1": l1.item(),
        "frames": float(pred_flat.shape[0]),
    }


def ssl_target_loss(
    pred_768: torch.Tensor,
    target_768: torch.Tensor,
    mask: torch.Tensor,
    l1_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    return feature_target_loss(pred_768, target_768, mask, l1_weight=l1_weight)


def masked_cosine(
    pred_768: torch.Tensor,
    target_768: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    pred = pred_768.transpose(1, 2)
    T = min(pred.shape[1], target_768.shape[1], mask.shape[1])
    pred = pred[:, :T]
    target = target_768[:, :T]
    mask = mask[:, :T]
    if T == 0 or not mask.any():
        return None
    return F.cosine_similarity(pred[mask], target[mask], dim=-1).mean().item()


# --- Training ---


def train(args):
    if args.content_loss_weight <= 0:
        raise ValueError("--content-loss-weight must stay positive to anchor ce_768")
    if args.ssl_loss_weight < 0:
        raise ValueError("--ssl-loss-weight must be non-negative")

    device = torch.device(args.device)
    log.info(f"Device: {device}")

    # Encoder
    config = EncoderConfig()
    encoder = CausalContentEncoder(config).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    log.info(f"Encoder params: {n_params:,}")
    ssl_projection = nn.Sequential(
        nn.LayerNorm(config.content_dim),
        nn.Linear(config.content_dim, config.content_dim),
    ).to(device)
    trainable_params = list(encoder.parameters()) + list(ssl_projection.parameters())

    # Optionally warm-start from Phase 1
    if args.warm_start and Path(args.warm_start).exists():
        ckpt = torch.load(args.warm_start, map_location=device)
        encoder.load_state_dict(ckpt["state_dict"], strict=True)
        log.info(f"Warm-started from {args.warm_start} (step {ckpt.get('step', '?')})")

    # Dataset (from pre-extracted cache)
    from astrape.data import speaker_disjoint_split
    meta = np.load(Path(args.data_dir) / "meta.npz")
    n_samples = int(meta["n_samples"])
    speakers = meta["spk_names"][:n_samples].astype(str)

    # Verify SSL cache exists
    ssl_check = Path(args.data_dir) / "ssl_00000.npz"
    if not ssl_check.exists():
        raise RuntimeError(
            "SSL cache not found. Run extract_wavlm_targets.py first."
        )

    train_idx, val_idx = speaker_disjoint_split(speakers, 0.15, seed=42)
    train_dataset = SSLTargetDataset(args.data_dir, train_idx)
    val_dataset = SSLTargetDataset(args.data_dir, val_idx)
    log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    # Training loop
    encoder.train()
    ssl_projection.train()
    step = 0
    best_val_content_cosine = 0.0
    rng = random.Random(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        f"Starting SSL-target training for {args.steps} steps "
        f"(content_weight={args.content_loss_weight}, ssl_weight={args.ssl_loss_weight})"
    )

    while step < args.steps:
        # Sample a batch
        batch_items = []
        for _ in range(args.batch_size):
            idx = rng.randint(0, len(train_dataset) - 1)
            item = train_dataset[idx]
            if item is not None:
                batch_items.append(item)

        if len(batch_items) < 2:
            continue

        batch = collate_fn(batch_items)
        if batch is None:
            continue

        mel = batch["mel"].to(device)
        content_target = batch["target_content"].to(device)
        ssl_target = batch["ssl_target"].to(device)
        mel_lens = batch["mel_lens"]
        content_lens = batch["content_lens"]
        ssl_lens = batch["ssl_lens"]

        # Forward encoder
        out = encoder(mel)  # content: (B, 768, T_25hz)

        # Primary loss: keep deployed content in decoder-compatible ce_768 space.
        T_content = min(out.content.shape[2], content_target.shape[1])
        content_mask = make_frame_mask(mel_lens, content_lens, T_content, device)
        content_loss, content_metrics = feature_target_loss(
            out.content[:, :, :T_content],
            content_target[:, :T_content],
            content_mask,
            l1_weight=args.content_l1_weight,
        )

        # Auxiliary loss: project deployed content to raw WavLM feature space.
        ssl_pred = ssl_projection(out.content.transpose(1, 2)).transpose(1, 2)
        T_ssl = min(ssl_pred.shape[2], ssl_target.shape[1])
        ssl_mask = make_frame_mask(mel_lens, ssl_lens, T_ssl, device)
        ssl_loss, ssl_metrics = ssl_target_loss(
            ssl_pred[:, :, :T_ssl],
            ssl_target[:, :T_ssl],
            ssl_mask,
            l1_weight=args.ssl_l1_weight,
        )
        loss = args.content_loss_weight * content_loss + args.ssl_loss_weight * ssl_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % args.log_every == 0:
            log.info(
                f"step={step:06d} loss={loss.item():.4f} "
                f"content_cos={content_metrics['cosine_sim']:.4f} "
                f"content_l1={content_metrics['l1']:.4f} "
                f"ssl_cos={ssl_metrics['cosine_sim']:.4f} "
                f"ssl_l1={ssl_metrics['l1']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if step % args.eval_every == 0:
            encoder.eval()
            ssl_projection.eval()
            val_content_cosines = []
            val_ssl_cosines = []
            for _ in range(args.eval_batches):
                items = []
                for _ in range(args.batch_size):
                    idx = rng.randint(0, len(val_dataset) - 1)
                    item = val_dataset[idx]
                    if item is not None:
                        items.append(item)
                if len(items) < 2:
                    continue
                vbatch = collate_fn(items)
                if vbatch is None:
                    continue

                with torch.no_grad():
                    vmel = vbatch["mel"].to(device)
                    vcontent = vbatch["target_content"].to(device)
                    vssl = vbatch["ssl_target"].to(device)
                    vml = vbatch["mel_lens"]
                    vcontent_lens = vbatch["content_lens"]
                    vssl_lens = vbatch["ssl_lens"]

                    vout = encoder(vmel)
                    vT_content = min(vout.content.shape[2], vcontent.shape[1])
                    vcontent_mask = make_frame_mask(
                        vml, vcontent_lens, vT_content, device
                    )
                    content_cos = masked_cosine(
                        vout.content[:, :, :vT_content],
                        vcontent[:, :vT_content],
                        vcontent_mask,
                    )
                    if content_cos is not None:
                        val_content_cosines.append(content_cos)

                    vssl_pred = ssl_projection(
                        vout.content.transpose(1, 2)
                    ).transpose(1, 2)
                    vT_ssl = min(vssl_pred.shape[2], vssl.shape[1])
                    vssl_mask = make_frame_mask(vml, vssl_lens, vT_ssl, device)
                    ssl_cos = masked_cosine(
                        vssl_pred[:, :, :vT_ssl],
                        vssl[:, :vT_ssl],
                        vssl_mask,
                    )
                    if ssl_cos is not None:
                        val_ssl_cosines.append(ssl_cos)

            encoder.train()
            ssl_projection.train()
            if val_content_cosines:
                mean_content_cos = sum(val_content_cosines) / len(val_content_cosines)
                mean_ssl_cos = (
                    sum(val_ssl_cosines) / len(val_ssl_cosines)
                    if val_ssl_cosines
                    else 0.0
                )
                log.info(
                    f"  VAL cosine_content={mean_content_cos:.4f} "
                    f"cosine_ssl={mean_ssl_cos:.4f}"
                )

                if mean_content_cos > best_val_content_cosine:
                    best_val_content_cosine = mean_content_cos
                    torch.save({
                        "format_version": 2,
                        "model_type": "causal_content_encoder",
                        "training_target": "ce_768_anchor_plus_wavlm_aux",
                        "config_dict": {
                            "mel_dim": config.mel_dim,
                            "content_dim": config.content_dim,
                            "frontend_dim": config.frontend_dim,
                            "transformer_dim": config.transformer_dim,
                            "transformer_heads": config.transformer_heads,
                            "transformer_layers": config.transformer_layers,
                            "transformer_window": config.transformer_window,
                            "fsq_levels": list(config.fsq_levels),
                        },
                        "state_dict": encoder.state_dict(),
                        "ssl_projection_state_dict": ssl_projection.state_dict(),
                        "step": step,
                        "val_cosine_content": mean_content_cos,
                        "val_cosine_ssl": mean_ssl_cos,
                        "loss_weights": {
                            "content": args.content_loss_weight,
                            "ssl": args.ssl_loss_weight,
                        },
                    }, checkpoint_dir / "encoder_ssl.best.pt")
                    log.info(f"  New best content cosine: {mean_content_cos:.4f}")

        if step % args.save_every == 0:
            torch.save({
                "format_version": 2,
                "model_type": "causal_content_encoder",
                "training_target": "ce_768_anchor_plus_wavlm_aux",
                "state_dict": encoder.state_dict(),
                "ssl_projection_state_dict": ssl_projection.state_dict(),
                "step": step,
                "optimizer": optimizer.state_dict(),
                "loss_weights": {
                    "content": args.content_loss_weight,
                    "ssl": args.ssl_loss_weight,
                },
            }, checkpoint_dir / "encoder_ssl.last.pt")

    log.info(
        "Training complete. "
        f"Best val cosine (content): {best_val_content_cosine:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train encoder with WavLM SSL target")
    parser.add_argument("--data-dir", type=str, default="data/mio_vctk_full_compact")
    parser.add_argument("--warm-start", type=str, default="checkpoints/encoder_phase1.best.pt",
                        help="Phase 1 checkpoint to warm-start from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--content-loss-weight",
        type=float,
        default=1.0,
        help="Primary ce_768 anchor weight for deployed encoder content",
    )
    parser.add_argument(
        "--ssl-loss-weight",
        type=float,
        default=0.1,
        help="Auxiliary projected WavLM loss weight",
    )
    parser.add_argument("--content-l1-weight", type=float, default=0.3)
    parser.add_argument("--ssl-l1-weight", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()
    train(args)
