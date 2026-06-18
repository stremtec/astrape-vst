"""Train CausalContentEncoder to predict WavLM[6,9] features through FSQ bottleneck.

The encoder learns to extract content from mel spectrograms by predicting
frozen WavLM deep-layer features. Speaker separation emerges because:
  1. WavLM layers 6+9 are content-rich / speaker-poor (known from SSL probing)
  2. FSQ bottleneck (5d, 12800 codes) further destroys speaker information
  3. The encoder never sees speaker identity — it only predicts content features

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
import torch.nn.functional as F

from astrape.encoder import CausalContentEncoder, EncoderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# --- Dataset (from cache, no audio loading) ---


class SSLTargetDataset:
    """Loads mel + pre-extracted WavLM features from NPZ cache."""

    def __init__(self, data_dir: str | Path, indices: np.ndarray):
        self.data_dir = Path(data_dir)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        sample_path = self.data_dir / f"s_{index:05d}.npz"
        ssl_path = self.data_dir / f"ssl_{index:05d}.npz"

        try:
            with np.load(sample_path) as data:
                if "logmel" in data:
                    mel = torch.from_numpy(data["logmel"]).float()
                else:
                    mel_path = self.data_dir / f"m_{index:05d}.npz"
                    with np.load(mel_path) as mdata:
                        mel = torch.from_numpy(mdata["logmel"]).float()

            with np.load(ssl_path) as sdata:
                ssl_target = torch.from_numpy(sdata["wavlm_25hz"]).float()

            return {"mel": mel, "ssl_target": ssl_target}
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_mel_t = max(b["mel"].shape[1] for b in batch)
    max_ssl_t = max(b["ssl_target"].shape[0] for b in batch)

    if max_mel_t % 2:
        max_mel_t += 1

    mels, targets, mel_lens = [], [], []
    for b in batch:
        mel_len = b["mel"].shape[1]
        ssl_len = b["ssl_target"].shape[0]
        mels.append(F.pad(b["mel"], (0, max_mel_t - mel_len)))
        targets.append(F.pad(b["ssl_target"], (0, 0, 0, max_ssl_t - ssl_len)))
        mel_lens.append(mel_len)

    return {
        "mel": torch.stack(mels),
        "ssl_target": torch.stack(targets),
        "mel_lens": torch.tensor(mel_lens),
    }


# --- Loss ---


def ssl_target_loss(
    pred_768: torch.Tensor,
    target_768: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cosine + L1 loss against WavLM features, masked.

    Args:
        pred_768: (B, 768, T_25hz) from encoder
        target_768: (B, T_25hz, 768) from WavLM pooled
        mask: (B, T_25hz) valid frames
    """
    pred = pred_768.transpose(1, 2)  # (B, T, 768)

    T = min(pred.shape[1], target_768.shape[1], mask.shape[1])
    pred = pred[:, :T]
    target = target_768[:, :T]
    mask = mask[:, :T]

    pred_flat = pred[mask]
    target_flat = target[mask]

    if pred_flat.shape[0] == 0:
        zero = torch.tensor(0.0, device=pred.device)
        return zero, {"cosine_sim": 0.0, "l1": 0.0}

    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
    cosine_loss = 1.0 - cosine_sim
    l1 = F.l1_loss(pred_flat, target_flat)

    total = cosine_loss + 0.3 * l1
    return total, {
        "cosine_sim": cosine_sim.item(),
        "l1": l1.item(),
    }


# --- Training ---


def train(args):
    device = torch.device(args.device)
    log.info(f"Device: {device}")

    # Encoder
    config = EncoderConfig()
    encoder = CausalContentEncoder(config).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    log.info(f"Encoder params: {n_params:,}")

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
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    # Training loop
    encoder.train()
    step = 0
    best_val_cosine = 0.0
    rng = random.Random(args.seed)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Starting SSL-target training for {args.steps} steps")

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
        ssl_target = batch["ssl_target"].to(device)
        mel_lens = batch["mel_lens"]

        # Forward encoder
        out = encoder(mel)  # content: (B, 768, T_25hz)

        # Build mask: valid frames at 25Hz
        T_out = out.content.shape[2]
        T_ssl = ssl_target.shape[1]
        T = min(T_out, T_ssl)
        mask_lens_25hz = (mel_lens // 2).clamp(max=T)
        mask = torch.arange(T, device=device).unsqueeze(0) < mask_lens_25hz.unsqueeze(1).to(device)

        # Loss
        loss, metrics = ssl_target_loss(out.content[:, :, :T], ssl_target[:, :T], mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % args.log_every == 0:
            log.info(
                f"step={step:06d} loss={loss.item():.4f} "
                f"cos={metrics['cosine_sim']:.4f} l1={metrics['l1']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if step % args.eval_every == 0:
            encoder.eval()
            val_cosines = []
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
                    vssl = vbatch["ssl_target"].to(device)
                    vml = vbatch["mel_lens"]

                    vout = encoder(vmel)
                    vT = min(vout.content.shape[2], vssl.shape[1])
                    vmask_lens = (vml // 2).clamp(max=vT)
                    vmask = torch.arange(vT, device=device).unsqueeze(0) < vmask_lens.unsqueeze(1).to(device)

                    pred = vout.content[:, :, :vT].transpose(1, 2)
                    tgt = vssl[:, :vT]
                    if vmask.any():
                        cos = F.cosine_similarity(
                            pred[vmask], tgt[vmask], dim=-1
                        ).mean().item()
                        val_cosines.append(cos)

            encoder.train()
            if val_cosines:
                mean_cos = sum(val_cosines) / len(val_cosines)
                log.info(f"  VAL cosine_ssl={mean_cos:.4f}")

                if mean_cos > best_val_cosine:
                    best_val_cosine = mean_cos
                    torch.save({
                        "format_version": 2,
                        "model_type": "causal_content_encoder",
                        "training_target": "wavlm_6_9",
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
                        "step": step,
                        "val_cosine_ssl": mean_cos,
                    }, Path(args.checkpoint_dir) / "encoder_ssl.best.pt")
                    log.info(f"  New best: {mean_cos:.4f}")

        if step % args.save_every == 0:
            torch.save({
                "format_version": 2,
                "model_type": "causal_content_encoder",
                "training_target": "wavlm_6_9",
                "state_dict": encoder.state_dict(),
                "step": step,
                "optimizer": optimizer.state_dict(),
            }, Path(args.checkpoint_dir) / "encoder_ssl.last.pt")

    log.info(f"Training complete. Best val cosine (SSL): {best_val_cosine:.4f}")


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
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()
    train(args)
