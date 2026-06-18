"""Phase 1: Encoder warm-up with teacher guidance.

Trains the CausalContentEncoder using MSE + cosine loss against teacher ce_768.
The encoder learns to produce content embeddings near the teacher's space.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from astrape.data import ContentSample, MioContentDataset, crop_aligned, speaker_disjoint_split
from astrape.encoder import CausalContentEncoder, EncoderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# --- Collation ---


def collate_phase1(batch: list[ContentSample]) -> dict[str, torch.Tensor]:
    max_mel = max(s.mel.shape[1] for s in batch)
    max_content = max(s.content.shape[0] for s in batch)

    # Ensure even mel length for frame-pair downsampling
    if max_mel % 2:
        max_mel += 1

    mels = []
    targets = []
    masks = []

    for s in batch:
        mel_len = s.mel.shape[1]
        content_len = s.content.shape[0]

        mel_padded = F.pad(s.mel, (0, max_mel - mel_len))
        content_padded = F.pad(s.content, (0, 0, 0, max_content - content_len))
        mask = torch.zeros(max_content, dtype=torch.bool)
        mask[:content_len] = True

        mels.append(mel_padded)
        targets.append(content_padded)
        masks.append(mask)

    return {
        "mel": torch.stack(mels),
        "target_content": torch.stack(targets),
        "mask": torch.stack(masks),
    }


# --- Loss Functions ---


def teacher_loss(
    pred_768: torch.Tensor,
    target_768: torch.Tensor,
    mask: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    # pred_768: (B, 768, T), target_768: (B, T, 768), mask: (B, T)
    pred = pred_768.transpose(1, 2)  # (B, T, 768)

    # Align lengths
    T_pred = pred.shape[1]
    T_target = target_768.shape[1]
    T = min(T_pred, T_target)
    pred = pred[:, :T]
    target = target_768[:, :T]
    mask = mask[:, :T]

    # Flatten masked
    pred_flat = pred[mask]
    target_flat = target[mask]

    if pred_flat.shape[0] == 0:
        zero = pred.sum() * 0.0
        return zero, {"mse": 0.0, "cosine_loss": 0.0, "cosine_sim": 0.0}

    mse = F.mse_loss(pred_flat, target_flat)
    cosine = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()

    total = mse_weight * mse + cosine_weight * cosine
    return total, {
        "mse": mse.item(),
        "cosine_loss": cosine.item(),
        "cosine_sim": (1.0 - cosine).item(),
    }


# --- Training Loop ---


def train(args):
    device = torch.device(args.device)
    log.info(f"Device: {device}")

    # Encoder
    config = EncoderConfig()
    encoder = CausalContentEncoder(config).to(device)
    log.info(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Optionally load teacher proj_out
    proj_out_path = Path(args.teacher_proj_out)
    if proj_out_path.exists():
        encoder.fsq.load_teacher_proj_out(proj_out_path)
        log.info(f"Loaded teacher proj_out from {proj_out_path}")

    # Dataset
    dataset = MioContentDataset(args.data_dir, args.data_dir)
    train_idx, val_idx = speaker_disjoint_split(
        dataset.speakers, validation_fraction=0.15, seed=42
    )
    train_dataset = MioContentDataset(args.data_dir, args.data_dir, indices=train_idx)
    val_dataset = MioContentDataset(args.data_dir, args.data_dir, indices=val_idx)
    log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    rng = random.Random(args.seed)

    def make_batch(dataset, batch_size):
        samples = []
        for _ in range(batch_size):
            idx = rng.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            sample = crop_aligned(sample, args.max_mel_frames, rng)
            samples.append(sample)
        return collate_phase1(samples)

    # Optimizer
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    # Training
    encoder.train()
    step = 0
    best_val_cosine = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting Phase 1 encoder warm-up for {args.steps} steps")

    while step < args.steps:
        batch = make_batch(train_dataset, args.batch_size)
        mel = batch["mel"].to(device)
        target = batch["target_content"].to(device)
        mask = batch["mask"].to(device)

        # Forward
        out = encoder(mel)

        # Teacher loss
        loss, metrics = teacher_loss(
            out.content, target, mask,
            mse_weight=1.0, cosine_weight=0.1,
        )

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
                f"mse={metrics['mse']:.4f} cos={metrics['cosine_sim']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if step % args.eval_every == 0:
            encoder.eval()
            val_cosines = []
            for _ in range(args.eval_batches):
                batch = make_batch(val_dataset, args.batch_size)
                mel = batch["mel"].to(device)
                target = batch["target_content"].to(device)
                mask = batch["mask"].to(device)

                with torch.no_grad():
                    out = encoder(mel)
                    pred = out.content.transpose(1, 2)
                    T = min(pred.shape[1], target.shape[1])
                    pred = pred[:, :T]
                    target_t = target[:, :T]
                    m = mask[:, :T]
                    if m.any():
                        cos = F.cosine_similarity(
                            pred[m], target_t[m], dim=-1
                        ).mean().item()
                        val_cosines.append(cos)

            encoder.train()
            if val_cosines:
                mean_cos = sum(val_cosines) / len(val_cosines)
                log.info(f"  VAL cosine={mean_cos:.4f}")

                if mean_cos > best_val_cosine:
                    best_val_cosine = mean_cos
                    torch.save({
                        "format_version": 1,
                        "model_type": "causal_content_encoder",
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
                        "val_cosine": mean_cos,
                    }, checkpoint_dir / "encoder_phase1.best.pt")
                    log.info(f"  New best: {mean_cos:.4f}")

        if step % args.save_every == 0:
            torch.save({
                "format_version": 1,
                "model_type": "causal_content_encoder",
                "state_dict": encoder.state_dict(),
                "step": step,
                "optimizer": optimizer.state_dict(),
            }, checkpoint_dir / "encoder_phase1.last.pt")

    log.info(f"Phase 1 complete. Best val cosine: {best_val_cosine:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Encoder warm-up")
    parser.add_argument("--data-dir", type=str, default="data/mio_vctk_full_compact")
    parser.add_argument("--teacher-proj-out", type=str, default="checkpoints/teacher_fsq_proj_out.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-mel-frames", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()
    train(args)
