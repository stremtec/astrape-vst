#!/usr/bin/env python3
"""Train the Minimal Causal Student (MCS)."""

import argparse
from pathlib import Path
from astrape.mcs_model import McsConfig
from astrape.mcs_training import McsTrainingConfig, train_mcs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    p.add_argument("--projection-path", type=Path, default=Path("checkpoints/teacher_fsq_proj_out.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--run-name", default="mcs")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--probe-samples", type=int, default=1024)
    p.add_argument("--full-validation-every", type=int, default=5)
    p.add_argument("--target-cosine", type=float, default=0.85)
    p.add_argument("--conv-dim", type=int, default=256)
    p.add_argument("--gru-dim", type=int, default=384)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_mcs(McsConfig(conv_dim=args.conv_dim, gru_dim=args.gru_dim, gru_layers=args.gru_layers), McsTrainingConfig(
        data_dir=args.data_dir, projection_path=args.projection_path,
        output_dir=args.output_dir, run_name=args.run_name,
        device=args.device, batch_size=args.batch_size,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate, probe_samples=args.probe_samples,
        full_validation_every=args.full_validation_every,
        target_cosine=args.target_cosine, seed=args.seed,
        resume=args.resume, log_every=args.log_every,
    ))


if __name__ == "__main__":
    main()
