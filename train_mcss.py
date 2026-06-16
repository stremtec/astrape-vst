#!/usr/bin/env python3
"""Train the Minimal Causal Student – Scaled (MCSS)."""

import argparse
from pathlib import Path
from astrape.mcss_model import McssConfig
from astrape.mcss_training import McssTrainingConfig, train_mcss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    p.add_argument("--projection-path", type=Path, default=Path("checkpoints/teacher_fsq_proj_out.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--run-name", default="mcss")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--probe-samples", type=int, default=1024)
    p.add_argument("--full-validation-every", type=int, default=5)
    p.add_argument("--target-cosine", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    p.add_argument("--log-every", type=int, default=50)

    # Architecture overrides
    p.add_argument("--stem-dim", type=int, default=384)
    p.add_argument("--attn-dim", type=int, default=384)
    p.add_argument("--attn-layers", type=int, default=4)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--attn-context", type=int, default=100)
    p.add_argument("--gru-dim", type=int, default=512)
    p.add_argument("--gru-layers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_config = McssConfig(
        stem_dim=args.stem_dim,
        attn_dim=args.attn_dim,
        attn_layers=args.attn_layers,
        attn_heads=args.attn_heads,
        attn_context=args.attn_context,
        gru_dim=args.gru_dim,
        gru_layers=args.gru_layers,
    )
    train_config = McssTrainingConfig(
        data_dir=args.data_dir,
        projection_path=args.projection_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        probe_samples=args.probe_samples,
        full_validation_every=args.full_validation_every,
        target_cosine=args.target_cosine,
        seed=args.seed,
        resume=args.resume,
        log_every=args.log_every,
    )
    train_mcss(model_config, train_config)


if __name__ == "__main__":
    main()
