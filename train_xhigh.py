#!/usr/bin/env python3
"""Train a configured quality tier with the shared causal architecture."""

import argparse
from dataclasses import replace
from pathlib import Path

from astrape.training import TrainingConfig, train_content_student
from tiers import TIERS, get_tier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=sorted(TIERS), default="xhigh")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument("--mel-dir", type=Path, default=Path("data/mio_4k_mel"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--device",
        default="mps",
        choices=("cpu", "mps", "cuda"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-mel-frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--import-legacy", type=Path)
    parser.add_argument(
        "--safe-convs",
        action="store_true",
        help="Use unfold+matmul convolutions for older MPS runtimes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tier = get_tier(args.tier)
    model_config = replace(tier.model, safe_convs=args.safe_convs)
    train_content_student(
        model_config,
        TrainingConfig(
            data_dir=args.data_dir,
            mel_dir=args.mel_dir,
            output_dir=args.output_dir,
            run_name=f"content_student_{args.tier}_causal",
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs or tier.epochs,
            learning_rate=args.learning_rate or tier.learning_rate,
            max_mel_frames=args.max_mel_frames,
            seed=args.seed,
            resume=args.resume,
            import_legacy=args.import_legacy,
        ),
    )


if __name__ == "__main__":
    main()
