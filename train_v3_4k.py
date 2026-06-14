#!/usr/bin/env python3
"""Train the standard strictly causal content student."""

import argparse
from pathlib import Path

from astrape.model import ContentStudentConfig
from astrape.training import TrainingConfig, train_content_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument("--mel-dir", type=Path, default=Path("data/mio_4k_mel"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="content_student_v3_4k_causal")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-mel-frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--import-legacy", type=Path)
    parser.add_argument("--auxiliary-prefsq", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ContentStudentConfig(auxiliary_prefsq=args.auxiliary_prefsq)
    training = TrainingConfig(
        data_dir=args.data_dir,
        mel_dir=args.mel_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_mel_frames=args.max_mel_frames,
        seed=args.seed,
        num_workers=args.num_workers,
        resume=args.resume,
        import_legacy=args.import_legacy,
    )
    train_content_student(model, training)


if __name__ == "__main__":
    main()
