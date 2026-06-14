#!/usr/bin/env python3
"""Train the 768x10 FSQ-aware student with original-to-teacher curriculum."""

import argparse
from pathlib import Path

from astrape.curriculum import CurriculumConfig, train_curriculum
from astrape.model import ContentStudentConfig
from astrape.text import VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument("--mel-dir", type=Path, default=Path("data/mio_4k_mel"))
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument(
        "--fsq-projection",
        type=Path,
        default=Path("checkpoints/teacher_fsq_proj_out.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="content_student_768x10_fsq")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--original-epochs", type=int, default=5)
    parser.add_argument("--blend-epochs", type=int, default=10)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--attention-context-frames", type=int, default=100)
    parser.add_argument("--target-cosine", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ContentStudentConfig(
        hidden=768,
        n_layers=10,
        n_heads=12,
        auxiliary_prefsq=True,
        structured_fsq=True,
        text_vocab_size=VOCAB_SIZE,
        max_attention_context=args.attention_context_frames,
    )
    train_curriculum(
        model,
        CurriculumConfig(
            data_dir=args.data_dir,
            mel_dir=args.mel_dir,
            audio_root=args.audio_root,
            transcript_root=args.transcript_root,
            fsq_projection=args.fsq_projection,
            output_dir=args.output_dir,
            run_name=args.run_name,
            device=args.device,
            batch_size=args.batch_size,
            original_epochs=args.original_epochs,
            blend_epochs=args.blend_epochs,
            teacher_epochs=args.teacher_epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            seed=args.seed,
            num_workers=args.num_workers,
            resume=args.resume,
            target_cosine=args.target_cosine,
            log_every=args.log_every,
        ),
    )


if __name__ == "__main__":
    main()
