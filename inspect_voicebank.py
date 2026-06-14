#!/usr/bin/env python3
"""Print VoiceBank compatibility and reference-quality metadata."""

import argparse
import json
from pathlib import Path

from astrape.voicebank import VoiceBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("voicebank", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = VoiceBank.load(args.voicebank)
    print(
        json.dumps(
            {
                "embedding_shape": list(bank.global_embedding.shape),
                "embedding_norm": float(bank.global_embedding.norm()),
                "embedding_model": bank.embedding_model,
                "duration_seconds": bank.duration_seconds,
                "source_sample_rate": bank.source_sample_rate,
                "source_path": bank.source_path,
                "reference_sha256": bank.reference_sha256,
                "created_utc": bank.created_utc,
                "peak_amplitude": bank.peak_amplitude,
                "rms_dbfs": bank.rms_dbfs,
                "clipping_fraction": bank.clipping_fraction,
                "active_speech_ratio": bank.active_speech_ratio,
                "dc_offset": bank.dc_offset,
                "quality_warnings": list(bank.quality_warnings),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
