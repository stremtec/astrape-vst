#!/usr/bin/env python3
"""Build a zero-shot VoiceBank from one continuous reference recording."""

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from astrape.voicebank import (
    MIN_REFERENCE_SECONDS,
    MIO_GLOBAL_MODEL,
    VoiceBank,
    analyze_reference,
)


TEACHER_SAMPLE_RATE = 44100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from miocodec.model import MioCodecModel
    except ModuleNotFoundError as error:
        raise SystemExit("MioCodec is required to build a VoiceBank") from error
    audio, sample_rate = sf.read(args.reference, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    duration_seconds = len(audio) / sample_rate
    if duration_seconds < MIN_REFERENCE_SECONDS:
        raise SystemExit(
            f"Reference is {duration_seconds:.2f}s; "
            f"VoiceBank requires at least {MIN_REFERENCE_SECONDS:.1f}s"
        )
    quality = analyze_reference(audio, sample_rate)
    if sample_rate != TEACHER_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TEACHER_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TEACHER_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
    device = torch.device(args.device)
    teacher = MioCodecModel.from_pretrained(
        MIO_GLOBAL_MODEL
    ).to(device).eval()
    waveform = torch.from_numpy(np.asarray(audio)).float().to(device)
    with torch.inference_mode():
        features = teacher.encode(
            waveform,
            return_content=False,
            return_global=True,
        )
    bank = VoiceBank(
        global_embedding=features.global_embedding.detach().cpu(),
        duration_seconds=duration_seconds,
        source_sample_rate=sample_rate,
        source_path=str(args.reference.resolve()),
        embedding_model=MIO_GLOBAL_MODEL,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),
        peak_amplitude=quality.peak_amplitude,
        rms_dbfs=quality.rms_dbfs,
        clipping_fraction=quality.clipping_fraction,
        active_speech_ratio=quality.active_speech_ratio,
        dc_offset=quality.dc_offset,
        quality_warnings=quality.warnings,
    )
    bank.save(args.output)
    print(
        f"Saved {args.output} from one {duration_seconds:.2f}s reference "
        f"| embedding={tuple(bank.global_embedding.shape)} "
        f"| rms={bank.rms_dbfs:.1f}dBFS "
        f"| active={bank.active_speech_ratio:.1%}"
    )
    if bank.quality_warnings:
        print(f"Quality warnings: {', '.join(bank.quality_warnings)}")


if __name__ == "__main__":
    main()
