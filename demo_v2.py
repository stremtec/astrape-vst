#!/usr/bin/env python3
"""Offline teacher-decoder demo using the current content student."""

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from astrape.audio import StreamingLogMel
from astrape.checkpoint import load_content_checkpoint
from astrape.mel_decoder import load_mel_decoder


OUTPUT_SAMPLE_RATE = 44100


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        audio, target_rate // divisor, source_rate // divisor
    )


def load_audio(path: Path, sample_rate: int, seconds: float) -> np.ndarray:
    audio, source_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample(audio, source_rate, sample_rate)
    return audio[: int(sample_rate * seconds)].astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/content_student_v3_4k_causal.best.pt"),
    )
    parser.add_argument("--mel-decoder", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--allow-legacy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from miocodec.model import MioCodecModel
    except ModuleNotFoundError as error:
        raise SystemExit(
            "MioCodec is required for waveform decoding. Install the package "
            "that provides miocodec.model.MioCodecModel."
        ) from error
    device = torch.device(args.device)
    student, _ = load_content_checkpoint(
        args.checkpoint,
        device=device,
        allow_legacy=args.allow_legacy,
    )
    student.eval()
    teacher = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-44.1kHz-v2"
    ).eval()
    source = load_audio(args.source, OUTPUT_SAMPLE_RATE, args.seconds)
    reference = load_audio(args.reference, OUTPUT_SAMPLE_RATE, args.seconds)
    source_tensor = torch.from_numpy(source).unsqueeze(0)
    reference_tensor = torch.from_numpy(reference).unsqueeze(0)
    source_16k = resample(source, OUTPUT_SAMPLE_RATE, 16000)
    logmel = StreamingLogMel()(torch.from_numpy(source_16k).float()).to(device)

    with torch.inference_mode():
        source_features = teacher.encode(
            source_tensor, return_content=True, return_global=False
        )
        reference_features = teacher.encode(
            reference_tensor, return_content=False, return_global=True
        )
        content = student(logmel).content.squeeze(0).transpose(0, 1).cpu()
        teacher_waveform = teacher.decode(
            global_embedding=reference_features.global_embedding,
            content_token_indices=source_features.content_token_indices,
            target_audio_length=len(source),
        )
        student_waveform = teacher.decode(
            global_embedding=reference_features.global_embedding,
            content_embedding=content,
            target_audio_length=len(source),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_dir / "source.wav", source, OUTPUT_SAMPLE_RATE)
    sf.write(
        args.output_dir / "teacher_vc.wav",
        teacher_waveform.cpu().numpy()[: len(source)],
        OUTPUT_SAMPLE_RATE,
    )
    sf.write(
        args.output_dir / "student_vc.wav",
        student_waveform.cpu().numpy()[: len(source)],
        OUTPUT_SAMPLE_RATE,
    )
    if args.mel_decoder is not None:
        decoder = load_mel_decoder(args.mel_decoder, device).eval()
        global_embedding = reference_features.global_embedding.unsqueeze(0).to(device)
        predicted_mel = decoder(
            content.unsqueeze(0).to(device), global_embedding
        )
        np.save(
            args.output_dir / "student_mel.npy",
            predicted_mel.squeeze(0).cpu().numpy(),
        )
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
