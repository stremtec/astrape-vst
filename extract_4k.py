#!/usr/bin/env python3
"""Extract MioCodec teacher targets and aligned log-mel inputs."""

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from astrape.audio import StreamingLogMel


TEACHER_SAMPLE_RATE = 44100
MEL_SAMPLE_RATE = 16000


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        audio,
        target_rate // divisor,
        source_rate // divisor,
    )


def atomic_savez(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vctk-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument("--mel-dir", type=Path, default=Path("data/mio_4k_mel"))
    parser.add_argument("--utterances-per-speaker", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from miocodec.model import MioCodecModel
    except ModuleNotFoundError as error:
        raise SystemExit(
            "MioCodec is required for extraction. Install the MioCodec package "
            "that provides miocodec.model.MioCodecModel."
        ) from error

    if not args.vctk_root.is_dir():
        raise SystemExit(f"VCTK root does not exist: {args.vctk_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mel_dir.mkdir(parents=True, exist_ok=True)
    teacher = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-44.1kHz-v2"
    ).eval()
    mel_extractor = StreamingLogMel()
    rng = np.random.default_rng(args.seed)

    selected: list[tuple[str, Path]] = []
    speaker_dirs = sorted(
        path for path in args.vctk_root.iterdir() if path.is_dir() and path.name.startswith("p")
    )
    for speaker_dir in speaker_dirs:
        files = sorted(speaker_dir.glob(f"{speaker_dir.name}_*_mic1.flac"))
        if len(files) > args.utterances_per_speaker:
            chosen = rng.choice(
                len(files), size=args.utterances_per_speaker, replace=False
            )
            files = [files[index] for index in sorted(chosen)]
        selected.extend((speaker_dir.name, path) for path in files)

    speakers = []
    utterance_ids = []
    source_files = []
    for index, (speaker, path) in enumerate(selected):
        output_path = args.output_dir / f"s_{index:05d}.npz"
        mel_path = args.mel_dir / f"m_{index:05d}.npz"
        speakers.append(speaker)
        utterance_ids.append(path.stem.split("_")[1])
        source_files.append(str(path))
        if args.resume and output_path.exists() and mel_path.exists():
            continue

        audio, sample_rate = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample(audio, sample_rate, TEACHER_SAMPLE_RATE)
        audio = audio[: int(TEACHER_SAMPLE_RATE * args.max_seconds)]
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        padding = teacher._calculate_waveform_padding(len(audio))
        with torch.inference_mode():
            features = teacher.encode(
                waveform, return_content=True, return_global=True
            )
            local_ssl, _ = teacher.forward_ssl_features(waveform, padding=padding)
            local_encoded = teacher.local_encoder(local_ssl)
            local_encoded = teacher.conv_downsample(
                local_encoded.transpose(1, 2)
            ).transpose(1, 2)
            audio_16k = resample(audio, TEACHER_SAMPLE_RATE, MEL_SAMPLE_RATE)
            logmel = mel_extractor(torch.from_numpy(audio_16k).float()).squeeze(0)
        expected_content_frames = (logmel.shape[1] + 1) // 2
        aligned_content_frames = min(
            expected_content_frames,
            features.content_embedding.shape[0],
            features.content_token_indices.shape[0],
            local_encoded.shape[1],
        )
        if aligned_content_frames < expected_content_frames:
            logmel = logmel[:, : 2 * aligned_content_frames]
        atomic_savez(
            output_path,
            ce_768=features.content_embedding[:aligned_content_frames].cpu().numpy(),
            ct=features.content_token_indices[:aligned_content_frames].cpu().numpy(),
            ge_128=features.global_embedding.cpu().numpy(),
            audio=audio.astype(np.float32),
            pre_fsq_768=local_encoded[
                :, :aligned_content_frames
            ].squeeze(0).cpu().numpy(),
            aligned_frames_25hz=np.asarray(aligned_content_frames, dtype=np.int64),
        )
        atomic_savez(mel_path, logmel=logmel.cpu().numpy())
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{len(selected)}")

    atomic_savez(
        args.output_dir / "meta.npz",
        spk_names=np.asarray(speakers),
        utterance_ids=np.asarray(utterance_ids),
        source_files=np.asarray(source_files),
        n_samples=np.asarray(len(selected), dtype=np.int64),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    print(f"Done: {len(selected)} samples, {len(speaker_dirs)} speakers")


if __name__ == "__main__":
    main()
