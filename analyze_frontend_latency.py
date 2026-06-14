#!/usr/bin/env python3
"""Compare shorter causal log-mel windows against the 512-sample baseline."""

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from astrape.audio import StreamingLogMel
from astrape.checkpoint import load_content_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[512, 400, 320, 256],
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_audio(path: Path, seconds: float) -> torch.Tensor:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor,
        )
    return torch.from_numpy(
        np.asarray(audio[: int(seconds * 16000)])
    ).float()


def main() -> None:
    args = parse_args()
    if 512 not in args.windows:
        raise SystemExit("--windows must include the 512-sample baseline")
    files = sorted(args.audio_root.glob("p*/*_mic1.flac"))
    if not files:
        raise SystemExit(f"No VCTK mic1 files found under {args.audio_root}")
    selected = [
        files[index]
        for index in np.linspace(
            0,
            len(files) - 1,
            min(args.samples, len(files)),
            dtype=int,
        )
    ]
    extractors = {
        window: StreamingLogMel(n_fft=window) for window in args.windows
    }
    mel_cosines = {window: [] for window in args.windows if window != 512}
    mel_l1 = {window: [] for window in args.windows if window != 512}
    student_cosines = {
        window: [] for window in args.windows if window != 512
    }
    model = None
    device = torch.device(args.device)
    if args.checkpoint is not None:
        model, _ = load_content_checkpoint(args.checkpoint, device=device)
        model.eval()

    with torch.inference_mode():
        for path in selected:
            waveform = load_audio(path, args.seconds)
            features = {
                window: extractor(waveform).squeeze(0).transpose(0, 1)
                for window, extractor in extractors.items()
            }
            baseline = features[512]
            baseline_student = (
                model(baseline.transpose(0, 1).unsqueeze(0).to(device)).content
                if model is not None
                else None
            )
            for window, alternative in features.items():
                if window == 512:
                    continue
                length = min(len(baseline), len(alternative))
                left = baseline[:length]
                right = alternative[:length]
                left = left - left.mean(dim=-1, keepdim=True)
                right = right - right.mean(dim=-1, keepdim=True)
                mel_cosines[window].append(
                    F.cosine_similarity(left, right, dim=-1)
                )
                mel_l1[window].append(
                    (baseline[:length] - alternative[:length])
                    .abs()
                    .mean(dim=-1)
                )
                if model is not None and baseline_student is not None:
                    output = model(
                        alternative.transpose(0, 1).unsqueeze(0).to(device)
                    ).content
                    output_length = min(
                        baseline_student.shape[-1],
                        output.shape[-1],
                    )
                    student_cosines[window].append(
                        F.cosine_similarity(
                            baseline_student[:, :, :output_length],
                            output[:, :, :output_length],
                            dim=1,
                        ).flatten().cpu()
                    )

    print(
        f"{'n_fft':>6} {'collect':>9} {'mel cos':>9} "
        f"{'mel p05':>9} {'mel L1':>9} {'student cos':>12}"
    )
    for window in args.windows:
        if window == 512:
            continue
        cosine = torch.cat(mel_cosines[window])
        l1 = torch.cat(mel_l1[window])
        student_text = "-"
        if student_cosines[window]:
            student_text = f"{torch.cat(student_cosines[window]).mean():.4f}"
        print(
            f"{window:6d} {window / 16:.1f}ms "
            f"{cosine.mean():9.4f} {torch.quantile(cosine, 0.05):9.4f} "
            f"{l1.mean():9.4f} {student_text:>12}"
        )


if __name__ == "__main__":
    main()
