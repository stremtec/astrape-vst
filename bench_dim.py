#!/usr/bin/env python3
"""Synchronized full-sequence and streaming latency benchmark."""

import argparse
import math
import statistics
import time

import torch

from astrape.model import ContentStudent
from tiers import TIERS


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call, device: torch.device, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        call()
    synchronize(device)
    timings = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        call()
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIERS), default=list(TIERS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mel_frames = int(round(args.seconds * 50))
    mel_frames += mel_frames % 2
    x = torch.randn(1, 80, mel_frames, device=device)
    print(
        f"{'Tier':>8} {'Params':>9} {'Full p50':>10} {'Full p95':>10} "
        f"{'Stream/frame':>13} {'RTF':>8}"
    )
    for name in args.tiers:
        tier = TIERS[name]
        model = ContentStudent(tier.model).to(device).eval()
        with torch.inference_mode():
            full = measure(
                lambda: model(x),
                device,
                args.warmup,
                args.repeats,
            )

            def stream_once() -> None:
                state = None
                for start in range(0, mel_frames, 2):
                    _, state = model.forward_stream(x[:, :, start : start + 2], state)

            streamed = measure(
                stream_once,
                device,
                max(2, args.warmup // 5),
                max(5, args.repeats // 3),
            )
        params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        full_p50 = statistics.median(full)
        full_p95 = sorted(full)[min(len(full) - 1, math.ceil(len(full) * 0.95) - 1)]
        content_frames = mel_frames // 2
        stream_per_frame = statistics.median(streamed) / content_frames
        rtf = full_p50 / (args.seconds * 1000)
        print(
            f"{name:>8} {params:8.1f}M {full_p50:9.1f}ms {full_p95:9.1f}ms "
            f"{stream_per_frame:12.2f}ms {rtf:8.3f}"
        )


if __name__ == "__main__":
    main()
