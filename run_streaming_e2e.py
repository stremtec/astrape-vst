#!/usr/bin/env python3
"""Drive the E2E streaming runtime from an audio file."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from astrape.streaming_pipeline import StreamingVoiceConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--voicebank", type=Path, required=True)
    parser.add_argument("--content-checkpoint", type=Path, required=True)
    parser.add_argument("--wave-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-ms", type=float, default=5.0)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def load_source(path: Path, target_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        audio = resample_poly(
            audio,
            target_rate // divisor,
            sample_rate // divisor,
        )
    return np.asarray(audio, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.chunk_ms <= 0:
        raise SystemExit("--chunk-ms must be positive")
    converter = StreamingVoiceConverter.from_checkpoints(
        args.content_checkpoint,
        args.wave_checkpoint,
        args.voicebank,
        device=args.device,
    )
    chunk_samples = max(
        1,
        round(converter.input_sample_rate * args.chunk_ms / 1000.0),
    )
    if not args.no_warmup:
        converter.warmup(chunk_samples)
    source = load_source(args.input, converter.input_sample_rate)
    output_chunks = []
    first_output_at = None
    first_output_wall = None
    started = time.perf_counter()
    for start in range(0, source.size, chunk_samples):
        chunk = converter.process(
            torch.from_numpy(source[start : start + chunk_samples])
        )
        if chunk.output_samples:
            if first_output_at is None:
                first_output_at = converter.counters.input_samples
                first_output_wall = time.perf_counter() - started
            output_chunks.append(chunk.audio)
    final = converter.flush()
    if final.output_samples:
        output_chunks.append(final.audio)
    elapsed = time.perf_counter() - started
    if not output_chunks:
        raise RuntimeError("Input produced no waveform output")
    output = torch.cat(output_chunks, dim=-1).squeeze(0).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        args.output,
        output,
        converter.output_sample_rate,
        subtype="PCM_16",
    )
    audio_seconds = source.size / converter.input_sample_rate
    first_ms = (
        first_output_at / converter.input_sample_rate * 1000.0
        if first_output_at is not None
        else float("nan")
    )
    first_wall_ms = (
        first_output_wall * 1000.0
        if first_output_wall is not None
        else float("nan")
    )
    print(
        f"Saved {args.output} | input={audio_seconds:.2f}s "
        f"| output={output.size / converter.output_sample_rate:.2f}s "
        f"| first_input={first_ms:.1f}ms "
        f"| first_compute={first_wall_ms:.1f}ms "
        f"| compute_rtf={elapsed / audio_seconds:.3f} "
        f"| tail={converter.counters.unframed_tail_samples} samples"
    )


if __name__ == "__main__":
    main()
