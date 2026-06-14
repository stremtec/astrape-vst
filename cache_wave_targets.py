#!/usr/bin/env python3
"""Cache MioCodec teacher reconstructions for waveform decoder training."""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mio_4k_teacher_wave"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from miocodec.model import MioCodecModel
    except ModuleNotFoundError as error:
        raise SystemExit("MioCodec is required to cache wave targets") from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.data_dir / "meta.npz") as metadata:
        count = int(metadata["n_samples"])
    device = torch.device(args.device)
    teacher = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-44.1kHz-v2"
    ).to(device).eval()
    for index in range(count):
        output_path = args.output_dir / f"w_{index:05d}.npz"
        if args.resume and output_path.exists():
            continue
        with np.load(args.data_dir / f"s_{index:05d}.npz") as data:
            tokens = torch.from_numpy(data["ct"]).long().to(device)
            global_embedding = (
                torch.from_numpy(data["ge_128"]).float().to(device)
            )
            target_length = len(data["audio"])
        with torch.inference_mode():
            waveform = teacher.decode(
                global_embedding=global_embedding,
                content_token_indices=tokens,
                target_audio_length=target_length,
            )
        temporary = output_path.with_name(output_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            waveform=waveform.detach().cpu().numpy().astype(np.float32),
        )
        temporary.replace(output_path)
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{count}", flush=True)


if __name__ == "__main__":
    main()
