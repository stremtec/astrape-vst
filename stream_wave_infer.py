#!/usr/bin/env python3
"""Run cached mel through the causal content and direct waveform decoders."""

import argparse
from pathlib import Path

import numpy as np
import torch

from astrape.checkpoint import load_content_checkpoint
from astrape.voicebank import VoiceBank
from astrape.wave_decoder import load_wave_decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel", type=Path, required=True)
    parser.add_argument("--voicebank", type=Path, required=True)
    parser.add_argument("--content-checkpoint", type=Path, required=True)
    parser.add_argument("--wave-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/direct.wav.npy"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-mel-frames", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_mel_frames <= 0:
        raise SystemExit("--chunk-mel-frames must be positive")
    device = torch.device(args.device)
    content_model, _ = load_content_checkpoint(
        args.content_checkpoint,
        device=device,
    )
    wave_model = load_wave_decoder(args.wave_checkpoint, device)
    content_model.eval()
    wave_model.eval()
    bank = VoiceBank.load(args.voicebank)
    global_embedding = bank.global_embedding.unsqueeze(0).to(device)
    with np.load(args.mel) as data:
        mel = torch.from_numpy(data["logmel"]).float().unsqueeze(0).to(device)
    content_state = None
    wave_state = None
    waveform_chunks = []
    with torch.inference_mode():
        for start in range(0, mel.shape[-1], args.chunk_mel_frames):
            content, content_state = content_model.forward_stream(
                mel[:, :, start : start + args.chunk_mel_frames],
                content_state,
            )
            if content.content.shape[-1] == 0:
                continue
            waveform, wave_state = wave_model.forward_stream(
                content.content.transpose(1, 2),
                global_embedding,
                wave_state,
            )
            waveform_chunks.append(waveform.cpu())
        content, content_state = content_model.forward_stream(
            mel[:, :, :0],
            content_state,
            flush=True,
        )
        if content.content.shape[-1]:
            waveform, wave_state = wave_model.forward_stream(
                content.content.transpose(1, 2),
                global_embedding,
                wave_state,
            )
            waveform_chunks.append(waveform.cpu())
    if not waveform_chunks:
        raise RuntimeError("Input mel produced no content frames")
    waveform = torch.cat(waveform_chunks, dim=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, waveform.squeeze(0).numpy())
    print(
        f"Saved {args.output} | samples={waveform.shape[-1]} "
        f"| seconds={waveform.shape[-1] / wave_model.config.sample_rate:.2f}"
    )


if __name__ == "__main__":
    main()
