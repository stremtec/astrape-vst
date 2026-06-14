#!/usr/bin/env python3
"""Run the causal content and mel models incrementally on cached features."""

import argparse
from pathlib import Path

import numpy as np
import torch

from astrape.checkpoint import load_content_checkpoint
from astrape.mel_decoder import load_mel_decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mel-decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/stream_result.npz"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-mel-frames", type=int, default=2)
    parser.add_argument("--allow-legacy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_mel_frames <= 0:
        raise SystemExit("--chunk-mel-frames must be positive")
    device = torch.device(args.device)
    student, _ = load_content_checkpoint(
        args.checkpoint, device=device, allow_legacy=args.allow_legacy
    )
    decoder = load_mel_decoder(args.mel_decoder, device)
    student.eval()
    decoder.eval()
    with np.load(args.mel) as data:
        mel = torch.from_numpy(data["logmel"]).float().unsqueeze(0).to(device)
    with np.load(args.target) as data:
        global_embedding = (
            torch.from_numpy(data["ge_128"]).float().unsqueeze(0).to(device)
        )

    content_chunks = []
    output_mel_chunks = []
    content_state = None
    decoder_state = None
    with torch.inference_mode():
        for start in range(0, mel.shape[-1], args.chunk_mel_frames):
            content_output, content_state = student.forward_stream(
                mel[:, :, start : start + args.chunk_mel_frames],
                content_state,
            )
            if content_output.content.shape[-1] == 0:
                continue
            decoded, decoder_state = decoder.forward_stream(
                content_output.content.transpose(1, 2),
                global_embedding,
                decoder_state,
            )
            content_chunks.append(content_output.content)
            output_mel_chunks.append(decoded)
        content_output, content_state = student.forward_stream(
            mel[:, :, :0], content_state, flush=True
        )
        if content_output.content.shape[-1]:
            decoded, decoder_state = decoder.forward_stream(
                content_output.content.transpose(1, 2),
                global_embedding,
                decoder_state,
            )
            content_chunks.append(content_output.content)
            output_mel_chunks.append(decoded)
        streamed_content = torch.cat(content_chunks, dim=-1)
        streamed_mel = torch.cat(output_mel_chunks, dim=-1)
        full_content = student(mel).content
        full_mel = decoder(full_content.transpose(1, 2), global_embedding)
    content_error = (streamed_content - full_content).abs().max().item()
    mel_error = (streamed_mel - full_mel).abs().max().item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        content=streamed_content.squeeze(0).cpu().numpy(),
        mel=streamed_mel.squeeze(0).cpu().numpy(),
        content_streaming_max_error=np.float32(content_error),
        mel_streaming_max_error=np.float32(mel_error),
    )
    print(
        f"Saved {args.output} | content max error={content_error:.3e} "
        f"| mel max error={mel_error:.3e}"
    )


if __name__ == "__main__":
    main()
