"""Inference script for Codex VC."""

from __future__ import annotations

import argparse
import sys

import soundfile as sf
import torch
from moshi.models import loaders
from pathlib import Path
from scipy import signal

from codex_vc.model import CodeGenerator, convert

# ── Constants ──
SR = 24000
STRIDE = 1920
SPK_DIM = 256


def load_audio(path: str, duration: float | None = None) -> torch.Tensor:
    """Load and preprocess audio file.

    Args:
        path: Path to audio file (any format supported by soundfile).
        duration: If set, trim to this many seconds.

    Returns:
        Tensor of shape (1, 1, T).
    """
    data, orig_sr = sf.read(path)
    if orig_sr != SR:
        data = signal.resample(data, int(len(data) * SR / orig_sr), axis=0)

    if duration is not None:
        L = int(duration * SR)
        L -= L % STRIDE
        data = data[:L]
    else:
        L = len(data) - (len(data) % STRIDE)
        data = data[:L]

    if data.ndim > 1:
        data = data.mean(axis=1)

    return torch.from_numpy(data.copy()).float().unsqueeze(0).unsqueeze(0)


def load_speaker_embedding(path: str, speaker_id: str) -> torch.Tensor:
    """Load pre-computed speaker embedding.

    Args:
        path: Path to the speaker embedding .pt file.
        speaker_id: Speaker key (e.g., 'p225').

    Returns:
        Tensor of shape (1, 256).
    """
    emb_dict = torch.load(path, weights_only=True)
    if speaker_id not in emb_dict:
        available = sorted(emb_dict.keys())
        raise KeyError(
            f"Speaker '{speaker_id}' not found. Available: {available[:10]}..."
        )
    return emb_dict[speaker_id].unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Codex VC inference")
    parser.add_argument("--source", required=True, help="Source audio file")
    parser.add_argument("--target-speaker", required=True, help="Target speaker ID (e.g., p226)")
    parser.add_argument("--spk-emb", default="runs/vctk_full_spk.pt",
                        help="Speaker embeddings file")
    parser.add_argument("--model", default="runs/codex_model.pt",
                        help="Model checkpoint")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Audio duration in seconds")
    args = parser.parse_args()

    # Load Mimi (frozen)
    print("Loading Mimi...")
    mimi = loaders.get_mimi(
        Path.home() / ".cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/"
        "snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/"
        "tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    for p in mimi.parameters():
        p.requires_grad_(False)

    # Load model
    print("Loading model...")
    model = CodeGenerator()
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()

    # Load speaker embedding
    spk_emb = load_speaker_embedding(args.spk_emb, args.target_speaker)

    # Load source audio
    src_audio = load_audio(args.source, duration=args.duration)

    # Convert
    print("Converting...")
    vc_audio = convert(model, mimi, src_audio, spk_emb)

    # Save
    T_out = min(vc_audio.shape[2], src_audio.shape[2])
    sf.write(args.output, vc_audio[0, 0, :T_out].numpy(), SR)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
