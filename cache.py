"""Unified caching CLI (replaces cache_wavlm_*.py + cache_speaker_embeddings.py).

  --what wavlm     WavLM L4 raw 200Hz features → <data>/wavlm_L4_200hz/   (encoder frontend)
                   5 causal conv layers @16kHz, saved (T, 512) @200Hz, 10ms delay.
  --what speakers  per-speaker MioCodec global centroids → <data>/spk_centroids.npz (decoder)
                   chunked + energy-gated + averaged over several utterances.

Examples:
  .venv/bin/python cache.py --what wavlm --limit 0
  .venv/bin/python cache.py --what speakers --utts-per-speaker 8
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from astrape.miocodec import load_mio, load_wave, SAMPLE_RATE, extract_chunk_embeddings


def cache_wavlm(args):
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); srcs = meta["source_files"][:n].astype(str)
    end = n if args.limit == 0 else min(n, args.start + args.limit)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "wavlm_L4_200hz"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching WavLM L4 raw (200Hz): {args.start}..{end-1} ({end-args.start} total)")

    mio = load_mio(args.device).eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor
    for i in range(args.start, end):
        out_path = out_dir / f"s_{i:05d}.npy"
        if out_path.exists():
            continue
        wav = load_wave(Path(str(srcs[i])), SAMPLE_RATE, max_seconds=args.max_s)
        wav_16 = torchaudio.functional.resample(wav.unsqueeze(0), SAMPLE_RATE, 16000).squeeze(0)
        with torch.no_grad():
            x = wav_16.unsqueeze(0)
            for layer_idx in range(5):
                layer = fe.conv_layers[layer_idx]
                x = layer.conv(x)
                if hasattr(layer, "layer_norm") and layer.layer_norm is not None:
                    x = (layer.layer_norm(x.unsqueeze(0)).squeeze(0)
                         if x.dim() == 2 else layer.layer_norm(x))
                x = F.gelu(x)
        cnn = x.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)  # (T, 512)
        np.save(out_path, cnn)
        if (i - args.start) % 100 == 0:
            print(f"{i - args.start}/{end - args.start}")
    print(f"Done: {end - args.start} samples")


def cache_speakers(args):
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"])
    spk_names = meta["spk_names"][:n].astype(str); src = meta["source_files"][:n].astype(str)
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(spk_names):
        by_spk[s].append(i)
    speakers = sorted(by_spk)
    print(f"{len(speakers)} speakers, up to {args.utts_per_speaker} utterances each", flush=True)

    mio = load_mio(args.device).eval()
    rng = np.random.default_rng(0)
    embeddings = []
    for j, spk in enumerate(speakers):
        chosen = rng.choice(by_spk[spk], size=min(args.utts_per_speaker, len(by_spk[spk])), replace=False)
        chunks: list[torch.Tensor] = []
        for i in chosen:
            w = load_wave(Path(src[int(i)]), SAMPLE_RATE)
            chunks += extract_chunk_embeddings(mio, w, SAMPLE_RATE, device=args.device)
        embeddings.append(torch.stack(chunks).mean(0).numpy())
        if j % 10 == 0:
            print(f"  {j}/{len(speakers)} ({spk})", flush=True)

    out = Path(args.out_dir) if args.out_dir else data_dir / "spk_centroids.npz"
    np.savez(out, speakers=np.array(speakers), embeddings=np.stack(embeddings).astype(np.float32))
    print(f"Wrote {out}: {len(speakers)} speaker centroids (128-d)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["wavlm", "speakers"], required=True)
    ap.add_argument("--data-dir", default="data/mio_vctk_full_compact")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    # wavlm
    ap.add_argument("--max-s", type=float, default=6.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    # speakers
    ap.add_argument("--utts-per-speaker", type=int, default=8)
    args = ap.parse_args()
    (cache_wavlm if args.what == "wavlm" else cache_speakers)(args)


if __name__ == "__main__":
    main()
