"""Build a high-quality .astrape voicebank from target-speaker audio.

The inline extraction in the eval scripts is a SINGLE `mio.encode` forward over
the first ≤5 s of ONE clip (`eval_q2d2_vc.py`, `eval_mcs_trans_audio.py`). The
speaker embedding (MioCodec global branch: WavLM L1–L2 → ConvNeXt → attentive
stats pool) is more representative/stable when fed:

  - the FULL reference (not truncated to 5 s),
  - CHUNKED + ENERGY-GATED windows (skip silence) averaged over the clip,
  - multiple utterances (enrollment) — `audio` accepts several files.

All embeddings live in MioCodec's global space (same as the decoder's training
condition), so the chunk-mean is a cleaner centroid, not a distribution shift.

Usage:
  .venv/bin/python build_voicebank.py ref1.wav [ref2.wav ...] -o p225.astrape
"""

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from astrape.miocodec import load_mio, load_wave, SAMPLE_RATE, extract_chunk_embeddings
from astrape.voicebank import VoiceBank, analyze_reference, MIN_REFERENCE_SECONDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="target-speaker reference file(s)")
    ap.add_argument("-o", "--out", required=True, help="output .astrape path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk-s", type=float, default=8.0)
    ap.add_argument("--hop-s", type=float, default=4.0)
    ap.add_argument("--energy-gate-db", type=float, default=-45.0)
    args = ap.parse_args()

    mio = load_mio(args.device).eval()

    waves, all_embs, total_s = [], [], 0.0
    for p in args.audio:
        w = load_wave(Path(p), SAMPLE_RATE)                # FULL clip (no 5 s cap)
        waves.append(w)
        total_s += w.shape[0] / SAMPLE_RATE
        all_embs.extend(extract_chunk_embeddings(
            mio, w, SAMPLE_RATE, args.chunk_s, args.hop_s, args.energy_gate_db, args.device))

    if total_s < MIN_REFERENCE_SECONDS:
        raise SystemExit(f"Need ≥{MIN_REFERENCE_SECONDS:.0f}s of reference, got {total_s:.1f}s")

    embedding = torch.stack(all_embs).mean(0)              # centroid over all windows/files

    allwav = torch.cat(waves).numpy().astype(np.float32)
    q = analyze_reference(allwav, SAMPLE_RATE)
    bank = VoiceBank(
        global_embedding=embedding,
        duration_seconds=float(total_s),
        source_sample_rate=SAMPLE_RATE,
        source_path=str(args.audio[0]),
        reference_sha256=hashlib.sha256(allwav.tobytes()).hexdigest(),
        created_utc=datetime.datetime.utcnow().isoformat() + "Z",
        peak_amplitude=q.peak_amplitude, rms_dbfs=q.rms_dbfs,
        clipping_fraction=q.clipping_fraction, active_speech_ratio=q.active_speech_ratio,
        dc_offset=q.dc_offset, quality_warnings=q.warnings,
    )
    out = bank.save(args.out)
    print(f"Wrote {out}  ({len(all_embs)} windows over {total_s:.1f}s from {len(args.audio)} file(s))")
    if q.warnings:
        print(f"  ⚠ reference quality: {', '.join(q.warnings)}")


if __name__ == "__main__":
    main()
