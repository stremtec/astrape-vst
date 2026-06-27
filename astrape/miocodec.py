"""MioCodec bridge — load the frozen MioCodec model, audio IO, speaker extraction.

Single home for what `eval_mcs_trans_audio.py` and the cache/voicebank scripts all
needed (was duplicated as load_mio / load_wave / write_wave / SAMPLE_RATE, plus the
chunked speaker-embedding extraction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

_mio = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mio) not in sys.path:
    sys.path.insert(0, str(_mio))

SAMPLE_RATE = 44100

MIO_CONFIG = Path(
    "/Users/asill/.cache/huggingface/hub/models--Aratako--MioCodec-25Hz-44.1kHz-v2/"
    "snapshots/67faba34153fe74e6665991c432a7327e23c5c1c/config.yaml"
)
MIO_WEIGHTS = Path(
    "/Users/asill/.cache/huggingface/hub/models--Aratako--MioCodec-25Hz-44.1kHz-v2/"
    "snapshots/67faba34153fe74e6665991c432a7327e23c5c1c/model.safetensors"
)


def load_wave(path: Path, sample_rate: int, max_seconds: float | None = None) -> torch.Tensor:
    import soundfile as sf
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.asarray(data))
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
    except Exception:
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if max_seconds is not None:
        wav = wav[: int(round(max_seconds * sample_rate))]
    return wav.contiguous().float()


def write_wave(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.detach().cpu().float().numpy(), sample_rate)


def load_mio(device):
    import logging
    import miocodec
    logging.disable(logging.INFO)
    model = miocodec.MioCodecModel.from_pretrained(
        config_path=str(MIO_CONFIG), weights_path=str(MIO_WEIGHTS),
    )
    return model.to(device).eval()


@torch.no_grad()
def extract_chunk_embeddings(mio, wave: torch.Tensor, sr: int = SAMPLE_RATE,
                             chunk_s: float = 8.0, hop_s: float = 4.0,
                             energy_gate_db: float = -45.0,
                             device: str = "cpu") -> list[torch.Tensor]:
    """MioCodec global embeddings over energy-gated windows of `wave` (1-D)."""
    n = wave.shape[0]
    chunk, hop = int(chunk_s * sr), int(hop_s * sr)
    starts = [0] if n <= chunk else list(range(0, n - chunk + 1, hop))
    embs: list[torch.Tensor] = []
    for st in starts:
        seg = wave[st:st + chunk]
        rms = seg.pow(2).mean().clamp(min=1e-12).sqrt()
        if 20.0 * torch.log10(rms) < energy_gate_db:        # skip near-silent windows
            continue
        f = mio.encode(seg.to(device), return_content=False, return_global=True)
        embs.append(f.global_embedding.squeeze(0).cpu())
    if not embs:                                            # all gated → whole clip
        f = mio.encode(wave.to(device), return_content=False, return_global=True)
        embs.append(f.global_embedding.squeeze(0).cpu())
    return embs
