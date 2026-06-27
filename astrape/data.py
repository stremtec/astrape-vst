"""Decoder-side dataset + audio helpers (Phase-0 / v5 decoder training).

Moved out of the old `train_decoder.py` so the training script stays a thin CLI.
`Phase0Dataset` loads the WavLM cache (rate-aware), the original audio, and a
per-speaker centroid embedding; `gaussian_blur_wave` is the cdecoder teacher
smoothing.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

S = 44100  # decoder / audio sample rate


def gaussian_blur_wave(wave: torch.Tensor, sigma_ms: float = 2.0) -> torch.Tensor:
    """Causal (left-padded) time-domain Gaussian blur of the teacher waveform."""
    if sigma_ms <= 0:
        return wave
    sigma_samples = int(sigma_ms / 1000 * S)
    if sigma_samples < 1:
        return wave
    radius = min(4 * sigma_samples, 512)
    kernel_size = 2 * radius + 1
    t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=wave.device)
    kernel = torch.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    wave_3d = wave.unsqueeze(1)  # (B, 1, T)
    padded = F.pad(wave_3d, (kernel_size - 1, 0), mode='reflect')  # left-only → causal
    blurred = F.conv1d(padded, kernel.expand(1, 1, -1))
    return blurred.squeeze(1)


class Phase0Dataset(Dataset):
    """Loads WavLM CNN features (rate-aware), original audio, and speaker embedding."""

    def __init__(self, indices, wavlm_dir, source_files, spk_embeds,
                 spk_names, max_content_frames=50, seed=42, wavlm_rate=50,
                 speaker_emb_map=None):
        self.indices = [int(i) for i in indices]
        self.wavlm_dir = Path(wavlm_dir)
        self.source_files = source_files
        self.spk_names = spk_names       # (N,) array of speaker IDs (e.g., 'p315')
        self.spk_embeds = spk_embeds     # (N_spk, 128) float32 tensor (legacy path)
        self.max_cf = max_content_frames
        # WavLM frames per 25Hz content frame: 50Hz cache=2, 200Hz L4 cache=8.
        self.R = max(1, wavlm_rate // 25)
        self.max_wavlm = max_content_frames * self.R
        self.max_samples = max_content_frames * 1764
        self.rng = random.Random(seed)

        # Preferred: per-speaker centroid map {speaker_id: (128,) tensor}
        # (from cache_speaker_embeddings.py — covers ALL speakers).
        self.speaker_emb_map = speaker_emb_map
        if speaker_emb_map is not None:
            self._spk_fallback = torch.stack(list(speaker_emb_map.values())).mean(0)
            print(f"  Speaker mapping: {len(speaker_emb_map)} per-speaker centroids")
        else:
            # Legacy: speaker ID → first-occurrence index in spk_embeds.
            n_emb = len(self.spk_embeds)
            self.spk_to_emb = {}
            for i in range(min(n_emb, len(spk_names))):
                spk_id = str(spk_names[i])
                if spk_id not in self.spk_to_emb:
                    self.spk_to_emb[spk_id] = i
            print(f"  Speaker mapping: {len(self.spk_to_emb)} unique speakers → "
                  f"{n_emb} embeddings")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import soundfile as sf
        idx = self.indices[i]

        # ── WavLM features ──
        wl_path = self.wavlm_dir / f"s_{idx:05d}.npy"
        wavlm = np.load(wl_path, allow_pickle=False).astype(np.float32)  # (T, 512)
        wavlm = torch.from_numpy(wavlm)

        # Crop content-aligned: 1 content frame = R wavlm frames = 1764 audio samples
        # (keeps wavlm and audio integer-aligned for any rate, incl. 200Hz where a
        # single wavlm frame is a non-integer 220.5 audio samples).
        avail_cf = wavlm.shape[0] // self.R
        if avail_cf < self.max_cf:
            wavlm = F.pad(wavlm, (0, 0, 0, self.max_wavlm - wavlm.shape[0]))
            cf_start = 0
        else:
            cf_start = self.rng.randint(0, avail_cf - self.max_cf)
            ws = cf_start * self.R
            wavlm = wavlm[ws:ws + self.max_wavlm]

        # ── Original audio ──
        src_path = str(self.source_files[idx])
        wave, sr = sf.read(src_path, dtype="float32")
        wave = torch.from_numpy(np.asarray(wave))
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != S:
            wave = torchaudio.functional.resample(wave.unsqueeze(0), sr, S).squeeze(0)

        # Crop audio to the SAME content window (1764 samples per content frame)
        audio_start = cf_start * 1764
        audio_end = audio_start + self.max_samples
        if audio_end > wave.shape[0]:
            wave = F.pad(wave, (0, audio_end - wave.shape[0]))
            wave = wave[audio_start:audio_end]
        elif audio_start + self.max_samples > wave.shape[0]:
            wave = wave[audio_start:]
            wave = F.pad(wave, (0, self.max_samples - wave.shape[0]))
        else:
            wave = wave[audio_start:audio_end]

        # ── Speaker embedding (match actual speaker) ──
        spk_id = str(self.spk_names[idx])
        if self.speaker_emb_map is not None:
            spk = self.speaker_emb_map.get(spk_id, self._spk_fallback).clone()
        else:
            spk_idx = self.spk_to_emb.get(spk_id, 0)  # fallback to first
            spk = self.spk_embeds[spk_idx].clone()

        return {"wavlm": wavlm, "audio": wave, "speaker": spk, "idx": idx}


def collate_phase0(batch):
    """Stack batch items → (wavlm, audio, speaker, indices)."""
    wavlm = torch.stack([b["wavlm"] for b in batch])      # (B, T_wl, 512)
    audio = torch.stack([b["audio"] for b in batch])      # (B, samples)
    speaker = torch.stack([b["speaker"] for b in batch])  # (B, 128)
    indices = [b["idx"] for b in batch]
    return wavlm, audio, speaker, indices
