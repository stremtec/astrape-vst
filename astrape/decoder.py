"""Causal Waveform Decoder v5 — Max-Quality streaming vocoder.

Strict-causal (0 look-ahead). Only algorithmic latency is the iSTFT group delay:
(n_fft - hop)/2 = (1512 - 252)/2 = 630 samples @ 44.1kHz = 14.3 ms.

Differences from v4 (and from MioCodec's wave decoder):
  - Phase 1 deepened: 4 causal AdaLN transformer layers @25Hz (v4: 2).
  - Phase 2 LEARNED causal upsampler (ConvTranspose + SnakeBeta), replacing
    v4's repeat_interleave zero-order-hold.
  - Phase 2b OPTIONAL NSF: causal F0 head → analytic harmonic-comb magnitude
    template (MPS-safe, no forward STFT), concatenated into the iSTFT bridge.
  - Phase 3 deepened + dilated causal SnakeBeta conv stack (v4: 2 shallow k=3).
  - Phase 4 iSTFT n_fft 1008→1512 (finer frequency resolution; +5.7ms latency).
  - vs MioCodec: causal (not bidirectional / symmetric-pad), attention stays at
    25Hz (MioCodec runs its decoder transformer at the upsampled ~172Hz), and the
    44.1kHz wave_upsampler is dropped (hop=252 hits 44.1kHz directly).

Trained adversarially (MPD + MR-STFT discriminators, see astrape/discriminators.py).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import CausalConv1d, RoPE, SnakeBeta, AdaLNTransformerLayer

_mio = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mio) not in sys.path:
    sys.path.insert(0, str(_mio))


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Learned causal upsampler (25 → 175 Hz)
# ═══════════════════════════════════════════════════════════════════

class LearnedCausalUpsampler(nn.Module):
    """ConvTranspose1d(stride=factor) + SnakeBeta. Causal: output[j] depends on
    input frames ≤ floor(j/factor), so the front-trim to `out_len` is causal."""

    def __init__(self, dim: int, factor: int):
        super().__init__()
        self.factor = factor
        self.up = nn.ConvTranspose1d(dim, dim, kernel_size=2 * factor, stride=factor)
        self.act = SnakeBeta(dim)

    def forward(self, x: torch.Tensor, out_len: int) -> torch.Tensor:
        h = x.transpose(1, 2)               # (B, dim, T)
        h = self.act(self.up(h))            # (B, dim, (T+1)*factor)
        h = h[:, :, :out_len]               # causal trim to stft length
        return h.transpose(1, 2)            # (B, out_len, dim)


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Causal dilated SnakeBeta conv block (AdaLN-Zero speaker)
# ═══════════════════════════════════════════════════════════════════

class SnakeConvBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        from miocodec.module.adaln_zero import AdaLNZero
        self.adaln = AdaLNZero(dim, cond_dim, return_gate=True)
        self.depthwise = CausalConv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.act = SnakeBeta(dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        normed, gate = self.adaln(x, condition)     # (B, T, dim)
        h = normed.transpose(1, 2)                  # (B, dim, T)
        h = self.depthwise(h)
        h = self.act(h)
        h = self.pointwise(h)
        h = self.dropout(h).transpose(1, 2)         # (B, T, dim)
        return residual + gate * h


# ═══════════════════════════════════════════════════════════════════
# Phase 2b: Neural source-filter — analytic harmonic comb (optional)
# ═══════════════════════════════════════════════════════════════════

class HarmonicComb(nn.Module):
    """Causal F0/voiced head → analytic harmonic-comb log-magnitude template.

    Avoids a forward STFT (unstable on MPS): the comb is built directly in the
    rfft frequency grid as a sum of soft peaks at k·F0. Differentiable in F0.
    Output: (B, T_stft, n_freq) to be concatenated into the iSTFT bridge.
    """

    def __init__(self, dim: int, cond_dim: int, sample_rate: int, n_fft: int,
                 n_harmonics: int = 16, f0_min: float = 50.0, f0_max: float = 600.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.f0_min, self.f0_max = f0_min, f0_max
        # Speaker conditioning: F0 must follow the TARGET speaker's pitch register
        # (low male → high female), so the predictor gets the global embedding
        # directly (broadcast over time), not just via the upstream AdaLN latent.
        self.spk_proj = nn.Linear(cond_dim, dim)
        self.head = nn.Sequential(
            CausalConv1d(dim, dim, 3),
            SnakeBeta(dim),
            nn.Conv1d(dim, 2, 1),                   # [f0_logit, voiced_logit]
        )
        n_freq = n_fft // 2 + 1
        freqs = torch.linspace(0.0, sample_rate / 2.0, n_freq)   # rfft bin centers (Hz)
        self.register_buffer("freqs", freqs, persistent=False)
        self.bandwidth = sample_rate / n_fft                     # ~1 bin

    def forward(self, latent_stft: torch.Tensor, speaker: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # latent_stft: (B, T_stft, dim), speaker: (B, cond_dim)
        x = latent_stft + self.spk_proj(speaker).unsqueeze(1)        # +target register
        h = self.head(x.transpose(1, 2)).transpose(1, 2)             # (B, T, 2)
        f0 = self.f0_min + (self.f0_max - self.f0_min) * torch.sigmoid(h[..., :1])  # (B,T,1)
        voiced = torch.sigmoid(h[..., 1:2])                                          # (B,T,1)
        k = torch.arange(1, self.n_harmonics + 1, device=f0.device).float()
        centers = f0 * k                                       # (B, T, K)
        diff = self.freqs.view(1, 1, -1, 1) - centers.unsqueeze(2)   # (B,T,F,K)
        comb = torch.exp(-(diff / self.bandwidth).pow(2)).sum(-1)    # (B,T,F)
        comb = comb * voiced                                          # gate by voicing
        return torch.log(comb + 1e-4), f0.squeeze(-1)                 # (B,T,F), (B,T)


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalDecoderV5Config:
    content_dim: int = 768
    condition_dim: int = 128
    content_rate: int = 25
    sample_rate: int = 44100
    input_std_scale: float = 0.46 / 0.38

    prenet_dim: int = 384
    prenet_layers: int = 4              # v4: 2 (deepen — free latency)
    prenet_heads: int = 8
    prenet_ff_mult: int = 4
    prenet_window: int = 64
    prenet_rope_theta: float = 10000.0
    prenet_dropout: float = 0.0

    upsample_factor: int = 7            # 25→175Hz (matches hop=252)

    convnet_dim: int = 384
    convnet_dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)   # v4: 2 shallow k=3
    convnet_kernel: int = 3
    dropout: float = 0.0

    use_nsf: bool = False               # Phase 2b (curriculum stage 2)
    nsf_harmonics: int = 16

    istft_bridge_dim: int = 512
    n_fft: int = 1512                   # v4: 1008  (→ 14.3ms algorithmic latency)
    hop_length: int = 252               # 44100/252 = 175Hz = 7×25, unchanged
    istft_padding: str = "same"


# ═══════════════════════════════════════════════════════════════════
# Full v5 decoder
# ═══════════════════════════════════════════════════════════════════

class CausalDecoderV5(nn.Module):
    def __init__(self, config: CausalDecoderV5Config = CausalDecoderV5Config()):
        super().__init__()
        c = config
        self.config = c
        D, W = c.prenet_dim, c.convnet_dim
        self.register_buffer("input_scale", torch.tensor(c.input_std_scale))

        # Phase 1: pointwise in + N causal AdaLN transformer layers @25Hz
        self.pw_in = nn.Linear(c.content_dim, D, bias=False)
        self.prenet_rope = RoPE(D // c.prenet_heads, max_len=c.prenet_window * 4,
                                theta=c.prenet_rope_theta)
        self.prenet_layers = nn.ModuleList([
            AdaLNTransformerLayer(D, c.prenet_heads, c.condition_dim,
                                  c.prenet_ff_mult, c.prenet_dropout)
            for _ in range(c.prenet_layers)])
        self.prenet_norm = nn.LayerNorm(D)

        # Phase 2: learned causal upsampler 25→175Hz
        self.upsampler = LearnedCausalUpsampler(D, c.upsample_factor)
        self.up_proj = nn.Conv1d(D, W, kernel_size=1)

        # Phase 2b: optional NSF
        self.nsf = (HarmonicComb(W, c.condition_dim, c.sample_rate, c.n_fft, c.nsf_harmonics)
                    if c.use_nsf else None)
        n_freq = c.n_fft // 2 + 1

        # Phase 3: deeper dilated causal SnakeBeta conv stack @175Hz
        self.convnet = nn.ModuleList([
            SnakeConvBlock(W, c.condition_dim, c.convnet_kernel, d, c.dropout)
            for d in c.convnet_dilations])
        self.convnet_norm = nn.LayerNorm(W)

        # Phase 4: bridge (+ NSF channels) + iSTFT head
        bridge_in = W + (n_freq if c.use_nsf else 0)
        self.istft_bridge = nn.Conv1d(bridge_in, c.istft_bridge_dim, kernel_size=1)
        from miocodec.module.istft_head import ISTFTHead
        self.istft_head = ISTFTHead(dim=c.istft_bridge_dim, n_fft=c.n_fft,
                                    hop_length=c.hop_length, padding=c.istft_padding)

    def _compute_stft_length(self, content_frames: int) -> int:
        return int(content_frames * self.config.sample_rate
                   / self.config.hop_length / self.config.content_rate)

    def forward(self, content: torch.Tensor, speaker: torch.Tensor,
                stft_length: int | None = None) -> torch.Tensor:
        B, T, _ = content.shape
        if stft_length is None:
            stft_length = self._compute_stft_length(T)
        spk = speaker.unsqueeze(1)                       # (B, 1, cond)

        # Phase 0 + 1
        h = content * self.input_scale.to(dtype=content.dtype)
        h = self.pw_in(h)
        for layer in self.prenet_layers:
            h = layer(h, spk, self.prenet_rope, self.config.prenet_window)
        h = self.prenet_norm(h)

        # Phase 2: learned causal upsample → 175Hz
        h = self.upsampler(h, stft_length)               # (B, T_stft, D)
        h = self.up_proj(h.transpose(1, 2)).transpose(1, 2)   # (B, T_stft, W)

        # Phase 2b: NSF harmonic template (optional)
        nsf_logmag = None
        if self.nsf is not None:
            nsf_logmag, _f0 = self.nsf(h, speaker)       # (B, T_stft, n_freq)

        # Phase 3: causal conv refine
        for block in self.convnet:
            h = block(h, spk)
        h = self.convnet_norm(h)

        # Phase 4: bridge + iSTFT
        if nsf_logmag is not None:
            h = torch.cat([h, nsf_logmag], dim=-1)
        h = self.istft_bridge(h.transpose(1, 2)).transpose(1, 2)   # (B, T_stft, bridge)
        return self.istft_head(h)                                   # (B, samples)


if __name__ == "__main__":
    for use_nsf in (False, True):
        cfg = CausalDecoderV5Config(use_nsf=use_nsf)
        m = CausalDecoderV5(cfg).eval()
        n = sum(p.numel() for p in m.parameters())
        B, T = 2, 50
        cont, spk = torch.randn(B, T, 768), torch.randn(B, 128)
        with torch.no_grad():
            wav = m(cont, spk)
        algo = (cfg.n_fft - cfg.hop_length) / 2 / cfg.sample_rate * 1000
        print(f"use_nsf={use_nsf}: params={n/1e6:.2f}M  out={list(wav.shape)} "
              f"({wav.shape[1]/cfg.sample_rate:.2f}s)  iSTFT algo-latency={algo:.1f}ms")
