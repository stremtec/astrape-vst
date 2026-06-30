"""Causal Waveform Decoder v6 — content-first, staged-refine architecture.

STRICTLY CAUSAL (0 look-ahead). Only latency is iSTFT group delay.

Why v5/v4/MCS/CausalWave all failed:
  - v5: single ×18 learned upsample (25→450) in one shot. AdaLN transformer
    at 25Hz can't model fine pitch/periodicity — it sees ONE frame per 40ms.
    The ×18 leap creates staircase even with ConvTranspose smoothing.
  - CausalWave: mirrors MioCodec structure but the bi-directional teacher's
    global context is structurally inaccessible to a causal transformer.
    GroupNorm→LayerNorm loses cross-channel normalisation.
  - MCS v3: all-conv ConvNeXt/TCN. No global context at all — the speaker
    dilation TCN has large RF but can't do the frame-level attention/spread
    that reconstructs prosody and fine spectral tilt.
  - SimpleGRU: too few params, wrong ISTFT grid.

Root cause analysis:
  The MioCodec teacher works because:
    1. Its prenet transformer runs at 25Hz with FULL bidirectional context —
       this is where long-range prosody (pitch contour, rhythm, energy envelope)
       gets baked into the latent. Causal can't match this; we need an
       alternative route to prosody.
    2. Its speaker transformer runs at 50Hz with bidirectional AdaLN — fine
       grain speaker × content interaction.
    3. The ×2 then ×9 upsample is gradual — each stage only needs to fill in
       local detail, not synthesize from a coarse 25Hz representation.

v6 design philosophy: CONTENT-FIRST, STAGED-REFINE

  Instead of trying to match the teacher's architecture (which is fundamentally
  bidirectional), we re-think the information flow for a CAUSAL system:

  ① Content Projection  @25Hz  768→384  (learned, no speaker — pure content)
  ② Prosody Embedding   @25Hz  speaker→condition via a small causal LSTM
     The LSTM reads content+speaker and produces a per-frame conditioning
     vector. This is the causal substitute for the teacher's bidirectional
     prenet: an RNN can accumulate prosodic context incrementally.
  ③ Rate-2 Upsample     25→50Hz  AA upsample (ConvNeXt smooth + SnakeBeta)
     First gentle doubling. Every output frame has 1 content parent.
  ④ Speaker-Content Fusion @50Hz  AdaLN-Zero causal transformer layers
     This is where speaker identity meets content at fine temporal resolution.
     At 50Hz we have 20ms granularity — enough to shape pitch/voicing.
  ⑤ Rate-9 Upsample     50→450Hz  AA upsample in 2 stages (3×, 3×)
     Gentle ×3 then ×3. Each stage: AA-repeat + CausalConv + SnakeBeta + lowpass
  ⑥ ISTFT Head          n_fft=392, hop=98  → 44.1kHz (3.3ms group delay)

Key differences from previous attempts:
  - No single giant ×N upsample. 25→50 and 50→450 are separate, each gentle.
  - LSTM prosody encoder (not transformer) as the causal substitute for the
    teacher's bidirectional prenet. RNNs accumulate context naturally.
  - Speaker enters at 50Hz (not 25Hz) — finer temporal resolution for pitch.
  - All norms are causal (LayerNorm per-position, CausalGRN).
  - Full AA (anti-aliased) upsample everywhere — no "지지직" artifact.
  - ConvNeXt-v2 blocks (CausalGRN) for content smoothing — proven in MCS v3.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import CausalConv1d, RoPE, SnakeBeta, AdaLNTransformerLayer

_mio = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mio) not in sys.path:
    sys.path.insert(0, str(_mio))


# ═══════════════════════════════════════════════════════════════════
# Anti-aliased upsample (proven in MCS v3 — kills "지지직")
# ═══════════════════════════════════════════════════════════════════

def _lowpass_kernel(cutoff: float, ksize: int) -> torch.Tensor:
    n = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    h = torch.where(n == 0, torch.tensor(2 * cutoff), torch.sin(2 * math.pi * cutoff * n) / (math.pi * n))
    h = h * torch.hamming_window(ksize, periodic=False)
    return h / h.sum()


class CausalLowPass(nn.Module):
    """Fixed linear-phase sinc low-pass, depthwise, causal (left-pad)."""
    def __init__(self, channels: int, cutoff: float, ksize: int):
        super().__init__()
        if ksize % 2 == 0:
            ksize += 1
        k = _lowpass_kernel(cutoff, ksize).view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer("k", k, persistent=False)
        self.ksize, self.ch = ksize, channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(F.pad(x, (self.ksize - 1, 0)), self.k, groups=self.ch)


class AAUpsample(nn.Module):
    """Anti-aliased ×factor upsample: nearest-repeat → low-pass."""
    def __init__(self, channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.lp = CausalLowPass(channels, cutoff=0.5 / factor, ksize=2 * factor * 4 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lp(F.interpolate(x, scale_factor=self.factor, mode="nearest"))


# ═══════════════════════════════════════════════════════════════════
# Causal GRN (ConvNeXt-v2, proven in MCS v3)
# ═══════════════════════════════════════════════════════════════════

class CausalGRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnt = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        Gx = torch.sqrt(torch.cumsum(x.pow(2), dim=-1) / cnt + 1e-6)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class _ChannelLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


# ═══════════════════════════════════════════════════════════════════
# ① & ② Content projection + Prosody LSTM
# ═══════════════════════════════════════════════════════════════════

class ProsodyLSTM(nn.Module):
    """Causal LSTM that fuses content + speaker → per-frame prosody conditioning.

    This is the causal substitute for the teacher's bidirectional prenet
    transformer. An LSTM accumulates pitch/energy/rhythm context incrementally
    — exactly what streaming needs, and what a causal transformer can't do
    well (it only sees its local window, not the full prefix).

    Output: (B, T, cond_dim) — one prosody vector per content frame.
    """
    def __init__(self, content_dim: int, speaker_dim: int, hidden: int, cond_dim: int):
        super().__init__()
        self.content_proj = nn.Linear(content_dim, hidden)
        self.speaker_proj = nn.Linear(speaker_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden, cond_dim)

    def forward(self, content: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """content: (B, T, content_dim), speaker: (B, speaker_dim) → (B, T, cond_dim)"""
        h = self.content_proj(content) + self.speaker_proj(speaker).unsqueeze(1)
        h, _ = self.lstm(h)
        return self.out(h)


# ═══════════════════════════════════════════════════════════════════
# ③ Causal ConvNeXt-v2 content smoothing block
# ═══════════════════════════════════════════════════════════════════

class CausalConvNeXtBlock(nn.Module):
    """ConvNeXt-v2: depthwise CausalConv → LN → PW expand → GELU → GRN → PW contract → +res."""
    def __init__(self, dim: int, kernel: int = 7, expand: int = 4):
        super().__init__()
        self.dw = CausalConv1d(dim, dim, kernel, groups=dim)
        self.norm = _ChannelLN(dim)
        self.pw1 = nn.Conv1d(dim, expand * dim, 1)
        self.grn = CausalGRN(expand * dim)
        self.pw2 = nn.Conv1d(expand * dim, dim, 1)
        self.scale = nn.Parameter(1e-6 * torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x); h = self.norm(h)
        h = F.gelu(self.pw1(h)); h = self.grn(h); h = self.pw2(h)
        return x + self.scale * h


# ═══════════════════════════════════════════════════════════════════
# ③ AA upsample stage: AA-upsample → CausalConv → SnakeBeta → lowpass
# ═══════════════════════════════════════════════════════════════════

class AAUpStage(nn.Module):
    """BigVGAN-v2-style anti-aliased upsample stage."""
    def __init__(self, c_in: int, c_out: int, factor: int, conv_k: int = 15):
        super().__init__()
        self.up = AAUpsample(c_in, factor)
        self.conv = CausalConv1d(c_in, c_out, conv_k)
        self.snake = SnakeBeta(c_out)
        self.lp = CausalLowPass(c_out, cutoff=0.45, ksize=2 * 4 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lp(self.snake(self.conv(self.up(x))))


# ═══════════════════════════════════════════════════════════════════
# ④ Dilated speaker TCN block (AdaLN-Zero, proven in MCS v3)
# ═══════════════════════════════════════════════════════════════════

class DilatedSpeakerBlock(nn.Module):
    """Dilated causal TCN block with AdaLN-Zero(speaker) + GRN."""
    def __init__(self, dim: int, cond_dim: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        from miocodec.module.adaln_zero import AdaLNZero
        from miocodec.module.istft_head import SnakeBeta as MioSnakeBeta
        self.adaln = AdaLNZero(dim, cond_dim, return_gate=True)
        self.dw = CausalConv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.snake = MioSnakeBeta(dim, alpha_logscale=True)
        self.grn = CausalGRN(dim)
        self.pw = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        normed, gate = self.adaln(x.transpose(1, 2), condition=cond)
        h = normed.transpose(1, 2)
        h = self.pw(self.grn(self.snake(self.dw(h))))
        return x + gate.transpose(1, 2) * h


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalDecoderV6Config:
    content_dim: int = 768
    speaker_dim: int = 128
    sample_rate: int = 44100
    content_rate: int = 25
    input_std_scale: float = 0.46 / 0.38

    # ① Content projection
    proj_dim: int = 384           # 768→384 (content lives in 384d after projection)

    # ② Prosody LSTM (causal substitute for teacher's bidirectional prenet)
    prosody_hidden: int = 384
    prosody_cond_dim: int = 128   # prosody conditioning vector dim

    # ③ Content smoothing @25Hz (ConvNeXt-v2, proven)
    smooth_blocks: int = 4
    smooth_kernel: int = 7

    # ③ AA Rate ×2  25→50Hz
    up2_dim: int = 512            # 384→512 at rate-2 (expand channels at higher rate)

    # ④ Speaker-Content Fusion @50Hz
    #    Two parallel pathways merged:
    #    - AdaLN causal transformer (global structure)
    #    - Dilated TCN + AdaLN (local fine-grained speaker interaction)
    fusion_layers: int = 4        # AdaLN transformer layers at 50Hz
    fusion_heads: int = 8
    fusion_window: int = 128      # causal window at 50Hz (~2.5s context)
    fusion_rope_theta: float = 10000.0
    speaker_dilations: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32)
    speaker_kernel: int = 5

    # ⑤ AA Upsampler 50→450Hz  (×3 then ×3)
    upsampler_factors: tuple[int, ...] = (3, 3)

    # ⑥ ISTFT Head
    istft_bridge_dim: int = 512
    n_fft: int = 392
    hop_length: int = 98          # 44100/98 = 450Hz = 18×25
    istft_padding: str = "same"


# ═══════════════════════════════════════════════════════════════════
# Full v6 decoder
# ═══════════════════════════════════════════════════════════════════

class CausalDecoderV6(nn.Module):
    """Content-first, staged-refine causal decoder.

    Pipeline:
        content(768) @25Hz + speaker(128)
          ① Content Projection       Linear 768→384                        @25Hz
          ② Prosody Embedding        LSTM(content+speaker) → condition     @25Hz
          ③ Content Smoothing        ConvNeXt-v2 ×4                       @25Hz
          ③ AA Rate ×2               BigVGAN-v2 upsample 384→512          @50Hz
          ④ Speaker-Content Fusion   AdaLN-Transformer + DilatedTCN       @50Hz
          ⑤ AA Upsampler ×9          (3,3)                                @450Hz
          ⑥ ISTFT Head               n_fft=392, hop=98                    → 44.1kHz
    """
    def __init__(self, config: CausalDecoderV6Config = CausalDecoderV6Config()):
        super().__init__()
        c = config
        self.config = c
        D, W = c.proj_dim, c.up2_dim
        self.register_buffer("input_scale", torch.tensor(c.input_std_scale))

        # ① Content projection: 768→384 (no speaker — pure content)
        self.content_proj = nn.Linear(c.content_dim, D, bias=False)

        # ② Prosody LSTM: causal RNN fuses content+speaker → conditioning
        self.prosody = ProsodyLSTM(
            content_dim=c.content_dim,
            speaker_dim=c.speaker_dim,
            hidden=c.prosody_hidden,
            cond_dim=c.prosody_cond_dim,
        )

        # ③ Content smoothing @25Hz (ConvNeXt-v2 — proven reliable)
        self.smooth = nn.ModuleList([
            CausalConvNeXtBlock(D, kernel=c.smooth_kernel) for _ in range(c.smooth_blocks)
        ])
        self.smooth_proj = nn.Conv1d(D, D, 1)

        # ③ AA rate ×2 upsample, 384→512 → 50Hz
        self.up2 = AAUpStage(D, W, factor=2, conv_k=15)

        # ④ Speaker-Content Fusion @50Hz
        #    Two parallel pathways, then merge:

        #    Path A: AdaLN causal transformer (global prosodic structure)
        #    The prosody LSTM output conditions this — it carries the accumulated
        #    pitch/energy context that the teacher's bidirectional prenet provided.
        self.fusion_rope = RoPE(W // c.fusion_heads, max_len=c.fusion_window * 4,
                                theta=c.fusion_rope_theta)
        self.fusion_layers = nn.ModuleList([
            AdaLNTransformerLayer(W, c.fusion_heads, c.prosody_cond_dim,
                                  ff_mult=4)
            for _ in range(c.fusion_layers)
        ])
        self.fusion_norm = nn.LayerNorm(W)

        #    Path B: Dilated speaker TCN (fine-grained local interaction)
        #    This is the high-resolution speaker pathway — dilations up to 32
        #    give ~640ms backward RF at 50Hz, plenty for formant texture.
        self.speaker_blocks = nn.ModuleList([
            DilatedSpeakerBlock(W, c.speaker_dim, c.speaker_kernel, d)
            for d in c.speaker_dilations
        ])

        #    Merge: gate-weighted sum of both pathways
        #    gate = sigmoid(learned) so init ≈ 0.5 each path
        self.fusion_gate = nn.Parameter(torch.zeros(1, W, 1))

        # ⑤ Post-fusion refinement + AA upsampler 50→450Hz
        self.post_smooth = nn.ModuleList([
            CausalConvNeXtBlock(W, kernel=5) for _ in range(2)
        ])
        self.upsampler = nn.ModuleList([
            AAUpStage(W, W, factor=f, conv_k=7) for f in c.upsampler_factors
        ])

        # ⑥ Bridge + ISTFT head
        self.istft_bridge = nn.Conv1d(W, c.istft_bridge_dim, kernel_size=1)
        from miocodec.module.istft_head import ISTFTHead
        self.istft_head = ISTFTHead(
            dim=c.istft_bridge_dim, n_fft=c.n_fft,
            hop_length=c.hop_length, padding=c.istft_padding,
        )

    def _compute_stft_length(self, content_frames: int) -> int:
        return int(content_frames * self.config.sample_rate
                   / self.config.hop_length / self.config.content_rate)

    def forward(self, content: torch.Tensor, speaker: torch.Tensor,
                stft_length: int | None = None, return_spec: bool = False):
        """
        Args:
            content: (B, T, 768) content embeddings @25Hz
            speaker: (B, 128) speaker embedding
            stft_length: number of STFT frames (computed if None)
            return_spec: if True, return (wav, mag, phase)
        Returns:
            wav: (B, samples)
            (mag, phase): only if return_spec=True
        """
        B, T, _ = content.shape
        if stft_length is None:
            stft_length = self._compute_stft_length(T)

        # ② Prosody embedding: LSTM reads full content+speaker prefix
        #    This is the key new component — causal substitute for bidirectional prenet
        prosody_cond = self.prosody(content, speaker)   # (B, T, prosody_cond_dim)

        # ① Content projection (after prosody, so we use raw content for both)
        h = content * self.input_scale.to(dtype=content.dtype)
        h = self.content_proj(h)                         # (B, T, 384)

        # ③ Content smoothing @25Hz
        h = h.transpose(1, 2)                            # (B, 384, T)
        for block in self.smooth:
            h = block(h)
        h = self.smooth_proj(h)

        # ③ AA rate ×2 → 50Hz
        h = self.up2(h)                                  # (B, 512, 2T)

        # ④ Speaker-Content Fusion @50Hz
        #    Upsample prosody condition to 50Hz to match
        T_50 = h.shape[-1]
        # Nearest-repeat is causal (each 50Hz frame copies its 25Hz parent)
        prosody_50 = prosody_cond.repeat_interleave(2, dim=1)[:, :T_50, :]  # (B, 2T, cond)
        spk = speaker.unsqueeze(1)                       # (B, 1, 128)

        #    Path A: AdaLN transformer (global)
        h_a = h.transpose(1, 2)                          # (B, 2T, 512)
        for layer in self.fusion_layers:
            h_a = layer(h_a, prosody_50, self.fusion_rope, self.config.fusion_window)
        h_a = self.fusion_norm(h_a).transpose(1, 2)     # (B, 512, 2T)

        #    Path B: Dilated speaker TCN (local)
        h_b = h.clone()
        for block in self.speaker_blocks:
            h_b = block(h_b, spk)

        #    Merge: gate-weighted sum
        gate = torch.sigmoid(self.fusion_gate)           # ∈ (0, 1)
        h = gate * h_a + (1 - gate) * h_b

        # ⑤ Post-fusion refinement + AA upsampler 50→450Hz
        for block in self.post_smooth:
            h = block(h)
        for block in self.upsampler:
            h = block(h)                                 # (B, 512, 18T)

        # ⑥ Bridge + ISTFT
        h = self.istft_bridge(h).transpose(1, 2)         # (B, 18T, bridge_dim)
        xo = self.istft_head.out(h).transpose(1, 2)     # (B, n_fft+2, 18T)
        mag_log, phase = xo.chunk(2, dim=1)
        mag = torch.exp(mag_log).clamp(max=1e2)
        wav = self.istft_head.istft(torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)))
        if return_spec:
            return wav, mag, phase
        return wav


if __name__ == "__main__":
    for fusion_layers in (2, 4):
        cfg = CausalDecoderV6Config(fusion_layers=fusion_layers)
        m = CausalDecoderV6(cfg).eval()
        n = sum(p.numel() for p in m.parameters())
        B, T = 2, 50
        cont, spk = torch.randn(B, T, 768), torch.randn(B, 128)
        with torch.no_grad():
            wav = m(cont, spk)
        algo = (cfg.n_fft - cfg.hop_length) / 2 / cfg.sample_rate * 1000
        print(f"fusion={fusion_layers}L: params={n/1e6:.2f}M  out={list(wav.shape)} "
              f"({wav.shape[1]/cfg.sample_rate:.2f}s)  iSTFT algo-latency={algo:.1f}ms")

    # strict-causal check
    c = CausalDecoderV6Config(fusion_layers=4)
    m = CausalDecoderV6(c).eval()
    cont, spk = torch.randn(1, 50, 768), torch.randn(1, 128)
    with torch.no_grad():
        a = m(cont, spk)
        c2 = cont.clone(); c2[:, 40:] += 5.0
        b = m(c2, spk)
    edge = a.shape[1] * 40 // 50
    look = (c.n_fft - c.hop_length) // 2
    print(f"\nstrict-causal: pre-boundary max={((a[0,:edge-look-200]-b[0,:edge-look-200]).abs().max()):.2e}")
