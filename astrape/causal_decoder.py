"""Causal Waveform Decoder v4 — Ultra-lightweight streaming vocoder (~6M, 29ms delay).

Design philosophy:
  Encoder already processed ALL long-range context (7L Transformer + Q2D2).
  Decoder only needs: speaker conditioning + waveform rendering.
  NO 8L Transformer prenet — that's redundant. 2L with AdaLN-Zero is enough.
  25Hz domain: ZERO temporal convs. 172Hz domain: 2-block shallow conv.

Pipeline:
  Phase 0: Q2D2 content(std=0.38) → ×1.21 → Teacher distribution
  Phase 1: Pointwise 768→384 + 2L CausalTransformer(AdaLN-Zero speaker) → 0ms
           Speaker conditioning happens HERE, not downstream.
  Phase 2: repeat_interleave(7) + trim + pointwise → 0ms
           No temporal convs. Content expansion only.
  Phase 3: 2× Shallow Causal Conv(k=3, AdaLN-Zero) → 23.2ms
           Local phase/formant refinement.
  Phase 4: Pointwise 384→512 + ISTFTHead → 5.8ms

Total algorithmic delay: 0 + 0 + 0 + 23.2 + 5.8 = 29.0ms
Total params: ~6M
"""

from __future__ import annotations

import math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_mio = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mio) not in sys.path:
    sys.path.insert(0, str(_mio))


@dataclass
class CausalDecoderConfig:
    content_dim: int = 768
    condition_dim: int = 128
    content_rate: int = 25
    sample_rate: int = 44100
    input_std_scale: float = 0.46 / 0.38

    # Phase 1: Ultra-light prenet (2L, AdaLN-Zero speaker conditioning)
    prenet_dim: int = 384
    prenet_layers: int = 2
    prenet_heads: int = 8
    prenet_ff_mult: int = 4
    prenet_window: int = 50
    prenet_rope_theta: float = 10000.0
    prenet_dropout: float = 0.0

    # Phase 2: Upsample (repeat + pointwise only)
    upsample_factor: int = 7  # 25→175Hz, then trim to ~172Hz

    # Phase 3: Shallow ConvNet (or Mamba)
    convnet_dim: int = 384
    convnet_blocks: int = 2
    convnet_kernel: int = 3
    use_mamba: bool = False        # Mamba SSM instead of ShallowConvBlock
    mamba_d_state: int = 16
    mamba_expand: int = 2

    # Phase 4: ISTFT head
    istft_bridge_dim: int = 512
    n_fft: int = 1024
    hop_length: int = 256
    istft_padding: str = "same"

    dropout: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# CausalConv1d
# ═══════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    @property
    def left_context(self) -> int:
        return self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        if self.left_context:
            x = F.pad(x, (self.left_context, 0))
        return super().forward(x)


# ═══════════════════════════════════════════════════════════════════
# RoPE
# ═══════════════════════════════════════════════════════════════════

class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096, theta=10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim; self.max_len = max_len; self.theta = theta
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x, offset=0):
        T = x.shape[-2]
        if T == 0: return x
        cd = x.dtype
        if cd in (torch.float16, torch.bfloat16): cd = torch.float32
        pos = torch.arange(offset, offset+T, device=x.device, dtype=cd)
        inv = self.inv_freq.to(device=x.device, dtype=cd)
        ang = torch.outer(pos, inv)
        cos = ang.cos().to(x.dtype).unsqueeze(0).unsqueeze(0)
        sin = ang.sin().to(x.dtype).unsqueeze(0).unsqueeze(0)
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        return torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), -1).flatten(-2)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Causal Transformer + AdaLN-Zero (speaker conditioning)
# ═══════════════════════════════════════════════════════════════════

class AdaLNTransformerLayer(nn.Module):
    """Causal windowed attention + SwiGLU + AdaLN-Zero speaker conditioning."""
    def __init__(self, dim, heads, cond_dim, ff_mult=4, dropout=0.0):
        super().__init__()
        from miocodec.module.adaln_zero import AdaLNZero
        hd = dim // heads
        self.heads = heads; self.hd = hd

        self.attn_adaln = AdaLNZero(dim, cond_dim, return_gate=True)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn_adaln = AdaLNZero(dim, cond_dim, return_gate=True)
        self.w1 = nn.Linear(dim, dim*ff_mult, bias=False)
        self.w3 = nn.Linear(dim, dim*ff_mult, bias=False)
        self.w2 = nn.Linear(dim*ff_mult, dim, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x, condition, rope, window, offset=0):
        """x:(B,T,D), condition:(B,1,D) or (B,D)"""
        B, T, D = x.shape
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)

        # Attention with AdaLN
        normed, gate_a = self.attn_adaln(x, condition)
        q = self.wq(normed).view(B,T,self.heads,self.hd)
        k = self.wk(normed).view(B,T,self.heads,self.hd)
        v = self.wv(normed).view(B,T,self.heads,self.hd)
        q = rope(q.transpose(1,2), offset).transpose(1,2)
        k = rope(k.transpose(1,2), offset).transpose(1,2)
        q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        mask = torch.ones(T,T,dtype=torch.bool,device=x.device)
        mask = torch.tril(mask)
        mask = torch.triu(mask, diagonal=-(window-1))
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn = F.scaled_dot_product_attention(
            q,k,v, attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x + gate_a * self.wo(attn.transpose(1,2).contiguous().view(B,T,D))

        # FFN with AdaLN
        normed, gate_f = self.ffn_adaln(x, condition)
        x = x + gate_f * self.ffn_drop(
            self.w2(F.silu(self.w1(normed))*self.w3(normed)))
        return x


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Shallow Causal Conv Block (AdaLN-Zero)
# ═══════════════════════════════════════════════════════════════════

class ShallowConvBlock(nn.Module):
    """Depthwise causal conv + AdaLN-Zero. k=3 → 2 frames RF @172Hz = 11.6ms."""
    def __init__(self, dim, cond_dim, kernel=3, dropout=0.0):
        super().__init__()
        from miocodec.module.adaln_zero import AdaLNZero
        self.adaln = AdaLNZero(dim, cond_dim, return_gate=True)
        self.depthwise = CausalConv1d(dim, dim, kernel, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()

    def forward(self, x, condition):
        residual = x
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        normed, gate = self.adaln(x, condition)
        h = normed.transpose(1,2)
        h = self.depthwise(h)
        h = F.silu(h)
        h = self.pointwise(h)
        h = self.dropout(h)
        h = h.transpose(1,2)
        return residual + gate * h


# ═══════════════════════════════════════════════════════════════════
# Full Causal Decoder v4
# ═══════════════════════════════════════════════════════════════════

class CausalDecoder(nn.Module):
    def __init__(self, config: CausalDecoderConfig = CausalDecoderConfig()):
        super().__init__()
        c = config; D, W = c.prenet_dim, c.convnet_dim
        self.config = c
        self.register_buffer("input_scale", torch.tensor(c.input_std_scale))

        # Phase 1: Pointwise in + 2L AdaLN Transformer
        self.pw_in = nn.Linear(c.content_dim, D, bias=False)
        hd = D // c.prenet_heads
        self.prenet_rope = RoPE(hd, max_len=c.prenet_window*4, theta=c.prenet_rope_theta)
        self.prenet_layers = nn.ModuleList([
            AdaLNTransformerLayer(D, c.prenet_heads, c.condition_dim,
                                  c.prenet_ff_mult, c.prenet_dropout)
            for _ in range(c.prenet_layers)])
        self.prenet_norm = nn.LayerNorm(D)

        # Phase 2: repeat + pointwise (0ms)
        self.up_pw = nn.Conv1d(D, W, kernel_size=1)

        # Phase 3: shallow convnet or Mamba
        if c.use_mamba:
            from astrape.mamba_block import CausalMambaBlock
            self.convnet = nn.ModuleList([
                CausalMambaBlock(W, c.condition_dim, c.mamba_d_state,
                                 c.mamba_expand, d_conv=2)
                for _ in range(c.convnet_blocks)])
        else:
            self.convnet = nn.ModuleList([
                ShallowConvBlock(W, c.condition_dim, c.convnet_kernel, c.dropout)
                for _ in range(c.convnet_blocks)])
        self.convnet_norm = nn.LayerNorm(W)

        # Phase 4: pointwise bridge + ISTFT head
        self.istft_bridge = nn.Conv1d(W, c.istft_bridge_dim, kernel_size=1)
        from miocodec.module.istft_head import ISTFTHead
        self.istft_head = ISTFTHead(
            dim=c.istft_bridge_dim, n_fft=c.n_fft,
            hop_length=c.hop_length, padding=c.istft_padding)

    def _compute_stft_length(self, content_frames):
        return int(content_frames * self.config.sample_rate
                   / self.config.hop_length / self.config.content_rate)

    def forward(self, content, speaker, stft_length=None):
        B, T, _ = content.shape
        if stft_length is None:
            stft_length = self._compute_stft_length(T)

        # Phase 0
        h = content * self.input_scale.to(dtype=content.dtype)

        # Phase 1: Pointwise + 2L AdaLN Transformer (0ms)
        h = self.pw_in(h)  # (B,T,D)
        spk = speaker.unsqueeze(1)  # (B,1,128)
        for layer in self.prenet_layers:
            h = layer(h, spk, self.prenet_rope, self.config.prenet_window)
        h = self.prenet_norm(h)

        # Phase 2: repeat_interleave(7) + pointwise (0ms)
        h = h.repeat_interleave(self.config.upsample_factor, dim=1)
        h = self.up_pw(h.transpose(1,2)).transpose(1,2)  # (B,T*7,W)
        h = h[:, :stft_length, :]

        # Phase 3: Shallow convnet (23.2ms)
        for block in self.convnet:
            h = block(h, spk)
        h = self.convnet_norm(h)

        # Phase 4: ISTFT head (5.8ms)
        h = self.istft_bridge(h.transpose(1,2))  # (B,512,T_stft)
        h = h.transpose(1,2)  # (B,T_stft,512)
        return self.istft_head(h)


# ═══════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = CausalDecoderConfig()
    model = CausalDecoder(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} ({total/1e6:.2f}M)")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            print(f"  {name:<25s} {n:>10,} ({n/1e6:.2f}M)")

    B, T = 2, 50
    cont = torch.randn(B, T, cfg.content_dim)
    spk = torch.randn(B, cfg.condition_dim)
    stft = model._compute_stft_length(T)
    print(f"\nInput: content={list(cont.shape)}, T={T} @25Hz → stft={stft}")

    model.eval()
    with torch.no_grad():
        wav = model(cont, spk, stft)
    print(f"Output: wave={list(wav.shape)} "
          f"({wav.shape[1]/cfg.sample_rate:.2f}s)")

    conv_delay = cfg.convnet_blocks * 2 * (cfg.hop_length / cfg.sample_rate * 1000)
    istft_delay = cfg.hop_length / cfg.sample_rate * 1000
    print(f"\nDelay: conv={conv_delay:.1f}ms + iSTFT={istft_delay:.1f}ms "
          f"= {conv_delay+istft_delay:.1f}ms")
    print(f"  + encoder 27ms = e2e {conv_delay+istft_delay+27:.0f}ms")
