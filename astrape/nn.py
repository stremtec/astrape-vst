"""Shared neural primitives for the Astrape decoders.

Single home for the building blocks that were previously duplicated across
`causal_decoder.py` / `causal_decoder_v5.py`:
  - CausalConv1d           left-padded (strict-causal) conv
  - RoPE                   rotary position embedding
  - SnakeBeta              periodic activation x + (1/β)·sin²(αx)  (BigVGAN/X-Codec)
  - AdaLNTransformerLayer  causal windowed attn + SwiGLU + AdaLN-Zero speaker cond
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding → strictly causal (0 look-ahead)."""

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


class SnakeBeta(nn.Module):
    """SnakeBeta(x) = x + (1/β)·sin²(αx), per-channel α,β in log-scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        a = self.alpha.exp().view(1, -1, 1)
        b = self.beta.exp().view(1, -1, 1)
        return x + (1.0 / (b + 1e-9)) * (a * x).sin().pow(2)


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
