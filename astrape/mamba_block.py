"""Causal Mamba Block — Mamba-minimal adapted for audio decoding.

Pure PyTorch selective scan (CPU/MPS compatible).
k=2 causal conv + SSM + AdaLN-Zero speaker conditioning.
Replaces ShallowConvBlock in Phase 3.
"""

from __future__ import annotations

import math, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# MioCodec import path
_mp = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mp) not in sys.path:
    sys.path.insert(0, str(_mp))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalMambaBlock(nn.Module):
    """Mamba block with AdaLN-Zero + causal conv (k=2, left-pad=1).

    Adapted from johnma2006/mamba-minimal.
    Adds speaker conditioning via AdaLN-Zero at input.
    Uses causal depthwise conv (left-only padding) for streaming.
    SSM via pure PyTorch sequential scan — CPU/MPS compatible.

    Args:
        dim: hidden dimension (d_model)
        cond_dim: speaker embedding dimension
        d_state: SSM state dimension (N in paper, default 16)
        expand: inner expansion factor (E in paper, default 2)
        d_conv: conv kernel size (default 2 → 1-frame causal RF)
        dt_rank: rank of delta projection (auto = dim/16)
    """
    def __init__(self, dim: int, cond_dim: int, d_state: int = 16,
                 expand: int = 2, d_conv: int = 2, dt_rank: int | str = 'auto'):
        super().__init__()
        from miocodec.module.adaln_zero import AdaLNZero

        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state
        self.d_conv = d_conv

        if dt_rank == 'auto':
            dt_rank = max(1, math.ceil(dim / 16))
        self.dt_rank = dt_rank

        # AdaLN-Zero: speaker conditioning at input (zero-init)
        self.adaln = AdaLNZero(dim, cond_dim, return_gate=True)

        # Input projection: x → (x_in, z) for main path and gate
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # Causal depthwise conv: k=2, left-pad=1
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=True,
                                kernel_size=d_conv, groups=self.d_inner,
                                padding=0)  # manual causal padding in forward

        # x_proj: input → delta, B, C
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)

        # dt_proj: delta from dt_rank → d_inner
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # A: SSM state transition (learned, input-independent)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, n)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))  # (d_inner, n)

        # D: skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, L, dim)  input sequence
            condition: (B, 1, cond_dim) or (B, cond_dim)  speaker embedding

        Returns:
            (B, L, dim)  output sequence with same shape as input
        """
        residual = x
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)

        B, L, D = x.shape

        # ── AdaLN-Zero: normalize + modulate with speaker ──
        normed, gate = self.adaln(x, condition)  # normed: (B, L, D), gate: (B, L, D)

        # ── Mamba forward ──
        # 1. Input projection
        xz = self.in_proj(normed)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 2. Causal depthwise conv (left-only padding)
        x_in = x_in.transpose(1, 2)  # (B, d_inner, L)
        x_in = F.pad(x_in, (self.d_conv - 1, 0))  # causal: left pad only
        x_in = self.conv1d(x_in)  # (B, d_inner, L)
        x_in = F.silu(x_in)
        x_in = x_in.transpose(1, 2)  # (B, L, d_inner)

        # 3. Selective scan
        y = self._ssm(x_in)  # (B, L, d_inner)

        # 4. Gate with z
        y = y * F.silu(z)

        # 5. Output projection
        out = self.out_proj(y)  # (B, L, D)

        # ── Residual + AdaLN-Zero gate ──
        return residual + gate * out

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state space model (sequential scan, pure PyTorch)."""
        B, L, d_inner = x.shape
        n = self.d_state

        A = -torch.exp(self.A_log.float())  # (d_inner, n)
        D = self.D.float()  # (d_inner,)

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*n)
        delta, ssm_B, ssm_C = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        # Discretize A, B
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * ssm_B.unsqueeze(-2) * x.unsqueeze(-1)

        state = torch.zeros(B, d_inner, n, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(L):
            state = deltaA[:, i] * state + deltaB_u[:, i]
            out_i = (state * ssm_C[:, i, :].unsqueeze(-2)).sum(-1)
            outputs.append(out_i)
        y = torch.stack(outputs, dim=1)

        y = y + x * D.unsqueeze(0).unsqueeze(0)
        return y


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, L, D, Cd = 2, 50, 384, 128
    x = torch.randn(B, L, D)
    cond = torch.randn(B, 1, Cd)

    block = CausalMambaBlock(D, Cd, d_state=16, expand=2, d_conv=2)
    n = sum(p.numel() for p in block.parameters())
    print(f"Params: {n:,} ({n/1e6:.3f}M)")

    out = block(x, cond)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(out.shape)}")
    assert out.shape == x.shape, "Shape mismatch!"

    # Causality test
    half = x[:, :L//2]
    cond_half = cond[:, :1]  # same condition
    out_half = block(half, cond_half)
    diff = (out[:, :L//2] - out_half).abs().max().item()
    print(f"Causality diff: {diff:.6f}")

    print("✅ OK")
