"""
Minimal Causal Student – Scaled (MCSS)

An evolution of MCS that adds the essential architectural components
identified through systematic ablation of the MioCodec teacher, while
staying strictly causal.

Key additions over MCS (each justified by measurement):

1. **Residual conv stem** (replaces flat sequential conv)
   → Depth degradation in 4-layer sequential conv without residuals.
   Residuals allow deeper/wider stem without vanishing gradients.

2. **Multi-scale skip paths** (3 dilations: 16, 32, 64)
   → Single dil=32 skip may miss optimal granularity. Per-channel
   gates let each path contribute where it helps most.

3. **Learnable causal downsample** (replaces odd-frame slicing)
   → Conv1d(stride=2) fuses neighbouring frames before decimation.
   Teacher uses this; odd-frame slicing discards half the mel frames
   with no learnable fusion.

4. **Causal self-attention + RoPE** (new, 4 blocks)
   → GRU alone cannot attend over >400ms history without compression
   loss. Causal windowed attention gives each frame direct access to
   its past context. RoPE provides explicit relative-position signal.

5. **SwiGLU feed-forward** (replaces no FFN)
   → Teacher and MAFS both use SwiGLU. 4× expansion per attention
   block adds representational capacity orthogonal to attention.

6. **Post-attention GRU** (wider: 512 dim)
   → Attention captures discrete dependencies; GRU smooths the
   phonetic trajectory across the full sequence.

7. **Independent per-axis heads** (replaces shared Linear)
   → probe_per_axis.py hypothesis: shared pre_fsq causes axis
   interference. Dedicated tiny heads eliminate the competition.

8. **SiLU-gated stem blocks** (replaces bare conv+SiLU)
   → Gated activation path adds nonlinear gating that bare SiLU lacks.

Parameters: ~16.3M
Target: 5d code cosine ≥ 0.90 (≈ 0.95 projected 768d)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── config ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class McssConfig:
    # Frontend
    in_dim: int = 80               # log-mel bins
    stem_dim: int = 384            # conv stem hidden dim
    stem_kernel: int = 5           # conv kernel size
    stem_dilations: tuple[int, ...] = (1, 2, 4)   # 3 residual blocks

    # Multi-scale skip paths (local-to-mid phonetic, attn handles long context)
    skip_dilations: tuple[int, ...] = (8, 16, 32)

    # Causal attention blocks (at 25 Hz after downsample)
    attn_dim: int = 384
    attn_layers: int = 4
    attn_heads: int = 8            # head_dim = 48
    attn_ff_mult: int = 4          # SwiGLU expansion: 4× → 1536
    attn_context: int = 100         # max causal attention window (frames)
    gate_floor: float = 0.03         # minimum residual gate target
    rope_theta: float = 10000.0

    # GRU (post-attention phonetic smoothing)
    gru_dim: int = 512
    gru_layers: int = 2

    # Heads
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    content_dim: int = 768
    residual_hidden: int = 384     # wider residual head

    dropout: float = 0.1


@dataclass
class McssOutput:
    codes: torch.Tensor              # [B, T25, 5] continuous 5d codes
    projected: torch.Tensor          # [B, 768, T25] final content
    ordinal_logits: torch.Tensor     # [B, 34, T25] per-axis logits
    # Auxiliary diagnostics
    pre_residual: torch.Tensor       # [B, 768, T25] FSQ-only projection
    residual_gate: torch.Tensor      # scalar


# ── building blocks ─────────────────────────────────────────────────────────────

class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding — strictly causal."""
    def __init__(self, *args, **kwargs):
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    @property
    def left_context(self) -> int:
        return self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_context:
            x = F.pad(x, (self.left_context, 0))
        return super().forward(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(dtype)


class GatedConvBlock(nn.Module):
    """RMSNorm → CausalConv(GLU gate) → SiLU → residual.

    Uses a gated linear unit path: input split into feature + gate,
    conv applied to both, SiLU on gate, elementwise multiply.
    """
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(dim)
        # Project to 2*dim for GLU split
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.depthwise = CausalConv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.contract = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  (operates in channel-first for conv)
        residual = x
        x_t = x.transpose(1, 2)                       # [B, T, C]
        normed = self.norm(x_t)
        a, b = self.expand(normed).chunk(2, dim=-1)    # [B, T, C] each
        # Conv path on gate branch
        b_conv = self.depthwise(b.transpose(1, 2))     # [B, C, T]
        b_out = self.contract(F.silu(b_conv).transpose(1, 2))  # [B, T, C]
        # Fused: a provides linear path, b_out provides gated nonlinear path
        out = (a + b_out).transpose(1, 2)              # [B, C, T]
        return residual + F.dropout(out, self.dropout, self.training)


def precompute_rope(dim: int, max_seq_len: int, theta: float, device: torch.device) -> torch.Tensor:
    """Precompute rotary position embeddings as complex numbers."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                      # [max_seq_len, dim//2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a tensor of shape [B, T, n_heads, head_dim]."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: [T, head_dim//2]; reshape for broadcast
    freqs_cis = freqs_cis[: x_.shape[1]].unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim//2]
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


class CausalAttention(nn.Module):
    """Causal multi-head self-attention with RoPE."""
    def __init__(self, dim: int, n_heads: int, max_context: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_context = max_context
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        # RoPE on q and k
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Causal mask + optional window
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        if self.max_context > 0:
            positions = torch.arange(T, device=x.device)
            too_old = positions.unsqueeze(0) < (positions.unsqueeze(1) - self.max_context + 1)
            mask = mask | too_old

        # SDPA: [B, n_heads, T, head_dim]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (LLaMA style)."""
    def __init__(self, dim: int, hidden_mult: int, dropout: float):
        super().__init__()
        hidden = dim * hidden_mult
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.dropout(
            F.silu(self.w1(x)) * self.w3(x),
            self.dropout, self.training,
        ))


class AttnBlock(nn.Module):
    """Pre-norm attention block: RMSNorm → Attention → residual → RMSNorm → SwiGLU → residual."""
    def __init__(self, dim: int, n_heads: int, ff_mult: int, max_context: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalAttention(dim, n_heads, max_context, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ff_mult, dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── FSQ helpers ─────────────────────────────────────────────────────────────────

def _fsq_quantize(z: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    """MioCodec FSQ: bound + round_ste. z: [..., D] → [..., D]"""
    _levels = z.new_tensor(levels, dtype=torch.long)
    half_l = (_levels - 1) * (1 - 1e-3) / 2
    offset = torch.where(_levels % 2 == 0, 0.5, 0.0)
    shift = (offset / half_l).tan()
    bounded = (z + shift).tanh() * half_l - offset
    quantized = bounded + (bounded.round() - bounded).detach()  # STE
    half_width = _levels // 2
    return quantized / half_width.to(dtype=z.dtype)


# ── MCSS model ──────────────────────────────────────────────────────────────────

class McssModel(nn.Module):
    """Minimal Causal Student – Scaled."""

    def __init__(self, config: McssConfig):
        super().__init__()
        self.config = config
        dim = config.stem_dim
        dropout = config.dropout
        n_axes = len(config.fsq_levels)

        # ── 1. Stem: input projection + gated residual conv blocks ──
        self.stem_proj = nn.Linear(config.in_dim, dim, bias=False)
        self.stem_blocks = nn.ModuleList([
            GatedConvBlock(dim, config.stem_kernel, d, dropout)
            for d in config.stem_dilations
        ])

        # ── 2. Multi-scale skip paths ──
        self.skips = nn.ModuleList([
            CausalConv1d(config.in_dim, dim, config.stem_kernel, dilation=d)
            for d in config.skip_dilations
        ])
        # Per-path, per-channel learned gates (init 0 → σ(0)=0.5)
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim, 1))
            for _ in config.skip_dilations
        ])

        # ── 3. Learnable causal downsample: 50 Hz → 25 Hz ──
        # Conv1d(k=2,s=2) fuses adjacent mel frames before decimation
        self.downsample = nn.Conv1d(dim, dim, kernel_size=2, stride=2)

        # ── 4. Causal self-attention blocks (25 Hz) ──
        self.attn_blocks = nn.ModuleList([
            AttnBlock(dim, config.attn_heads, config.attn_ff_mult,
                      config.attn_context, dropout)
            for _ in range(config.attn_layers)
        ])
        self.attn_norm = RMSNorm(dim)

        # ── 5. GRU + causal depthwise smoother ──
        self.gru = nn.GRU(
            dim, config.gru_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=dropout if config.gru_layers > 1 else 0.0,
        )
        gru_out = config.gru_dim
        # Light causal depthwise conv for temporal smoothing (no lookahead)
        self.post_gru_smooth = CausalConv1d(gru_out, gru_out, kernel_size=5, groups=gru_out)

        # ── 6. Heads ──
        # 6a. Independent per-axis pre_fsq heads (eliminates shared-Linear interference)
        self.axis_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_out, 64, bias=False),
                nn.SiLU(),
                nn.Linear(64, 1, bias=True),
            )
            for _ in range(n_axes)
        ])

        # 6b. FSQ projection (initialized from teacher, learnable)
        self.proj_out = nn.Linear(n_axes, config.content_dim)

        # 6c. Factorized ordinal head — auxiliary CE
        self.ordinal_head = nn.Linear(gru_out, sum(config.fsq_levels))

        # 6d. Residual 768d head (wider than MCS: 384 vs 256 hidden)
        self.residual_head = nn.Sequential(
            nn.Linear(gru_out, config.residual_hidden, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(config.residual_hidden, config.content_dim, bias=False),
        )
        self.residual_gate = nn.Parameter(torch.tensor(-2.5))  # σ(-2.5) ≈ 0.076

        # ── 7. RoPE frequencies (precomputed, not learned) ──
        max_seq = 4096
        head_dim = dim // config.attn_heads
        rope = precompute_rope(head_dim, max_seq, config.rope_theta, torch.device("cpu"))
        self.register_buffer("rope", rope, persistent=True)

        self._proj_out_initialized = False

    def load_fsq_projection(self, state: dict[str, torch.Tensor]) -> None:
        """Initialize proj_out from teacher's frozen FSQ projection."""
        self.proj_out.load_state_dict(state, strict=True)
        self._proj_out_initialized = True

    def _skip_fusion(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply all skip paths with learned per-channel gates, sum into stem output."""
        skip_sum: Optional[torch.Tensor] = None
        for skip_conv, gate_param in zip(self.skips, self.skip_gates):
            s = F.silu(skip_conv(mel))                    # [B, dim, T50]
            g = gate_param.sigmoid()                        # [1, dim, 1]
            if skip_sum is None:
                skip_sum = g * s
            else:
                skip_sum = skip_sum + g * s
        return skip_sum

    def forward(self, mel: torch.Tensor) -> McssOutput:
        # mel: [B, 80, T50]

        # ── Stem: input proj + gated conv residual blocks ──
        h = self.stem_proj(mel.transpose(1, 2)).transpose(1, 2)  # [B, dim, T50]
        for block in self.stem_blocks:
            h = block(h)                                          # [B, dim, T50]

        # ── Multi-scale skip fusion ──
        skip = self._skip_fusion(mel)                             # [B, dim, T50]
        h = h + skip

        # ── Learnable downsample: 50 Hz → 25 Hz ──
        h = self.downsample(h)                                    # [B, dim, T25]
        h = h.transpose(1, 2)                                     # [B, T25, dim]

        # ── Causal self-attention blocks ──
        rope = self.rope.to(h.device)
        for block in self.attn_blocks:
            h = block(h, rope)
        h = self.attn_norm(h)                                     # [B, T25, dim]

        # ── GRU refinement ──
        h_gru, _ = self.gru(h)                                    # [B, T25, gru_dim]
        # Causal depthwise temporal smoothing
        h_out = self.post_gru_smooth(h_gru.transpose(1, 2)).transpose(1, 2)  # [B, T25, gru_dim]

        # ── Heads ──
        # 6a. Independent per-axis 5d codes
        axis_outs = []
        for head in self.axis_heads:
            axis_outs.append(head(h_out))                         # [B, T25, 1]
        codes_5d = torch.cat(axis_outs, dim=-1)                  # [B, T25, 5]

        # 6b. FSQ quantize + frozen-ish projection
        codes_quant = _fsq_quantize(codes_5d, self.config.fsq_levels)
        pre_residual = self.proj_out(codes_quant).transpose(1, 2)  # [B, 768, T25]

        # 6c. Ordinal logits
        ordinal = self.ordinal_head(h_out).transpose(1, 2)       # [B, 34, T25]

        # 6d. Residual correction
        residual = self.residual_head(h_out)                       # [B, T25, 768]
        gate_val = self.residual_gate.sigmoid()                    # scalar
        projected = pre_residual + gate_val * residual.transpose(1, 2)

        return McssOutput(
            codes=codes_5d,
            projected=projected,
            ordinal_logits=ordinal,
            pre_residual=pre_residual,
            residual_gate=gate_val,
        )


# ── checkpointing ───────────────────────────────────────────────────────────────

MCSS_FORMAT_VERSION = 1


def save_mcss_checkpoint(
    path: str | Path, model: McssModel, *,
    epoch: int, metrics: dict[str, float],
    optimizer=None, scheduler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": MCSS_FORMAT_VERSION,
        "model_type": "mcss",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_mcss_checkpoint(path: str | Path, *, device="cpu") -> tuple[McssModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if payload.get("model_type") != "mcss":
        raise ValueError("not an MCSS checkpoint")
    config = McssConfig(**payload["config"])
    model = McssModel(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    return model, payload
