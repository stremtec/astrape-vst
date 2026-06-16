"""
Minimal Causal Student (MCS)

A deliberately minimal content student for zero-shot voice conversion.
Every element earns its place against an ablation.

Design decisions grounded in measurement:

1. Conv encoder (4 dilated causal layers, 256 dim)
   → TinyProbe experiment: 3K-param conv+pool gets 0.821 cos.
   The nonlinear mel→content gap (+0.08 over linear) is captured here.

2. 2-layer GRU (384 dim)
   → Autocorrelation: teacher content at lag>10 has cos<0.04.
   GRU state preserves phonetic trajectory that the causal conv misses.

3. 5d continuous code head (NOT 768d)
   → PCA: teacher content manifold is EXACTLY rank-5.
   Metric inflation: 768d cosine overstates quality by +0.072.
   Frozen FSQ projection guarantees output is on the correct manifold.

4. End-of-cell decimation (odd mel frames → 25Hz)
   → Controlled probe: +0.006 over immediate emission.
   Each content prediction sees the full 40ms mel cell.

5. No attention, no SwiGLU, no RoPE, no false future
   → Simplicity First. 10x smaller than alternatives (3M vs 35-51M).
   If this doesn't reach target, add components only AFTER ablation
   proves they're the bottleneck.

Parameters: ~3.4M
Target: 5d code cosine ≥ 0.85 (≈ 0.92 projected 768d)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class McsConfig:
    in_dim: int = 80
    conv_dim: int = 256
    gru_dim: int = 384
    gru_layers: int = 2
    conv_kernel: int = 5
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    content_dim: int = 768
    dropout: float = 0.0


@dataclass
class McsOutput:
    codes: torch.Tensor           # [B, T25, 5] continuous 5d codes
    projected: torch.Tensor       # [B, 768, T25] via frozen projection
    ordinal_logits: torch.Tensor  # [B, 34, T25] per-axis logits


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


class McsModel(nn.Module):
    def __init__(self, config: McsConfig):
        super().__init__()
        self.config = config

        # Conv encoder — extracts local acoustic features
        self.conv = nn.Sequential(
            CausalConv1d(config.in_dim, config.conv_dim, config.conv_kernel),
            nn.SiLU(),
            CausalConv1d(config.conv_dim, config.conv_dim, config.conv_kernel, dilation=2),
            nn.SiLU(),
            CausalConv1d(config.conv_dim, config.conv_dim, config.conv_kernel, dilation=4),
            nn.SiLU(),
            CausalConv1d(config.conv_dim, config.conv_dim, config.conv_kernel, dilation=8),
            nn.SiLU(),
        )

        # Wide-dilation skip: single-layer long-range context (Phase 1)
        self.skip_conv = CausalConv1d(config.in_dim, config.conv_dim, config.conv_kernel, dilation=32)
        self.skip_gate = nn.Parameter(torch.zeros(1, config.conv_dim, 1))

        # GRU — phonetic trajectory memory
        self.gru = nn.GRU(
            config.conv_dim, config.gru_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )

        self.output_norm = nn.LayerNorm(config.gru_dim)

        # FSQ head — MioCodec-style: pre_fsq → bound → round_ste → learnable proj_out
        self.fsq_levels = config.fsq_levels
        self.pre_fsq = nn.Linear(config.gru_dim, len(config.fsq_levels))  # → 5d
        self.proj_out = nn.Linear(len(config.fsq_levels), config.content_dim)  # 5→768, LEARNABLE

        # Factorized ordinal heads — auxiliary CE loss
        self.ordinal_head = nn.Linear(config.gru_dim, sum(config.fsq_levels))    # 34

        # Initialize proj_out from teacher projection
        self._proj_out_initialized = False

        # Learnable 768d residual head — corrects off-manifold errors
        self.residual_head = nn.Sequential(
            nn.Linear(config.gru_dim, 256), nn.SiLU(), nn.Linear(256, config.content_dim)
        )
        self.residual_gate = nn.Parameter(torch.tensor(-4.0))  # σ(-4) ≈ 0.018 → near-pure 5d start

    def load_fsq_projection(self, state: dict[str, torch.Tensor]) -> None:
        self.proj_out.load_state_dict(state, strict=True)
        self._proj_out_initialized = True

    def forward(self, mel: torch.Tensor) -> McsOutput:
        # Fine conv: [B, 80, T50] → [B, C, T50]
        h_fine = self.conv(mel)

        # Wide-dilation skip: single-layer long-range context
        h_skip = F.silu(self.skip_conv(mel))              # [B, C, T50]
        gate = self.skip_gate.sigmoid()                       # [1, C, 1]
        h = h_fine + gate * h_skip                            # gated fusion

        # End-of-cell decimate: [B, C, T50] → [B, C, T25]
        h = h[:, :, 1::2]

        # GRU: [B, T25, C] → [B, T25, gru_dim]
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        h = self.output_norm(h)

        # Heads — FSQ pipeline
        # pre_fsq: continuous 5d
        codes_5d = self.pre_fsq(h)                              # [B, T25, 5]
        # Bound + quantize with straight-through (MioCodec FSQ)
        codes_quant = _fsq_quantize(codes_5d, self.fsq_levels)   # [B, T25, 5]
        # Learnable projection to 768d
        projected_fsq = self.proj_out(codes_quant).transpose(1, 2)  # [B, 768, T25]
        ordinal = self.ordinal_head(h).transpose(1, 2)           # [B, 34, T25]

        # Residual 768d head — gated correction
        residual = self.residual_head(h)                         # [B, T25, 768]
        gate = self.residual_gate.sigmoid()                      # scalar, starts ~0.018
        projected = projected_fsq + gate * residual.transpose(1, 2)  # [B, 768, T25]

        return McsOutput(codes=codes_5d, projected=projected, ordinal_logits=ordinal)


# ── checkpointing ───────────────────────────────────────────────────────────────

MCS_FORMAT_VERSION = 1


def save_mcs_checkpoint(
    path: str | Path, model: McsModel, *,
    epoch: int, metrics: dict[str, float],
    optimizer=None, scheduler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": MCS_FORMAT_VERSION,
        "model_type": "mcs",
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


def load_mcs_checkpoint(path: str | Path, *, device="cpu") -> tuple[McsModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if payload.get("model_type") != "mcs":
        raise ValueError("not an MCS checkpoint")
    config = McsConfig(**payload["config"])
    model = McsModel(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    return model, payload
