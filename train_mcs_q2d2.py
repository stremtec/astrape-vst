"""MCS-Trans with Q2D2 quantizer — drop-in FSQ replacement.

Clone of train_mcs_trans.py with the FSQ (5-axis independent scalar)
replaced by Q2D2 (3-pair 2D grid quantization from ICML 2026 paper
"Two-Dimensional Quantization for Geometry-Aware Audio Coding").

Same conv frontend + causal transformer.  Only the quantizer head
and losses change.

Reference:
  train_mcs_trans.py   — original MCS-Trans with FSQ
  mcs_q2d2.py          — Q2D2 quantizer implementation
  https://arxiv.org/abs/2512.01537
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

from mcs_common import (
    Batch, MioCompactDataset, ContentCollator,
    split_by_speaker, speaker_balanced_subset,
    move_batch, save_checkpoint,
    CausalConv1d, ResidualConvBlock, DepthwiseResidualBlock, CellDownsample,
    DEFAULT_DATA_DIR, DEFAULT_PROJECTION,
    _voiced_weights,
)
from mcs_q2d2 import Q2D2Projection, Q2D2Quantizer, compute_q2d2_perplexity

DEFAULT_OUT_DIR = Path("checkpoints/mcs_trans_q2d2")
DEFAULT_Q2D2_LEVELS = (7, 7, 7, 7, 7, 7)  # rhombic grid, 3 pairs


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class MCSTransQ2D2Config:
    in_dim: int = 80
    conv_dim: int = 320
    trans_dim: int = 512
    n_heads: int = 8
    n_layers: int = 2
    ffn_dim: int = 1024
    window: int = 256
    conv_kernel: int = 5
    stem_dilations: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    stem_block_type: str = "depthwise"  # "standard" | "depthwise"
    q2d2_noise_dropout: float = 0.0  # exploration noise for Q2D2
    q2d2_l2_norm: bool = False  # L2-normalize features before grid snapping
    skip_dilations: tuple[int, ...] = (16, 32)
    dropout: float = 0.0
    # Transformer improvements
    use_rope: bool = False      # Rotary Position Embedding
    use_swiglu: bool = False    # SwiGLU FFN (replaces SiLU)
    # Q2D2 specifics
    q2d2_dim: int = 6
    q2d2_levels: tuple[int, ...] = DEFAULT_Q2D2_LEVELS
    q2d2_grid: str = "rhombic"
    content_dim: int = 768
    # GRL speaker disentanglement
    grl_weight: float = 0.0          # 0 = disabled, ~0.1 is a good start
    grl_num_speakers: int = 0        # set automatically from dataset
    use_wavlm_frontend: bool = False  # use WavLM CNN instead of Mel
    # ── recovered features ──
    delta2_weight: float = 0.0       # 2nd-order temporal smoothness
    contrastive_weight: float = 0.0  # InfoNCE contrastive loss
    contrastive_tau: float = 0.1     # InfoNCE temperature
    ssl_weight: float = 0.0          # WavLM multi-target distillation
    ssl_layers: tuple[int, ...] = (0, 4, 8)  # WavLM layer targets
    # Mamba / SSM (replaces trailing transformer layers)
    mamba_layers: int = 0
    mamba_d_state: int = 16
    # Gumbel-Softmax Q2D2 relaxation annealing (0 = disabled)
    q2d2_gumbel_start: float = 0.0
    q2d2_gumbel_end: float = 0.0
    # WavLM frontend adapter dims
    wavlm_in_dim: int = 512
    wavlm_rate: int = 50      # Hz (50=default, 200=L4 raw)


# ─────────────────────────────────────────────
# GRL: Gradient Reversal Layer
# ─────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    """Reverses gradient sign during backward pass.  Forward is identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


class SpeakerClassifier(nn.Module):
    """Predict speaker ID from content embedding (used with GRL)."""

    def __init__(self, dim: int = 768, num_speakers: int = 108,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_speakers),
        )

    def forward(self, content: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # content: (B, 768, T) → temporal mean → (B, 768)
        if mask is not None:
            # Masked mean: ignore right-padding frames so the pooled speaker
            # embedding isn't diluted by zeros (which biases the GRL signal by
            # a batch-composition-dependent factor).
            m = mask.unsqueeze(1).to(content.dtype)  # (B, 1, T)
            pooled = (content * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
        else:
            pooled = content.mean(dim=-1)  # average over time
        return self.net(pooled)  # (B, num_speakers)


# ─────────────────────────────────────────────
# RoPE (unchanged)
# ─────────────────────────────────────────────

def _precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0,
                           device: torch.device | None = None) -> torch.Tensor:
    """Precompute RoPE cos/sin tables.  dim must be even."""
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, freqs)                     # (max_len, dim/2)
    return torch.cat([freqs, freqs], dim=-1)          # (max_len, dim)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding: x_rot = x*cos + rotate_half(x)*sin."""
    # x: (B, n_heads, T, head_dim)
    x_rot = x * cos + _rotate_half(x) * sin
    return x_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate second half of last dim: [-x2, x1, -x4, x3, ...]"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


# ─────────────────────────────────────────────
# Causal window mask (unchanged)
# ─────────────────────────────────────────────

def _causal_window_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)
    can_attend = (diff >= 0) & (diff < window)
    return torch.zeros(seq_len, seq_len, device=device).masked_fill(~can_attend, -1e4)


# ─────────────────────────────────────────────
# Transformer block (unchanged)
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Causal transformer block with optional RoPE and SwiGLU.

    Args:
        dim: Hidden dimension.
        n_heads: Number of attention heads.
        ffn_dim: FFN intermediate dimension.
        dropout: Dropout rate.
        use_rope: Apply rotary position embeddings to Q/K.
        use_swiglu: Use SwiGLU (gated SiLU) instead of plain SiLU FFN.
    """

    def __init__(self, dim: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.0, use_rope: bool = False,
                 use_swiglu: bool = False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        if use_swiglu:
            # SwiGLU: W2(SiLU(W1(x)) ⊙ W3(x))
            self.w1 = nn.Linear(dim, ffn_dim)      # gate projection
            self.w2 = nn.Linear(ffn_dim, dim)       # output projection
            self.w3 = nn.Linear(dim, ffn_dim)       # value projection
        else:
            # Original: W2(SiLU(W1(x)))
            self.ff = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.SiLU(),
                nn.Linear(ffn_dim, dim),
            )

        self.dropout = dropout

    def _apply_rope_to_qk(self, q: torch.Tensor, k: torch.Tensor,
                          offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        q, k: (B, n_heads, T, head_dim)
        offset: Position offset for the first token (used in streaming).
        """
        T = q.shape[2]
        device = q.device
        # Precompute freqs on first call or reuse cached
        if not hasattr(self, '_rope_cos') or self._rope_cos.shape[0] < offset + T:
            max_len = max(512, offset + T + 256)
            freqs = _precompute_rope_freqs(self.head_dim, max_len, device=device)
            self._rope_cos = freqs.cos()
            self._rope_sin = freqs.sin()

        cos = self._rope_cos[offset:offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,hd)
        sin = self._rope_sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        return q, k

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ── Attention ──
        h = self.norm1(x)

        if self.use_rope:
            # Manual QKV projection (need Q,K separately for RoPE)
            B, T, D = h.shape
            qkv = self.attn.in_proj_weight  # (3*D, D)
            bias = self.attn.in_proj_bias
            # Split projection manually
            qkv_out = F.linear(h, qkv, bias)  # (B, T, 3*D)
            q, k, v = qkv_out.chunk(3, dim=-1)
            # Reshape to multi-head
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            # Apply RoPE
            q, k = self._apply_rope_to_qk(q, k)
            # Merge back for nn.MultiheadAttention's internal scaled_dot_product_attention
            # Actually, use torch's native attention which supports separate q,k,v
            h_attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            h_attn = h_attn.transpose(1, 2).reshape(B, T, D)
            h_attn = self.attn.out_proj(h_attn)
        else:
            h_attn, _ = self.attn(h, h, h, attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False)

        x = x + F.dropout(h_attn, self.dropout, self.training)

        # ── FFN ──
        h = self.norm2(x)
        if hasattr(self, 'w1'):
            # SwiGLU
            gate = F.silu(self.w1(h))
            value = self.w3(h)
            ffn_out = self.w2(gate * value)
        else:
            ffn_out = self.ff(h)
        x = x + F.dropout(ffn_out, self.dropout, self.training)

        return x


# ─────────────────────────────────────────────
# Mamba / SSM block (conditional fallback)
# ─────────────────────────────────────────────

def _mamba_available() -> bool:
    """True only if mamba_ssm is importable AND MPS is not active.

    The spec records selective_scan as unstable on MPS, so we fall back to a
    pure-torch SSM approximation on Apple Silicon (and anywhere mamba_ssm is
    missing).
    """
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
        return True
    except Exception:
        return False


class MambaBlock(nn.Module):
    """Selective state-space block with conditional fallback.

    If mamba_ssm is available and the device supports it, uses the real
    Mamba module.  Otherwise falls back to a pure-torch selective SSM
    approximation (causal, no external dependency), so the feature is
    usable everywhere including MPS / CPU.
    """

    def __init__(self, dim: int, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dropout_p = dropout
        self._use_native = _mamba_available()

        if self._use_native:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=dim, d_state=d_state, expand=1,
            )
        else:
            # ── pure-torch selective SSM approximation ──
            self.norm = nn.LayerNorm(dim)
            self.in_proj = nn.Linear(dim, dim, bias=False)
            # input-dependent gates (selectivity)
            self.gate = nn.Linear(dim, dim, bias=False)
            # A (decay), B, C, D (skip) parameters
            self.A_log = nn.Parameter(torch.zeros(d_state))
            self.B_proj = nn.Linear(dim, d_state, bias=False)
            self.C_proj = nn.Linear(dim, d_state, bias=False)
            self.D = nn.Parameter(torch.ones(1))
            self.out_proj = nn.Linear(dim, dim, bias=False)

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal selective scan in pure torch (RNN-like recurrence)."""
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)                       # (B, T, D)
        g = torch.sigmoid(self.gate(h))           # selectivity gate
        u = u * g
        B_, T_, D_ = u.shape
        S = self.d_state

        A = -torch.exp(self.A_log)                # (S,) negative for decay
        b = self.B_proj(u)                        # (B, T, S)
        c = self.C_proj(u)                        # (B, T, S)

        # per-channel state: (B, S, D)
        state = u.new_zeros(B_, S, D_)
        ys: list[torch.Tensor] = []
        A_decay = torch.exp(A)                    # (S,)
        for t in range(T_):
            # b_t: (B, S)  u_t: (B, D) → update: (B, S, D)
            update = b[:, t].unsqueeze(-1) * u[:, t].unsqueeze(-2)   # (B,S,1)*(B,1,D)
            state = state * A_decay.view(1, S, 1) + update           # (B, S, D)
            # c_t: (B, S)  → output: (B, D)
            y_t = (c[:, t].unsqueeze(-1) * state).sum(dim=1)         # (B, D)
            ys.append(y_t.unsqueeze(1))
        y = torch.cat(ys, dim=1)                  # (B, T, D)
        y = y + self.D * u                        # skip connection
        out = self.out_proj(y)
        return residual + F.dropout(out, self.dropout_p, self.training)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # extra args (attn_mask, kpm) are accepted but ignored by SSM.
        if self._use_native:
            return x + F.dropout(self.mamba(x), self.dropout_p, self.training)
        return self._fallback_forward(x)


# ─────────────────────────────────────────────
# WavLM CNN frontend adapter
# ─────────────────────────────────────────────

class WavLMFrontendAdapter(nn.Module):
    """Projects cached WavLM CNN features (512d) to mel-like 80d.

    Supports rate conversion via stride: when wavlm_rate > 50Hz,
    uses CausalConv1d(s=rate/50) for learned, low-delay downsampling.

    Args:
        in_dim: input feature dim (512 for WavLM)
        out_dim: output dim (80 for mel-compatible)
        wavlm_rate: input rate in Hz (50 default, 200 for L4 raw)
        dropout: dropout rate
    """
    def __init__(self, in_dim: int = 512, out_dim: int = 80,
                 hidden: int = 256, wavlm_rate: int = 50,
                 dropout: float = 0.0):
        super().__init__()
        self.wavlm_rate = wavlm_rate
        stride = max(1, wavlm_rate // 50)  # e.g., 200//50 = 4

        if stride > 1:
            # Learned stride-down: CausalConv + projection.
            # kernel_size == stride ⇒ the decimation window covers ALL input
            # frames (every 200Hz frame reaches exactly one output).  A smaller
            # kernel (e.g. 2) would skip stride-2 of every stride frames, i.e.
            # silently discard ~50% of the cached WavLM features.  Causal
            # (left-padded) ⇒ the wider window is backward context only and
            # adds NO look-ahead latency.
            self.down = CausalConv1d(in_dim, in_dim//2, kernel_size=stride,
                                     stride=stride, groups=in_dim//2)
        else:
            self.down = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(in_dim if stride==1 else in_dim//2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_input, in_dim)
        if self.wavlm_rate > 50:
            x = self.down(x.transpose(1, 2)).transpose(1, 2)  # conv
        return self.net(x)



# ─────────────────────────────────────────────
# MCS-Trans with Q2D2 quantizer
# ─────────────────────────────────────────────

class MCSTransQ2D2(nn.Module):
    """MCS-Trans encoder with Q2D2 quantization replacing FSQ.

    Forward pipeline:
        mel → causal conv → downsample → transformer
            → q2d2_projection → 768d content (MioCodec-compatible)

    The key difference from MCSTrans:
      - No per-axis FSQ heads / ordinal heads.
      - Single latent head (trans_dim → q2d2_dim) fed into Q2D2Projection.
      - Output is the 768d projected content (no intermediate discrete codes).
    """

    def __init__(self, config: MCSTransQ2D2Config):
        super().__init__()
        self.config = config
        dim = config.conv_dim

        # ── conv frontend (depthwise-separable for deeper receptive field) ──
        Block = DepthwiseResidualBlock if config.stem_block_type == "depthwise" else ResidualConvBlock
        self.input_conv = CausalConv1d(config.in_dim, dim, config.conv_kernel)
        self.blocks = nn.ModuleList([
            Block(dim, config.conv_kernel, d, config.dropout)
            for d in config.stem_dilations
        ])
        self.skips = nn.ModuleList([
            CausalConv1d(config.in_dim, dim, config.conv_kernel, dilation=d)
            for d in config.skip_dilations
        ])
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.full((1, dim, 1), -2.0))
            for _ in config.skip_dilations
        ])
        self.downsample = CellDownsample(dim)
        self.proj_in = (
            nn.Linear(dim, config.trans_dim, bias=False)
            if dim != config.trans_dim else nn.Identity()
        )

        # ── transformer (with optional RoPE + SwiGLU) ──
        # Hybrid: leading layers are TransformerBlock, trailing layers may be
        # Mamba/SSM blocks (--mamba-layers).
        n_mamba = max(0, min(config.mamba_layers, config.n_layers))
        n_trans = config.n_layers - n_mamba
        layers: list[nn.Module] = [
            TransformerBlock(config.trans_dim, config.n_heads,
                             config.ffn_dim, config.dropout,
                             use_rope=config.use_rope,
                             use_swiglu=config.use_swiglu)
            for _ in range(n_trans)
        ]
        layers += [
            MambaBlock(config.trans_dim, config.mamba_d_state, config.dropout)
            for _ in range(n_mamba)
        ]
        self.trans_layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(config.trans_dim)
        self.smooth = CausalConv1d(
            config.trans_dim, config.trans_dim, kernel_size=3,
            groups=config.trans_dim,
        )

        # ── NEW: Q2D2 quantizer replaces FSQ ──
        self.q2d2 = Q2D2Projection(
            encoder_dim=config.trans_dim,
            q2d2_dim=config.q2d2_dim,
            content_dim=config.content_dim,
            levels=list(config.q2d2_levels),
            vq_type=config.q2d2_grid,
            noise_dropout=config.q2d2_noise_dropout,
            use_l2_norm=config.q2d2_l2_norm,
            gumbel_temperature=max(config.q2d2_gumbel_start, 0.0),
        )

        # ── optional WavLM frontend adapter ──
        self.use_wavlm_frontend = config.use_wavlm_frontend
        if self.use_wavlm_frontend:
            self.wavlm_adapter: nn.Module | None = WavLMFrontendAdapter(
                in_dim=config.wavlm_in_dim, out_dim=config.in_dim,
                wavlm_rate=config.wavlm_rate, dropout=config.dropout,
            )
        else:
            self.wavlm_adapter = None

        # ── forecast heads: predict teacher[t+1], teacher[t+2] ──
        self.forecast_head_1 = nn.Linear(config.trans_dim, config.content_dim)
        self.forecast_head_2 = nn.Linear(config.trans_dim, config.content_dim)

        # ── SSL distillation projection heads (one per WavLM layer target) ──
        self.ssl_heads = nn.ModuleList([
            nn.Linear(config.trans_dim, config.content_dim)
            for _ in config.ssl_layers
        ])

        # ── optional GRL speaker classifier ──
        self.speaker_classifier: SpeakerClassifier | None = None
        if config.grl_weight > 0 and config.grl_num_speakers > 0:
            self.speaker_classifier = SpeakerClassifier(
                dim=config.content_dim,
                num_speakers=config.grl_num_speakers,
            )

    def forward(
        self, mel: torch.Tensor, padding_mask: torch.Tensor | None = None,
    ) -> dict:
        # ── optional WavLM frontend ──
        # When enabled, expects cached WavLM CNN features (B, 512, T).
        # Adapter projects 512 → 80 → conv stem.
        if self.wavlm_adapter is not None:
            if mel.shape[1] != self.config.wavlm_in_dim:
                raise RuntimeError(
                    f"WavLM frontend expects input dim {self.config.wavlm_in_dim}, "
                    f"got {mel.shape[1]}. Ensure WavLMFrontendDataset is wrapping "
                    f"the base dataset and all samples have 'wavlm_cnn' cache."
                )
            mel = self.wavlm_adapter(mel.transpose(1, 2)).transpose(1, 2)
# 16kHz pipeline: native 50Hz, no interpolation needed

        # ── conv frontend ──
        h = F.silu(self.input_conv(mel))
        for block in self.blocks:
            h = block(h)
        for skip, gate in zip(self.skips, self.skip_gates):
            h = h + torch.sigmoid(gate) * F.silu(skip(mel))
        h = self.downsample(h).transpose(1, 2)       # (B, T25, conv_dim)
        h = self.proj_in(h)                            # (B, T25, trans_dim)

        # ── causal transformer ──
        T = h.shape[1]
        attn_mask = _causal_window_mask(T, self.config.window, h.device)
        kpm = (~padding_mask[:, :T]).float() * -1e4 if padding_mask is not None else None
        for layer in self.trans_layers:
            h = layer(h, attn_mask, kpm)
        h = self.norm(h)
        h = h + self.smooth(h.transpose(1, 2)).transpose(1, 2)

        # ── Q2D2 quantization (replaces FSQ) ──
        content, q2d2_codes = self.q2d2(h, return_codes=True)
        # content:  (B, T, 768)  — MioCodec compatible
        # q2d2_codes: (B, T, 6) — raw quantized latent (for utilization stats)

        # ── forecast predictions ──
        fc1 = self.forecast_head_1(h)  # (B, T, 768)
        fc2 = self.forecast_head_2(h)

        return {
            "projected": content.transpose(1, 2),   # (B, 768, T)
            "q2d2_codes": q2d2_codes,                # (B, T, 6)
            "ordinal": None,
            "forecast_1": fc1.transpose(1, 2),        # (B, 768, T)
            "forecast_2": fc2.transpose(1, 2),
            "hidden": h.transpose(1, 2),              # (B, trans_dim, T) — SSL distill
        }


# ─────────────────────────────────────────────
# Q2D2-aware losses
# ─────────────────────────────────────────────

def contrastive_loss(
    pred_768: torch.Tensor,
    tgt_768: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss to prevent content centroid hedging.

    For each frame, the positive is the matching teacher frame; all other
    frames in the batch are negatives.  Operates per-frame over masked
    positions only.

    Args:
        pred_768: (B, 768, L) student projected content.
        tgt_768: (B, L, 768) teacher content.
        mask: (B, L) bool mask of valid frames.
        tau: temperature.

    Returns:
        scalar contrastive loss.
    """
    pred = pred_768.permute(0, 2, 1)[mask]            # (N, 768)
    tgt = tgt_768[mask]                                 # (N, 768)
    if pred.shape[0] < 2:
        return pred.sum() * 0.0
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(tgt, dim=-1)
    # (N, N) similarity; diagonal = positive
    sim = pred_n @ tgt_n.t() / tau
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(sim, labels)


def ssl_distill_loss(
    hidden: torch.Tensor | None,
    batch: Batch,
    mask: torch.Tensor,
    ssl_heads: nn.ModuleList | None,
    ssl_layers: tuple[int, ...] = (0, 4, 8),
    ts: int = 0,
) -> torch.Tensor:
    """WavLM multi-target distillation (Mimi-style).

    The student's pre-quantization hidden state is projected through
    ``ssl_heads`` (one per target layer) and matched to the cached WavLM
    layer outputs by cosine similarity.

    Args:
        hidden: (B, trans_dim, T) student pre-quantization state.
        batch: training batch carrying ssl_L* targets.
        mask: (B, L) valid-frame mask.
        ssl_heads: ModuleList of Linear(trans_dim → 768), one per target.
        ssl_layers: which WavLM layers to target (used to build attr names).
        ts: time-shift offset.

    Returns:
        scalar distillation loss (mean of 1 - cos over masked frames & layers).
    """
    if hidden is None or ssl_heads is None or len(ssl_heads) == 0:
        return torch.tensor(0.0, device=hidden.device if hidden is not None else "cpu")
    h = hidden[:, :, ts:ts + mask.shape[1]].permute(0, 2, 1)  # (B, L, trans_dim)
    L = h.shape[1]
    ssl_keys = [f"ssl_L{lv}" for lv in ssl_layers[:len(ssl_heads)]]
    cos_terms: list[torch.Tensor] = []
    for head, k in zip(ssl_heads, ssl_keys):
        tgt = getattr(batch, k, None)
        if tgt is None or tgt.numel() == 0:
            continue
        tgt = tgt[:, :L]                                   # (B, L, 768)
        pred = head(h)                                     # (B, L, 768)
        a = F.normalize(pred, dim=-1)
        b = F.normalize(tgt, dim=-1)
        cos = (a * b).sum(dim=-1)                          # (B, L)
        cos_terms.append(
            (1.0 - cos * mask.float()).sum() / mask.float().sum().clamp(min=1)
        )
    if not cos_terms:
        return hidden.sum() * 0.0
    return torch.stack(cos_terms).mean()


def q2d2_losses(
    output: dict,
    batch: Batch,
    args: argparse.Namespace,
    quantizer: Q2D2Quantizer | None = None,
    speaker_classifier: nn.Module | None = None,
    speaker_ids: torch.Tensor | None = None,
    time_shift: int = 0,
    ssl_heads: nn.ModuleList | None = None,
    ssl_layers: tuple[int, ...] = (0, 4, 8),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute losses for Q2D2 quantized output.

    Since Q2D2 has no per-axis ordinal structure, losses are:
      - content_cos:  cosine similarity between projected 768d and teacher content
      - content_l1:   L1 between projected and teacher content
      - q2d2_perplexity (metrics only): codebook utilization

    Args:
        output: Model output dict with 'projected' and 'q2d2_codes'.
        batch: Training batch.
        args: Training arguments.
        quantizer: Optional Q2D2Quantizer for utilization stats.

    Returns:
        loss, metrics dict.
    """
    projected = output["projected"]                     # (B, 768, T)
    q2d2_codes = output.get("q2d2_codes")               # (B, T, 6) or None

    ts = time_shift
    length = min(projected.shape[2] - ts, batch.content.shape[1] - ts,
                 batch.mask.shape[1] - ts)
    if length < 2:
        zero = projected.sum() * 0.0
        return zero, {"loss": float(zero.detach().cpu()), "cos768": 0.0}
    mask = batch.mask[:, ts:ts + length]

    # ── time-shifted alignment ──
    # student[t] compares with teacher[t-ts]
    pred_768 = projected[:, :, ts:ts + length]           # (B, 768, L)
    if ts > 0:
        tgt_768 = batch.content[:, :length]               # student[ts..] ↔ teacher[0..]
    else:
        tgt_768 = batch.content[:, :length]               # (B, L, 768)

    # voiced weighting
    voiced_boost = getattr(args, "voiced_boost", 1.0)
    vw = _voiced_weights(batch.mel, length, voiced_boost)  # (B, L)
    weighted_mask_sum = (vw * mask.float()).sum().clamp(min=1)

    # ── content cosine (primary quality metric) ──
    # Compute over masked frames: cos per batch item then average
    pred_masked = pred_768.permute(0, 2, 1)[mask]       # (N_valid, 768)
    tgt_masked = tgt_768[mask]                            # (N_valid, 768)
    cos768 = F.cosine_similarity(pred_masked, tgt_masked, dim=-1).mean()
    cos768_loss = 1.0 - cos768

    # ── content L1 ──
    pred_flat = pred_768.permute(0, 2, 1)                # (B, L, 768)
    l1_per_frame = (pred_flat - tgt_768).abs().mean(dim=-1)  # (B, L)
    content_l1 = ((l1_per_frame * vw * mask.float()).sum() / weighted_mask_sum)

    # ── delta (temporal smoothness) ──
    if length >= 2:
        delta_mask = mask[:, 1:] & mask[:, :-1]
        pred_delta = pred_flat[:, 1:] - pred_flat[:, :-1]
        tgt_delta = tgt_768[:, 1:] - tgt_768[:, :-1]
        delta_weights = 0.5 * (vw[:, 1:] + vw[:, :-1])
        delta = F.smooth_l1_loss(
            pred_delta[delta_mask], tgt_delta[delta_mask], reduction="mean"
        )
    else:
        delta = projected.sum() * 0.0

    # ── delta2 (2nd-order temporal smoothness) ──
    delta2 = projected.sum() * 0.0
    if length >= 3:
        d2_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        if d2_mask.any():
            pred_d2 = pred_flat[:, 2:] - 2 * pred_flat[:, 1:-1] + pred_flat[:, :-2]
            tgt_d2 = tgt_768[:, 2:] - 2 * tgt_768[:, 1:-1] + tgt_768[:, :-2]
            delta2 = F.smooth_l1_loss(
                pred_d2[d2_mask], tgt_d2[d2_mask], reduction="mean"
            )

    # ── total loss ──
    loss = (args.content_cos_weight * cos768_loss +
            args.content_l1_weight * content_l1 +
            args.delta_weight * delta +
            getattr(args, "delta2_weight", 0.0) * delta2)

    # ── forecast loss ──
    forecast_weight = getattr(args, "forecast_weight", 0.0)
    forecast_loss_val: float = 0.0
    if forecast_weight > 0:
        fc1 = output.get("forecast_1")
        fc2 = output.get("forecast_2")
        # Lf bounded so the target/mask shifts (t+1, t+2) stay in range.
        Lf = min(length - 2, batch.content.shape[1] - 2)
        if fc1 is not None and fc2 is not None and length >= 3 and Lf >= 1:
            fc1_flat = fc1[:, :, ts:ts + length].permute(0, 2, 1)[:, :Lf, :]
            fc2_flat = fc2[:, :, ts:ts + length].permute(0, 2, 1)[:, :Lf, :]
            tgt_fc1 = batch.content[:, 1:1 + Lf]
            tgt_fc2 = batch.content[:, 2:2 + Lf]
            # Mask: predict only where BOTH the source frame and the future
            # target frame are valid (excludes right-padding).
            m1 = (mask[:, :Lf] & mask[:, 1:1 + Lf]).float()
            m2 = (mask[:, :Lf] & mask[:, 2:2 + Lf]).float()
            fl1 = (F.mse_loss(fc1_flat, tgt_fc1, reduction="none").mean(-1) * m1
                   ).sum() / m1.sum().clamp(min=1)
            fl2 = (F.mse_loss(fc2_flat, tgt_fc2, reduction="none").mean(-1) * m2
                   ).sum() / m2.sum().clamp(min=1)
            fl = (fl1 + fl2) * 0.5
            forecast_loss_val = float(fl.detach().cpu())
            loss = loss + forecast_weight * fl

    # ── GRL speaker disentanglement loss ──
    grl_loss_val: float = 0.0
    grl_acc_val: float = 0.0
    if speaker_classifier is not None and speaker_ids is not None:
        grl_weight = getattr(args, "grl_weight", 0.0)
        if grl_weight > 0:
            # Reverse gradient: classifier tries to predict speaker,
            # but encoder gets reversed gradient → strips speaker info.
            # Pool over the valid (masked) loss region only.
            grl_content = grad_reverse(projected[:, :, ts:ts + length], grl_weight)
            speaker_logits = speaker_classifier(grl_content, mask)
            grl_loss = F.cross_entropy(speaker_logits, speaker_ids)
            loss = loss + grl_loss
            grl_loss_val = float(grl_loss.detach().cpu())
            grl_acc_val = float(
                (speaker_logits.argmax(dim=-1) == speaker_ids).float().mean().cpu()
            )

    # ── InfoNCE contrastive loss ──
    contrastive_loss_val: float = 0.0
    contrastive_weight = getattr(args, "contrastive_weight", 0.0)
    if contrastive_weight > 0:
        c_loss = contrastive_loss(
            pred_768, tgt_768, mask,
            tau=getattr(args, "contrastive_tau", 0.1),
        )
        contrastive_loss_val = float(c_loss.detach().cpu())
        loss = loss + contrastive_weight * c_loss

    # ── WavLM SSL multi-target distillation ──
    ssl_loss_val: float = 0.0
    ssl_weight = getattr(args, "ssl_weight", 0.0)
    if ssl_weight > 0:
        hidden = output.get("hidden")              # (B, trans_dim, T)
        s_loss = ssl_distill_loss(hidden, batch, mask, ssl_heads, ssl_layers, ts)
        ssl_loss_val = float(s_loss.detach().cpu())
        loss = loss + ssl_weight * s_loss

    # ── metrics ──
    metrics: dict[str, float] = {
        "loss": float(loss.detach().cpu()),
        "cos768": float(cos768.detach().cpu()),
        "content_l1": float(content_l1.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "delta2": float(delta2.detach().cpu()),
        "grl_loss": grl_loss_val,
        "grl_acc": grl_acc_val,
        "forecast_loss": forecast_loss_val,
        "contrastive_loss": contrastive_loss_val,
        "ssl_loss": ssl_loss_val,
    }

    # Q2D2 utilization stats (diagnostic, no gradient)
    if quantizer is not None and q2d2_codes is not None:
        with torch.no_grad():
            stats = compute_q2d2_perplexity(quantizer, q2d2_codes)
            metrics["q2d2_usage"] = stats["overall_usage"]
            for i in range(quantizer.num_pairs):
                metrics[f"q2d2_pair{i}_usage"] = stats[f"pair_{i}_usage"]

    return loss, metrics


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

class CenterFalseMelWrapper(Dataset):
    """Wraps a base dataset, recomputing mel from raw audio with center=False."""

    SAMPLE_RATE = 44100

    def __init__(self, base, srcs):
        self.base = base
        self.src = srcs

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        import soundfile as sf
        import torchaudio

        sample = self.base[idx]
        si = int(sample['idx'])
        w, sr = sf.read(str(Path(self.src[si])), dtype='float32')
        w = torch.from_numpy(np.asarray(w))
        if w.ndim == 2:
            w = w.mean(1)
        if sr != self.SAMPLE_RATE:
            w = torchaudio.functional.resample(
                w.unsqueeze(0), sr, self.SAMPLE_RATE
            ).squeeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            self.SAMPLE_RATE, 2048, 882, n_mels=80, f_min=0.0,
            f_max=self.SAMPLE_RATE / 2.0, power=1, center=False,
        )(w.unsqueeze(0))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        sample['mel'] = mel[0]
        return sample


class WavLMFrontendDataset(Dataset):
    """Replaces 'mel' with cached WavLM CNN features (cache_wavlm_cnn.py).

    The npz cache stores 'wavlm_cnn' as (T, 512) float32 at ~46 Hz.
    We transpose to (512, T) so ContentCollator treats the second dim
    (time) correctly for cropping/padding.

    ContentCollator's mel_frames then controls the number of WavLM
    output frames to keep, just like for mel.
    """

    def __init__(self, base: Dataset, data_dir: Path, wavlm_dir: str = "wavlm_16k"):
        self.base = base
        self.data_dir = data_dir
        self.wavlm_dir = wavlm_dir

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        si = int(sample['idx'])
        # Load cached WavLM CNN from .npy file
        cnn_path = Path(self.wavlm_dir) / f's_{si:05d}.npy' if Path(self.wavlm_dir).is_absolute() else self.data_dir / self.wavlm_dir / f's_{si:05d}.npy'
        if not cnn_path.exists():
            raise RuntimeError(
                f"WavLM frontend: sample s_{si:05d} has no wavlm_cnn cache. "
                f"Re-run cache_wavlm_cnn.py or disable --wavlm-frontend."
            )
        cnn = np.load(cnn_path, allow_pickle=False)
        # wavlm_cnn: (T, 512) → transpose → (512, T)
        sample['mel'] = torch.from_numpy(cnn.astype(np.float32)).t()
        return sample



@torch.inference_mode()
def evaluate(
    model: MCSTransQ2D2,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    quantizer: Q2D2Quantizer | None = None,
    speaker_to_id: dict[str, int] | None = None,
) -> dict[str, float]:
    model.eval()
    buckets: dict[str, list[float]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        speaker_ids = None
        if model.speaker_classifier is not None and speaker_to_id is not None:
            speaker_ids = torch.tensor(
                [speaker_to_id[s] for s in batch.speakers],
                device=device, dtype=torch.long,
            )
        output = model(batch.mel, padding_mask=batch.mask)
        _, metrics = q2d2_losses(output, batch, args, quantizer,
                                 model.speaker_classifier, speaker_ids,
                                 time_shift=args.time_shift,
                                 ssl_heads=model.ssl_heads,
                                 ssl_layers=model.config.ssl_layers)
        for key, value in metrics.items():
            buckets.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(vs)) if vs else 0.0
            for key, vs in buckets.items()}


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_q2d2_levels(s: str) -> tuple[int, ...]:
    """Parse comma-separated level string, e.g. '7,7,7,7,7,7'."""
    parts = [v.strip() for v in s.split(",") if v.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("empty level list")
    levels = tuple(int(p) for p in parts)
    if len(levels) % 2 != 0:
        raise argparse.ArgumentTypeError(
            f"Q2D2 requires even number of levels, got {len(levels)}"
        )
    return levels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCS-Trans with Q2D2 quantizer (replaces FSQ)",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mel-frames", type=int, default=200)
    p.add_argument("--eval-mel-frames", type=int, default=300)
    p.add_argument("--probe-samples", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--init-from", type=Path, default=None,
                   help="Initialize from FSQ checkpoint (proj_out skipped).")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--run-name", default="mcs_trans_q2d2")

    # Transformer architecture (unchanged)
    p.add_argument("--conv-dim", type=int, default=320)
    p.add_argument("--trans-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--window", type=int, default=256)

    # Transformer improvements
    p.add_argument("--rope", action="store_true",
                   help="Use Rotary Position Embedding in transformer attention.")
    p.add_argument("--swiglu", action="store_true",
                   help="Use SwiGLU FFN (gated SiLU) instead of plain SiLU.")

    # Q2D2-specific
    p.add_argument("--q2d2-dim", type=int, default=6,
                   help="Q2D2 latent dimension (must be even).")
    p.add_argument("--q2d2-levels", type=parse_q2d2_levels,
                   default=DEFAULT_Q2D2_LEVELS,
                   help="Per-dimension levels, comma-separated (even count). "
                        "Default: 7,7,7,7,7,7 → 3 rhombic pairs.")
    p.add_argument("--q2d2-grid", default="rhombic",
                   choices=["rhombic", "hexagon", "rectangle"],
                   help="2D grid geometry type.")
    p.add_argument("--q2d2-l2-norm", action="store_true",
                   help="L2-normalize Q2D2 feature pairs before grid snap.")

    # Loss weights
    p.add_argument("--content-cos-weight", type=float, default=1.0,
                   help="Weight on (1 - cos768) content fidelity loss.")
    p.add_argument("--content-l1-weight", type=float, default=0.5,
                   help="Weight on L1 content loss.")
    p.add_argument("--delta-weight", type=float, default=0.04,
                   help="Weight on temporal delta smoothness.")
    p.add_argument("--voiced-boost", type=float, default=1.0,
                   help="Voiced frame weight multiplier.")

    # GRL speaker disentanglement
    p.add_argument("--grl-weight", type=float, default=0.0,
                   help="GRL speaker disentanglement weight (0=disabled, ~0.1).")
    p.add_argument("--grl-num-speakers", type=int, default=0,
                   help="Number of speakers for GRL classifier (auto if 0).")
    p.add_argument("--time-shift", type=int, default=0,
                   help="Shift teacher target by Δ frames. 1 frame = 40ms.")
    p.add_argument("--forecast-weight", type=float, default=0.0,
                   help="Weight on forecast heads.")
    p.add_argument("--stem-block-type", default="depthwise",
                   choices=["standard","depthwise"],
                   help="Conv stem block type.")
    p.add_argument("--center-false", action="store_true",
                   help="Compute center=False mel on-the-fly from raw audio.")

    # Decoder-in-loop (original audio feedback)
    p.add_argument("--decoder-wave-weight", type=float, default=0.0,
                   help="Weight for MR-STFT loss against original wav "
                        "via frozen Mio decoder.")
    p.add_argument("--decoder-wave-prob", type=float, default=0.5,
                   help="Fraction of steps that compute decoder wave loss.")
    p.add_argument("--decoder-n-ffts", default="512,1024,2048")

    # ── recovered features ──
    p.add_argument("--delta2-weight", type=float, default=0.0,
                   help="Weight on 2nd-order temporal smoothness.")
    p.add_argument("--contrastive-weight", type=float, default=0.0,
                   help="Weight on InfoNCE contrastive loss.")
    p.add_argument("--contrastive-tau", type=float, default=0.1,
                   help="InfoNCE temperature τ.")
    p.add_argument("--ssl-weight", type=float, default=0.0,
                   help="Weight on WavLM multi-target distillation (L0,L4,L8).")
    p.add_argument("--ssl-layers", default="0,4,8",
                   help="Comma-separated WavLM layer targets for SSL distillation.")
    p.add_argument("--mamba-layers", type=int, default=0,
                   help="Number of trailing layers replaced by Mamba/SSM.")
    p.add_argument("--mamba-d-state", type=int, default=16,
                   help="Mamba SSM state dimension (d_state).")
    p.add_argument("--q2d2-gumbel-start", type=float, default=0.0,
                   help="Initial Gumbel temperature for Q2D2 relaxation.")
    p.add_argument("--q2d2-gumbel-end", type=float, default=0.0,
                   help="Final Gumbel temperature for Q2D2 relaxation (anneal).")
    p.add_argument("--wavlm-frontend", action="store_true",
                   help="Use cached WavLM CNN features instead of mel.")
    p.add_argument("--wavlm-dir", default="wavlm_16k",
                   help="Subdirectory for WavLM cache (default: wavlm_16k, use wavlm_L4 for L4)")
    p.add_argument("--wavlm-rate", type=int, default=50,
                   help="WavLM feature rate in Hz (50=default, 200=L4 raw)")

    return p.parse_args()


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.resume_from is not None and args.init_from is not None:
        raise ValueError("--resume-from and --init-from are mutually exclusive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── data ──
    with np.load(args.data_dir / "meta.npz", allow_pickle=False) as meta:
        n_samples = int(meta["n_samples"])
        speakers = meta["spk_names"][:n_samples].astype(str)
        source_files = meta["source_files"][:n_samples].astype(str)
        cache_format = str(meta["cache_format"].item())
    if cache_format != "compact-fp16-ctc-v2":
        raise ValueError(f"unexpected cache format: {cache_format}")

    # Speaker ID mapping for GRL
    unique_speakers = sorted(set(speakers))
    speaker_to_id = {s: i for i, s in enumerate(unique_speakers)}
    print(f"Speakers: {len(unique_speakers)} unique, "
          f"GRL={args.grl_weight > 0}", flush=True)

    train_idx, val_idx = split_by_speaker(speakers, args.val_fraction, args.seed)
    probe_idx = speaker_balanced_subset(val_idx, speakers, args.probe_samples, args.seed)

    train_ds = MioCompactDataset(args.data_dir, train_idx, speakers)
    probe_ds = MioCompactDataset(args.data_dir, probe_idx, speakers)

    if args.center_false:
        train_ds = CenterFalseMelWrapper(train_ds, source_files)
        probe_ds = CenterFalseMelWrapper(probe_ds, source_files)
        print("center=False mel: computing on-the-fly from raw audio", flush=True)

    if args.wavlm_frontend:
        train_ds = WavLMFrontendDataset(train_ds, args.data_dir, args.wavlm_dir)
        probe_ds = WavLMFrontendDataset(probe_ds, args.data_dir, args.wavlm_dir)
        print(f"WavLM frontend: using cached {args.wavlm_dir} (512d) instead of mel",
              flush=True)

    # Frontend frames per 25Hz teacher token: Mel/50Hz-WavLM = 2, 200Hz L4 = 8.
    frames_per_token = (args.wavlm_rate // 25) if args.wavlm_frontend else 2
    # persistent_workers: keep workers alive across epochs instead of respawning
    # them each epoch (the respawn leaks ~17 fds/epoch and hits macOS's default
    # ulimit -n=256 after ~15 epochs). Requires num_workers > 0.
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        collate_fn=ContentCollator(args.mel_frames, args.seed,
                                   pad_mel_multiple=frames_per_token,
                                   frames_per_token=frames_per_token),
        generator=torch.Generator().manual_seed(args.seed),
    )
    probe_loader = DataLoader(
        probe_ds,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        collate_fn=ContentCollator(args.eval_mel_frames, args.seed + 999,
                                   pad_mel_multiple=frames_per_token,
                                   frames_per_token=frames_per_token),
    )

    # ── config ──
    ssl_layer_list = tuple(
        int(v.strip()) for v in args.ssl_layers.split(",") if v.strip()
    )
    config = MCSTransQ2D2Config(
        conv_dim=args.conv_dim,
        trans_dim=args.trans_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        window=args.window,
        use_rope=args.rope,
        use_swiglu=args.swiglu,
        q2d2_dim=args.q2d2_dim,
        q2d2_levels=args.q2d2_levels,
        q2d2_grid=args.q2d2_grid,
        grl_weight=args.grl_weight,
        grl_num_speakers=args.grl_num_speakers if args.grl_num_speakers > 0 else len(unique_speakers),
        stem_block_type=args.stem_block_type,
        use_wavlm_frontend=args.wavlm_frontend,
        wavlm_rate=args.wavlm_rate,
        delta2_weight=args.delta2_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_tau=args.contrastive_tau,
        ssl_weight=args.ssl_weight,
        ssl_layers=ssl_layer_list,
        mamba_layers=args.mamba_layers,
        mamba_d_state=args.mamba_d_state,
        q2d2_gumbel_start=args.q2d2_gumbel_start,
        q2d2_gumbel_end=args.q2d2_gumbel_end,
        q2d2_l2_norm=args.q2d2_l2_norm,
    )

    # ── model ──
    checkpoint = None
    checkpoint_mode = None
    start_epoch, best_cos, global_step = 0, -1.0, 0
    checkpoint_path = args.resume_from or args.init_from

    if checkpoint_path is not None:
        checkpoint_mode = "resume" if args.resume_from is not None else "init"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if checkpoint_mode == "init":
            # Loading from FSQ checkpoint: extract config and weights
            # but skip axis_heads, ordinal_heads, proj_out (replaced by Q2D2)
            saved_cfg = checkpoint.get("config", {})
            # Map old config to new Q2D2 config (only keep shared fields)
            config = MCSTransQ2D2Config(
                conv_dim=saved_cfg.get("conv_dim", config.conv_dim),
                trans_dim=saved_cfg.get("trans_dim", config.trans_dim),
                n_heads=saved_cfg.get("n_heads", config.n_heads),
                n_layers=saved_cfg.get("n_layers", config.n_layers),
                ffn_dim=saved_cfg.get("ffn_dim", config.ffn_dim),
                window=saved_cfg.get("window", config.window),
                use_rope=args.rope,
                use_swiglu=args.swiglu,
                q2d2_dim=args.q2d2_dim,
                q2d2_levels=args.q2d2_levels,
                q2d2_grid=args.q2d2_grid,
                grl_weight=args.grl_weight,
                grl_num_speakers=len(unique_speakers),
                use_wavlm_frontend=args.wavlm_frontend,
                delta2_weight=args.delta2_weight,
                contrastive_weight=args.contrastive_weight,
                contrastive_tau=args.contrastive_tau,
                ssl_weight=args.ssl_weight,
                ssl_layers=ssl_layer_list,
                mamba_layers=args.mamba_layers,
                mamba_d_state=args.mamba_d_state,
                q2d2_gumbel_start=args.q2d2_gumbel_start,
                q2d2_gumbel_end=args.q2d2_gumbel_end,
                q2d2_l2_norm=args.q2d2_l2_norm,
            )
            # Resume from Q2D2 checkpoint
            saved_cfg = checkpoint.get("config", {})
            known = {f.name for f in __import__("dataclasses").fields(MCSTransQ2D2Config)}
            cfg_filtered = {}
            for k, v in saved_cfg.items():
                if k in known:
                    cfg_filtered[k] = tuple(v) if isinstance(v, list) else v
            config = MCSTransQ2D2Config(**cfg_filtered)

    model = MCSTransQ2D2(config).to(device)

    # Load shared weights from FSQ checkpoint (conv + transformer)
    if checkpoint is not None and checkpoint_mode == "init":
        fsq_state = checkpoint.get("state_dict", checkpoint)
        # Load all matching keys. Skip only old FSQ-specific heads.
        # Q2D2 weights ARE loaded if present (Q2D2→Q2D2 transfer).
        shared_state = {}
        for k, v in fsq_state.items():
            if k.startswith("axis_heads."):
                continue  # old FSQ, removed in Q2D2
            if k.startswith("ordinal_heads."):
                continue  # old FSQ, removed in Q2D2
            if k.startswith("proj_out."):
                continue  # old FSQ proj_out, Q2D2 has its own
            shared_state[k] = v

        missing, unexpected = model.load_state_dict(shared_state, strict=False)
        if missing:
            print(f"Q2D2 init: missing keys (new Q2D2 head, expected): {len(missing)}")
            for m in missing[:5]:
                print(f"  {m}")
        if unexpected:
            print(f"Q2D2 init: unexpected keys (old FSQ): {len(unexpected)}")
        print(f"Initialized conv+transformer from {checkpoint_path}", flush=True)

    elif checkpoint is not None and checkpoint_mode == "resume":
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}", flush=True)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("metrics", {}).get("global_step", 0))
        current_cos = float(checkpoint.get("metrics", {}).get("probe", {}).get("cos768", -1.0))
        best_cos = max(float(checkpoint.get("best_probe_cos768", -1.0)), current_cos)
        print(f"Resumed from {checkpoint_path} at epoch={start_epoch} "
              f"global_step={global_step} best_cos768={best_cos:.4f}", flush=True)

    # ── optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs),
    )

    if checkpoint is not None and checkpoint_mode == "resume":
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                print("Optimizer mismatch, starting fresh")
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    # ── decoder-in-loop setup ──
    mio = None
    decoder_n_ffts: tuple[int, ...] = ()
    if args.decoder_wave_weight > 0:
        from eval_mcs_trans_audio import load_mio, load_wave, SAMPLE_RATE  # noqa: E402
        from mcs_common import multi_resolution_stft_loss  # noqa: E402
        print("Loading frozen MioCodec for decoder-in-loop ...", flush=True)
        mio = load_mio(device).eval()
        for p in mio.parameters():
            p.requires_grad_(False)
        decoder_n_ffts = tuple(
            int(v) for v in args.decoder_n_ffts.split(",") if v.strip()
        )
        print(f"  decoder_wave_weight={args.decoder_wave_weight} "
              f"prob={args.decoder_wave_prob} n_ffts={decoder_n_ffts}", flush=True)

    # ── info ──
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    q2d2_cbs = model.q2d2.codebook_size

    print(f"Train={len(train_idx)} Probe={len(probe_idx)} "
          f"speakers={len(set(speakers))} "
          f"params={params:,} trainable={trainable:,} device={device}", flush=True)
    print(f"Arch: trans_dim={config.trans_dim} n_heads={config.n_heads} "
          f"n_layers={config.n_layers} ffn_dim={config.ffn_dim} "
          f"window={config.window} "
          f"rope={config.use_rope} swiglu={config.use_swiglu}", flush=True)
    print(f"Q2D2: grid={config.q2d2_grid} dim={config.q2d2_dim} "
          f"levels={list(config.q2d2_levels)} "
          f"codebook={q2d2_cbs:,} (FSQ=12,800)", flush=True)
    print(f"Objective: content_cos={args.content_cos_weight} "
          f"content_l1={args.content_l1_weight} "
          f"delta={args.delta_weight} "
          f"delta2={args.delta2_weight} "
          f"decoder_wave={args.decoder_wave_weight}", flush=True)
    print(f"Extra: contrastive={args.contrastive_weight} "
          f"ssl={args.ssl_weight} "
          f"mamba_layers={config.mamba_layers}/{config.n_layers} "
          f"gumbel={args.q2d2_gumbel_start:.1f}→{args.q2d2_gumbel_end:.1f} "
          f"wavlm_frontend={args.wavlm_frontend} "
          f"stem={args.stem_block_type}"
          f"l2_norm={config.q2d2_l2_norm}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        probe = evaluate(model, probe_loader, device, args, model.q2d2.quantizer,
                          speaker_to_id)
        print(f"eval probe cos768={probe.get('cos768', 0):.4f} "
              f"loss={probe.get('loss', 0):.4f} "
              f"q2d2_usage={probe.get('q2d2_usage', 0):.3f}", flush=True)
        return

    # ── training loop ──
    quantizer = model.q2d2.quantizer  # for utilization stats
    run_started = time.time()

    def _gumbel_temp(epoch: int) -> float:
        """Linear annealing from q2d2_gumbel_start → q2d2_gumbel_end."""
        start = args.q2d2_gumbel_start
        end = args.q2d2_gumbel_end
        if start <= 0.0:
            return 0.0
        E = max(1, args.epochs - start_epoch)
        frac = min(1.0, epoch / max(1, E - 1))
        return start + (end - start) * frac

    for epoch in range(start_epoch, args.epochs):
        # ── Gumbel temperature annealing ──
        tau = _gumbel_temp(epoch)
        if args.q2d2_gumbel_start > 0:
            quantizer.gumbel_temperature = tau

        model.train()
        totals: dict[str, float] = {}
        step_started = time.time()

        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = move_batch(batch, device)

            # Build speaker IDs for GRL
            speaker_ids = None
            if model.speaker_classifier is not None:
                speaker_ids = torch.tensor(
                    [speaker_to_id[s] for s in batch.speakers],
                    device=device, dtype=torch.long,
                )

            output = model(batch.mel, padding_mask=batch.mask)
            loss, metrics = q2d2_losses(output, batch, args, quantizer,
                                        model.speaker_classifier, speaker_ids,
                                        time_shift=args.time_shift,
                                        ssl_heads=model.ssl_heads,
                                        ssl_layers=config.ssl_layers)

            # Decoder-in-loop: MR-STFT vs original wav
            if (mio is not None and
                    args.decoder_wave_weight > 0 and
                    random.random() < args.decoder_wave_prob):
                if True:  # no try/except - let errors surface
                    idx_in_batch = random.randrange(len(batch.speakers))
                    row = int(batch.indices[idx_in_batch].item())
                    src_path = Path(str(source_files[row]))
                    if src_path.exists():
                        from eval_mcs_trans_audio import load_wave, SAMPLE_RATE  # noqa: E402
                        from mcs_common import multi_resolution_stft_loss  # noqa: E402
                        orig_wave = load_wave(src_path, SAMPLE_RATE,
                                              max_seconds=10.0).to(device)
                        # Crop to match the EXACT mel window used in training
                        mel_crop_start = int(batch.crop_starts[idx_in_batch].item())
                        wav_start = mel_crop_start * 882
                        wav_len = int(args.mel_frames * 882)
                        orig_wave = orig_wave[wav_start:wav_start + wav_len]
                        with torch.no_grad():
                            feats = mio.encode(orig_wave.unsqueeze(0),
                                              return_content=True,
                                              return_global=True)
                            global_emb = feats.global_embedding.unsqueeze(0)
                            stft_len = mio._calculate_target_stft_length(
                                orig_wave.numel()
                            )
                        content_i = output["projected"][idx_in_batch].unsqueeze(0)
                        content_i_t = content_i.transpose(1, 2)
                        n_frames = min(content_i_t.shape[1],
                                       feats.content_embedding.shape[0])
                        pred_wave = mio.forward_wave(
                            content_i_t[:, :n_frames], global_emb,
                            stft_length=stft_len,
                        ).squeeze(0)
                        tgt_len = min(pred_wave.shape[-1],
                                      orig_wave.shape[-1])
                        wave_loss = multi_resolution_stft_loss(
                            pred_wave[:tgt_len], orig_wave[:tgt_len],
                            decoder_n_ffts,
                        )
                        loss = loss + args.decoder_wave_weight * wave_loss
                        metrics["decoder_wave"] = float(
                            wave_loss.detach().cpu()
                        )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            global_step += 1

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value

            if step % args.log_every == 0 or step == args.steps_per_epoch:
                denom = max(step, 1)
                elapsed = time.time() - step_started
                print(
                    f"E{epoch:03d} step={step:04d}/{args.steps_per_epoch} "
                    f"loss={totals['loss']/denom:.4f} "
                    f"cos768={totals['cos768']/denom:.4f} "
                    f"l1={totals.get('content_l1',0)/denom:.4f} "
                    f"usage={totals.get('q2d2_usage',0)/denom:.3f} "
                    f"{elapsed/max(step,1):.3f}s/step",
                    flush=True,
                )

        scheduler.step()
        probe = evaluate(model, probe_loader, device, args, quantizer,
                          speaker_to_id)
        metrics_full = {
            "epoch": epoch,
            "global_step": global_step,
            "probe": probe,
            "elapsed_seconds": time.time() - run_started,
        }

        print(
            f"E{epoch:03d} probe cos768={probe.get('cos768',0):.4f} "
            f"loss={probe.get('loss',0):.4f} "
            f"l1={probe.get('content_l1',0):.4f} "
            f"q2d2_usage={probe.get('q2d2_usage',0):.3f}",
            flush=True,
        )

        save_checkpoint(
            args.out_dir / f"{args.run_name}.last.pt",
            model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
        )

        current_cos = probe.get("cos768", 0.0)
        if current_cos > best_cos:
            best_cos = current_cos
            save_checkpoint(
                args.out_dir / f"{args.run_name}.best.pt",
                model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
            )

        if args.save_every_epoch:
            save_checkpoint(
                args.out_dir / f"{args.run_name}.epoch{epoch:03d}.pt",
                model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
            )

        (args.out_dir / f"{args.run_name}.summary.json").write_text(
            json.dumps(metrics_full, indent=2, sort_keys=True) + "\n"
        )

    print(f"done best_cos768={best_cos:.4f} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
