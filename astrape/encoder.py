"""Strict-causal Q2D2 content encoder (StridingAdapter + RoPE Transformer).

The `MCSTransQ2D2` model and its building blocks, moved verbatim out of the old
`train_mcs_q2d2.py` so the training script stays a thin CLI.  Submodule names are
unchanged, so existing checkpoints (e.g. striding_8l_200hz) load as-is.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import CausalConv1d
from .quantizer import Q2D2Projection

DEFAULT_Q2D2_LEVELS = (7, 7, 7, 7, 7, 7)  # rhombic grid, 3 pairs


# ── GRL speaker classifier (the gradient-reversal op lives in losses.py) ──

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


# ── RoPE + causal window mask ──

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


# ── Transformer block ──

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


# ── Mamba / SSM block (conditional fallback) ──

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


# ── WavLM CNN frontend adapter ──

class WavLMFrontendAdapter(nn.Module):
    """Projects cached WavLM CNN features (512d) to mel-like 80d.

    Rate conversion: when wavlm_rate > 50Hz, uses CausalReplicatePool
    (alpha-decaying replicate-pad + avg_pool) for 5ms delay, 0 params.
    """
    def __init__(self, in_dim: int = 512, out_dim: int = 80,
                 hidden: int = 256, wavlm_rate: int = 50,
                 dropout: float = 0.0):
        super().__init__()
        self.wavlm_rate = wavlm_rate
        stride = max(1, wavlm_rate // 50)

        if stride > 1:
            self.down = nn.Sequential(
                CausalReplicatePad(stride, alpha=0.8),
                nn.AvgPool1d(kernel_size=stride, stride=stride))
        else:
            self.down = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.wavlm_rate > 50:
            x = self.down(x.transpose(1, 2)).transpose(1, 2)
        return self.net(x)


class CausalReplicatePad(nn.Module):
    """Replicate-pad on the LEFT only (causal). α-decay for boundary smoothing."""
    def __init__(self, stride: int = 4, alpha: float = 0.8):
        super().__init__()
        self.stride = stride
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) @200Hz. Pad left with α-decay replicas of x0.
        first = x[:, :, :1]  # (B, C, 1)
        w = self.alpha ** torch.arange(self.stride - 1, 0, -1,
                                       device=x.device, dtype=x.dtype)
        pad = first * w.view(1, 1, -1)  # (B, C, stride-1)
        return torch.cat([pad, x], dim=-1)


# ── conv stem blocks ──

class ResidualConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = CausalConv1d(dim, dim, kernel, dilation=dilation)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = F.silu(self.conv(h))
        return residual + F.dropout(h, self.dropout, self.training)


class DepthwiseResidualBlock(nn.Module):
    """Causal depthwise separable conv block (WavTokenizer/ConvNeXt style).

    LayerNorm → DepthwiseConv1d(groups=dim) → SiLU → PointwiseConv1d → residual.
    Much more parameter-efficient than standard conv, enabling deeper stems.
    """
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = CausalConv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.depthwise(h)
        h = F.silu(h)
        h = self.pointwise(h)
        return residual + F.dropout(h, self.dropout, self.training)


class CellDownsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2:
            x = F.pad(x, (0, 1))
        return self.conv(x)


# ── config + model ──

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

        # ── Post-Q2D2 content expansion (compensates std compression) ──
        self.content_expand = nn.Sequential(
            nn.Linear(config.content_dim, 256),
            nn.GELU(),
            nn.Linear(256, config.content_dim),
        )
        nn.init.zeros_(self.content_expand[-1].weight)
        nn.init.zeros_(self.content_expand[-1].bias)

        # ── optional WavLM frontend adapter ──
        self.use_wavlm_frontend = config.use_wavlm_frontend
        if self.use_wavlm_frontend:
            self.wavlm_adapter: nn.Module | None = WavLMFrontendAdapter(
                in_dim=config.wavlm_in_dim, out_dim=config.in_dim,
                wavlm_rate=config.wavlm_rate,
                dropout=config.dropout,
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

        # ── Content expansion (compensates Q2D2 std compression) ──
        if hasattr(self, 'content_expand'):
            content = content + self.content_expand(content)

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
