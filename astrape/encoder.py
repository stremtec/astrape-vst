"""Causal Content Encoder for Astrape VC.

Transforms 80-bin log-mel at 50Hz into 768d content embeddings at 25Hz,
with speaker identity stripped by the FSQ bottleneck.

Strictly causal: every output depends only on current and past inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsq import DEFAULT_LEVELS, indices_to_codes
from .wave_decoder import CausalConv1d, ChannelLayerNorm


# --- Configuration ---


@dataclass(frozen=True)
class EncoderConfig:
    mel_dim: int = 80
    content_dim: int = 768
    frontend_dim: int = 384
    frontend_kernel: int = 5
    convnext_kernel: int = 7
    convnext_ff_mult: int = 4
    n_convnext_blocks: int = 2
    transformer_dim: int = 384
    transformer_heads: int = 6
    transformer_layers: int = 4
    transformer_ff_mult: int = 4
    transformer_window: int = 64
    rope_theta: float = 10000.0
    fsq_levels: tuple[int, ...] = DEFAULT_LEVELS
    dropout: float = 0.0

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def n_codes(self) -> int:
        product = 1
        for level in self.fsq_levels:
            product *= level
        return product

    def __post_init__(self) -> None:
        if self.mel_dim <= 0 or self.content_dim <= 0 or self.frontend_dim <= 0:
            raise ValueError("dimensions must be positive")
        if self.transformer_heads <= 0 or self.transformer_dim % self.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if self.transformer_window <= 0:
            raise ValueError("transformer_window must be positive")
        if not self.fsq_levels or any(l <= 0 for l in self.fsq_levels):
            raise ValueError("fsq_levels must be non-empty positive integers")


# --- Streaming State ---


@dataclass
class ContentEncoderState:
    frontend_caches: list[Optional[torch.Tensor]]
    pending_frame: Optional[torch.Tensor]
    kv_cache: Optional[torch.Tensor]
    cache_len: int
    frames_emitted: int


# --- Output ---


@dataclass
class ContentOutput:
    content: torch.Tensor
    pre_fsq: Optional[torch.Tensor] = None
    hidden_384: Optional[torch.Tensor] = None
    token_indices: Optional[torch.Tensor] = None


# --- Building Blocks ---


class CausalConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, ff_mult: int = 4):
        super().__init__()
        self.dw_conv = CausalConv1d(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw_up = nn.Linear(dim, dim * ff_mult)
        self.pw_down = nn.Linear(dim * ff_mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_down(F.gelu(self.pw_up(x)))
        x = x.transpose(1, 2)
        return x + residual

    def forward_stream(
        self, x: torch.Tensor, cache: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x, cache = self.dw_conv.forward_stream(x, cache)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_down(F.gelu(self.pw_up(x)))
        x = x.transpose(1, 2)
        return x + residual, cache


class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("RoPE dimension must be a positive even integer")
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        if offset < 0:
            raise ValueError("RoPE offset must be non-negative")
        seq_len = x.shape[-2]
        if seq_len == 0:
            return x
        compute_dtype = x.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        positions = torch.arange(
            offset,
            offset + seq_len,
            device=x.device,
            dtype=compute_dtype,
        )
        inv_freq = self.inv_freq.to(device=x.device, dtype=compute_dtype)
        angles = torch.outer(positions, inv_freq)
        cos = angles.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = angles.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.attn_norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim * ff_mult, bias=False)
        self.w3 = nn.Linear(dim, dim * ff_mult, bias=False)
        self.w2 = nn.Linear(dim * ff_mult, dim, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        window: int,
        offset: int = 0,
    ) -> torch.Tensor:
        B, T, D = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(B, T, self.heads, self.head_dim)
        k = self.wk(h).view(B, T, self.heads, self.head_dim)
        v = self.wv(h).view(B, T, self.heads, self.head_dim)

        q = rope(q.transpose(1, 2), offset).transpose(1, 2)
        k = rope(k.transpose(1, 2), offset).transpose(1, 2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)
        mask = torch.triu(mask, diagonal=-(window - 1))
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(attn)

        h = self.ffn_norm(x)
        x = x + self.ffn_drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        rope: RoPE,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_len: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(B, T, self.heads, self.head_dim)
        k_new = self.wk(h).view(B, T, self.heads, self.head_dim)
        v_new = self.wv(h).view(B, T, self.heads, self.head_dim)

        q = rope(q.transpose(1, 2), cache_len).transpose(1, 2)
        k_new = rope(k_new.transpose(1, 2), cache_len).transpose(1, 2)

        # Update circular cache
        max_len = k_cache.shape[1]
        for i in range(T):
            pos = (cache_len + i) % max_len
            k_cache[:, pos] = k_new[:, i]
            v_cache[:, pos] = v_new[:, i]

        new_cache_len = cache_len + T
        valid_len = min(new_cache_len, max_len)

        # Gather valid keys/values in order
        if new_cache_len <= max_len:
            k_all = k_cache[:, :valid_len]
            v_all = v_cache[:, :valid_len]
        else:
            start = new_cache_len % max_len
            indices = [(start + i) % max_len for i in range(max_len)]
            idx = torch.tensor(indices, device=k_cache.device)
            k_all = k_cache[:, idx]
            v_all = v_cache[:, idx]

        # Window: only attend to last `window` entries
        if valid_len > window:
            k_all = k_all[:, -window:]
            v_all = v_all[:, -window:]

        q = q.transpose(1, 2)
        k_all = k_all.transpose(1, 2)
        v_all = v_all.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k_all, v_all)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(attn)

        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x, k_cache, v_cache


class CausalTransformer(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        dim = config.transformer_dim
        self.rope = RoPE(
            dim // config.transformer_heads,
            max_len=config.transformer_window * 4,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            CausalTransformerLayer(
                dim, config.transformer_heads, config.transformer_ff_mult, config.dropout
            )
            for _ in range(config.transformer_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.rope, self.config.transformer_window)
        return self.norm(x)

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        cache_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        B, T, D = x.shape
        head_dim = self.config.transformer_dim // self.config.transformer_heads
        window = self.config.transformer_window

        if kv_cache is None:
            kv_cache = torch.zeros(
                self.config.transformer_layers, 2, B, window,
                self.config.transformer_heads, head_dim,
                device=x.device, dtype=x.dtype,
            )

        for i, layer in enumerate(self.layers):
            x, kv_cache[i, 0], kv_cache[i, 1] = layer.forward_stream(
                x, self.rope, kv_cache[i, 0], kv_cache[i, 1], cache_len, window
            )
        return self.norm(x), kv_cache, cache_len + T


# --- FSQ ---


class FSQBottleneck(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.levels = config.fsq_levels
        self.proj_in = nn.Linear(config.transformer_dim, config.fsq_dim)
        self.proj_out = nn.Linear(config.fsq_dim, config.content_dim)

    @staticmethod
    def _bound(z: torch.Tensor, levels: Sequence[int]) -> torch.Tensor:
        eps = 1e-3
        half_levels = torch.tensor(
            [l // 2 for l in levels], device=z.device, dtype=z.dtype
        )
        shift = torch.where(
            torch.tensor([l % 2 == 0 for l in levels], device=z.device),
            0.5,
            0.0,
        ).to(z.dtype)
        bounded = torch.tanh(z)
        bounded = bounded * (half_levels - eps + shift)
        return bounded

    @staticmethod
    def _quantize(z: torch.Tensor, levels: Sequence[int]) -> torch.Tensor:
        quantized = z.round()
        lower = torch.tensor(
            [-(level // 2) for level in levels], device=z.device, dtype=z.dtype
        )
        upper = torch.tensor(
            [level - level // 2 - 1 for level in levels],
            device=z.device,
            dtype=z.dtype,
        )
        quantized = quantized.clamp(min=lower, max=upper)
        return z + (quantized - z).detach()

    @staticmethod
    def _normalize(z: torch.Tensor, levels: Sequence[int]) -> torch.Tensor:
        half_levels = torch.tensor(
            [l // 2 for l in levels], device=z.device, dtype=z.dtype
        )
        return z / half_levels

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj_in(x)
        bounded = self._bound(z, self.levels)
        quantized = self._quantize(bounded, self.levels)
        normalized = self._normalize(quantized, self.levels)

        # Compute flat indices
        basis = []
        product = 1
        for level in self.levels:
            basis.append(product)
            product *= level
        basis_t = torch.tensor(basis, device=z.device, dtype=torch.long)
        half_w = torch.tensor(
            [l // 2 for l in self.levels], device=z.device, dtype=torch.long
        )
        level_indices = quantized.long() + half_w
        indices = (level_indices * basis_t).sum(dim=-1)

        return normalized, indices

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized, indices = self.encode(x)
        content = self.proj_out(normalized)
        return content, normalized, indices

    def load_teacher_proj_out(self, path: str | Path) -> None:
        weights = torch.load(path, map_location="cpu")
        self.proj_out.weight.data.copy_(weights["weight"])
        self.proj_out.bias.data.copy_(weights["bias"])


# --- Speaker Adversarial ---


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


class SpeakerAdversarial(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_speakers: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_speakers),
        )

    def forward(
        self, x: torch.Tensor, scale: float = 0.1
    ) -> torch.Tensor:
        x = GradientReversal.apply(x, scale)
        return self.classifier(x)


# --- Main Encoder ---


class CausalContentEncoder(nn.Module):
    def __init__(self, config: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.config = config

        # Frontend: mel → frontend_dim at 50Hz
        self.input_conv = CausalConv1d(config.mel_dim, config.frontend_dim, config.frontend_kernel)
        self.input_norm = nn.LayerNorm(config.frontend_dim)
        self.convnext_blocks = nn.ModuleList([
            CausalConvNeXtBlock(config.frontend_dim, config.convnext_kernel, config.convnext_ff_mult)
            for _ in range(config.n_convnext_blocks)
        ])

        # Downsample: 50Hz → 25Hz via frame-pair concat + projection
        self.downsample_proj = nn.Linear(config.frontend_dim * 2, config.transformer_dim)

        # Transformer at 25Hz
        self.transformer = CausalTransformer(config)

        # FSQ bottleneck
        self.fsq = FSQBottleneck(config)

    def forward(self, mel: torch.Tensor) -> ContentOutput:
        B, C, T = mel.shape

        # Frontend
        x = self.input_conv(mel)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        for block in self.convnext_blocks:
            x = block(x)

        # Downsample: pair adjacent frames
        x = x.transpose(1, 2)  # (B, T, D)
        T_trunc = (x.shape[1] // 2) * 2
        x = x[:, :T_trunc]
        x = x.reshape(B, T_trunc // 2, self.config.frontend_dim * 2)
        x = self.downsample_proj(x)

        # Transformer
        hidden = self.transformer(x)

        # FSQ
        content, pre_fsq, indices = self.fsq(hidden)

        return ContentOutput(
            content=content.transpose(1, 2),
            pre_fsq=pre_fsq.transpose(1, 2) if pre_fsq is not None else None,
            hidden_384=hidden.transpose(1, 2),
            token_indices=indices,
        )

    def initial_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> ContentEncoderState:
        n_caches = 1 + self.config.n_convnext_blocks
        return ContentEncoderState(
            frontend_caches=[None] * n_caches,
            pending_frame=None,
            kv_cache=None,
            cache_len=0,
            frames_emitted=0,
        )

    @torch.inference_mode()
    def forward_stream(
        self,
        mel: torch.Tensor,
        state: Optional[ContentEncoderState] = None,
        flush: bool = False,
    ) -> tuple[ContentOutput, ContentEncoderState]:
        if self.training:
            raise RuntimeError("forward_stream requires model.eval()")

        state = state or self.initial_state(
            batch_size=mel.shape[0], device=mel.device
        )

        B = mel.shape[0]
        content_frames = []

        # Frontend — skip if no new mel frames (flush-only call)
        if mel.shape[2] > 0:
            x, state.frontend_caches[0] = self.input_conv.forward_stream(
                mel, state.frontend_caches[0]
            )
            x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
            x = F.gelu(x)
            for i, block in enumerate(self.convnext_blocks):
                x, state.frontend_caches[i + 1] = block.forward_stream(
                    x, state.frontend_caches[i + 1]
                )

            # x is (B, D, T_new) at 50Hz — process each frame for downsampling
            x = x.transpose(1, 2)  # (B, T_new, D)

            for t in range(x.shape[1]):
                frame = x[:, t:t+1, :]  # (B, 1, D)
                if state.pending_frame is None:
                    state.pending_frame = frame
                else:
                    paired = torch.cat((state.pending_frame, frame), dim=-1)
                    state.pending_frame = None
                    down = self.downsample_proj(paired)

                    out, state.kv_cache, state.cache_len = self.transformer.forward_stream(
                        down, state.kv_cache, state.cache_len
                    )

                    content_768, _, indices = self.fsq(out)
                    content_frames.append(content_768)
                    state.frames_emitted += 1

        if flush and state.pending_frame is not None:
            pad = torch.zeros_like(state.pending_frame)
            paired = torch.cat((state.pending_frame, pad), dim=-1)
            state.pending_frame = None
            down = self.downsample_proj(paired)

            out, state.kv_cache, state.cache_len = self.transformer.forward_stream(
                down, state.kv_cache, state.cache_len
            )
            content_768, _, indices = self.fsq(out)
            content_frames.append(content_768)
            state.frames_emitted += 1

        if content_frames:
            content = torch.cat(content_frames, dim=1).transpose(1, 2)
        else:
            content = mel.new_empty(B, self.config.content_dim, 0)

        return ContentOutput(content=content), state
