"""Causal Synthesis Decoder for Astrape VC.

Transforms 768d content embeddings at 25Hz into 44.1kHz PCM audio,
conditioned on a 128d speaker embedding from a VoiceBank.

Architecture:
  1. Input projection (25Hz)
  2. Causal synthesis transformer with AdaLN-Zero (25Hz)
  3. Causal upsample (25Hz → 50Hz)
  4. Causal local refinement ResNet (50Hz)
  5. Causal waveform chain (50Hz → 44.1kHz, 882×)
  6. Output head

Strictly causal: every output sample depends only on current and past inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RoPE
from .voicebank import MIO_GLOBAL_MODEL
from .wave_decoder import (
    CausalConv1d,
    CausalUpsampleStage,
    ChannelLayerNorm,
    Snake1d,
    UpsampleStageState,
)


# --- Configuration ---


@dataclass(frozen=True)
class SynthesisDecoderConfig:
    content_dim: int = 768
    condition_dim: int = 128
    condition_model: str = MIO_GLOBAL_MODEL
    sample_rate: int = 44100
    content_rate: int = 25
    # Transformer
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_ff_mult: int = 3
    transformer_window: int = 32
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # ResNet refinement
    resnet_blocks: int = 2
    resnet_kernel: int = 5
    resnet_dilations: tuple[int, ...] = (1, 3)
    residual_scale: float = 0.3
    # Waveform chain
    stage_channels: tuple[int, ...] = (384, 256, 128, 96, 64)
    upsample_factors: tuple[int, ...] = (7, 7, 3, 3, 2)
    mrf_kernel_sizes: tuple[int, ...] = (3, 7)
    mrf_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5))
    output_kernel_size: int = 7

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate // self.content_rate

    @property
    def internal_rate(self) -> int:
        return self.content_rate * 2

    def __post_init__(self) -> None:
        if self.content_dim <= 0 or self.condition_dim <= 0:
            raise ValueError("dimensions must be positive")
        if self.sample_rate <= 0 or self.content_rate <= 0:
            raise ValueError("sample_rate and content_rate must be positive")
        if (
            self.transformer_heads <= 0
            or self.transformer_dim % self.transformer_heads != 0
        ):
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if (self.transformer_dim // self.transformer_heads) % 2:
            raise ValueError("transformer head dimension must be even for RoPE")
        if self.transformer_window <= 0:
            raise ValueError("transformer_window must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.resnet_blocks < 0:
            raise ValueError("resnet_blocks must be non-negative")
        if len(self.resnet_dilations) != self.resnet_blocks:
            raise ValueError("resnet_blocks must match len(resnet_dilations)")
        if self.resnet_kernel <= 0 or any(d <= 0 for d in self.resnet_dilations):
            raise ValueError("resnet kernel and dilations must be positive")
        if len(self.stage_channels) != len(self.upsample_factors):
            raise ValueError("stage_channels and upsample_factors must align")
        if self.sample_rate % self.internal_rate:
            raise ValueError("sample_rate must be divisible by internal_rate")
        expected_product = self.sample_rate // self.internal_rate
        actual_product = math.prod(self.upsample_factors)
        if actual_product != expected_product:
            raise ValueError(
                f"upsample factor product ({actual_product}) must equal "
                f"sample_rate/internal_rate ({expected_product})"
            )


# --- Streaming State ---


@dataclass
class ResNetBlockState:
    caches: list[Optional[torch.Tensor]]


@dataclass
class SynthesisDecoderState:
    upsample_conv_cache: Optional[torch.Tensor]
    kv_cache: Optional[torch.Tensor]
    cache_len: int
    resnet_states: list[ResNetBlockState]
    waveform_stages: list[UpsampleStageState]
    output_cache: Optional[torch.Tensor]
    frames_50hz: int


# --- AdaLN-Zero ---


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, dim * 3),
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.proj(condition)
        if params.ndim == 2:
            params = params.unsqueeze(1)
        gamma, beta, alpha = params.chunk(3, dim=-1)
        normalized = self.norm(x) * (1 + gamma) + beta
        return normalized, alpha


# --- Causal Synthesis Transformer ---


class AdaLNTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        condition_dim: int,
        ff_mult: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.attn_adaln = AdaLNZero(dim, condition_dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn_adaln = AdaLNZero(dim, condition_dim)
        self.w1 = nn.Linear(dim, dim * ff_mult, bias=False)
        self.w3 = nn.Linear(dim, dim * ff_mult, bias=False)
        self.w2 = nn.Linear(dim * ff_mult, dim, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        rope: RoPE,
        window: int,
        offset: int = 0,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Attention with AdaLN-Zero
        normed, gate_a = self.attn_adaln(x, condition)
        q = self.wq(normed).view(B, T, self.heads, self.head_dim)
        k = self.wk(normed).view(B, T, self.heads, self.head_dim)
        v = self.wv(normed).view(B, T, self.heads, self.head_dim)

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
        x = x + gate_a * self.wo(attn)

        # FFN with AdaLN-Zero
        normed, gate_f = self.ffn_adaln(x, condition)
        x = x + gate_f * self.ffn_drop(self.w2(F.silu(self.w1(normed)) * self.w3(normed)))
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        rope: RoPE,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_len: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        normed, gate_a = self.attn_adaln(x, condition)
        q = self.wq(normed).view(B, T, self.heads, self.head_dim)
        k_new = self.wk(normed).view(B, T, self.heads, self.head_dim)
        v_new = self.wv(normed).view(B, T, self.heads, self.head_dim)

        q = rope(q.transpose(1, 2), cache_len).transpose(1, 2)
        k_new = rope(k_new.transpose(1, 2), cache_len).transpose(1, 2)

        max_len = k_cache.shape[1]
        for i in range(T):
            pos = (cache_len + i) % max_len
            k_cache[:, pos] = k_new[:, i]
            v_cache[:, pos] = v_new[:, i]

        new_cache_len = cache_len + T
        valid_len = min(new_cache_len, max_len)

        if new_cache_len <= max_len:
            k_all = k_cache[:, :valid_len]
            v_all = v_cache[:, :valid_len]
        else:
            start = new_cache_len % max_len
            idx = (torch.arange(max_len, device=k_cache.device) + start) % max_len
            k_all = k_cache[:, idx]
            v_all = v_cache[:, idx]

        if valid_len > window:
            k_all = k_all[:, -window:]
            v_all = v_all[:, -window:]

        q = q.transpose(1, 2)
        k_all = k_all.transpose(1, 2)
        v_all = v_all.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k_all, v_all)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + gate_a * self.wo(attn)

        normed, gate_f = self.ffn_adaln(x, condition)
        x = x + gate_f * self.w2(F.silu(self.w1(normed)) * self.w3(normed))
        return x, k_cache, v_cache


class SynthesisTransformer(nn.Module):
    def __init__(self, config: SynthesisDecoderConfig):
        super().__init__()
        self.config = config
        dim = config.transformer_dim
        head_dim = dim // config.transformer_heads
        self.rope = RoPE(
            head_dim,
            max_len=config.transformer_window * 4,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            AdaLNTransformerLayer(
                dim, config.transformer_heads, config.condition_dim,
                config.transformer_ff_mult,
                config.dropout,
            )
            for _ in range(config.transformer_layers)
        ])
        self.final_adaln = AdaLNZero(dim, config.condition_dim)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, condition, self.rope, self.config.transformer_window)
        normed, _ = self.final_adaln(x, condition)
        return normed

    def forward_stream(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
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
                x, condition, self.rope,
                kv_cache[i, 0], kv_cache[i, 1],
                cache_len, window,
            )

        normed, _ = self.final_adaln(x, condition)
        return normed, kv_cache, cache_len + T


# --- Causal ResNet Refinement ---


class CausalResNetBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: float = 0.3):
        super().__init__()
        self.scale = scale
        self.norm1 = ChannelLayerNorm(channels)
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm2 = ChannelLayerNorm(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.gelu(self.norm1(x)))
        residual = self.conv2(F.gelu(self.norm2(residual)))
        return x + self.scale * residual

    def forward_stream(
        self, x: torch.Tensor, caches: list[Optional[torch.Tensor]]
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        residual, caches[0] = self.conv1.forward_stream(F.gelu(self.norm1(x)), caches[0])
        residual, caches[1] = self.conv2.forward_stream(F.gelu(self.norm2(residual)), caches[1])
        return x + self.scale * residual, caches


# --- Main Decoder ---


class CausalSynthesisDecoder(nn.Module):
    """Causal 25Hz content + 128d speaker → 44.1kHz PCM."""

    def __init__(self, config: SynthesisDecoderConfig = SynthesisDecoderConfig()):
        super().__init__()
        self.config = config

        # Stage 1: Input projection (25Hz)
        self.input_proj = nn.Linear(config.content_dim, config.transformer_dim)

        # Stage 2: Synthesis transformer at 25Hz with AdaLN-Zero
        self.transformer = SynthesisTransformer(config)

        # Stage 3: Upsample 25Hz → 50Hz
        self.upsample_conv = CausalConv1d(
            config.transformer_dim, config.transformer_dim, kernel_size=5
        )
        self.upsample_norm = ChannelLayerNorm(config.transformer_dim)

        # Stage 4: Local refinement ResNet at 50Hz
        self.resnet_blocks = nn.ModuleList([
            CausalResNetBlock(
                config.transformer_dim, config.resnet_kernel,
                dilation=d, scale=config.residual_scale,
            )
            for d in config.resnet_dilations
        ])

        # Stage 5: Waveform upsample chain (50Hz → 44.1kHz)
        stages = []
        in_ch = config.transformer_dim
        for out_ch, factor in zip(config.stage_channels, config.upsample_factors):
            stages.append(
                CausalUpsampleStage(
                    in_ch, out_ch, factor, config.condition_dim,
                    config.mrf_kernel_sizes, config.mrf_dilations,
                    config.residual_scale,
                )
            )
            in_ch = out_ch
        self.waveform_stages = nn.ModuleList(stages)

        # Stage 6: Output head
        self.output_norm = ChannelLayerNorm(in_ch)
        self.output_act = Snake1d(in_ch)
        self.output_conv = CausalConv1d(in_ch, 1, config.output_kernel_size)
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_conv.bias)

    def initial_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> SynthesisDecoderState:
        return SynthesisDecoderState(
            upsample_conv_cache=None,
            kv_cache=None,
            cache_len=0,
            resnet_states=[
                ResNetBlockState(caches=[None, None])
                for _ in self.resnet_blocks
            ],
            waveform_stages=[stage.initial_state() for stage in self.waveform_stages],
            output_cache=None,
            frames_50hz=0,
        )

    def forward(
        self,
        content: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if content.ndim != 3:
            raise ValueError("content must have shape [batch, frames, content_dim]")

        # Stage 1: input projection (25Hz)
        x = self.input_proj(content)  # (B, T25, dim)

        # Stage 2: transformer at 25Hz
        x = self.transformer(x, global_embedding)  # (B, T25, dim)

        # Stage 3: upsample to 50Hz
        x = x.transpose(1, 2)  # (B, dim, T25)
        x = x.repeat_interleave(2, dim=-1)  # (B, dim, T50)
        x = self.upsample_conv(x)
        x = F.gelu(self.upsample_norm(x))

        # Stage 4: ResNet refinement at 50Hz
        for block in self.resnet_blocks:
            x = block(x)

        # Stage 5: waveform chain
        for stage in self.waveform_stages:
            x = stage(x, global_embedding)

        # Stage 6: output
        x = self.output_act(self.output_norm(x))
        return torch.tanh(self.output_conv(x)).squeeze(1)

    @torch.inference_mode()
    def forward_stream(
        self,
        content: torch.Tensor,
        global_embedding: torch.Tensor,
        state: Optional[SynthesisDecoderState] = None,
    ) -> tuple[torch.Tensor, SynthesisDecoderState]:
        if self.training:
            raise RuntimeError("forward_stream requires model.eval()")
        if content.ndim != 3:
            raise ValueError("content must have shape [batch, frames, content_dim]")
        if content.shape[1] > 1:
            raise ValueError(
                "forward_stream accepts at most one content frame per call; "
                "call it once per frame to preserve causal decoding"
            )

        state = state or self.initial_state(
            batch_size=content.shape[0], device=content.device
        )

        if content.shape[1] == 0:
            empty = content.new_empty(content.shape[0], 0)
            return empty, state

        # Stage 1: input projection (25Hz)
        x = self.input_proj(content)  # (B, T25, dim)

        # Stage 2: transformer at 25Hz
        x, state.kv_cache, state.cache_len = self.transformer.forward_stream(
            x, global_embedding, state.kv_cache, state.cache_len
        )

        # Stage 3: upsample to 50Hz
        x = x.transpose(1, 2)  # (B, dim, T25)
        x = x.repeat_interleave(2, dim=-1)  # (B, dim, T50)
        x, state.upsample_conv_cache = self.upsample_conv.forward_stream(
            x, state.upsample_conv_cache
        )
        x = F.gelu(self.upsample_norm(x))

        # Stage 4: ResNet
        for i, block in enumerate(self.resnet_blocks):
            x, state.resnet_states[i].caches = block.forward_stream(
                x, state.resnet_states[i].caches
            )

        # Stage 5: waveform chain
        for i, stage in enumerate(self.waveform_stages):
            x, state.waveform_stages[i] = stage.forward_stream(
                x, global_embedding, state.waveform_stages[i]
            )

        # Stage 6: output
        x = self.output_act(self.output_norm(x))
        x, state.output_cache = self.output_conv.forward_stream(x, state.output_cache)
        state.frames_50hz += content.shape[1] * 2

        return torch.tanh(x).squeeze(1), state


# --- Checkpoint helpers ---


def save_decoder_checkpoint(
    path: str | Path,
    model: CausalSynthesisDecoder,
    *,
    step: int,
    metrics: dict[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_type": "causal_synthesis_decoder",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "step": step,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)


def load_decoder(
    path: str | Path, device: torch.device | str = "cpu"
) -> CausalSynthesisDecoder:
    payload = torch.load(path, map_location=device)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or payload.get("model_type") != "causal_synthesis_decoder"
    ):
        raise ValueError("Not a valid causal synthesis decoder checkpoint")
    config = SynthesisDecoderConfig(**payload["config"])
    model = CausalSynthesisDecoder(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model
