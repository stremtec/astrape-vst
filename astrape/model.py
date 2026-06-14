from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ContentStudentConfig:
    in_dim: int = 80
    hidden: int = 384
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    kernel_size: int = 5
    content_dim: int = 768
    dropout: float = 0.0
    auxiliary_prefsq: bool = False
    structured_fsq: bool = False
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    text_vocab_size: int = 0
    safe_convs: bool = False
    max_attention_context: Optional[int] = None


@dataclass
class ContentStudentOutput:
    content: torch.Tensor
    pre_fsq: Optional[torch.Tensor] = None
    hard_content: Optional[torch.Tensor] = None
    fsq_logits: Optional[tuple[torch.Tensor, ...]] = None
    fsq_codes: Optional[torch.Tensor] = None
    text_logits: Optional[torch.Tensor] = None


@dataclass
class StreamingState:
    stem_caches: list[Optional[torch.Tensor]]
    block_histories: list[Optional[torch.Tensor]]
    down_cache: Optional[torch.Tensor] = None
    pending_mel: Optional[torch.Tensor] = None
    position: int = 0


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding and a stateful streaming path."""

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

    def forward_stream(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor],
        *,
        position: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.left_context
        if cache is None:
            cache = x.new_zeros(x.shape[0], x.shape[1], context)
        if cache.shape[:2] != x.shape[:2] or cache.shape[2] != context:
            raise ValueError("Invalid convolution streaming cache shape")
        joined = torch.cat((cache, x), dim=-1)
        if self.stride[0] == 1:
            out = super().forward(joined)
        else:
            all_outputs = F.conv1d(
                joined,
                self.weight,
                self.bias,
                stride=1,
                dilation=self.dilation,
                groups=self.groups,
            )
            global_positions = position + torch.arange(
                x.shape[-1], device=x.device
            )
            out = all_outputs[
                :, :, global_positions.remainder(self.stride[0]) == 0
            ]
        next_cache = joined[:, :, -context:] if context else joined[:, :, :0]
        return out, next_cache


class SafeCausalConv1d(nn.Module):
    """MPS-safe causal Conv1d implemented with unfold and matmul."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def left_context(self) -> int:
        return self.dilation * (self.kernel_size - 1)

    def _convolve(self, x: torch.Tensor, stride: Optional[int] = None) -> torch.Tensor:
        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            dilation=(self.dilation, 1),
            stride=(self.stride if stride is None else stride, 1),
        )
        unfolded = unfolded.transpose(1, 2)
        out = torch.matmul(unfolded, self.weight.reshape(self.out_channels, -1).T)
        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, -1)
        return out.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_context:
            x = F.pad(x, (self.left_context, 0))
        return self._convolve(x)

    def forward_stream(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor],
        *,
        position: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.left_context
        if cache is None:
            cache = x.new_zeros(x.shape[0], x.shape[1], context)
        if cache.shape[:2] != x.shape[:2] or cache.shape[2] != context:
            raise ValueError("Invalid convolution streaming cache shape")
        joined = torch.cat((cache, x), dim=-1)
        if self.stride == 1:
            out = self._convolve(joined)
        else:
            all_outputs = self._convolve(joined, stride=1)
            global_positions = position + torch.arange(
                x.shape[-1], device=x.device
            )
            out = all_outputs[
                :, :, global_positions.remainder(self.stride) == 0
            ]
        next_cache = joined[:, :, -context:] if context else joined[:, :, :0]
        return out, next_cache


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(
            offset, offset + x.shape[1], device=x.device, dtype=torch.float32
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        pe = x.new_zeros(x.shape[1], self.dim)
        pe[:, 0::2] = torch.sin(positions * div).to(x.dtype)
        pe[:, 1::2] = torch.cos(positions * div).to(x.dtype)
        return x + pe.unsqueeze(0)


class CausalTransformerBlock(nn.Module):
    def __init__(self, config: ContentStudentConfig):
        super().__init__()
        dim = config.hidden
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * config.ff_mult),
            nn.GELU(),
            nn.Linear(dim * config.ff_mult, dim),
        )
        self.dropout = config.dropout
        self.max_context = config.max_attention_context

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        length = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        if self.max_context is not None:
            positions = torch.arange(length, device=x.device)
            too_old = positions.unsqueeze(0) < (
                positions.unsqueeze(1) - self.max_context + 1
            )
            causal_mask = causal_mask | too_old
        normalized = self.norm1(x)
        attended = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + F.dropout(attended, self.dropout, self.training)
        fed = self.ff(self.norm2(x))
        return x + F.dropout(fed, self.dropout, self.training)

    def forward_stream(
        self,
        x: torch.Tensor,
        history: Optional[torch.Tensor],
        max_context: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history is None:
            history = x[:, :0]
        keys = torch.cat((history, x), dim=1)
        history_length = history.shape[1]
        query_length = x.shape[1]
        key_length = keys.shape[1]
        query_positions = history_length + torch.arange(
            query_length, device=x.device
        )
        key_positions = torch.arange(key_length, device=x.device)
        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        if max_context is not None:
            mask = mask | (
                key_positions.unsqueeze(0)
                < query_positions.unsqueeze(1) - max_context + 1
            )
        attended = self.attn(
            self.norm1(x),
            self.norm1(keys),
            self.norm1(keys),
            attn_mask=mask,
            need_weights=False,
        )[0]
        x = x + attended
        x = x + self.ff(self.norm2(x))
        next_history = keys
        if max_context is not None:
            next_history = next_history[:, -max_context:]
        return x, next_history


class ContentStudent(nn.Module):
    """Strictly causal mel-to-content student with stateful streaming inference."""

    def __init__(self, config: ContentStudentConfig):
        super().__init__()
        if config.hidden % config.n_heads:
            raise ValueError("hidden must be divisible by n_heads")
        if config.max_attention_context is not None and config.max_attention_context <= 0:
            raise ValueError("max_attention_context must be positive")
        if config.structured_fsq and (
            not config.fsq_levels or any(level < 2 for level in config.fsq_levels)
        ):
            raise ValueError("structured FSQ levels must all be at least two")
        self.config = config
        conv_type = SafeCausalConv1d if config.safe_convs else CausalConv1d
        self.stem = nn.Sequential(
            conv_type(config.in_dim, config.hidden, config.kernel_size),
            nn.GELU(),
            conv_type(config.hidden, config.hidden, config.kernel_size),
            nn.GELU(),
        )
        self.pos_enc = SinusoidalPositionEncoding(config.hidden)
        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden)
        self.down = conv_type(config.hidden, config.hidden, 3, stride=2)
        self.content_head = (
            None
            if config.structured_fsq
            else conv_type(config.hidden, config.content_dim, 1)
        )
        self.fsq_head = (
            conv_type(config.hidden, sum(config.fsq_levels), 1)
            if config.structured_fsq
            else None
        )
        self.fsq_projection = (
            nn.Linear(len(config.fsq_levels), config.content_dim)
            if config.structured_fsq
            else None
        )
        if self.fsq_projection is not None:
            self.fsq_projection.requires_grad_(False)
        self.text_head = (
            nn.Linear(config.hidden, config.text_vocab_size)
            if config.text_vocab_size > 0
            else None
        )
        self.prefsq_head = (
            conv_type(config.hidden, config.content_dim, 1)
            if config.auxiliary_prefsq
            else None
        )

    def load_fsq_projection(self, state: dict[str, torch.Tensor]) -> None:
        if self.fsq_projection is None:
            raise RuntimeError("Model does not use a structured FSQ head")
        self.fsq_projection.load_state_dict(state, strict=True)
        self.fsq_projection.requires_grad_(False)

    def _decode_fsq(
        self, hidden: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        torch.Tensor,
    ]:
        if self.fsq_head is None or self.fsq_projection is None:
            raise RuntimeError("Structured FSQ head is not configured")
        packed_logits = self.fsq_head(hidden)
        logits = torch.split(packed_logits, self.config.fsq_levels, dim=1)
        soft_codes = []
        hard_codes = []
        level_indices = []
        for axis_logits, levels in zip(logits, self.config.fsq_levels):
            values = (
                torch.arange(levels, device=hidden.device, dtype=hidden.dtype)
                - levels // 2
            ) / (levels // 2)
            indices = axis_logits.argmax(dim=1)
            level_indices.append(indices)
            hard_codes.append(values[indices])
            if self.training:
                probabilities = torch.softmax(axis_logits, dim=1)
                soft_codes.append(
                    torch.sum(probabilities * values.view(1, -1, 1), dim=1)
                )
        hard_code = torch.stack(hard_codes, dim=-1)
        hard_content = self.fsq_projection(hard_code).transpose(1, 2)
        if self.training:
            soft_code = torch.stack(soft_codes, dim=-1)
            soft_content = self.fsq_projection(soft_code).transpose(1, 2)
        else:
            soft_content = hard_content
        indices = torch.stack(level_indices, dim=-1)
        return soft_content, hard_content, logits, indices

    @staticmethod
    def output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(input_lengths + 1, 2, rounding_mode="floor")

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> ContentStudentOutput:
        h = self.stem(x).transpose(1, 2)
        h = self.pos_enc(h)
        padding_mask = None
        if lengths is not None:
            positions = torch.arange(h.shape[1], device=h.device)
            padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        for block in self.blocks:
            h = block(h, padding_mask)
        h = self.norm(h)
        text_logits = self.text_head(h) if self.text_head is not None else None
        h = self.down(h.transpose(1, 2))
        hard_content = None
        fsq_logits = None
        fsq_codes = None
        if self.fsq_head is not None:
            soft_content, hard_content, fsq_logits, fsq_codes = self._decode_fsq(h)
            content = soft_content if self.training else hard_content
        else:
            content = self.content_head(h)
        pre_fsq = self.prefsq_head(h) if self.prefsq_head is not None else None
        return ContentStudentOutput(
            content=content,
            pre_fsq=pre_fsq,
            hard_content=hard_content,
            fsq_logits=fsq_logits,
            fsq_codes=fsq_codes,
            text_logits=text_logits,
        )

    def initial_streaming_state(self) -> StreamingState:
        return StreamingState(
            stem_caches=[None, None],
            block_histories=[None] * len(self.blocks),
        )

    @torch.inference_mode()
    def forward_stream(
        self,
        x: torch.Tensor,
        state: Optional[StreamingState] = None,
        *,
        flush: bool = False,
    ) -> tuple[ContentStudentOutput, StreamingState]:
        if self.training:
            raise RuntimeError("forward_stream requires model.eval()")
        state = state or self.initial_streaming_state()
        if state.pending_mel is not None:
            if x.shape[0] != state.pending_mel.shape[0]:
                raise ValueError("Streaming batch size changed")
            x = torch.cat((state.pending_mel, x), dim=-1)
        if flush:
            process_length = x.shape[-1]
        elif state.position == 0:
            process_length = (
                x.shape[-1]
                if x.shape[-1] % 2
                else max(0, x.shape[-1] - 1)
            )
        else:
            process_length = x.shape[-1] - x.shape[-1] % 2
        state.pending_mel = x[:, :, process_length:]
        x = x[:, :, :process_length]
        if x.shape[-1] == 0:
            if flush:
                state.pending_mel = None
            empty = x.new_empty(x.shape[0], self.config.content_dim, 0)
            return ContentStudentOutput(content=empty), state
        h, state.stem_caches[0] = self.stem[0].forward_stream(
            x, state.stem_caches[0]
        )
        h = self.stem[1](h)
        h, state.stem_caches[1] = self.stem[2].forward_stream(
            h, state.stem_caches[1]
        )
        h = self.stem[3](h).transpose(1, 2)
        h = self.pos_enc(h, offset=state.position)
        for index, block in enumerate(self.blocks):
            h, state.block_histories[index] = block.forward_stream(
                h,
                state.block_histories[index],
                self.config.max_attention_context,
            )
        h = self.norm(h)
        h, state.down_cache = self.down.forward_stream(
            h.transpose(1, 2),
            state.down_cache,
            position=state.position,
        )
        state.position += x.shape[-1]
        if h.shape[-1] == 0:
            if flush:
                state.pending_mel = None
            empty = x.new_empty(x.shape[0], self.config.content_dim, 0)
            return ContentStudentOutput(content=empty), state
        hard_content = None
        fsq_logits = None
        fsq_codes = None
        if self.fsq_head is not None:
            _, hard_content, fsq_logits, fsq_codes = self._decode_fsq(h)
            content = hard_content
        else:
            content = self.content_head(h)
        if flush:
            state.pending_mel = None
        return (
            ContentStudentOutput(
                content=content,
                pre_fsq=None,
                hard_content=hard_content,
                fsq_logits=fsq_logits,
                fsq_codes=fsq_codes,
                text_logits=None,
            ),
            state,
        )
