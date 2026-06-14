from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MelDecoderConfig:
    content_dim: int = 768
    condition_dim: int = 128
    hidden: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    n_mels: int = 80
    dropout: float = 0.1
    max_attention_context: Optional[int] = None


@dataclass
class MelStreamingState:
    block_histories: list[Optional[torch.Tensor]]


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, condition_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 3 * dim))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.proj(condition).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


class CausalDecoderBlock(nn.Module):
    def __init__(self, config: MelDecoderConfig):
        super().__init__()
        self.adaln = AdaLNZero(config.hidden, config.condition_dim)
        self.attn = nn.MultiheadAttention(
            config.hidden,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.adaln2 = AdaLNZero(config.hidden, config.condition_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden, config.hidden * config.ff_mult),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden * config.ff_mult, config.hidden),
            nn.Dropout(config.dropout),
        )
        self.max_context = config.max_attention_context

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        length = x.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        if self.max_context is not None:
            positions = torch.arange(length, device=x.device)
            too_old = positions.unsqueeze(0) < (
                positions.unsqueeze(1) - self.max_context + 1
            )
            mask = mask | too_old
        normalized, gate = self.adaln(x, condition)
        attended = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + gate * attended
        normalized, gate = self.adaln2(x, condition)
        return x + gate * self.ff(normalized)

    def forward_stream(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        history: Optional[torch.Tensor],
        max_context: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history is None:
            history = x[:, :0]
        keys = torch.cat((history, x), dim=1)
        history_length = history.shape[1]
        query_positions = history_length + torch.arange(x.shape[1], device=x.device)
        key_positions = torch.arange(keys.shape[1], device=x.device)
        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        if max_context is not None:
            mask = mask | (
                key_positions.unsqueeze(0)
                < query_positions.unsqueeze(1) - max_context + 1
            )
        query, gate = self.adaln(x, condition)
        key_values, _ = self.adaln(keys, condition)
        attended = self.attn(
            query, key_values, key_values, attn_mask=mask, need_weights=False
        )[0]
        x = x + gate * attended
        normalized, gate = self.adaln2(x, condition)
        x = x + gate * self.ff(normalized)
        next_history = keys
        if max_context is not None:
            next_history = next_history[:, -max_context:]
        return x, next_history


class CausalMelDecoder(nn.Module):
    def __init__(self, config: MelDecoderConfig = MelDecoderConfig()):
        super().__init__()
        self.config = config
        self.proj_in = nn.Linear(config.content_dim, config.hidden)
        self.blocks = nn.ModuleList(
            [CausalDecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_out = nn.LayerNorm(config.hidden)
        self.proj_out = nn.Linear(config.hidden, config.n_mels)

    def forward(
        self,
        content: torch.Tensor,
        global_embedding: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.proj_in(content)
        condition = global_embedding.unsqueeze(1)
        padding_mask = None
        if lengths is not None:
            positions = torch.arange(x.shape[1], device=x.device)
            padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        for block in self.blocks:
            x = block(x, condition, padding_mask)
        return self.proj_out(self.norm_out(x)).transpose(1, 2)

    def initial_streaming_state(self) -> MelStreamingState:
        return MelStreamingState(block_histories=[None] * len(self.blocks))

    @torch.inference_mode()
    def forward_stream(
        self,
        content: torch.Tensor,
        global_embedding: torch.Tensor,
        state: Optional[MelStreamingState] = None,
    ) -> tuple[torch.Tensor, MelStreamingState]:
        if self.training:
            raise RuntimeError("forward_stream requires model.eval()")
        state = state or self.initial_streaming_state()
        x = self.proj_in(content)
        condition = global_embedding.unsqueeze(1)
        for index, block in enumerate(self.blocks):
            x, state.block_histories[index] = block.forward_stream(
                x,
                condition,
                state.block_histories[index],
                self.config.max_attention_context,
            )
        mel = self.proj_out(self.norm_out(x)).transpose(1, 2)
        return mel, state


def load_mel_decoder(path: str, device: torch.device | str = "cpu") -> CausalMelDecoder:
    payload = torch.load(path, map_location=device)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    config_data = payload.get("config", {}) if isinstance(payload, dict) else {}
    config = MelDecoderConfig(**config_data) if config_data else MelDecoderConfig()
    model = CausalMelDecoder(config).to(device)
    model.load_state_dict(state, strict=True)
    return model
