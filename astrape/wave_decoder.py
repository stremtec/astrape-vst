from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .voicebank import MIO_GLOBAL_MODEL


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.left_context
        if cache is None:
            cache = x.new_zeros(x.shape[0], x.shape[1], context)
        joined = torch.cat((cache, x), dim=-1)
        out = super().forward(joined)
        next_cache = joined[:, :, -context:] if context else joined[:, :, :0]
        return out, next_cache


# Legacy direct-wave decoder config/state -----------------------------------------


@dataclass(frozen=True)
class WaveDecoderConfig:
    """Legacy config for DirectWaveDecoder checkpoint compatibility."""

    content_dim: int = 768
    condition_dim: int = 128
    sample_rate: int = 44100
    content_rate: int = 25
    condition_model: str = MIO_GLOBAL_MODEL
    supports_f0_conditioning: bool = False
    supports_formant_conditioning: bool = False
    f0_model: str = ""
    initial_channels: int = 384
    stage_channels: tuple[int, ...] = (256, 192, 128, 96, 64, 48)
    upsample_factors: tuple[int, ...] = (7, 7, 3, 3, 2, 2)
    mrf_kernel_sizes: tuple[int, ...] = (7,)
    mrf_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5),)
    residual_scale: float = 0.3
    output_kernel_size: int = 7

    def __post_init__(self) -> None:
        if (
            self.content_dim <= 0
            or self.condition_dim <= 0
            or self.initial_channels <= 0
        ):
            raise ValueError("feature and channel dimensions must be positive")
        if not self.condition_model:
            raise ValueError("condition_model must be recorded")
        if self.sample_rate <= 0 or self.content_rate <= 0:
            raise ValueError("sample and content rates must be positive")
        if self.sample_rate % self.content_rate:
            raise ValueError("sample_rate must be divisible by content_rate")
        if len(self.stage_channels) != len(self.upsample_factors):
            raise ValueError("stage_channels and upsample_factors must align")
        if len(self.mrf_kernel_sizes) != len(self.mrf_dilations):
            raise ValueError("MRF kernels and dilation groups must align")
        if not self.mrf_kernel_sizes:
            raise ValueError("At least one residual branch is required")
        if any(channels <= 0 for channels in self.stage_channels):
            raise ValueError("stage channels must be positive")
        if any(factor <= 0 for factor in self.upsample_factors):
            raise ValueError("upsample factors must be positive")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.mrf_kernel_sizes):
            raise ValueError("MRF kernels must be positive odd values")
        if any(
            dilation <= 0
            for dilation_group in self.mrf_dilations
            for dilation in dilation_group
        ):
            raise ValueError("MRF dilations must be positive")
        if self.output_kernel_size <= 0 or self.output_kernel_size % 2 == 0:
            raise ValueError("output kernel must be a positive odd value")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        if math.prod(self.upsample_factors) != self.samples_per_frame:
            raise ValueError(
                "upsample factor product must equal sample_rate/content_rate"
            )

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate // self.content_rate

    @classmethod
    def quality_profile(cls) -> WaveDecoderConfig:
        return cls(
            initial_channels=512,
            stage_channels=(384, 256, 192, 128, 96, 64),
            mrf_kernel_sizes=(3, 7),
            mrf_dilations=((1, 3, 5), (1, 3, 5)),
        )


@dataclass
class ResidualBranchState:
    first_caches: list[Optional[torch.Tensor]]
    second_caches: list[Optional[torch.Tensor]]


@dataclass
class MRFState:
    branches: list[ResidualBranchState]


@dataclass
class UpsampleStageState:
    upsample_cache: Optional[torch.Tensor]
    mrf: MRFState


@dataclass
class WaveDecoderState:
    """Legacy streaming state for DirectWaveDecoder checkpoint compatibility."""

    stages: list[UpsampleStageState]
    output_cache: Optional[torch.Tensor] = None
    content_frames: int = 0


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each time step."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Snake1d(nn.Module):
    """Pointwise periodic activation suitable for voiced waveform synthesis."""

    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        return x + torch.sin(alpha * x).square() / alpha.clamp_min(1e-6)


class FiLM(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.projection = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError("condition must have shape [batch, condition_dim]")
        scale, shift = self.projection(condition).chunk(2, dim=-1)
        return (
            x * (1.0 + scale.unsqueeze(-1))
            + shift.unsqueeze(-1)
        )


class CausalResidualBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        residual_scale: float,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.first_norms = nn.ModuleList(
            [ChannelLayerNorm(channels) for _ in dilations]
        )
        self.second_norms = nn.ModuleList(
            [ChannelLayerNorm(channels) for _ in dilations]
        )
        self.first_activations = nn.ModuleList(
            [Snake1d(channels) for _ in dilations]
        )
        self.second_activations = nn.ModuleList(
            [Snake1d(channels) for _ in dilations]
        )
        self.first_convs = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.second_convs = nn.ModuleList(
            [
                CausalConv1d(channels, channels, kernel_size)
                for _ in dilations
            ]
        )

    def initial_state(self) -> ResidualBranchState:
        return ResidualBranchState(
            first_caches=[None] * len(self.first_convs),
            second_caches=[None] * len(self.second_convs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm1, act1, conv1, norm2, act2, conv2 in zip(
            self.first_norms,
            self.first_activations,
            self.first_convs,
            self.second_norms,
            self.second_activations,
            self.second_convs,
        ):
            residual = conv1(act1(norm1(x)))
            residual = conv2(act2(norm2(residual)))
            x = x + self.residual_scale * residual
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        state: Optional[ResidualBranchState],
    ) -> tuple[torch.Tensor, ResidualBranchState]:
        state = state or self.initial_state()
        for index, (norm1, act1, conv1, norm2, act2, conv2) in enumerate(
            zip(
                self.first_norms,
                self.first_activations,
                self.first_convs,
                self.second_norms,
                self.second_activations,
                self.second_convs,
            )
        ):
            residual, state.first_caches[index] = conv1.forward_stream(
                act1(norm1(x)),
                state.first_caches[index],
            )
            residual, state.second_caches[index] = conv2.forward_stream(
                act2(norm2(residual)),
                state.second_caches[index],
            )
            x = x + self.residual_scale * residual
        return x, state


class CausalMRF(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_dim: int,
        kernel_sizes: tuple[int, ...],
        dilation_groups: tuple[tuple[int, ...], ...],
        residual_scale: float,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                CausalResidualBranch(
                    channels,
                    kernel_size,
                    dilations,
                    residual_scale,
                )
                for kernel_size, dilations in zip(
                    kernel_sizes,
                    dilation_groups,
                )
            ]
        )
        self.output_norm = ChannelLayerNorm(channels)
        self.condition = FiLM(channels, condition_dim)

    def initial_state(self) -> MRFState:
        return MRFState(
            branches=[branch.initial_state() for branch in self.branches]
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        x = torch.stack(outputs, dim=0).mean(dim=0)
        return self.condition(self.output_norm(x), condition)

    def forward_stream(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        state: Optional[MRFState],
    ) -> tuple[torch.Tensor, MRFState]:
        state = state or self.initial_state()
        outputs = []
        for index, branch in enumerate(self.branches):
            output, state.branches[index] = branch.forward_stream(
                x,
                state.branches[index],
            )
            outputs.append(output)
        x = torch.stack(outputs, dim=0).mean(dim=0)
        return self.condition(self.output_norm(x), condition), state


class CausalUpsampleStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        factor: int,
        condition_dim: int,
        kernel_sizes: tuple[int, ...],
        dilation_groups: tuple[tuple[int, ...], ...],
        residual_scale: float,
    ):
        super().__init__()
        self.factor = factor
        self.upsample_conv = CausalConv1d(
            input_channels,
            output_channels,
            kernel_size=2 * factor + 1,
        )
        self.upsample_norm = ChannelLayerNorm(output_channels)
        self.upsample_activation = Snake1d(output_channels)
        self.mrf = CausalMRF(
            output_channels,
            condition_dim,
            kernel_sizes,
            dilation_groups,
            residual_scale,
        )

    def initial_state(self) -> UpsampleStageState:
        return UpsampleStageState(
            upsample_cache=None,
            mrf=self.mrf.initial_state(),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        x = x.repeat_interleave(self.factor, dim=-1)
        x = self.upsample_conv(x)
        x = self.upsample_activation(self.upsample_norm(x))
        return self.mrf(x, condition)

    def forward_stream(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        state: Optional[UpsampleStageState],
    ) -> tuple[torch.Tensor, UpsampleStageState]:
        state = state or self.initial_state()
        x = x.repeat_interleave(self.factor, dim=-1)
        x, state.upsample_cache = self.upsample_conv.forward_stream(
            x,
            state.upsample_cache,
        )
        x = self.upsample_activation(self.upsample_norm(x))
        x, state.mrf = self.mrf.forward_stream(x, condition, state.mrf)
        return x, state


class DirectWaveDecoder(nn.Module):
    """Legacy direct 25 Hz content + VoiceBank global to 44.1 kHz PCM.

    The current synthesis path lives in astrape.decoder.CausalSynthesisDecoder.
    This class remains loadable so old direct-wave checkpoints can be inspected
    or migrated.
    """

    def __init__(self, config: WaveDecoderConfig = WaveDecoderConfig()):
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(
            config.content_dim,
            config.initial_channels,
            kernel_size=1,
        )
        self.input_norm = ChannelLayerNorm(config.initial_channels)
        self.input_activation = Snake1d(config.initial_channels)
        self.input_condition = FiLM(
            config.initial_channels,
            config.condition_dim,
        )
        stages = []
        input_channels = config.initial_channels
        for output_channels, factor in zip(
            config.stage_channels,
            config.upsample_factors,
        ):
            stages.append(
                CausalUpsampleStage(
                    input_channels,
                    output_channels,
                    factor,
                    config.condition_dim,
                    config.mrf_kernel_sizes,
                    config.mrf_dilations,
                    config.residual_scale,
                )
            )
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)
        self.output_norm = ChannelLayerNorm(input_channels)
        self.output_activation = Snake1d(input_channels)
        self.output_projection = CausalConv1d(
            input_channels,
            1,
            config.output_kernel_size,
        )
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def initial_streaming_state(self) -> WaveDecoderState:
        return WaveDecoderState(
            stages=[stage.initial_state() for stage in self.stages]
        )

    def _prepare_input(
        self,
        content: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if content.ndim != 3:
            raise ValueError("content must have shape [batch, frames, content_dim]")
        if content.shape[-1] != self.config.content_dim:
            raise ValueError("content feature dimension does not match config")
        if global_embedding.shape != (
            content.shape[0],
            self.config.condition_dim,
        ):
            raise ValueError(
                "global_embedding must have shape [batch, condition_dim]"
            )
        x = self.input_projection(content.transpose(1, 2))
        x = self.input_activation(self.input_norm(x))
        return self.input_condition(x, global_embedding)

    def forward(
        self,
        content: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = self._prepare_input(content, global_embedding)
        for stage in self.stages:
            x = stage(x, global_embedding)
        x = self.output_activation(self.output_norm(x))
        return torch.tanh(self.output_projection(x)).squeeze(1)

    @torch.inference_mode()
    def forward_stream(
        self,
        content: torch.Tensor,
        global_embedding: torch.Tensor,
        state: Optional[WaveDecoderState] = None,
    ) -> tuple[torch.Tensor, WaveDecoderState]:
        if self.training:
            raise RuntimeError("forward_stream requires model.eval()")
        state = state or self.initial_streaming_state()
        if content.shape[1] == 0:
            empty = content.new_empty(content.shape[0], 0)
            return empty, state
        x = self._prepare_input(content, global_embedding)
        for index, stage in enumerate(self.stages):
            x, state.stages[index] = stage.forward_stream(
                x,
                global_embedding,
                state.stages[index],
            )
        x = self.output_activation(self.output_norm(x))
        x, state.output_cache = self.output_projection.forward_stream(
            x,
            state.output_cache,
        )
        state.content_frames += content.shape[1]
        return torch.tanh(x).squeeze(1), state


def save_wave_decoder_checkpoint(
    path: str | Path,
    model: DirectWaveDecoder,
    *,
    step: int,
    metrics: dict[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_type": "direct_wave_decoder",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "step": step,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def load_wave_decoder(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> DirectWaveDecoder:
    payload = torch.load(path, map_location=device)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or payload.get("model_type") != "direct_wave_decoder"
        or "state_dict" not in payload
    ):
        raise ValueError("Not a versioned direct waveform decoder checkpoint")
    config = WaveDecoderConfig(**payload["config"])
    model = DirectWaveDecoder(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model
