"""
MioStudentCodec-Causal v1

Full streaming neural audio codec — encoder, FSQ quantizer, and decoder
are all strictly causal. No future lookahead, no bidirectional attention,
no center padding, no GroupNorm over time.

Teacher MioCodec is used ONLY as a label generator during training.
At inference, the entire pipeline is self-contained and streaming.

Architecture:
    Audio 44.1kHz
    -> causal 80-bin log-mel @50Hz
    -> 384d causal conv stem x4
    -> 384d causal dual-path blocks x8 (conv + GRU, gated merge)
    -> pair-downsample 50Hz -> 25Hz (concat h[2t-1],h[2t] -> Linear)
    -> FSQ [8,8,8,5,5] + factorized ordinal heads
    -> 768d content embedding
    -> concat offline global 128d -> linear(896->512)
    -> 512d causal code prenet x4
    -> 512d causal dilated decoder blocks x8
    -> repeat upsample [2,2,3,3,7,7]
    -> causal waveform head
    -> Audio 44.1kHz
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsq import DEFAULT_LEVELS

# ── config ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CausalCodecConfig:
    # Frontend
    in_dim: int = 80          # log-mel bins
    frontend_rate: int = 50   # Hz
    codec_rate: int = 25      # Hz (FSQ output rate)
    sample_rate: int = 44100

    # Encoder
    enc_dim: int = 384
    enc_conv_blocks: int = 4
    enc_dual_blocks: int = 8
    enc_conv_kernel: int = 5
    enc_ff_mult: int = 3
    enc_gru_layers: int = 1

    # FSQ
    fsq_levels: tuple[int, ...] = DEFAULT_LEVELS  # [8,8,8,5,5]
    content_dim: int = 768

    # Global
    global_dim: int = 128

    # Decoder
    dec_dim: int = 512
    dec_prenet_blocks: int = 4
    dec_dilated_blocks: int = 8
    dec_conv_kernel: int = 5
    dec_ff_mult: int = 3
    upsample_factors: tuple[int, ...] = (2, 2, 3, 3, 7, 7)
    dec_final_channels: int = 64

    dropout: float = 0.0


# ── building blocks ─────────────────────────────────────────────────────────────

class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding."""
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


class CausalConvStemBlock(nn.Module):
    """RMSNorm -> Linear -> SiLU -> CausalConv -> Linear -> residual"""
    def __init__(self, dim: int, kernel: int, dilation: int = 1, ff_mult: int = 3):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        inner_dim = dim * ff_mult
        self.expand = nn.Linear(dim, inner_dim)
        self.depthwise = CausalConv1d(inner_dim, inner_dim, kernel, dilation=dilation, groups=inner_dim)
        self.project = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.expand(h))
        h = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        h = self.project(h)
        return x + h


class CausalDualPathBlock(nn.Module):
    """Conv (local detail) + GRU (temporal memory) with gated merge"""
    def __init__(self, dim: int, kernel: int, dilation: int = 1,
                 ff_mult: int = 3, gru_layers: int = 1):
        super().__init__()
        # Conv path
        self.conv_norm = nn.RMSNorm(dim)
        self.conv_expand = nn.Linear(dim, dim * ff_mult)
        self.conv_depthwise = CausalConv1d(dim * ff_mult, dim * ff_mult, kernel,
                                           dilation=dilation, groups=dim * ff_mult)
        self.conv_project = nn.Linear(dim * ff_mult, dim)

        # GRU path
        self.gru_norm = nn.RMSNorm(dim)
        self.gru = nn.GRU(dim, dim, num_layers=gru_layers, batch_first=True)

        # Gate
        self.gate = nn.Linear(dim * 2, dim)

        # FFN
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn_expand = nn.Linear(dim, dim * ff_mult)
        self.ffn_project = nn.Linear(dim * ff_mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv path
        hc = self.conv_norm(x)
        hc = F.silu(self.conv_expand(hc))
        hc = self.conv_depthwise(hc.transpose(1, 2)).transpose(1, 2)
        hc = self.conv_project(hc)

        # GRU path
        hg = self.gru_norm(x)
        hg, _ = self.gru(hg)

        # Gated merge
        gate = torch.sigmoid(self.gate(torch.cat([hc, hg], dim=-1)))
        h = gate * hc + (1.0 - gate) * hg
        h = x + h

        # FFN
        hf = self.ffn_norm(h)
        hf = F.silu(self.ffn_expand(hf))
        hf = self.ffn_project(hf)
        return h + hf


class PairDownsample(nn.Module):
    """50Hz -> 25Hz: concat h[2t-1], h[2t] then linear"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T50, D]
        B, T, D = x.shape
        # Align: ensure even length
        if T % 2 == 1:
            x = x[:, :-1]
            T -= 1
        # Reshape pairs: (h[0],h[1]), (h[2],h[3]), ...
        # But we want (h[2t-1], h[2t]) for end-of-cell — use h[1],h[3],h[5]...
        # Which means take odd indices first, then even indices:
        odd = x[:, 1::2]   # h[1], h[3], h[5], ...
        even = x[:, 0::2]  # h[0], h[2], h[4], ...
        # Concatenate: [h[1],h[0]], [h[3],h[2]], ...  — past frame first, then current
        paired = torch.cat([even[:, :odd.shape[1]], odd[:, :odd.shape[1]]], dim=-1)
        return self.proj(paired)


class FactorizedFSQHead(nn.Module):
    """5-axis FSQ with factorized ordinal heads"""
    def __init__(self, in_dim: int, levels: tuple[int, ...], content_dim: int = 768):
        super().__init__()
        self.levels = levels
        n_axes = len(levels)

        self.norm = nn.RMSNorm(in_dim)
        self.pre_fsq = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(), nn.Linear(256, n_axes),
        )
        # Factorized ordinal heads
        self.ordinal_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 128), nn.SiLU(), nn.Linear(128, L))
            for L in levels
        ])
        # Frozen projection to content space
        self.proj_out = nn.Linear(n_axes, content_dim)
        self.proj_out.requires_grad_(False)

    def load_projection(self, state: dict[str, torch.Tensor]) -> None:
        self.proj_out.load_state_dict(state, strict=True)
        self.proj_out.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # h: [B, T, D]
        h_norm = self.norm(h)
        codes = self.pre_fsq(h_norm)  # [B, T, 5]
        logits = [head(h_norm) for head in self.ordinal_heads]  # list of [B, T, L_i]
        content = self.proj_out(codes).transpose(1, 2)  # [B, 768, T]
        return codes, content, logits


class CausalDecoderBlock(nn.Module):
    """Dilated conv + AdaLN from global embedding"""
    def __init__(self, dim: int, condition_dim: int, kernel: int,
                 dilation: int = 1, ff_mult: int = 3):
        super().__init__()
        # AdaLN for conv path
        self.conv_norm = nn.RMSNorm(dim)
        self.adaln_conv = nn.Linear(condition_dim, dim * 2)
        self.conv_depthwise = CausalConv1d(dim, dim, kernel, dilation=dilation, groups=dim)

        # AdaLN for FFN
        self.ffn_norm = nn.RMSNorm(dim)
        self.adaln_ffn = nn.Linear(condition_dim, dim * 2)
        self.ffn_expand = nn.Linear(dim, dim * ff_mult)
        self.ffn_project = nn.Linear(dim * ff_mult, dim)

        nn.init.zeros_(self.adaln_conv.weight)
        nn.init.zeros_(self.adaln_conv.bias)
        nn.init.zeros_(self.adaln_ffn.weight)
        nn.init.zeros_(self.adaln_ffn.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # condition: [B, D] — global embedding, repeated per frame
        shift_c, scale_c = self.adaln_conv(condition).chunk(2, dim=-1)
        h = self.conv_norm(x) * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)
        h = self.conv_depthwise(h.transpose(1, 2)).transpose(1, 2)
        h = x + h

        shift_f, scale_f = self.adaln_ffn(condition).chunk(2, dim=-1)
        hf = self.ffn_norm(h) * (1 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)
        hf = F.silu(self.ffn_expand(hf))
        hf = self.ffn_project(hf)
        return h + hf


class RepeatUpsampleBlock(nn.Module):
    """Repeat each frame `factor` times, then causal conv"""
    def __init__(self, in_dim: int, out_dim: int, factor: int, kernel: int = 7):
        super().__init__()
        self.factor = factor
        self.norm = nn.RMSNorm(in_dim)
        self.conv = CausalConv1d(in_dim, out_dim, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x.repeat_interleave(self.factor, dim=1)
        x = self.norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x


# ── full model ──────────────────────────────────────────────────────────────────

class MioStudentCodec(nn.Module):
    """Full causal neural audio codec."""

    def __init__(self, config: CausalCodecConfig):
        super().__init__()
        self.config = config

        # ── Encoder ──
        self.enc_proj = nn.Linear(config.in_dim, config.enc_dim)
        self.enc_conv_stem = nn.ModuleList([
            CausalConvStemBlock(config.enc_dim, config.enc_conv_kernel,
                                dilation=2**i, ff_mult=config.enc_ff_mult)
            for i in range(config.enc_conv_blocks)
        ])
        self.enc_dual_blocks = nn.ModuleList([
            CausalDualPathBlock(config.enc_dim, config.enc_conv_kernel,
                                dilation=2**(i % 4), ff_mult=config.enc_ff_mult,
                                gru_layers=config.enc_gru_layers)
            for i in range(config.enc_dual_blocks)
        ])
        self.downsample = PairDownsample(config.enc_dim, config.enc_dim * 2 // 3)
        self.enc_out_dim = config.enc_dim * 2 // 3  # after pair-downsample

        # ── FSQ ──
        self.fsq_head = FactorizedFSQHead(
            self.enc_out_dim, config.fsq_levels, config.content_dim
        )

        # ── Decoder ──
        dec_input_dim = config.content_dim + config.global_dim  # 768 + 128 = 896
        self.dec_proj = nn.Linear(dec_input_dim, config.dec_dim)
        self.dec_prenet = nn.ModuleList([
            CausalConvStemBlock(config.dec_dim, config.dec_conv_kernel,
                                dilation=1, ff_mult=config.dec_ff_mult)
            for _ in range(config.dec_prenet_blocks)
        ])
        self.dec_dilated = nn.ModuleList([
            CausalDecoderBlock(config.dec_dim, config.global_dim,
                               config.dec_conv_kernel,
                               dilation=2**(i % 4),
                               ff_mult=config.dec_ff_mult)
            for i in range(config.dec_dilated_blocks)
        ])
        # Upsampler — progressive channel reduction
        current_dim = config.dec_dim
        up_factors = list(config.upsample_factors)
        up_dims = []
        for f in up_factors[:-1]:
            current_dim = max(current_dim * 3 // 4, 64)
            up_dims.append(current_dim)
        up_dims.append(config.dec_final_channels)
        self.upsampler = nn.ModuleList([
            RepeatUpsampleBlock(d_in, d_out, f, kernel=7)
            for (d_in, d_out, f) in zip(
                [config.dec_dim] + up_dims[:-1], up_dims, up_factors
            )
        ])
        # Waveform head
        self.wave_head = nn.Sequential(
            CausalConv1d(config.dec_final_channels, 32, 7),
            nn.SiLU(),
            CausalConv1d(32, 1, 7),
            nn.Tanh(),
        )

    def load_fsq_projection(self, state: dict[str, torch.Tensor]) -> None:
        self.fsq_head.load_projection(state)

    def encode(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Encode mel spectrogram to FSQ codes and content embedding.
        Args:
            mel: [B, 80, T50] log-mel @ 50Hz
        Returns:
            codes: [B, T25, 5] continuous 5d FSQ codes
            content: [B, 768, T25] content embedding
            logits: list of [B, T25, L_i] per-axis ordinal logits
        """
        # Encoder
        h = self.enc_proj(mel.transpose(1, 2))  # [B, T50, enc_dim]
        for block in self.enc_conv_stem:
            h = block(h)
        for block in self.enc_dual_blocks:
            h = block(h)
        # Downsample
        h = self.downsample(h)  # [B, T25, enc_out_dim]
        # FSQ
        codes, content, logits = self.fsq_head(h)
        return codes, content, logits

    def decode(self, content_embedding: torch.Tensor,
               global_embedding: torch.Tensor) -> torch.Tensor:
        """Decode content + global to waveform.
        Args:
            content_embedding: [B, 768, T25]
            global_embedding: [B, 128]
        Returns:
            waveform: [B, T_audio] 44.1kHz
        """
        B, _, T = content_embedding.shape
        # Concatenate content + global per frame
        h = content_embedding.transpose(1, 2)  # [B, T25, 768]
        g = global_embedding.unsqueeze(1).expand(-1, T, -1)  # [B, T25, 128]
        h = torch.cat([h, g], dim=-1)  # [B, T25, 896]

        h = self.dec_proj(h)  # [B, T25, dec_dim]
        for block in self.dec_prenet:
            h = block(h)
        for block in self.dec_dilated:
            h = block(h, global_embedding)

        # Upsample
        for up_block in self.upsampler:
            h = up_block(h)  # [B, T_audio, channels]

        # Waveform
        waveform = self.wave_head(h.transpose(1, 2))  # [B, 1, T_audio]
        return waveform.squeeze(1)  # [B, T_audio]

    def forward(self, mel: torch.Tensor,
                global_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor, list[torch.Tensor]]:
        """Full encode-decode forward pass."""
        codes, content, logits = self.encode(mel)
        waveform = self.decode(content, global_embedding)
        return waveform, codes, content, logits


# ── checkpointing ───────────────────────────────────────────────────────────────

CODEC_FORMAT_VERSION = 1


def save_codec_checkpoint(path: str | Path, model: MioStudentCodec, *,
                          epoch: int, metrics: dict[str, float],
                          optimizer=None, scheduler=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CODEC_FORMAT_VERSION,
        "model_type": "mio_student_codec",
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


def load_codec_checkpoint(path: str | Path, *, device="cpu") -> tuple[MioStudentCodec, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if payload.get("model_type") != "mio_student_codec":
        raise ValueError("not a MioStudentCodec checkpoint")
    config = CausalCodecConfig(**payload["config"])
    model = MioStudentCodec(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    return model, payload
