"""
FlowVC用 F³-Decoder。

MRF (Multi-Receptive Field) アップサンプラ付き因果的ConvNeXt v2デコーダ。
F³-Encoderの逆演算: TransposedConvでストライドを反転。
各段で話者埋め込みからFiLM条件付け。

出力: 44.1kHz 波形。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d, CausalConvTranspose1d, ConvNeXtV2Block, FiLM
from .config import DecoderConfig


class MRFBlock(nn.Module):
    """
    Multi-Receptive Field ブロック（HiFi-GAN 方式）。
    異なるカーネルサイズとダイレーションの並列Conv1dパス。
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tuple[int, ...] = (3, 7, 11),
        dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.paths = nn.ModuleList()
        for ks, dils in zip(kernel_sizes, dilations):
            path = nn.ModuleList()
            for d in dils:
                path.append(
                    nn.Sequential(
                        CausalConv1d(dim, dim, ks, dilation=d, groups=dim),
                        nn.LeakyReLU(0.1),
                    )
                )
            self.paths.append(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residuals = []
        for path in self.paths:
            h = x
            for layer in path:
                h = layer(h)
            residuals.append(h)

        h = sum(residuals) / len(residuals)
        return x + h  # 残差


class DecoderStage(nn.Module):
    """1段のアップサンプリング: TransposedConv → MRF ×2 → FiLM。"""

    def __init__(
        self, in_ch: int, out_ch: int, stride: int,
        kernel_size: int = 7, mrf_config: dict | None = None,
    ):
        super().__init__()
        stride_kernel = stride * 3  # kernel must be multiple of stride to avoid checkerboard
        self.upsample = CausalConvTranspose1d(
            in_ch, out_ch, kernel_size=stride_kernel, stride=stride
        )
        self.mrf1 = MRFBlock(out_ch, **(mrf_config or {}))
        self.mrf2 = MRFBlock(out_ch, **(mrf_config or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.mrf1(x)
        x = self.mrf2(x)
        return x


class F3Decoder(nn.Module):
    """
    因果的デコーダ: 潜在 → 波形。
    """
    def __init__(self, cfg: DecoderConfig, speaker_dim: int = 192):
        super().__init__()
        self.cfg = cfg

        latent_dim = cfg.latent_dim

        # 入力FiLM
        self.film_in = FiLM(latent_dim, speaker_dim)

        # アップサンプリング前 ConvNeXt ブロック
        self.pre_blocks = nn.ModuleList([
            ConvNeXtV2Block(
                latent_dim, kernel_size=cfg.kernel_size,
                mlp_expansion=4, use_grn=cfg.use_grn,
            )
            for _ in range(cfg.pre_upsample_blocks)
        ])

        # アップサンプリング段（エンコーダの逆順）
        in_ch = latent_dim
        mrf_cfg = {
            "kernel_sizes": cfg.mrf_kernel_sizes,
            "dilations": cfg.mrf_dilations,
        }
        stages = []
        for out_ch, stride in zip(cfg.stages, cfg.strides):
            stages.append(DecoderStage(in_ch, out_ch, stride, mrf_config=mrf_cfg))
            # 段ごとにFiLM
            stages.append(FiLM(out_ch, speaker_dim))
            in_ch = out_ch

        self.upsample_stages = nn.ModuleList(stages)

        # 最終射影
        self.final_conv = CausalConv1d(cfg.stages[-1], 1, kernel_size=7)

    def forward(self, z: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T_lat, latent_dim) 潜在 @ 25Hz
            speaker_emb: (B, speaker_dim) ターゲット話者
        Returns:
            wav: (B, 1, T_audio) 波形 @ 44.1kHz
        """
        # 入力FiLM
        x = z.transpose(1, 2)  # (B, dim, T)
        x = self.film_in(x, speaker_emb)

        # アップサンプリング前精緻化
        for block in self.pre_blocks:
            x = block(x)

        # アップサンプリング段
        for stage in self.upsample_stages:
            if isinstance(stage, FiLM):
                x = stage(x, speaker_emb)
            else:
                x = stage(x)

        # 最終
        x = self.final_conv(x)
        x = torch.tanh(x)

        return x


def make_decoder(speaker_dim: int = 192, **kwargs) -> F3Decoder:
    cfg = DecoderConfig(**kwargs)
    return F3Decoder(cfg, speaker_dim=speaker_dim)
