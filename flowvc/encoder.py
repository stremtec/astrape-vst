"""
FlowVC用 F³-Encoder。

完全因果的ConvNeXt v2エンコーダ。KLフリー（VQなし、コミットメント損失なし）。
ノイズ正則化: z_reg = z + σ·ε（学習時のみ）。

アーキテクチャ:
  波形 (44.1kHz) → 6段 ConvNeXt v2 (ストライド: 2,2,3,3,7,7)
  → z_raw (768次元 @ 25Hz)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CausalConv1d, ConvNeXtV2Block
from .config import EncoderConfig


class F3Encoder(nn.Module):
    """
    因果的連続エンコーダ。
    
    総ダウンサンプル: 2×2×3×3×7×7 = 1764 → 44100/1764 = 25Hz。
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.noise_sigma = cfg.noise_sigma

        in_ch = 1
        stages = []
        for out_ch, stride in zip(cfg.stages, cfg.strides):
            # ストライド付き畳み込み
            stages.append(
                CausalConv1d(in_ch, out_ch, kernel_size=stride * 3, stride=stride)
            )
            # ConvNeXt v2 ブロック
            for _ in range(cfg.blocks_per_stage):
                stages.append(
                    ConvNeXtV2Block(
                        out_ch,
                        kernel_size=cfg.kernel_size,
                        mlp_expansion=cfg.mlp_expansion,
                        use_grn=cfg.use_grn,
                    )
                )
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        # 最終正規化
        self.norm = nn.LayerNorm(cfg.stages[-1])

    def forward(self, wav: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Args:
            wav: (B, 1, T_audio) 波形 @ 44.1kHz
            training: Trueの場合、ノイズ正則化を適用
        Returns:
            z: (B, T_lat, content_dim) @ 25Hz
        """
        x = self.stages(wav)  # (B, C_out, T_lat)

        # → (B, T_lat, C_out)
        z = x.transpose(1, 2)
        z = self.norm(z)

        # ノイズ正則化（F³-Tokenizer 方式）— 学習時のみ
        if training and self.noise_sigma > 0:
            z = z + torch.randn_like(z) * self.noise_sigma

        return z

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """推論モードエンコード（ノイズなし）。"""
        return self.forward(wav, training=False)


def make_encoder(**kwargs) -> F3Encoder:
    cfg = EncoderConfig(**kwargs)
    return F3Encoder(cfg)
