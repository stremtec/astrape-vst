"""
FlowVC用 因果的ConvNeXt v2 構成ブロック。

全畳み込みが左パディングのみ — 未来情報の漏洩なし。
GRN (Global Response Normalization) は ConvNeXt v2 より。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 因果的 Conv1d ──────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """左パディングのみの1次元畳み込み（因果的）。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.pad_total = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            stride=stride, dilation=dilation,
            groups=groups, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.pad_total > 0:
            x = F.pad(x, (self.pad_total, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """因果的出力トリミング付き転置畳み込み。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_in)
        out = self.conv(x)
        expected_len = x.shape[2] * self.stride
        return out[:, :, :expected_len]


# ── GRN (Global Response Normalization) ────────────────────────

class GRN(nn.Module):
    """
    ConvNeXt v2 の Global Response Normalization。
    チャネル方向 L2ノルム → 除算正規化 → 学習可能スケール/バイアス。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


# ── ConvNeXt v2 ブロック ────────────────────────────────────────

class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt v2: DWConv7 → LayerNorm → 1×1拡張 → GELU → GRN → 1×1収縮。

    因果的: DWConvは左パディングのみ。
    確率的深さのためのDropPath。
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        mlp_expansion: int = 4,
        drop_path: float = 0.0,
        use_grn: bool = True,
    ):
        super().__init__()
        self.use_grn = use_grn

        self.dwconv = CausalConv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * mlp_expansion)
        self.act = nn.GELU()
        if use_grn:
            self.grn = GRN(dim * mlp_expansion)
        self.pwconv2 = nn.Linear(dim * mlp_expansion, dim)

        # LayerScale（ゼロ初期化で初期恒等写像）
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))

        # DropPath
        self.drop_path = drop_path > 0.0
        if self.drop_path:
            self.drop_path_prob = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — チャネルファースト
        shortcut = x

        # DWConv
        x = self.dwconv(x)  # (B, C, T)

        # チャネル方向LayerNorm
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)

        # MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # (B, mlp*C, T) GRN用
        if self.use_grn:
            x = self.grn(x)
        x = x.transpose(1, 2)  # (B, T, mlp*C)
        x = self.pwconv2(x)

        # LayerScale
        x = self.gamma * x

        x = x.transpose(1, 2)  # (B, C, T)

        # DropPath（確率的深さ）
        if self.drop_path and self.training:
            if torch.rand(1).item() < self.drop_path_prob:
                return shortcut
            x = x / (1.0 - self.drop_path_prob)

        return shortcut + x


# ── AdaLN-Zero（Flow変換器用）───────────────────────────────────

class AdaLNZero(nn.Module):
    """
    ゼロ初期化ゲーティング付き適応的LayerNorm。
    条件 → MLP → (スケール, シフト, ゲート)。
    全ゼロ初期化 → ブロックが初期恒等写像。
    """

    def __init__(self, dim: int, cond_dim: int, mlp_hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, dim * 3),
        )
        # 最終層ゼロ初期化
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: (B, T, dim)
            cond: (B, T, cond_dim) または (B, cond_dim) — 2Dならブロードキャスト
        Returns:
            (x_modulated, gate)
        """
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, x.size(1), -1)

        params = self.mlp(cond)
        shift, scale, gate = params.chunk(3, dim=-1)

        x_norm = self.norm(x)
        x_mod = x_norm * (1.0 + scale) + shift

        return x_mod, gate


# ── FiLM（デコーダ条件付け用）───────────────────────────────────

class FiLM(nn.Module):
    """ゼロ初期化付き特徴量線形変調。"""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — チャネルファースト
            cond: (B, cond_dim)
        Returns:
            (B, C, T) 変調済み
        """
        gamma, beta = self.proj(cond).chunk(2, dim=-1)  # (B, C)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta
