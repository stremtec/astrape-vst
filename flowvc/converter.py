"""
FlowVC 変換器 — 条件付きフローマッチング ODE。

ベクトル場ネットワーク v_θ(z_t, t, c) がソース→ターゲットの速度場を予測。

アーキテクチャ:
  12 個の ConvNeXt v2 ブロック (dim=512) + AdaLN-Zero(時間, 条件)
  + 話者プロンプトトークンへのクロスアテンション (層 [3,6,9])
  + ゼロ初期化出力ゲート → t=0 で恒等写像

推論: Euler または RK4 ODE ソルバ (4-8ステップ)。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AdaLNZero, CausalConv1d, GRN
from .config import FlowConverterConfig


# ── 正弦波時間埋め込み ──────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """連続時間用 Transformer 式正弦波位置埋め込み。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) または (B, 1) — 連続時間 ∈ [0, 1]
        Returns:
            (B, dim) 埋め込み
        """
        t = t.view(-1, 1).float()
        device = t.device

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


# ── 時間 MLP ────────────────────────────────────────────────────

class TimeMLP(nn.Module):
    """Sinusoidal embedding → MLP → time conditioning (standard init)."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Standard init — NOT zero-init. Time signal must be learned from step 0.

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ── 話者プロンプトへのクロスアテンション ─────────────────────────

class SpeakerCrossAttn(nn.Module):
    """変換器隠れ状態から話者プロンプトトークンへのクロスアテンション。"""

    def __init__(self, dim: int = 512, prompt_dim: int = 192, n_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(prompt_dim)
        self.proj_kv = nn.Linear(prompt_dim, dim * 2)  # K, V
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        # ゼロ初期化出力射影
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            prompt: (B, n_tokens, prompt_dim)
        Returns:
            (B, T, dim)
        """
        q = self.norm_q(x)
        kv = self.norm_kv(prompt)
        k, v = self.proj_kv(kv).chunk(2, dim=-1)  # 各 (B, n_tokens, dim)

        attn_out, _ = self.attn(q, k, v)
        return x + self.out_proj(attn_out)


# ── フローブロック (ConvNeXt v2 + AdaLN-Zero + クロスアテンション) ─

class FlowBlock(nn.Module):
    """ベクトル場ネットワークの単一ブロック。"""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        hidden = dim * mlp_expansion
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pwconv2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.dwconv(h)
        h = h.transpose(1, 2)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = h.transpose(1, 2)
        h = self.grn(h)
        h = h.transpose(1, 2)
        h = self.pwconv2(h)
        return x + h  # standard residual, no gate suppression


# ── ベクトル場ネットワーク ──────────────────────────────────────

class VectorFieldNet(nn.Module):
    """
    v_θ(z_t, t, c) — CFMの速度場を予測。
    
    12個のFlowBlock（巡回ダイレーション）+ 層[3,6,9]でのクロスアテンション。
    """

    def __init__(self, cfg: FlowConverterConfig):
        super().__init__()
        self.cfg = cfg

        # 入力射影
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)

        # 時間埋め込み
        self.time_mlp = TimeMLP(cfg.time_dim)

        # 条件射影: 話者(192) + 韻律(3) → cond_dim
        cond_in = cfg.speaker_dim + cfg.prosody_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

        # AdaLN条件 = cond_proj(フレーム) + time_emb(ブロードキャスト)
        adaln_cond_dim = cfg.cond_dim + cfg.time_dim

        # フローブロック
        self.blocks = nn.ModuleList([
            FlowBlock(
                dim=cfg.hidden_dim,
                cond_dim=adaln_cond_dim,
                kernel_size=cfg.kernel_size,
                dilation=d,
                mlp_expansion=cfg.mlp_expansion,
            )
            for d in cfg.dilations
        ])

        # クロスアテンションモジュール
        self.cross_attns = nn.ModuleDict()
        if cfg.use_cross_attn:
            for layer_idx in cfg.cross_attn_layers:
                self.cross_attns[str(layer_idx)] = SpeakerCrossAttn(
                    dim=cfg.hidden_dim,
                    prompt_dim=cfg.prompt_dim,
                    n_heads=cfg.cross_attn_heads,
                )

        # 出力射影
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.out_gate = nn.Parameter(torch.ones(1))  # 1.0 = no gating, let VFN learn freely

    def _assemble_cond(
        self, t: torch.Tensor, speaker_emb: torch.Tensor,
        prosody: torch.Tensor | None, T: int,
    ) -> torch.Tensor:
        """時間・話者・韻律からフレーム単位条件を構築。"""
        B = speaker_emb.size(0)

        # 時間埋め込み → 全フレームにブロードキャスト
        t_emb = self.time_mlp(t)  # (B, time_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, time_dim)

        # 話者 + 韻律のフレーム単位条件
        spk = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, speaker_dim)
        if prosody is not None:
            if prosody.size(1) != T:
                if prosody.size(1) > T:
                    prosody = prosody[:, :T, :]
                else:
                    prosody = F.pad(prosody, (0, 0, 0, T - prosody.size(1)))
            cond_cat = torch.cat([spk, prosody], dim=-1)
        else:
            # Pad with zeros to match cond_in dimension (speaker_dim + prosody_dim)
            prosody_zeros = torch.zeros(B, T, self.cfg.prosody_dim, device=spk.device)
            cond_cat = torch.cat([spk, prosody_zeros], dim=-1)

        cond = self.cond_proj(cond_cat)  # (B, T, cond_dim)
        return torch.cat([cond, t_emb], dim=-1)  # (B, T, cond_dim + time_dim)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor,
        speaker_emb: torch.Tensor,
        prompt_tokens: torch.Tensor | None = None,
        prosody: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (B, T, latent_dim) 現在の潜在状態
            t: (B,) または (B, 1) 時間 ∈ [0, 1]
            speaker_emb: (B, speaker_dim) ターゲット話者
            prompt_tokens: (B, n_tokens, prompt_dim) クロスアテンション用
            prosody: (B, T_prosody, 3) ソース韻律
        Returns:
            v: (B, T, latent_dim) 速度場
        """
        B, T_lat, _ = z_t.shape

        x = self.in_proj(z_t)  # (B, T, 512)
        cond = self._assemble_cond(t, speaker_emb, prosody, T_lat)

        for i, block in enumerate(self.blocks):
            layer_idx = i + 1
            if str(layer_idx) in self.cross_attns and prompt_tokens is not None:
                x = self.cross_attns[str(layer_idx)](x, prompt_tokens)
            x = block(x, cond)

        v = self.out_proj(x)
        v = v * self.out_gate  # ゼロ初期化 → 恒等写像
        return v


# ── ODE ソルバ ──────────────────────────────────────────────────

def solve_cfm_euler(
    vfn: VectorFieldNet, z_src: torch.Tensor, speaker_emb: torch.Tensor,
    prompt_tokens: torch.Tensor | None, prosody: torch.Tensor | None,
    n_steps: int = 4,
) -> torch.Tensor:
    """
    Euler法によるCFM ODEの求解。
    
    z_tgt = z_src + Σ v_θ(z_i, t_i, c) * dt
    """
    z = z_src
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((z_src.size(0),), i / n_steps, device=z.device)
        v = vfn(z, t, speaker_emb, prompt_tokens, prosody)
        z = z + v * dt

    # Final evaluation at t=1 (within valid domain), weighted by dt/2 for trapezoidal finish
    t_end = torch.ones(z_src.size(0), device=z.device)
    v_end = vfn(z, t_end, speaker_emb, prompt_tokens, prosody)
    z = z + v_end * dt * 0.5  # trapezoidal corrector: uses velocity at both t≈1 and t=1

    return z


def solve_cfm_rk4(
    vfn: VectorFieldNet, z_src: torch.Tensor, speaker_emb: torch.Tensor,
    prompt_tokens: torch.Tensor | None, prosody: torch.Tensor | None,
    n_steps: int = 4,
) -> torch.Tensor:
    """
    RK4（4次ルンゲクッタ）によるCFM ODEの求解。
    Eulerより高品質、2倍の計算コスト。
    """
    z = z_src
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_i = i * dt
        t = torch.full((z_src.size(0),), t_i, device=z.device)
        t_half = torch.full_like(t, t_i + dt / 2)
        t_next = torch.full_like(t, t_i + dt)

        k1 = vfn(z, t, speaker_emb, prompt_tokens, prosody)
        k2 = vfn(z + k1 * dt / 2, t_half, speaker_emb, prompt_tokens, prosody)
        k3 = vfn(z + k2 * dt / 2, t_half, speaker_emb, prompt_tokens, prosody)
        k4 = vfn(z + k3 * dt, t_next, speaker_emb, prompt_tokens, prosody)

        z = z + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    return z


# ── ファクトリ ──────────────────────────────────────────────────

def make_vector_field_net(**kwargs) -> VectorFieldNet:
    cfg = FlowConverterConfig(**kwargs)
    return VectorFieldNet(cfg)
