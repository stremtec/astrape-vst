"""
CFM (Conditional Flow Matching) 損失。

L_cfm = MSE(v_θ(z_t, t, c), v_target)
ここで:
  z_t = (1-t)*z_src + t*z_tgt + σ_min·ε
  v_target = z_tgt - z_src
  t ~ U[0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .converter import solve_cfm_euler


class CFMLoss(nn.Module):
    """
    最適輸送パスによる条件付きフローマッチング。

    OTパス: z_t = (1-t)·z_src + t·z_tgt（直線）
    目標速度: v = z_tgt - z_src
    """

    def __init__(self, sigma_min: float = 0.001):
        super().__init__()
        self.sigma_min = sigma_min

    def forward(
        self, vfn: nn.Module, z_src: torch.Tensor, z_tgt: torch.Tensor,
        speaker_emb: torch.Tensor,
        prompt_tokens: torch.Tensor | None = None,
        prosody: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            vfn: VectorFieldNet
            z_src: (B, T, latent_dim) ソース潜在
            z_tgt: (B, T, latent_dim) ターゲット潜在
            speaker_emb: (B, speaker_dim)
            prompt_tokens: (B, n_tokens, prompt_dim) または None
            prosody: (B, T_prosody, 3) ソース韻律 または None
        Returns:
            (loss, logs)
        """
        B = z_src.size(0)
        device = z_src.device

        # バッチ要素ごとに時間をサンプリング
        t = torch.rand(B, device=device)  # U[0, 1]

        # 補間: z_t = (1-t)·z_src + t·z_tgt + σ_min·ε
        t_expanded = t.view(B, 1, 1)
        z_t = (1 - t_expanded) * z_src + t_expanded * z_tgt
        z_t = z_t + torch.randn_like(z_t) * self.sigma_min

        # 目標速度: ソースからターゲットへの直線
        v_target = z_tgt - z_src

        # 予測速度
        v_pred = vfn(z_t, t, speaker_emb, prompt_tokens, prosody)

        # MSE損失
        loss = F.mse_loss(v_pred, v_target)

        logs = {
            "cfm_loss": loss.item(),
            "v_pred_norm": v_pred.norm().item(),
            "v_target_norm": v_target.norm().item(),
        }
        return loss, logs


class FlowVCLoss(nn.Module):
    """
    FlowVCの統合損失:
    - L_cfm: フローマッチング損失
    - L_recon: エンコーダ-デコーダ再構成（フェーズ0）
    - L_aux: 補助損失（話者一貫性）
    """

    def __init__(
        self, cfm: CFMLoss,
        lambda_recon: float = 1.0, lambda_cfm: float = 1.0,
        lambda_spk: float = 0.1,
    ):
        super().__init__()
        self.cfm = cfm
        self.lambda_recon = lambda_recon
        self.lambda_cfm = lambda_cfm
        self.lambda_spk = lambda_spk

    def forward(
        self, vfn: nn.Module, z_src: torch.Tensor, z_tgt: torch.Tensor,
        speaker_emb: torch.Tensor,
        recon_src: torch.Tensor | None = None,
        recon_tgt: torch.Tensor | None = None,
        wav_src: torch.Tensor | None = None,
        wav_tgt: torch.Tensor | None = None,
        prompt_tokens: torch.Tensor | None = None,
        prosody: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """オプションの再構成項付き統合損失。"""
        total = 0.0
        logs = {}

        # CFM損失
        if self.lambda_cfm > 0:
            cfm_loss, cfm_logs = self.cfm(
                vfn, z_src, z_tgt, speaker_emb, prompt_tokens, prosody
            )
            total = total + self.lambda_cfm * cfm_loss
            logs.update(cfm_logs)

        # 再構成損失（波形L1）
        if self.lambda_recon > 0 and wav_src is not None and recon_src is not None:
            recon_l1 = F.l1_loss(recon_src, wav_src)
            if wav_tgt is not None and recon_tgt is not None:
                recon_l1 = recon_l1 + F.l1_loss(recon_tgt, wav_tgt)
            total = total + self.lambda_recon * recon_l1
            logs["recon_l1"] = recon_l1.item()

        # 話者一貫性
        if self.lambda_spk > 0 and z_tgt is not None:
            z_cfm = solve_cfm_euler(
                vfn, z_src, speaker_emb, prompt_tokens, prosody, n_steps=4
            )
            spk_loss = F.mse_loss(z_cfm, z_tgt.detach())
            total = total + self.lambda_spk * spk_loss
            logs["spk_consistency"] = spk_loss.item()

        logs["loss_total"] = total.item() if isinstance(total, torch.Tensor) else total
        return total, logs
