"""
WavLM SSL distillation teacher for FlowVC Phase 0.

Frozen WavLM-large provides target features. Encoder learns to predict
these features via a lightweight projection head, preventing collapse.

Ref: USAD 2.0 (arXiv:2606.06444), F³-Tokenizer (arXiv:2606.06357)
"""

from __future__ import annotations
import torch, torch.nn as nn


WAVLM_SR = 16000  # WavLM native sample rate


class WavLMDistiller(nn.Module):
    """
    Frozen WavLM teacher + learnable projection head.

    encoder(audio) → latent (768-dim)
    latent → proj_head → pred (1024-dim)
    WavLM(audio) → target (1024-dim)
    loss = MSE(pred, target.detach()) + (1 - cos_sim(pred, target))
    """

    def __init__(self, encoder: nn.Module, device: str = "cpu"):
        super().__init__()
        self.encoder = encoder
        self.device = device

        # Projection head: 768 → 1024 (WavLM-large hidden dim)
        self.proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )

    def _ensure_wavlm(self):
        if not hasattr(self, "_wavlm"):
            try:
                from transformers import WavLMModel
                self._wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
                self._wavlm.eval()
                for p in self._wavlm.parameters():
                    p.requires_grad = False
            except ImportError:
                raise ImportError("transformers required: pip install transformers")

    @torch.no_grad()
    def _get_wavlm_target(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Extract WavLM features at ~25Hz frame rate."""
        self._ensure_wavlm()

        if orig_sr != WAVLM_SR:
            wav = torchaudio.functional.resample(wav, orig_sr, WAVLM_SR)

        wav = wav.to(self._wavlm.device)
        out = self._wavlm(wav.squeeze(1), output_hidden_states=True)
        feats = out.last_hidden_state  # (B, T_wlm, 1024)

        # Resample from WavLM frame rate (~50Hz) to 25Hz via temporal pooling
        B, T_wlm, D = feats.shape
        T_out = T_wlm // 2  # 50Hz → 25Hz
        feats = feats[:, :T_out * 2, :].reshape(B, T_out, 2, D).mean(dim=2)  # (B, T_out, 1024)
        return feats

    def forward(self, src_wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent: (B, T, 768) encoder output @ 25Hz
            loss: frame-level distillation loss
        """
        z = self.encoder(src_wav, training=True)  # (B, T_lat, 768)

        # Predict WavLM features per frame
        pred = self.proj(z)  # (B, T_lat, 1024)

        # WavLM target at matching frame rate
        target = self._get_wavlm_target(src_wav, 44100)
        target = target.to(pred.device)

        # Align frame counts
        n = min(pred.size(1), target.size(1))
        pred, target = pred[:, :n, :], target[:, :n, :]

        # MSE + cosine loss per frame
        mse = nn.functional.mse_loss(pred, target)
        cos = 1 - nn.functional.cosine_similarity(
            pred.reshape(-1, 1024), target.reshape(-1, 1024), dim=-1
        ).mean()

        return z, mse + 0.1 * cos


import torchaudio
