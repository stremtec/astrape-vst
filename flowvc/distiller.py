"""
WavLM direct feature matching for FlowVC Phase 0.

Encoder output is matched directly to WavLM features (no decoder, no learnable
projection). Fixed random projection maps WavLM 1024-dim → 768-dim.

This eliminates decoder compensation as a collapse mechanism.
"""

from __future__ import annotations
import torch, torch.nn as nn, torchaudio

WAVLM_SR = 16000


class WavLMDirectDistiller(nn.Module):
    """
    Encoder → WavLM direct feature matching.

    encoder(audio) → z (B, T, 768) @ 25Hz
    WavLM(audio) → feats (B, T_wlm, 1024) @ ~50Hz
    feats → fixed_proj → target (B, T_wlm, 768) @ ~50Hz
    z → upsample 25→50Hz → pred (B, T_wlm, 768)
    loss = MSE(pred, target)
    """

    def __init__(self, encoder: nn.Module, device: str = "cpu"):
        super().__init__()
        self.encoder = encoder
        self.device = device

        # Fixed random projection: 1024 → 768 (non-learnable, prevents memorization)
        proj = torch.randn(1024, 768) * 0.01
        self.register_buffer("fixed_proj", proj)

    def _ensure_wavlm(self):
        if not hasattr(self, "_wavlm"):
            from transformers import WavLMModel
            self._wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
            self._wavlm.eval()
            for p in self._wavlm.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _wavlm_features(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """WavLM features → fixed projection → (B, T, 768) @ ~50Hz."""
        self._ensure_wavlm()

        if orig_sr != WAVLM_SR:
            wav = torchaudio.functional.resample(wav, orig_sr, WAVLM_SR)

        wav = wav.to(self._wavlm.device)
        out = self._wavlm(wav.squeeze(1), output_hidden_states=True)
        feats = out.last_hidden_state  # (B, T, 1024)

        # Fixed projection to 768-dim
        target = feats @ self.fixed_proj.to(feats.device)  # (B, T, 768)
        return target

    def forward(self, src_wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z: (B, T_enc, 768) encoder output @ 25Hz
            loss: direct MSE matching loss
        """
        z = self.encoder(src_wav, training=True)  # (B, T_enc, 768) @ 25Hz

        # WavLM target @ ~50Hz
        target = self._wavlm_features(src_wav, 44100)  # (B, T_wlm, 768)
        target = target.to(z.device)

        # Upsample encoder output 25→50Hz via repeat-interleave
        z_up = z.repeat_interleave(2, dim=1)  # (B, T_enc*2, 768)

        # Align lengths
        n = min(z_up.size(1), target.size(1))
        z_up, target = z_up[:, :n, :], target[:, :n, :]

        # Direct MSE (no learnable projection — encoder must match raw features)
        mse = nn.functional.mse_loss(z_up, target)

        # Anti-collapse: per-channel temporal variance
        var_loss = torch.relu(0.5 - z.var(dim=1).mean())

        return z, mse + 0.1 * var_loss
