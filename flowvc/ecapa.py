"""
Pretrained ECAPA-TDNN speaker encoder for FlowVC.

Uses speechbrain/spkrec-ecapa-voxceleb (frozen).
Output: 192-dim speaker embedding + dummy prompt tokens.

Replaces the randomly-initialized ConvNeXt SpeakerEncoder
which produced meaningless speaker vectors.
"""

from __future__ import annotations
import torch, torch.nn as nn


class ECAPASpeakerEncoder(nn.Module):
    """
    Frozen ECAPA-TDNN speaker encoder.
    Returns (spk_emb, prompt_tokens) matching the VFN interface.
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._model = None
        
    def _ensure_model(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/speechbrain_ecapa",
            )
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False
    
    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wav: (B, 1, T) waveform (any sample rate, ECAPA resamples internally)
        Returns:
            spk_emb: (B, 192)
            prompt_tokens: (B, 4, 192) — dummy tokens (ECAPA has no prompt concept)
        """
        self._ensure_model()
        B = wav.size(0)
        
        # ECAPA expects (B, T) or (B, 1, T) at 16kHz
        wav_in = wav.squeeze(1)  # (B, T)
        wav_in = wav_in.to(self._model.device)
        
        spk_emb = self._model.encode_batch(wav_in).squeeze(1)  # (B, 1, 192) → (B, 192)
        
        # Dummy prompt tokens (ECAPA doesn't have P-Flow style prompts)
        prompt_tokens = spk_emb.unsqueeze(1).expand(-1, 4, -1)  # (B, 4, 192)
        
        return spk_emb.to(wav.device), prompt_tokens.to(wav.device)
    
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        spk_emb, _ = self.forward(wav)
        return spk_emb


def make_ecapa_encoder(device: str = "cpu") -> ECAPASpeakerEncoder:
    return ECAPASpeakerEncoder(device=device)
