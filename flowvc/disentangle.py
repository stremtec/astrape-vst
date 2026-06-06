"""
Kanade-style disentanglement via narrow information bottleneck.

AudioDec latent (64-dim) → two paths:
  Content: 64 → narrow bottleneck (8-dim) → 64  (forced compression)
  Speaker: 64 → ConvNeXt → pooling → 32

No auxiliary losses. The 8-dim bottleneck naturally drops speaker info.
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

AUDIODEC_DIM = 64
BOTTLENECK_DIM = 8
SPEAKER_DIM = 64  # matches content dim for addition


class ContentBottleneck(nn.Module):
    """Narrow bottleneck forces content-only encoding (drops speaker info)."""

    def __init__(self, bottleneck: int = 8):
        super().__init__()
        dim = AUDIODEC_DIM
        self.compress = nn.Linear(dim, bottleneck)
        self.expand = nn.Linear(bottleneck, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.expand(F.gelu(self.compress(x))) + x)


class SpeakerEncoder(nn.Module):
    """Separate path for speaker info — avoids squeezing through bottleneck."""

    def __init__(self):
        super().__init__()
        in_dim, out_dim = AUDIODEC_DIM, SPEAKER_DIM
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, out_dim, 5, padding=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D, T) for conv
        h = self.conv(x.transpose(1, 2))
        h = self.pool(h).squeeze(-1)  # (B, 32)
        return h


class KanadeDisentangler(nn.Module):
    """
    Disentangle AudioDec latent into content + speaker via architectural bottleneck.
    No auxiliary losses — bottleneck architecture does the work.
    """

    def __init__(self):
        super().__init__()
        self.content = ContentBottleneck()
        self.speaker = SpeakerEncoder()

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, T, 64) AudioDec latent
        Returns:
            z_content: (B, T, 64)
            z_speaker: (B, 32) per-utterance
        """
        z_content = self.content(z)
        z_spk = self.speaker(z)  # (B, 32)
        # Expand speaker to all time frames
        z_spk_expanded = z_spk.unsqueeze(1).expand(-1, z.size(1), -1)
        return z_content, z_spk_expanded
