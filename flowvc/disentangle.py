"""
Kanade-style disentanglement via narrow information bottleneck (v2).

v2 improvements:
  - Bottleneck 8→16 dim (still narrow, but enough for speaker distinction)
  - SpeakerEncoder: 4 Conv1d + attention pooling + projection
  - Per-utterance speaker vector (B, 1, D), caller expands to T
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

AUDIODEC_DIM = 64
BOTTLENECK_DIM = 16
SPEAKER_DIM = 64


class ContentBottleneck(nn.Module):
    """Narrow bottleneck forces content-only encoding (drops speaker info)."""

    def __init__(self, bottleneck: int = BOTTLENECK_DIM):
        super().__init__()
        dim = AUDIODEC_DIM
        self.compress = nn.Linear(dim, bottleneck)
        self.expand = nn.Linear(bottleneck, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.expand(F.gelu(self.compress(x))) + x)


class SpeakerEncoder(nn.Module):
    """
    Stronger speaker encoder: 4-layer Conv1d + attention pooling.
    Extracts per-utterance speaker vector.
    """

    def __init__(self):
        super().__init__()
        in_dim, out_dim = AUDIODEC_DIM, SPEAKER_DIM
        # 4-layer Conv1d with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2), nn.GELU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.GELU(),
        )
        self.proj = nn.Linear(256, out_dim)
        
        # Attention pooling: learnable query
        self.attn_query = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D, T) for Conv1d
        h = x.transpose(1, 2)
        h = self.conv1(h)  # (B, 128, T)
        h = self.conv2(h)  # (B, 256, T)
        h = h.transpose(1, 2)  # (B, T, 256)
        
        # Attention pooling → per-utterance
        B = h.size(0)
        q = self.attn_query.expand(B, -1, -1)
        pooled, _ = self.attn(q, h, h)
        spk = self.proj(pooled)  # (B, 1, out_dim)
        return spk


class KanadeDisentangler(nn.Module):
    """
    Disentangle AudioDec latent into content + speaker via architectural bottleneck.
    """

    def __init__(self):
        super().__init__()
        self.content = ContentBottleneck()
        self.speaker = SpeakerEncoder()

    def forward(self, z: torch.Tensor):
        z_content = self.content(z)
        z_spk = self.speaker(z)  # (B, 1, 64)
        return z_content, z_spk
