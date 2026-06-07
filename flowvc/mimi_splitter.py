"""
Mimi Splitter v2 — content/speaker disentanglement for 512-dim continuous latent.

Architecture:
  z (512-dim, 12.5Hz) → ContentBottleneck(512→128→512) → z_content
                       → SpeakerTemporalPool → z_speaker

Training:
  L_recon:   |decoder(z_content + z_speaker) - audio|₁
  L_spk_inv: |cos(z_content_A, z_content_B) - 1|²  (same text, diff spk → content should match)
  L_ctt_inv: |speaker_extractor(z_speaker) - ECAPA(audio)|₂  (speaker subspace matches ECAPA)
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

MIMI_DIM = 512
BOTTLENECK = 128


class ContentBottleneck(nn.Module):
    """Narrow bottleneck forces speaker info to drop from content path."""

    def __init__(self, dim: int = MIMI_DIM, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.compress = nn.Conv1d(dim, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        h = self.compress(x)
        h = F.gelu(h)
        h = self.expand(h)
        h = h.transpose(1, 2)  # (B, T, D) for LayerNorm
        h = self.norm(h)
        return (h + x.transpose(1, 2)).transpose(1, 2)  # residual


class SpeakerEncoder(nn.Module):
    """Extract per-utterance speaker vector from latent."""

    def __init__(self, dim: int = MIMI_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, 256, 5, padding=2), nn.GELU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.GELU(),
        )
        # Attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.proj = nn.Linear(256, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) → (B, 256, T)
        h = self.conv(x)  # (B, 256, T)
        h = h.transpose(1, 2)  # (B, T, 256)
        B = h.size(0)
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, h, h)
        spk = self.proj(pooled)  # (B, 1, D)
        return spk.transpose(1, 2)  # (B, D, 1) — per-utterance, expandable to T


class MimiSplitterV2(nn.Module):
    """Content/Speaker disentanglement for Mimi latent space."""

    def __init__(self, dim: int = MIMI_DIM, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.content = ContentBottleneck(dim, bottleneck)
        self.speaker = SpeakerEncoder(dim)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D, T) — Mimi latent @ 12.5Hz
        Returns:
            z_content: (B, D, T) time-varying content
            z_spk: (B, D, 1) per-utterance speaker (expand by caller)
        """
        z_content = self.content(z)
        z_spk = self.speaker(z)  # (B, D, 1)
        return z_content, z_spk


def splitter_loss(
    mimi,
    splitter: MimiSplitterV2,
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    audio_src: torch.Tensor,
    audio_tgt: torch.Tensor,
    ecapa_emb_src: torch.Tensor,
    ecapa_emb_tgt: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Combined splitter loss.

    z_src, z_tgt: same text, different speakers
    """
    B = z_src.size(0)

    # Split both sources
    c_src, s_src = splitter(z_src)
    c_tgt, s_tgt = splitter(z_tgt)

    # Expand speaker to match temporal dim
    T = z_src.size(2)
    s_src_exp = s_src.expand(-1, -1, T)
    s_tgt_exp = s_tgt.expand(-1, -1, T)

    # Reconstruction: decode from src content + src speaker (keep on gradient path!)
    z_recon = c_src + s_src_exp
    codes = mimi.quantizer.encode(z_recon.transpose(1, 2))
    audio_recon = mimi.decode(codes)
    recon_loss = F.l1_loss(audio_recon, audio_src)

    # Content invariance: src content ≈ tgt content (same text!)
    cos_content = F.cosine_similarity(
        c_src.transpose(1, 2).reshape(-1, MIMI_DIM),
        c_tgt.transpose(1, 2).reshape(-1, MIMI_DIM),
        dim=-1
    ).mean()
    content_loss = (1 - cos_content) ** 2

    # Speaker consistency: speaker vector should match ECAPA
    s_src_pooled = s_src.squeeze(-1)  # (B, D)
    spk_loss = 1 - F.cosine_similarity(s_src_pooled, ecapa_emb_src, dim=-1).mean()

    total = recon_loss + 0.5 * content_loss + 0.3 * spk_loss

    logs = {"recon": recon_loss.item(), "content": content_loss.item(), "spk": spk_loss.item()}
    return total, logs
