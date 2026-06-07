"""
Codex VC: LV0 + Text-Invariant Speaker → LV1-7 Code Generator.

Architecture:
    source audio → Mimi encode → LV0 codes ──────────┐
    target audio → Resemblyzer → spk embedding ───────┤
                                                       ↓
                                         Bidirectional Transformer
                                                       ↓
                                         LV1-7 codes (7×T)
                                                       ↓
                                         Mimi decoder → VC audio

Reference:
    Kyutai Mimi: https://github.com/kyutai-labs/moshi
    Resemblyzer: https://github.com/resemble-ai/Resemblyzer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeGenerator(nn.Module):
    """Bidirectional transformer predicting LV1-7 codes from LV0 + speaker.

    Args:
        vocab: Codebook size (2048 for Mimi).
        lv0_dim: LV0 embedding dimension.
        spk_dim: Speaker embedding input dimension (Resemblyzer = 256).
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab: int = 2048,
        lv0_dim: int = 128,
        spk_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab = vocab

        # ── Input embeddings ──
        self.lv0_emb = nn.Embedding(vocab, lv0_dim)
        self.spk_proj = nn.Linear(spk_dim, lv0_dim)

        # ── Input projection + position encoding ──
        self.input_proj = nn.Linear(lv0_dim * 2, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)

        # ── Bidirectional transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Output heads (one per LV1-7 level) ──
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab) for _ in range(7)])

    def forward(self, lv0: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """Predict LV1-7 code logits.

        Args:
            lv0: (B, T) integer tensor, source LV0 content codes [0, vocab).
            spk_emb: (B, spk_dim) float tensor, target speaker embedding.

        Returns:
            logits: (B, 7, T, vocab) float tensor.
        """
        B, T = lv0.shape

        # Embed
        lv0_e = self.lv0_emb(lv0)                               # (B, T, lv0_dim)
        spk_e = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)  # (B, T, lv0_dim)
        h = torch.cat([lv0_e, spk_e], dim=-1)                   # (B, T, 2*lv0_dim)

        # Project + position
        h = self.input_proj(h)                                  # (B, T, d_model)
        h = h + self.pos_emb[:, :T, :]

        # Bidirectional encoding (no causal mask → full context)
        h = self.transformer(h)                                 # (B, T, d_model)

        # Level-wise prediction
        logits = torch.stack([head(h) for head in self.heads], dim=1)  # (B, 7, T, vocab)
        return logits

    @torch.no_grad()
    def predict(self, lv0: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """Generate discrete codes via argmax.

        Args:
            lv0: (B, T) integer tensor.
            spk_emb: (B, spk_dim) float tensor.

        Returns:
            codes: (B, 7, T) integer tensor, predicted LV1-7 codes.
        """
        logits = self.forward(lv0, spk_emb)
        return logits.argmax(dim=-1)


def compute_loss(
    model: CodeGenerator,
    lv0: torch.Tensor,
    lv1_7_gt: torch.Tensor,
    spk_emb: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute cross-entropy loss across all 7 LV1-7 levels.

    Args:
        model: CodeGenerator instance.
        lv0: (B, T) source LV0 codes.
        lv1_7_gt: (B, 7, T) target LV1-7 ground truth.
        spk_emb: (B, spk_dim) target speaker embedding.
        criterion: nn.CrossEntropyLoss instance.

    Returns:
        Scalar loss averaged over 7 levels.
    """
    logits = model(lv0, spk_emb)  # (B, 7, T, vocab)
    loss = sum(
        criterion(logits[:, i].reshape(-1, model.vocab), lv1_7_gt[:, i].reshape(-1))
        for i in range(7)
    )
    return loss / 7.0  # average per level


@torch.no_grad()
def convert(
    model: CodeGenerator,
    mimi,
    src_audio: torch.Tensor,
    spk_emb: torch.Tensor,
) -> torch.Tensor:
    """Full voice conversion pipeline.

    Args:
        model: CodeGenerator instance.
        mimi: MimiModel (frozen).
        src_audio: (B, 1, T_audio) source waveform.
        spk_emb: (B, spk_dim) target speaker embedding.

    Returns:
        vc_audio: (B, 1, T_audio') converted waveform.
    """
    codes_src = mimi.encode(src_audio)                          # (B, 8, T)
    lv0 = codes_src[:, 0, :]                                    # (B, T)
    pred_lv1_7 = model.predict(lv0, spk_emb)                    # (B, 7, T)
    codes_vc = torch.cat([lv0.unsqueeze(1), pred_lv1_7], dim=1)  # (B, 8, T)
    return mimi.decode(codes_vc)
