"""
Mimi RVQ Converter: refines token-swap VC by predicting improved LV1-7 codes.
src LV0 (content) + tgt LV1-7 (speaker) → refined LV1-7 codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodebookEmbedding(nn.Module):
    """Learnable codebook embedding for discrete codes (optional: init from Mimi's)."""
    def __init__(self, vocab_size=2048, embed_dim=256, num_levels=8):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for _ in range(num_levels)
        ])

    def forward(self, codes):
        # codes: (B, num_levels, T) integers 0..2047
        B, Q, T = codes.shape
        embs = []
        for q in range(Q):
            emb = self.embeddings[q](codes[:, q, :])  # (B, T, D)
            embs.append(emb)
        return torch.stack(embs, dim=1)  # (B, Q, T, D)


class MimiRVQConverter(nn.Module):
    """
    Takes source LV0 codes + target LV1-7 codes → predicts improved LV1-7 codes.
    
    Architecture: lightweight causal transformer over codebook embeddings.
    """
    def __init__(self, vocab_size=2048, embed_dim=256, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_levels = 8

        # Codebook embeddings (can be initialized from Mimi's pretrained codebook)
        self.code_emb = CodebookEmbedding(vocab_size, embed_dim, num_levels=self.num_levels)

        # Positional encoding for time dimension
        self.time_pe = nn.Parameter(torch.randn(1, 1, 1024, embed_dim) * 0.02)

        # Input projection: concatenate 8 levels' embeddings → d_model
        self.input_proj = nn.Linear(embed_dim * self.num_levels, d_model)

        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: predict codes for levels 1-7
        self.output_proj = nn.Linear(d_model, 7 * vocab_size)

    def forward(self, src_lv0, tgt_lv1_7):
        """
        src_lv0: (B, T) integers 0..2047 — source content codes
        tgt_lv1_7: (B, 7, T) integers 0..2047 — target speaker codes
        Returns: logits (B, 7, T, vocab_size) for levels 1-7
        """
        B, T = src_lv0.shape

        # Embed all codes
        src_lv0_exp = src_lv0.unsqueeze(1)  # (B, 1, T)
        codes_full = torch.cat([src_lv0_exp, tgt_lv1_7], dim=1)  # (B, 8, T)
        emb = self.code_emb(codes_full)  # (B, 8, T, D)

        # Add positional encoding (time)
        emb = emb + self.time_pe[:, :, :T, :]

        # Reshape: (B, T, 8*D)
        h = emb.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, 8*D)

        # Project to d_model
        h = self.input_proj(h)  # (B, T, d_model)

        # Causal transformer (can only attend to past)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h = self.transformer(h, mask=causal_mask)  # (B, T, d_model)

        # Predict codes for each level
        logits = self.output_proj(h)  # (B, T, 7*vocab_size)
        logits = logits.view(B, T, 7, self.vocab_size)  # (B, 7, T, vocab_size) after permute
        logits = logits.permute(0, 2, 1, 3)  # (B, 7, T, vocab_size)

        return logits

    def predict(self, src_lv0, tgt_lv1_7, temperature=0.0):
        """Generate improved codes with optional temperature sampling."""
        logits = self.forward(src_lv0, tgt_lv1_7)
        if temperature == 0.0:
            return logits.argmax(dim=-1)  # (B, 7, T)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs.view(-1, self.vocab_size), 1).view(src_lv0.shape[0], 7, -1)
