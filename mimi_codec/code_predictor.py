"""
Cross-text VC: src LV0 + tgt_speaker_emb → compatible LV1-7 codes.
Trained on VCTK parallel utterances: learn code compatibility grammar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEmbedder(nn.Module):
    """Extract global speaker embedding from LV1-7 codes."""
    def __init__(self, vocab_size=2048, embed_dim=128, speaker_dim=512):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for _ in range(7)
        ])
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 7, 512),
            nn.GELU(),
            nn.Linear(512, speaker_dim),
        )
    
    def forward(self, lv1_7_codes):
        """lv1_7_codes: (B, 7, T) → (B, speaker_dim)"""
        B, _, T = lv1_7_codes.shape
        embs = []
        for lv in range(7):
            emb = self.embeddings[lv](lv1_7_codes[:, lv, :])  # (B, T, D)
            embs.append(emb.mean(dim=1))  # (B, D) time-average
        h = torch.cat(embs, dim=-1)  # (B, 7*D)
        return self.proj(h)


class CodePredictor(nn.Module):
    """
    Predicts LV1-7 codes given LV0 codes + speaker embedding.
    Causal transformer over code sequence.
    """
    def __init__(self, vocab_size=2048, embed_dim=256, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # LV0 embedding
        self.lv0_emb = nn.Embedding(vocab_size, embed_dim)
        
        # Speaker projection
        self.spk_proj = nn.Linear(512, embed_dim)
        
        # Positional encoding (d_model dim, added after input projection)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        # Input: LV0 emb + speaker emb (broadcast across time)
        self.input_proj = nn.Linear(embed_dim * 2, d_model)
        
        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads: predict code for each LV1-7 level
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(7)
        ])
    
    def forward(self, lv0, spk_emb):
        """
        lv0: (B, T) integers
        spk_emb: (B, 512)
        Returns: list of (B, T, vocab_size) logits for each of 7 levels
        """
        B, T = lv0.shape
        
        # Embed LV0
        lv0_e = self.lv0_emb(lv0)  # (B, T, D)
        
        # Speaker embedding broadcast
        spk_e = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        
        # Combine
        h = torch.cat([lv0_e, spk_e], dim=-1)  # (B, T, 2D)
        
        # Project
        h = self.input_proj(h)  # (B, T, d_model)
        
        # Add positional encoding
        h = h + self.pos_emb[:, :T, :]
        
        # Causal mask (can only see past)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h = self.transformer(h, mask=causal_mask)  # (B, T, d_model)
        
        # Predict each LV1-7 level
        logits_list = [head(h) for head in self.output_heads]  # list of (B, T, vocab)
        
        return logits_list
    
    def predict(self, lv0, spk_emb, temperature=0.0):
        """Generate LV1-7 codes."""
        logits_list = self.forward(lv0, spk_emb)
        if temperature == 0.0:
            return torch.stack([logits.argmax(dim=-1) for logits in logits_list], dim=1)  # (B, 7, T)
        else:
            codes = []
            for logits in logits_list:
                probs = F.softmax(logits / temperature, dim=-1)
                codes.append(torch.multinomial(probs.view(-1, self.vocab_size), 1).view(lv0.shape))
            return torch.stack(codes, dim=1)
