"""
Transformer-based acoustic adapter for Mimi Splitter VC.
Replaces FiLM with multi-head self-attention + speaker conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class SpeakerFiLM(nn.Module):
    """FiLM conditioning inside transformer block."""
    def __init__(self, dim, spk_dim):
        super().__init__()
        self.scale = nn.Linear(spk_dim, dim)
        self.bias = nn.Linear(spk_dim, dim)

    def forward(self, x, spk):
        # x: (B, T, D), spk: (B, spk_dim)
        s = self.scale(spk).unsqueeze(1)   # (B, 1, D)
        b = self.bias(spk).unsqueeze(1)    # (B, 1, D)
        return x * (1 + torch.tanh(s)) + b


class TransformerAdapterBlock(nn.Module):
    """Self-attention + FFN with speaker FiLM conditioning."""
    def __init__(self, dim=512, spk_dim=256, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.film1 = SpeakerFiLM(dim, spk_dim)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.film2 = SpeakerFiLM(dim, spk_dim)

    def forward(self, x, spk):
        # x: (B, T, D), spk: (B, spk_dim)
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, need_weights=False)[0]
        x = self.film1(x, spk)
        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.film2(x, spk)
        x = residual + x
        return x


class TransformerAcousticAdapter(nn.Module):
    """
    Replaces FiLM-based AcousticAdapter.
    
    Architecture:
      z_acoustic (q1-q7) → project to dim
      C_src     (content) → project to dim
      S_tgt     (speaker) → spk_dim
      
      Input = z_acoustic + C_src + pos_encoding
      → N × TransformerAdapterBlock(spk_cond)
      → output projection → acoustic residual
    """
    def __init__(self, dim=512, spk_dim=256, num_layers=4, num_heads=8, 
                 ff_mult=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.spk_dim = spk_dim
        
        # Input projections
        self.proj_acoustic = nn.Conv1d(dim, dim, 1)
        self.proj_content = nn.Conv1d(dim, dim, 1)
        self.spk_proj = nn.Linear(dim, spk_dim)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerAdapterBlock(dim, spk_dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Conv1d(dim, dim, 1)

    def forward(self, z_acoustic, spk_emb, z_content):
        """
        z_acoustic: (B, D, T) — acoustic codebooks (q1-q7)
        spk_emb:    (B, D)     — target speaker embedding
        z_content:  (B, D, T) — content latent (q0)
        
        Returns: acoustic residual (B, D, T)
        """
        # Project to transformer dim
        a = self.proj_acoustic(z_acoustic)   # (B, D, T)
        c = self.proj_content(z_content)     # (B, D, T)
        
        # Project speaker
        spk = self.spk_proj(spk_emb)  # (B, spk_dim)
        
        # Combine: acoustic + content as input
        x = (a + c).transpose(1, 2)   # (B, T, D)
        x = self.pos_enc(x)
        
        # Transformer blocks with speaker conditioning
        for block in self.blocks:
            x = block(x, spk)
        
        # Output
        x = self.norm_out(x)
        x = x.transpose(1, 2)          # (B, D, T)
        out = self.proj_out(x)         # (B, D, T)
        
        return out


# ── Replace in MimiSplitterV2 ─────────────────────────────────────────────
def create_transformer_splitter(mimi, dim=512, spk_dim=256, num_layers=4):
    """Create a MimiSplitterV2 with Transformer adapter."""
    import sys
    sys.path.insert(0, '/Users/asill/btrv5')
    from mimi_splitter_v2 import MimiSplitterV2, ContentExtractor, SpeakerEncoder
    
    splitter = MimiSplitterV2.__new__(MimiSplitterV2)
    nn.Module.__init__(splitter)
    
    splitter.mimi = mimi
    splitter.content_extractor = ContentExtractor(dim, bottleneck=64)
    splitter.speaker_encoder = SpeakerEncoder(dim, spk_dim)
    splitter.acoustic_adapter = TransformerAcousticAdapter(
        dim=dim, spk_dim=spk_dim, num_layers=num_layers,
        num_heads=8, ff_mult=4, dropout=0.1
    )
    
    return splitter


if __name__ == "__main__":
    # Quick test
    import sys, os
    sys.path.insert(0, '/Users/asill/btrv5')
    from mimi_splitter_v2 import load_mimi, mimi_encode
    
    device = torch.device('cpu')
    mimi = load_mimi(device).to(device)
    splitter = create_transformer_splitter(mimi, num_layers=2)
    
    print(f"Transformer splitter params: {sum(p.numel() for p in splitter.parameters() if p.requires_grad)}")
    
    # Test forward
    import soundfile as sf, numpy as np
    from scipy import signal
    
    d, sr = sf.read('/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p225/p225_001_mic1.flac')
    d = d[:48000]
    if sr != 24000:
        d = signal.resample(d, int(len(d)*24000/sr))
    x = torch.from_numpy(d).float().view(1, 1, -1).to(device)
    
    with torch.no_grad():
        z, codes = mimi_encode(x, mimi)
        z_vc, C, S, A = splitter(z, codes)
    
    print(f"z_vc: {z_vc.shape}, C: {C.shape}, S: {S.shape}, A: {A.shape}")
    print("Transformer adapter test OK")
