"""
Mimi Latent Converter: continuous latent space VC.
z_src + s_tgt → z_vc → quantize → decode.
Trained with discrete code supervision (LV0=content, LV1-7=speaker).
"""

import torch, torch.nn as nn, torch.nn.functional as F


class LatentConverter(nn.Module):
    """
    Converter that maps source continuous latent + target speaker embedding
    to a target-speaker continuous latent.
    """
    def __init__(self, latent_dim=512, speaker_dim=512, hidden_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Content extraction: z → c (remove speaker)
        self.content_conv = nn.Conv1d(latent_dim, latent_dim, 3, padding=1)
        self.content_norm = nn.LayerNorm(latent_dim)
        
        # Speaker injection: concatenate c + s_target, project
        self.injector = nn.Sequential(
            nn.Conv1d(latent_dim * 2, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=min(hidden_dim, 32)),
            nn.GELU(),
            nn.Conv1d(hidden_dim, latent_dim, 1),
        )
        
    def forward(self, z_src, s_tgt):
        """
        z_src: (B, D, T) continuous latent from source
        s_tgt: (B, D) target speaker embedding
        Returns z_vc: (B, D, T)
        """
        # Extract content (attempt to neutralize speaker)
        c = self.content_conv(z_src)
        c = c.transpose(1, 2)
        c = self.content_norm(c)
        c = c.transpose(1, 2)  # (B, D, T)
        
        # Inject target speaker
        s_expanded = s_tgt.unsqueeze(-1).expand(-1, -1, z_src.shape[2])
        h = torch.cat([c, s_expanded], dim=1)  # (B, 2D, T)
        z_vc = z_src + self.injector(h)  # residual connection
        
        return z_vc


class MimiVCTrainer:
    """Trains LatentConverter with discrete code supervision."""
    
    def __init__(self, mimi):
        self.mimi = mimi
        self.converter = LatentConverter()
        # Freeze Mimi
        for p in mimi.parameters(): p.requires_grad_(False)
        
    def speaker_embed(self, z):
        """Simple mean-pool speaker embedding."""
        return z.mean(dim=2)  # (B, D)
        
    def compute_loss(self, z_src, z_tgt):
        """
        Train converter to produce z_vc that:
        - Has source content (LV0 codes match source)
        - Has target speaker (LV1-7 codes match target)
        """
        s_tgt = self.speaker_embed(z_tgt)
        z_vc = self.converter(z_src, s_tgt)
        
        # Quantize to get codes
        codes_src = self.mimi.quantizer.encode(z_src)  # (B, 8, T)
        codes_tgt = self.mimi.quantizer.encode(z_tgt)
        codes_vc = self.mimi.quantizer.encode(z_vc)    # (B, 8, T)
        
        T = min(codes_src.shape[2], codes_tgt.shape[2], codes_vc.shape[2])
        codes_src = codes_src[:, :, :T]
        codes_tgt = codes_tgt[:, :, :T]
        codes_vc = codes_vc[:, :, :T]
        
        # Content loss: LV0 from VC should match LV0 from source
        content_match = (codes_vc[:, 0, :] == codes_src[:, 0, :]).float().mean()
        content_loss = 1.0 - content_match
        
        # Speaker loss: LV1-7 from VC should match LV1-7 from target
        speaker_match = (codes_vc[:, 1:, :] == codes_tgt[:, 1:, :]).float().mean()
        speaker_loss = 1.0 - speaker_match
        
        # Quantization consistency: z_vc should quantize to similar codes
        # (this helps gradient flow)
        z_vc_recon = self.mimi.quantizer.decode(codes_vc)
        quant_loss = F.mse_loss(z_vc, z_vc_recon)
        
        loss = content_loss * 2.0 + speaker_loss * 1.0 + quant_loss * 0.1
        
        return loss, {
            'content_match': content_match.item(),
            'speaker_match': speaker_match.item(),
            'quant_loss': quant_loss.item(),
        }
    
    def convert(self, src_audio, tgt_audio):
        """Full VC pipeline: source audio + target audio → VC audio."""
        with torch.no_grad():
            z_src = self.mimi.encode_to_latent(src_audio, quantize=False)
            z_tgt = self.mimi.encode_to_latent(tgt_audio, quantize=False)
            s_tgt = self.speaker_embed(z_tgt)
            z_vc = self.converter(z_src, s_tgt)
            codes_vc = self.mimi.quantizer.encode(z_vc)
            return self.mimi.decode(codes_vc)
