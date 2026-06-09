"""
AudioDec StreamVC: HuBERT content (0% speaker) + AudioDec differentiable decoder.

HuBERT layers 1-3 avg → ContentProjector → AudioDec latent (64-dim)
→ AudioDec decoder (differentiable!) → audio
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from transformers import HubertModel


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class SpeakerAdversarial(nn.Module):
    def __init__(self, dim=768, n_speakers=109, alpha=3.0):
        super().__init__()
        self.alpha = alpha
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, n_speakers),
        )
    def forward(self, x):
        x_rev = GradientReversal.apply(x, self.alpha)
        return self.classifier(x_rev)


class ContentProjector(nn.Module):
    """HuBERT 768-dim @ 50Hz → AudioDec 64-dim @ 50Hz (same rate!)."""
    def __init__(self, in_dim=768, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 384, 5, padding=2), nn.GELU(),
            nn.Conv1d(384, 192, 5, padding=2), nn.GELU(),
            nn.Conv1d(192, out_dim, 5, padding=2),
        )
        self.output_scale = nn.Parameter(torch.ones(1) * 40.0)  # Match real latent std≈1.6
        self.output_bias = nn.Parameter(torch.zeros(1))        # Match real latent mean≈0.02
    def forward(self, hubert_feat):
        """hubert_feat: (B, T, 768) → (B, T, 64)"""
        x = self.net(hubert_feat.transpose(1, 2)).transpose(1, 2)
        return x * self.output_scale + self.output_bias


class AudioDecStreamVC(nn.Module):
    """
    HuBERT content → AudioDec vocoder.
    No Mimi dependency — uses AudioDec differentiable decoder directly.
    """
    def __init__(self, hubert, audiodec_codec, n_speakers=109, spk_dim=256):
        super().__init__()
        self.hubert = hubert
        self.codec = audiodec_codec
        for p in hubert.parameters(): p.requires_grad_(False)
        for p in audiodec_codec.parameters(): p.requires_grad_(False)
        
        self.content_proj = ContentProjector(in_dim=768, out_dim=64)
        self.spk_adversarial = SpeakerAdversarial(dim=768, n_speakers=n_speakers, alpha=3.0)
        
        # Speaker conditioning on projected content
        self.spk_gamma = nn.Linear(spk_dim, 64)
        self.spk_beta = nn.Linear(spk_dim, 64)
    
    def forward(self, src_audio_16k, tgt_spk_emb):
        """
        Returns: z_vc (B, T, 64) AudioDec-compatible latent + audio (B, T_samples)
        """
        # HuBERT layers 1-3 average (0% speaker!)
        with torch.no_grad():
            hs = self.hubert(src_audio_16k, output_hidden_states=True).hidden_states
            h_avg = (hs[1] + hs[2] + hs[3]) / 3.0  # (B, T_h, 768)
        
        # Adversarial
        _ = self.spk_adversarial(h_avg.transpose(1, 2))
        
        # Project to AudioDec latent
        z_content = self.content_proj(h_avg)  # (B, T_h, 64)
        
        # Speaker injection
        gamma = self.spk_gamma(tgt_spk_emb).unsqueeze(1)
        beta = self.spk_beta(tgt_spk_emb).unsqueeze(1)
        mean = z_content.mean(dim=1, keepdim=True)
        std = z_content.std(dim=1, keepdim=True) + 1e-5
        z_vc = (z_content - mean) / std * gamma + beta
        
        return z_vc
    
    @torch.no_grad()
    def convert(self, src_audio_16k, tgt_spk_emb):
        z_vc = self.forward(src_audio_16k, tgt_spk_emb)
        z_vc_2d = z_vc.squeeze(0)  # (T, 64)
        return self.codec.decode(z_vc_2d)  # differentiable!
