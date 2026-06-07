"""
AudioDec VC System: splitter + converter with continuous latent.
Key advantage: NO quantizer — gradient flows through decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentExtractor(nn.Module):
    """Extract speaker-neutralized content from AudioDec latent (64-dim)."""
    def __init__(self, dim=64, bottleneck=16):
        super().__init__()
        self.compress = nn.Conv1d(dim, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, dim, 1)
        self.norm = nn.InstanceNorm1d(dim)

    def forward(self, z):
        """z: (B, T, D) or (T, D) → content: same shape"""
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        z_t = z.transpose(1, 2)  # (B, D, T)
        h = self.compress(z_t)
        h = F.gelu(h)
        h = self.expand(h)
        out = self.norm(z_t + h)  # residual
        out = out.transpose(1, 2)  # (B, T, D)
        if squeeze:
            out = out.squeeze(0)
        return out


class SpeakerExtractor(nn.Module):
    """Extract global speaker embedding from AudioDec latent."""
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z):
        """z: (B, T, D) or (T, D) → speaker: (D,) or (B, D)"""
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        z_t = z.transpose(1, 2)  # (B, D, T)
        s = self.net(z_t)  # (B, D)
        if squeeze:
            s = s.squeeze(0)
        return s


class LatentConverter(nn.Module):
    """
    Convert source content + target speaker → VC latent.
    Simple concatenation + MLP approach.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.spk_proj = nn.Linear(dim, dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim * 2, 128, 3, padding=1), nn.GELU(),
            nn.Conv1d(128, dim, 3, padding=1),
        )

    def forward(self, c_src, s_tgt):
        """
        c_src: (B, T, D) content from source
        s_tgt: (B, D) speaker from target
        → z_vc: (B, T, D)
        """
        if c_src.dim() == 2:
            c_src = c_src.unsqueeze(0)
            s_tgt = s_tgt.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, T, D = c_src.shape
        s_exp = s_tgt.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        h = torch.cat([c_src, s_exp], dim=-1)  # (B, T, 2D)
        h = h.transpose(1, 2)  # (B, 2D, T)
        z_vc = c_src + self.conv(h).transpose(1, 2)  # residual, both (B, T, D)
        # z_vc: (B, T, D)
        
        if squeeze:
            z_vc = z_vc.squeeze(0)
        return z_vc


class AudioDecVC(nn.Module):
    """Full AudioDec VC pipeline: splitter + converter + decoder."""
    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        self.content_ext = ContentExtractor(dim=64, bottleneck=16)
        self.speaker_ext = SpeakerExtractor(dim=64, hidden=128)
        self.converter = LatentConverter(dim=64)

    def forward(self, z_src, z_tgt):
        """Training: z_src, z_tgt from parallel utterances."""
        c_src = self.content_ext(z_src)
        s_tgt = self.speaker_ext(z_tgt)
        z_vc = self.converter(c_src, s_tgt)
        return z_vc

    def convert(self, src_audio, tgt_audio):
        """Inference: audio → audio."""
        with torch.no_grad():
            z_src = self.codec.encode(src_audio)
            z_tgt = self.codec.encode(tgt_audio)
            z_vc = self.forward(z_src.unsqueeze(0), z_tgt.unsqueeze(0))
            return self.codec.decode(z_vc.squeeze(0))

    def training_loss(self, src_audio, tgt_audio):
        """Loss: decode(z_vc) should match target audio."""
        z_src = self.codec.encode(src_audio).unsqueeze(0)  # (1, T, 64)
        z_tgt = self.codec.encode(tgt_audio).unsqueeze(0)
        T = min(z_src.shape[1], z_tgt.shape[1])
        z_src, z_tgt = z_src[:, :T], z_tgt[:, :T]
        
        z_vc = self.forward(z_src, z_tgt)
        audio_vc = self.codec.decode(z_vc.squeeze(0))
        
        # Trim to same length as target
        T_audio = min(len(audio_vc), len(src_audio), len(tgt_audio))
        audio_vc = audio_vc[:T_audio]
        tgt_audio_trim = tgt_audio[:T_audio]
        
        loss = F.mse_loss(audio_vc.unsqueeze(0), tgt_audio_trim.unsqueeze(0))
        return loss
