"""
Clean architecture: Splitter (disentanglement) + Converter (combination).

Splitter: z_q → content (speaker-independent) + speaker (content-independent)
Converter: c_src + s_tgt → z_q_vc (simple combination)

Kanade-style: multi-layer decoder transformer features for split.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentExtractor(nn.Module):
    """Extract speaker-independent content from decoder transformer features."""
    def __init__(self, dim=512, bottleneck=64):
        super().__init__()
        self.compress = nn.Conv1d(dim, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, dim, 1)

    def forward(self, features):
        """features: (B, T, D) or list of (B, T, D) → content: (B, D, T)"""
        if isinstance(features, list):
            f = torch.stack(features, dim=0).mean(0)  # avg across layers
        else:
            f = features
        f = f.transpose(1, 2)  # (B, D, T)
        h = self.compress(f)
        h = F.gelu(h)
        h = self.expand(h)
        return f + h  # residual, (B, D, T)


class SpeakerExtractor(nn.Module):
    """Extract content-independent speaker embedding."""
    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, dim),
        )

    def forward(self, features):
        """features: (B, T, D) or list → speaker: (B, D)"""
        if isinstance(features, list):
            f = torch.stack(features, dim=0).mean(0)
        else:
            f = features
        f = f.transpose(1, 2)  # (B, D, T)
        return self.net(f)


class Converter(nn.Module):
    """Simple combination: c_src + s_tgt → z_q_vc. No separation logic."""
    def __init__(self, dim=512):
        super().__init__()
        # Speaker modulation (FiLM)
        self.gamma = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.beta = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # Lightweight refinement
        self.refine = nn.Conv1d(dim, dim, 3, padding=1)

    def forward(self, c_src, s_tgt):
        """
        c_src: (B, D, T) content features
        s_tgt: (B, D) target speaker
        → z_q_vc: (B, D, T)
        """
        gamma = self.gamma(s_tgt).unsqueeze(-1)
        beta = self.beta(s_tgt).unsqueeze(-1)

        # Normalize and modulate
        mean = c_src.mean(dim=2, keepdim=True)
        std = c_src.std(dim=2, keepdim=True) + 1e-5
        c_norm = (c_src - mean) / std
        c_mod = c_norm * gamma + beta

        # Residual refinement
        z_q_vc = c_src + self.refine(c_mod)
        return z_q_vc


class DecoderFeatureExtractor(nn.Module):
    """Extract intermediate features from Mimi's decoder transformer."""
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        self.dt = mimi.decoder_transformer
        self.tt = self.dt.transformer

    def extract(self, audio):
        """Get decoder transformer features for an audio."""
        with torch.no_grad():
            z = self.mimi.encode_to_latent(audio, quantize=False)
            codes = self.mimi.quantizer.encode(z)
            z_q = self.mimi.quantizer.decode(codes)
            z_q_up = self.mimi._to_encoder_framerate(z_q)  # (B, D, T_up)

        h = z_q_up.transpose(1, 2)  # (B, T, D)
        if self.dt.input_proj is not None:
            h = self.dt.input_proj(h)

        shallow = []  # layers 0-1 (content-ish)
        deep = []     # layers 5-7 (speaker-ish)

        for i, layer in enumerate(self.tt.layers):
            h = layer(h)
            if i in [0, 1]:
                shallow.append(h.clone())
            if i in [5, 6, 7]:
                deep.append(h.clone())

        # Also get the z_q itself (quantizer space, for reconstruction target)
        return {
            'shallow': shallow,      # list of (B, T, D) — content features
            'deep': deep,            # list of (B, T, D) — speaker features
            'z_q': z_q,              # (B, D, T_z) — quantizer space latent
            'z_q_up': z_q_up,        # (B, D, T_up) — upsampled
        }


class MimiSplitterVC(nn.Module):
    """
    Clean splitter + converter VC.
    
    1. FeatureExtractor: audio → decoder transformer features
    2. ContentExtractor: shallow features → content (speaker-independent)
    3. SpeakerExtractor: deep features → speaker (content-independent)
    4. Converter: c_src + s_tgt → z_q_vc
    5. Decode: z_q_vc → audio (through frozen decoder pipeline)
    """
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        self.feature_ext = DecoderFeatureExtractor(mimi)
        self.content_ext = ContentExtractor(dim=512, bottleneck=64)
        self.speaker_ext = SpeakerExtractor(dim=512, hidden=256)
        self.converter = Converter(dim=512)

    def split(self, audio):
        """Extract content and speaker from audio."""
        feats = self.feature_ext.extract(audio)
        c = self.content_ext(feats['shallow'])  # (B, D, T_up)
        s = self.speaker_ext(feats['deep'])      # (B, D)
        z_q = feats['z_q']                       # (B, D, T_z)
        return c, s, z_q

    def convert(self, src_audio, tgt_audio):
        """Full VC pipeline."""
        with torch.no_grad():
            c_src, _, z_q_src = self.split(src_audio)
            _, s_tgt, _ = self.split(tgt_audio)
            z_q_vc = self.converter(c_src, s_tgt)

            # Decode through frozen pipeline
            z_q_vc_up = self.mimi._to_encoder_framerate(z_q_vc)
            if self.mimi.decoder_transformer is not None:
                (z_q_tr,) = self.mimi.decoder_transformer(z_q_vc_up)
            else:
                z_q_tr = z_q_vc_up
            return self.mimi.decoder(z_q_tr)

    def training_loss(self, src_audio, tgt_audio):
        """
        Train splitter + converter together.
        
        Losses:
        - Content: c_src ≈ c_tgt for same text
        - Speaker: s_src ≠ s_tgt for different speakers
        - Reconstruction: decode(convert(c_src, s_tgt)) ≈ tgt_audio
        """
        c_src, s_src, z_q_src = self.split(src_audio)
        c_tgt, s_tgt, z_q_tgt = self.split(tgt_audio)

        # 1. Content invariance: same text → same content
        T = min(c_src.shape[2], c_tgt.shape[2])
        loss_content = F.mse_loss(c_src[:, :, :T], c_tgt[:, :, :T])

        # 2. Speaker separation: different speakers → different speaker
        cos_spk = F.cosine_similarity(s_src, s_tgt, dim=-1).mean()
        loss_speaker = torch.relu(cos_spk - 0.1) + \
                       (1 - F.cosine_similarity(s_src, -s_tgt.detach(), dim=-1).mean()) * 0.1

        # 3. Reconstruction: convert + decode should match target
        z_q_vc = self.converter(c_src, s_tgt)
        z_q_vc_up = self.mimi._to_encoder_framerate(z_q_vc)
        if self.mimi.decoder_transformer is not None:
            (z_q_tr,) = self.mimi.decoder_transformer(z_q_vc_up)
        else:
            z_q_tr = z_q_vc_up
        audio_vc = self.mimi.decoder(z_q_tr)

        T_audio = min(audio_vc.shape[2], src_audio.shape[2], tgt_audio.shape[2])
        audio_vc = audio_vc[:, :, :T_audio]
        tgt_audio_trim = tgt_audio[:, :, :T_audio]
        loss_recon = F.mse_loss(audio_vc, tgt_audio_trim)

        # 4. Latent consistency: z_q_vc ≈ z_q_tgt
        T_z = min(z_q_vc.shape[2], z_q_tgt.shape[2])
        loss_latent = 0.1 * F.mse_loss(z_q_vc[:, :, :T_z], z_q_tgt[:, :, :T_z])

        total = loss_recon + 0.5 * loss_content + 0.3 * loss_speaker + loss_latent

        return total, {
            'recon': loss_recon.item(),
            'content': loss_content.item(),
            'speaker': loss_speaker.item(),
            'cos_spk': cos_spk.item(),
        }
