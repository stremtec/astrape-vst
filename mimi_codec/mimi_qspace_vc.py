"""
Mimi Quantizer-Space VC Converter.
Operates on z_q (quantizer output space) — compatible with decoder.
Training: audio reconstruction loss through frozen decoder pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerExtractor(nn.Module):
    """Extract global speaker embedding from quantizer-space latent."""
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, 256, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, dim),
        )
    def forward(self, z_q):
        """z_q: (B, D, T) → (B, D)"""
        return self.net(z_q)


class QSpaceConverter(nn.Module):
    """
    Converts z_q_src + s_tgt → z_q_vc in quantizer space.
    Uses FiLM (feature-wise linear modulation) for speaker conditioning.
    """
    def __init__(self, dim=512):
        super().__init__()
        # FiLM: predict gamma and beta from speaker embedding
        self.gamma_net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        # Refinement convs
        self.refine = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.Conv1d(dim, dim, 5, padding=2),
        )

    def forward(self, z_q_src, s_tgt):
        """
        z_q_src: (B, D, T) quantizer-space source latent
        s_tgt: (B, D) target speaker embedding
        → z_q_vc: (B, D, T)
        """
        gamma = self.gamma_net(s_tgt).unsqueeze(-1)  # (B, D, 1)
        beta = self.beta_net(s_tgt).unsqueeze(-1)    # (B, D, 1)

        # Normalize source (instance norm style)
        mean = z_q_src.mean(dim=2, keepdim=True)
        std = z_q_src.std(dim=2, keepdim=True) + 1e-5
        z_norm = (z_q_src - mean) / std

        # Apply target speaker modulation
        z_mod = z_norm * gamma + beta

        # Refine with residual
        z_q_vc = z_q_src + self.refine(z_mod)
        return z_q_vc


class MimiQSpaceVC(nn.Module):
    """
    Full Mimi VC in quantizer space.
    z_q_src + tgt_audio → decoder → audio_vc
    """
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        self.speaker_ext = SpeakerExtractor(dim=512)
        self.converter = QSpaceConverter(dim=512)

    def get_z_q(self, audio):
        """Encode audio → quantized latent in quantizer space."""
        with torch.no_grad():
            z = self.mimi.encode_to_latent(audio, quantize=False)
            codes = self.mimi.quantizer.encode(z)
            return self.mimi.quantizer.decode(codes)

    def encode_speaker(self, audio):
        """Extract speaker embedding from audio."""
        z_q = self.get_z_q(audio)
        return self.speaker_ext(z_q)

    def forward(self, z_q_src, s_tgt):
        """Convert: z_q_src + s_tgt → z_q_vc."""
        return self.converter(z_q_src, s_tgt)

    def decode_to_audio(self, z_q):
        """Decode quantizer-space latent to audio through full Mimi decoder."""
        with torch.no_grad():
            z_q_up = self.mimi._to_encoder_framerate(z_q)
            if self.mimi.decoder_transformer is not None:
                (z_q_tr,) = self.mimi.decoder_transformer(z_q_up)
            else:
                z_q_tr = z_q_up
            return self.mimi.decoder(z_q_tr)

    def convert(self, src_audio, tgt_audio):
        """Full VC: audio → audio."""
        with torch.no_grad():
            z_q_src = self.get_z_q(src_audio)
            s_tgt = self.encode_speaker(tgt_audio)
            z_q_vc = self.forward(z_q_src, s_tgt)
            return self.decode_to_audio(z_q_vc)

    def training_loss(self, src_audio, tgt_audio):
        """
        Train on parallel utterances.
        z_q_vc = converter(z_q_src, s_tgt)
        audio_vc = decode(z_q_vc)
        Loss: MSE(audio_vc, tgt_audio) + regularization
        """
        z_q_src = self.get_z_q(src_audio)
        z_q_tgt = self.get_z_q(tgt_audio)
        s_tgt = self.speaker_ext(z_q_tgt)

        z_q_vc = self.forward(z_q_src, s_tgt)

        # Decode through FROZEN decoder (gradient flows through here!)
        z_q_vc_up = self.mimi._to_encoder_framerate(z_q_vc)
        if self.mimi.decoder_transformer is not None:
            (z_q_tr,) = self.mimi.decoder_transformer(z_q_vc_up)
        else:
            z_q_tr = z_q_vc_up
        audio_vc = self.mimi.decoder(z_q_tr)

        # Trim to target length
        T_audio = min(audio_vc.shape[2], src_audio.shape[2], tgt_audio.shape[2])
        audio_vc = audio_vc[:, :, :T_audio]
        tgt_audio_trim = tgt_audio[:, :, :T_audio]

        # MSE on audio
        loss_audio = F.mse_loss(audio_vc, tgt_audio_trim)

        # Latent-level regularization: z_q_vc should be close to z_q_tgt
        T_lat = z_q_vc.shape[2]
        loss_latent = F.mse_loss(z_q_vc[:, :, :T_lat], z_q_tgt[:, :, :T_lat])

        # Speaker consistency: speaker(z_q_vc) should match s_tgt
        s_vc = self.speaker_ext(z_q_vc)
        loss_spk = F.mse_loss(s_vc, s_tgt)

        total = loss_audio + 0.1 * loss_latent + 0.05 * loss_spk
        return total
