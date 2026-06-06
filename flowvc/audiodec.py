"""
AudioDec encoder wrapper for FlowVC.

AudioDec: pretrained causal neural codec, 48kHz, hop=300, 64-dim continuous latent.
No collapse risk — pretrained encoder with proven speech representations.

Replaces F³-Encoder (scratch-trained, collapse-prone).
"""

from __future__ import annotations
import torch, torch.nn as nn

AUDIODEC_SR = 48000
AUDIODEC_HOP = 300
AUDIODEC_DIM = 64
AUDIODEC_HZ = AUDIODEC_SR / AUDIODEC_HOP  # 160 Hz


class AudioDecEncoder(nn.Module):
    """
    Pretrained AudioDec encoder, adapted for FlowVC pipeline.

    Wraps btrvrc0's AudioDecCodec.encode() as a drop-in replacement
    for F³-Encoder.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._codec = None

    def _ensure_codec(self):
        if self._codec is None:
            import sys
            sys.path.insert(0, "/Users/asill/btrvrc0")
            from v3lite.codec_audiodec import AudioDecCodec
            self._codec = AudioDecCodec(device=self.device)
            self._codec.eval()
            for p in self._codec.parameters():
                p.requires_grad = False

    @property
    def sample_rate(self) -> int:
        return AUDIODEC_SR

    @property
    def content_dim(self) -> int:
        return AUDIODEC_DIM

    @property
    def latent_hz(self) -> float:
        return AUDIODEC_HZ

    def forward(self, wav: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Args:
            wav: (B, 1, T) waveform @ 48kHz
        Returns:
            z: (B, T_lat, 64) continuous latent @ 160Hz
        """
        self._ensure_codec()
        B = wav.size(0)
        z_list = []
        for b in range(B):
            wb = wav[b, 0].to(self.device)  # ensure correct device
            z = self._codec.encode(wb)
            z_list.append(z)
        return torch.stack(z_list, dim=0)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Inference encode (noise-free)."""
        return self.forward(wav, training=False)


class AudioDecDecoder(nn.Module):
    """
    Pretrained AudioDec decoder, adapted for FlowVC pipeline.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._codec = None

    def _ensure_codec(self):
        if self._codec is None:
            import sys
            sys.path.insert(0, "/Users/asill/btrvrc0")
            from v3lite.codec_audiodec import AudioDecCodec
            self._codec = AudioDecCodec(device=self.device)
            self._codec.eval()
            for p in self._codec.parameters():
                p.requires_grad = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T_lat, 64) continuous latent @ 160Hz
        Returns:
            wav: (B, 1, T_audio) waveform @ 48kHz
        """
        self._ensure_codec()
        B = z.size(0)
        wav_list = []
        for b in range(B):
            zb = z[b].to(self.device)
            w = self._codec.decode(zb)  # (T_audio,)
            wav_list.append(w)
        return torch.stack(wav_list, dim=0).unsqueeze(1)  # (B, 1, T_audio)


def make_audiodec_encoder(device: str = "cpu") -> AudioDecEncoder:
    return AudioDecEncoder(device=device)


def make_audiodec_decoder(device: str = "cpu") -> AudioDecDecoder:
    return AudioDecDecoder(device=device)
