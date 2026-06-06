"""
FlowVC prosody extractor using FCPE.

FCPE runs at 16kHz CPU, output resampled to 25Hz.
Provides log_f0, voiced flag, and log_energy per frame.
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio

FCPE_SR, TARGET_SR, TARGET_HZ = 16000, 44100, 25


class FCPEProsodyExtractor(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._fcpe_model = None
        self._resampler_16k = None
        self._resampler_25hz = None

    def _ensure_fcpe(self):
        if self._fcpe_model is None:
            try:
                import torchfcpe
                self._fcpe_model = torchfcpe.spawn_bundled_infer_model(device=self.device)
            except ImportError:
                raise ImportError("torchfcpe not installed: pip install git+https://github.com/CNChTu/FCPE.git")

    def _ensure_resamplers(self):
        if self._resampler_16k is None:
            self._resampler_16k = torchaudio.transforms.Resample(TARGET_SR, FCPE_SR).to(self.device)
        if self._resampler_25hz is None:
            self._resampler_25hz = torchaudio.transforms.Resample(100, TARGET_HZ).to(self.device)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        self._ensure_fcpe()
        self._ensure_resamplers()
        wav = wav.to(self.device)

        B = wav.size(0)
        results = []
        for b in range(B):
            wb = wav[b:b+1]  # (1, 1, T)
            w16 = self._resampler_16k(wb)  # (1, 1, T16)

            f0 = self._fcpe_model.infer(w16, sr=FCPE_SR, decoder_mode="local_argmax", threshold=0.006)
            f0 = f0.to(self.device)  # (1, Tf)

            # Resample F0 to 25Hz using nearest-neighbor (avoids sinc ringing on pitch contour)
            f0_np = f0.squeeze(0).cpu().numpy() if f0.is_cuda or f0.device.type == "mps" else f0.squeeze(0).numpy()
            import numpy as np
            old_len = len(f0_np)
            new_len = max(1, int(old_len * TARGET_HZ / 100))
            indices = np.linspace(0, old_len - 1, new_len).round().astype(int)
            f0_25 = torch.from_numpy(f0_np[indices]).to(self.device)
            if f0_25.dim() == 0:
                f0_25 = f0_25.unsqueeze(0)
            elif f0_25.dim() == 2:
                f0_25 = f0_25.squeeze(0)
            T25 = f0_25.shape[0]

            # Voiced
            voiced = (f0_25 > 1.0).float()

            # Log F0
            log_f0 = torch.where(f0_25 > 1.0, torch.log(f0_25 + 1e-8), torch.zeros_like(f0_25))

            # Energy: RMS per 25Hz frame
            audio_1d = wb.reshape(-1)  # always 1D, robust to any input shape
            hop = max(1, audio_1d.shape[0] // max(T25, 1))
            if T25 > 1:
                chunks = audio_1d.unfold(0, hop, hop)[:T25]
                rms = chunks.pow(2).mean(dim=-1).sqrt()
                log_energy = torch.log(rms + 1e-8)
            else:
                rms = audio_1d.pow(2).mean().sqrt()
                log_energy = torch.log(rms + 1e-8).expand(T25)

            feat = torch.stack([log_f0, voiced, log_energy], dim=-1)  # (T25, 3)
            results.append(feat)

        return torch.stack(results, dim=0)  # (B, T_lat, 3)


def make_prosody_extractor(device: str = "cpu") -> FCPEProsodyExtractor:
    return FCPEProsodyExtractor(device=device)
