from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio


@dataclass
class LogMelState:
    waveform_buffer: Optional[torch.Tensor] = None


class StreamingLogMel:
    """Exact center=False log-mel extraction for full or chunked 16 kHz audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 320,
        n_mels: int = 80,
        f_min: float = 80.0,
        f_max: float = 7600.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self._kernel_cache: dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _kernels(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (waveform.device, waveform.dtype)
        cached = self._kernel_cache.get(key)
        if cached is None:
            window = torch.hann_window(
                self.n_fft, device=waveform.device, dtype=waveform.dtype
            )
            filter_bank = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
            ).to(device=waveform.device, dtype=waveform.dtype)
            cached = (window, filter_bank)
            self._kernel_cache[key] = cached
        return cached

    def _extract(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] < self.n_fft:
            return waveform.new_empty(waveform.shape[0], self.n_mels, 0)
        window, filter_bank = self._kernels(waveform)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=False,
            return_complex=True,
        )
        power = spectrum.abs().square().transpose(1, 2)
        mel = torch.matmul(power, filter_bank).transpose(1, 2)
        return torch.log(mel.clamp_min(1e-5))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                "waveform must have shape (samples,) or (batch, samples)"
            )
        return self._extract(waveform)

    def forward_stream(
        self, waveform: torch.Tensor, state: Optional[LogMelState] = None
    ) -> tuple[torch.Tensor, LogMelState]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                "waveform must have shape (samples,) or (batch, samples)"
            )
        state = state or LogMelState()
        if state.waveform_buffer is None:
            joined = waveform
        else:
            if state.waveform_buffer.shape[0] != waveform.shape[0]:
                raise ValueError("Streaming batch size changed")
            joined = torch.cat((state.waveform_buffer, waveform), dim=-1)
        mel = self._extract(joined)
        consumed = mel.shape[-1] * self.hop_length
        state.waveform_buffer = joined[:, consumed:]
        return mel, state
