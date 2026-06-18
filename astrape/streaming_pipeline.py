from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import torch

from .audio import LogMelState, StreamingLogMel
from .decoder import CausalSynthesisDecoder, SynthesisDecoderState
from .encoder import CausalContentEncoder, ContentEncoderState
from .voicebank import VoiceBank


@dataclass
class PipelineCounters:
    input_samples: int = 0
    mel_frames: int = 0
    content_frames: int = 0
    output_samples: int = 0
    unframed_tail_samples: int = 0


@dataclass(frozen=True)
class StreamChunk:
    audio: torch.Tensor
    input_samples: int
    mel_frames: int
    content_frames: int

    @property
    def output_samples(self) -> int:
        return self.audio.shape[-1]


class OutputRingBuffer:
    """CPU waveform buffer bridging model blocks to audio callbacks."""

    def __init__(self, capacity_samples: int = 44100):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity_samples = capacity_samples
        self._chunks: deque[torch.Tensor] = deque()
        self._offset = 0
        self._lock = Lock()
        self.buffered_samples = 0
        self.underrun_samples = 0
        self.overrun_samples = 0

    def _discard(self, sample_count: int) -> None:
        remaining = sample_count
        while remaining and self._chunks:
            chunk = self._chunks[0]
            available = chunk.numel() - self._offset
            take = min(remaining, available)
            self._offset += take
            self.buffered_samples -= take
            remaining -= take
            if self._offset == chunk.numel():
                self._chunks.popleft()
                self._offset = 0

    def write(self, audio: torch.Tensor) -> None:
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.ndim != 1:
            raise ValueError("Output audio must have shape [samples] or [1, samples]")
        audio = audio.detach().to(device="cpu", dtype=torch.float32)
        if not audio.numel():
            return
        with self._lock:
            if audio.numel() >= self.capacity_samples:
                dropped = self.buffered_samples + audio.numel() - self.capacity_samples
                self._chunks.clear()
                self._offset = 0
                self.buffered_samples = 0
                audio = audio[-self.capacity_samples:]
                self.overrun_samples += dropped
            elif self.buffered_samples + audio.numel() > self.capacity_samples:
                dropped = (
                    self.buffered_samples
                    + audio.numel()
                    - self.capacity_samples
                )
                self._discard(dropped)
                self.overrun_samples += dropped
            self._chunks.append(audio)
            self.buffered_samples += audio.numel()

    def read(self, sample_count: int) -> torch.Tensor:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        output = torch.zeros(sample_count, dtype=torch.float32)
        with self._lock:
            written = 0
            while written < sample_count and self._chunks:
                chunk = self._chunks[0]
                available = chunk.numel() - self._offset
                take = min(sample_count - written, available)
                output[written: written + take] = chunk[
                    self._offset: self._offset + take
                ]
                written += take
                self._offset += take
                self.buffered_samples -= take
                if self._offset == chunk.numel():
                    self._chunks.popleft()
                    self._offset = 0
            self.underrun_samples += sample_count - written
        return output


class StreamingVoiceConverter:
    """Stateful 16kHz PCM to 44.1kHz zero-shot voice conversion."""

    def __init__(
        self,
        encoder: CausalContentEncoder,
        decoder: CausalSynthesisDecoder,
        global_embedding: torch.Tensor,
        *,
        frontend: Optional[StreamingLogMel] = None,
        return_cpu: bool = True,
    ):
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.frontend = frontend or StreamingLogMel()
        self.return_cpu = return_cpu
        self.device = next(encoder.parameters()).device
        decoder_device = next(decoder.parameters()).device
        if decoder_device != self.device:
            raise ValueError("Encoder and decoder must be on the same device")

        mel_rate = self.frontend.sample_rate / self.frontend.hop_length
        expected_content_rate = mel_rate / 2.0
        if self.frontend.n_mels != encoder.config.mel_dim:
            raise ValueError(
                f"Frontend n_mels ({self.frontend.n_mels}) does not match "
                f"encoder mel_dim ({encoder.config.mel_dim})"
            )
        if abs(expected_content_rate - decoder.config.content_rate) > 1e-6:
            raise ValueError(
                f"Frontend mel rate ({mel_rate} Hz) / 2 does not match "
                f"decoder content rate ({decoder.config.content_rate} Hz)"
            )

        if encoder.config.content_dim != decoder.config.content_dim:
            raise ValueError(
                f"Encoder content_dim ({encoder.config.content_dim}) does not "
                f"match decoder content_dim ({decoder.config.content_dim})"
            )

        embedding = global_embedding.detach()
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        expected_shape = (1, decoder.config.condition_dim)
        if embedding.shape != expected_shape:
            raise ValueError(f"Global embedding must have shape {expected_shape}")
        if not torch.isfinite(embedding).all():
            raise ValueError("Global embedding must be finite")
        self.global_embedding = embedding.to(self.device)
        self.reset()

    @property
    def input_sample_rate(self) -> int:
        return self.frontend.sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self.decoder.config.sample_rate

    def reset(self) -> None:
        self.logmel_state: Optional[LogMelState] = None
        self.encoder_state: Optional[ContentEncoderState] = None
        self.decoder_state: Optional[SynthesisDecoderState] = None
        self.counters = PipelineCounters()
        self.finalized = False

    def _empty_audio(self) -> torch.Tensor:
        device = torch.device("cpu") if self.return_cpu else self.device
        return torch.empty(1, 0, device=device)

    def _decode_mel(
        self,
        mel: torch.Tensor,
        *,
        flush: bool,
        input_samples: int,
    ) -> StreamChunk:
        content_output, self.encoder_state = self.encoder.forward_stream(
            mel, self.encoder_state, flush=flush,
        )
        content_frames = content_output.content.shape[-1]
        if content_frames == 0:
            audio = self._empty_audio()
        else:
            # Decode one frame at a time to preserve strict causality
            # (multi-frame chunks lack an intra-chunk causal mask in the decoder)
            content_bt = content_output.content.transpose(1, 2)  # (B, T, D)
            audio_chunks = []
            for t in range(content_frames):
                frame = content_bt[:, t:t+1, :]
                chunk_audio, self.decoder_state = self.decoder.forward_stream(
                    frame, self.global_embedding, self.decoder_state,
                )
                audio_chunks.append(chunk_audio)
            audio = torch.cat(audio_chunks, dim=-1)
            if self.return_cpu:
                audio = audio.cpu()
        self.counters.mel_frames += mel.shape[-1]
        self.counters.content_frames += content_frames
        self.counters.output_samples += audio.shape[-1]
        return StreamChunk(
            audio=audio,
            input_samples=input_samples,
            mel_frames=mel.shape[-1],
            content_frames=content_frames,
        )

    @torch.inference_mode()
    def process(self, waveform: torch.Tensor) -> StreamChunk:
        if self.finalized:
            raise RuntimeError("Pipeline was flushed; call reset before reuse")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("waveform must have shape [samples] or [1, samples]")
        if not torch.isfinite(waveform).all():
            raise ValueError("waveform must contain finite samples")
        input_samples = waveform.shape[-1]
        self.counters.input_samples += input_samples
        if input_samples == 0:
            return StreamChunk(self._empty_audio(), 0, 0, 0)
        mel, self.logmel_state = self.frontend.forward_stream(
            waveform.to(dtype=torch.float32),
            self.logmel_state,
        )
        if mel.shape[-1] == 0:
            return StreamChunk(self._empty_audio(), input_samples, 0, 0)
        return self._decode_mel(
            mel.to(self.device),
            flush=False,
            input_samples=input_samples,
        )

    @torch.inference_mode()
    def flush(self) -> StreamChunk:
        if self.finalized:
            raise RuntimeError("Pipeline has already been flushed")
        overlap = self.frontend.n_fft - self.frontend.hop_length
        if (
            self.logmel_state is not None
            and self.logmel_state.waveform_buffer is not None
        ):
            buffered = self.logmel_state.waveform_buffer.shape[-1]
            self.counters.unframed_tail_samples = max(0, buffered - overlap)
        empty_mel = torch.empty(1, self.frontend.n_mels, 0, device=self.device)
        chunk = self._decode_mel(empty_mel, flush=True, input_samples=0)
        self.finalized = True
        return chunk

    def warmup(self, chunk_samples: int = 80) -> None:
        if self.counters.input_samples:
            raise RuntimeError("Warmup must run before processing user audio")
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        sample_count = self.frontend.n_fft + 2 * self.frontend.hop_length
        waveform = torch.zeros(sample_count)
        for start in range(0, sample_count, chunk_samples):
            self.process(waveform[start: start + chunk_samples])
        self.flush()
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reset()
