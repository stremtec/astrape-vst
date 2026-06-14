from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


VOICEBANK_FORMAT_VERSION = 2
MIN_REFERENCE_SECONDS = 5.0
MIO_GLOBAL_MODEL = "Aratako/MioCodec-25Hz-44.1kHz-v2"


@dataclass(frozen=True)
class ReferenceQuality:
    peak_amplitude: float
    rms_dbfs: float
    clipping_fraction: float
    active_speech_ratio: float
    dc_offset: float
    warnings: tuple[str, ...]


def analyze_reference(
    audio: np.ndarray,
    sample_rate: int,
) -> ReferenceQuality:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("VoiceBank reference is empty")
    if sample_rate <= 0:
        raise ValueError("VoiceBank source sample rate must be positive")
    if not np.isfinite(audio).all():
        raise ValueError("VoiceBank reference must contain finite samples")

    absolute = np.abs(audio)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    clipping_fraction = float(np.mean(absolute >= 0.999))
    dc_offset = float(np.mean(audio, dtype=np.float64))

    frame_length = max(1, round(sample_rate * 0.02))
    hop_length = max(1, round(sample_rate * 0.01))
    if audio.size < frame_length:
        frame_rms = np.asarray([rms], dtype=np.float64)
    else:
        starts = np.arange(0, audio.size - frame_length + 1, hop_length)
        frame_rms = np.asarray(
            [
                np.sqrt(
                    np.mean(
                        np.square(audio[start : start + frame_length]),
                        dtype=np.float64,
                    )
                )
                for start in starts
            ]
        )
    active_speech_ratio = float(np.mean(frame_rms >= 10 ** (-50.0 / 20.0)))

    warnings = []
    if clipping_fraction > 1e-4:
        warnings.append("clipping_detected")
    if rms_dbfs < -35.0:
        warnings.append("reference_too_quiet")
    if rms_dbfs > -8.0:
        warnings.append("reference_too_loud")
    if active_speech_ratio < 0.6:
        warnings.append("low_active_speech_ratio")
    if abs(dc_offset) > 0.01:
        warnings.append("dc_offset_detected")
    return ReferenceQuality(
        peak_amplitude=peak,
        rms_dbfs=rms_dbfs,
        clipping_fraction=clipping_fraction,
        active_speech_ratio=active_speech_ratio,
        dc_offset=dc_offset,
        warnings=tuple(warnings),
    )


def _optional_scalar(
    data: np.lib.npyio.NpzFile,
    key: str,
    default: float,
) -> float:
    return float(data[key]) if key in data else default


def _optional_string(
    data: np.lib.npyio.NpzFile,
    key: str,
    default: str,
) -> str:
    return str(data[key].item()) if key in data else default


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=np.str_)


@dataclass(frozen=True)
class VoiceBank:
    global_embedding: torch.Tensor
    duration_seconds: float
    source_sample_rate: int
    source_path: str
    embedding_model: str = MIO_GLOBAL_MODEL
    reference_sha256: str = ""
    created_utc: str = ""
    peak_amplitude: float = float("nan")
    rms_dbfs: float = float("nan")
    clipping_fraction: float = float("nan")
    active_speech_ratio: float = float("nan")
    dc_offset: float = float("nan")
    quality_warnings: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.global_embedding.shape != (128,):
            raise ValueError("VoiceBank global embedding must have shape [128]")
        if not torch.isfinite(self.global_embedding).all():
            raise ValueError("VoiceBank global embedding must be finite")
        if not np.isfinite(self.duration_seconds):
            raise ValueError("VoiceBank duration must be finite")
        if self.duration_seconds < MIN_REFERENCE_SECONDS:
            raise ValueError(
                f"VoiceBank reference must be at least "
                f"{MIN_REFERENCE_SECONDS:.1f} seconds"
            )
        if self.source_sample_rate <= 0:
            raise ValueError("VoiceBank source sample rate must be positive")
        if not self.embedding_model:
            raise ValueError("VoiceBank embedding model must be recorded")
        for name, value in (
            ("peak amplitude", self.peak_amplitude),
            ("RMS dBFS", self.rms_dbfs),
            ("clipping fraction", self.clipping_fraction),
            ("active speech ratio", self.active_speech_ratio),
            ("DC offset", self.dc_offset),
        ):
            if not np.isnan(value) and not np.isfinite(value):
                raise ValueError(f"VoiceBank {name} must be finite")
        for name, value in (
            ("clipping fraction", self.clipping_fraction),
            ("active speech ratio", self.active_speech_ratio),
        ):
            if not np.isnan(value) and not 0.0 <= value <= 1.0:
                raise ValueError(f"VoiceBank {name} must be in [0, 1]")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            format_version=np.asarray(
                VOICEBANK_FORMAT_VERSION,
                dtype=np.int64,
            ),
            global_embedding=self.global_embedding.detach().cpu().numpy(),
            duration_seconds=np.asarray(self.duration_seconds, dtype=np.float32),
            source_sample_rate=np.asarray(
                self.source_sample_rate,
                dtype=np.int64,
            ),
            source_path=np.asarray(self.source_path),
            embedding_model=np.asarray(self.embedding_model),
            reference_sha256=np.asarray(self.reference_sha256),
            created_utc=np.asarray(self.created_utc),
            peak_amplitude=np.asarray(self.peak_amplitude, dtype=np.float32),
            rms_dbfs=np.asarray(self.rms_dbfs, dtype=np.float32),
            clipping_fraction=np.asarray(
                self.clipping_fraction,
                dtype=np.float32,
            ),
            active_speech_ratio=np.asarray(
                self.active_speech_ratio,
                dtype=np.float32,
            ),
            dc_offset=np.asarray(self.dc_offset, dtype=np.float32),
            quality_warnings=_string_array(self.quality_warnings),
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> VoiceBank:
        with np.load(path) as data:
            version = int(data["format_version"])
            if version not in (1, VOICEBANK_FORMAT_VERSION):
                raise ValueError(f"Unsupported VoiceBank version: {version}")
            bank = cls(
                global_embedding=torch.from_numpy(
                    data["global_embedding"]
                ).float(),
                duration_seconds=float(data["duration_seconds"]),
                source_sample_rate=int(data["source_sample_rate"]),
                source_path=str(data["source_path"].item()),
                embedding_model=_optional_string(
                    data,
                    "embedding_model",
                    MIO_GLOBAL_MODEL,
                ),
                reference_sha256=_optional_string(
                    data,
                    "reference_sha256",
                    "",
                ),
                created_utc=_optional_string(data, "created_utc", ""),
                peak_amplitude=_optional_scalar(
                    data,
                    "peak_amplitude",
                    float("nan"),
                ),
                rms_dbfs=_optional_scalar(data, "rms_dbfs", float("nan")),
                clipping_fraction=_optional_scalar(
                    data,
                    "clipping_fraction",
                    float("nan"),
                ),
                active_speech_ratio=_optional_scalar(
                    data,
                    "active_speech_ratio",
                    float("nan"),
                ),
                dc_offset=_optional_scalar(
                    data,
                    "dc_offset",
                    float("nan"),
                ),
                quality_warnings=(
                    tuple(map(str, data["quality_warnings"].tolist()))
                    if "quality_warnings" in data
                    else ()
                ),
            )
        bank.validate()
        return bank
