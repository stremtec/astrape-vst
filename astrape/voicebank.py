"""Astrape voice bank file format.

The legacy `.npz` (versions 1 and 2) layout is still loadable, so existing
profiles do not break. New profiles default to the `.astrape` (version 3)
container, which is a small fixed-byte header followed by a raw float32 LE
embedding and a JSON metadata blob.

The `.astrape` layout trades zlib compression for random-access. The header
is exactly 48 bytes, so a length-bounded prefix read returns enough
information to validate the file, the embedding tensor's offset/length, and
where the JSON metadata block lives. The streaming pipeline reads only the
embedding during conversion startup; the JSON metadata is consumed lazily
for display, quality checks, and build diagnostics.

    HEADER  (48 bytes, fixed, little-endian)
        magic                  4s   "ASR1"
        version                I    = 3
        flags                  I    bit0 = has_embedding (always set)
        reserved               I    = 0
        embedding_offset       Q
        embedding_length       I    = 512 (no fallback)
        metadata_offset        Q
        metadata_length        I
        reserved2              Q    = 0

    EMBEDDING  (512 bytes, raw f32, 128 dimensions)
    METADATA   (UTF-8 JSON, dict of strings/numbers)

The format is fully **lossless**: the embedding is preserved at f32
precision, and the metadata is encoded as a JSON document with no rounding
or lossy compression. Round-trip equality is bit-exact.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


# Public constants ----------------------------------------------------------------

VOICEBANK_FORMAT_VERSION = 3                 # current write format
VOICEBANK_LEGACY_NPZ_FORMAT_VERSION = 2      # last .npz version that is round-trip loadable
VOICEBANK_LEGACY_NPZ_V1_FORMAT_VERSION = 1   # legacy v1, only loadable, no metadata parity
VOICEBANK_EXTENSION = ".astrape"
VOICEBANK_LEGACY_EXTENSION = ".npz"
ASTRAPE_MAGIC = b"ASR1"
ASTRAPE_HEADER_SIZE = 48
ASTRAPE_HEADER_FMT = "<4sIIIQIQIQ"
ASTRAPE_EMBEDDING_DIM = 128
ASTRAPE_EMBEDDING_BYTES = ASTRAPE_EMBEDDING_DIM * 4
ASTRAPE_EMBEDDING_DTYPE = np.dtype("<f4")
ASTRAPE_FLAG_HAS_EMBEDDING = 1 << 0

MIN_REFERENCE_SECONDS = 5.0
MIO_GLOBAL_MODEL = "Aratako/MioCodec-25Hz-44.1kHz-v2"

_UNSET = object()
_EMBEDDING_KEY = "global_embedding"
_NAN = float("nan")


# Reference quality -----------------------------------------------------------------

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


# Helpers --------------------------------------------------------------------------

def _optional_scalar(
    data: Mapping[str, Any] | np.lib.npyio.NpzFile,
    key: str,
    default: float,
) -> float:
    if key in data:
        return float(data[key])  # type: ignore[index]
    return default


def _optional_string(
    data: Mapping[str, Any] | np.lib.npyio.NpzFile,
    key: str,
    default: str,
) -> str:
    if key in data:
        if hasattr(data[key], "item"):
            return str(data[key].item())
        return str(data[key])
    return default


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=np.str_)


# VoiceBank dataclass --------------------------------------------------------------

@dataclass(frozen=True)
class VoiceBank:
    global_embedding: torch.Tensor
    duration_seconds: float
    source_sample_rate: int
    source_path: str
    embedding_model: str = MIO_GLOBAL_MODEL
    reference_sha256: str = ""
    created_utc: str = ""
    peak_amplitude: float = _NAN
    rms_dbfs: float = _NAN
    clipping_fraction: float = _NAN
    active_speech_ratio: float = _NAN
    dc_offset: float = _NAN
    quality_warnings: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.global_embedding.shape != (ASTRAPE_EMBEDDING_DIM,):
            raise ValueError(
                f"VoiceBank global embedding must have shape "
                f"[{ASTRAPE_EMBEDDING_DIM}]"
            )
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

    def save(
        self,
        path: str | Path,
        *,
        force_format: str | None = None,
    ) -> Path:
        """Persist a VoiceBank. The new layout is selected by file extension
        (`.astrape` -> v3 JSON layout) and `force_format` overrides that.

        ``force_format`` accepts:

        - ``"astrape"`` / ``"v3"`` : write the new `.astrape` layout.
        - ``"npz"`` / ``"v2"`` : legacy compressed `.npz` layout.
        """
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_format = _resolve_format(path, force_format)
        if resolved_format == "astrape":
            return _write_astrape(path, self)
        return _write_npz(path, self)

    @classmethod
    def load(cls, path: str | Path) -> "VoiceBank":
        """Load a VoiceBank from `.astrape` or legacy `.npz` storage."""
        path = Path(path)
        with path.open("rb") as handle:
            ident = handle.read(4)
        if ident == ASTRAPE_MAGIC:
            with path.open("rb") as handle:
                header = parse_astrape_header(handle.read(ASTRAPE_HEADER_SIZE))
            with path.open("rb") as handle:
                handle.seek(header["embedding_offset"])
                embedding_blob = handle.read(header["embedding_length"])
                handle.seek(header["metadata_offset"])
                metadata_blob = handle.read(header["metadata_length"])
        else:
            return _read_npz(path)
        bank = _build_from_astrape(embedding_blob, metadata_blob)
        bank.validate()
        return bank


# Format detection -----------------------------------------------------------------

def _resolve_format(path: Path, force_format: str | None) -> str:
    if force_format is not None:
        fmt = force_format.lower()
        if fmt in {"astrape", "v3"}:
            return "astrape"
        if fmt in {"npz", "v2"}:
            return "npz"
        raise ValueError(f"Unknown VoiceBank force_format: {force_format!r}")
    return "astrape" if path.suffix == VOICEBANK_EXTENSION else "npz"


def detect_format(path: str | Path) -> str:
    """Return ``"astrape"`` or ``"npz"`` based on the file header/extension."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            ident = handle.read(4)
    except FileNotFoundError:
        return "npz" if path.suffix == VOICEBANK_LEGACY_EXTENSION else "astrape"
    if ident == ASTRAPE_MAGIC:
        return "astrape"
    return "npz"


# `.astrape` writer/reader ---------------------------------------------------------

def _metadata_json_bytes(self: VoiceBank) -> bytes:
    payload = {
        "duration_seconds": float(self.duration_seconds),
        "source_sample_rate": int(self.source_sample_rate),
        "source_path": str(self.source_path),
        "embedding_model": str(self.embedding_model),
        "reference_sha256": str(self.reference_sha256),
        "created_utc": str(self.created_utc),
        "peak_amplitude": _encode_nan(self.peak_amplitude),
        "rms_dbfs": _encode_nan(self.rms_dbfs),
        "clipping_fraction": _encode_nan(self.clipping_fraction),
        "active_speech_ratio": _encode_nan(self.active_speech_ratio),
        "dc_offset": _encode_nan(self.dc_offset),
        "quality_warnings": list(self.quality_warnings),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _encode_nan(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _decode_nan(value: Any) -> float:
    if value is None:
        return _NAN
    return float(value)


def _write_astrape(path: Path, bank: VoiceBank) -> Path:
    embedding = bank.global_embedding.detach().cpu().contiguous().float()
    if embedding.shape != (ASTRAPE_EMBEDDING_DIM,):
        raise ValueError(
            f"VoiceBank embedding must have shape "
            f"[{ASTRAPE_EMBEDDING_DIM}] (got {tuple(embedding.shape)})"
        )
    embedding_bytes = embedding.numpy().astype(
        ASTRAPE_EMBEDDING_DTYPE, copy=False
    ).tobytes()
    if len(embedding_bytes) != ASTRAPE_EMBEDDING_BYTES:
        raise ValueError("VoiceBank embedding must be float32 contiguous")
    metadata_bytes = _metadata_json_bytes(bank)
    embedding_offset = ASTRAPE_HEADER_SIZE
    metadata_offset = ASTRAPE_HEADER_SIZE + ASTRAPE_EMBEDDING_BYTES
    header = struct.pack(
        ASTRAPE_HEADER_FMT,
        ASTRAPE_MAGIC,
        VOICEBANK_FORMAT_VERSION,
        ASTRAPE_FLAG_HAS_EMBEDDING,
        0,
        embedding_offset,
        ASTRAPE_EMBEDDING_BYTES,
        metadata_offset,
        len(metadata_bytes),
        0,
    )
    payload = header + embedding_bytes + metadata_bytes
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def parse_astrape_header(buffer: bytes) -> dict[str, int]:
    if len(buffer) < ASTRAPE_HEADER_SIZE:
        raise ValueError("header too short")
    if buffer[:4] != ASTRAPE_MAGIC:
        raise ValueError(f"bad magic {buffer[:4]!r}")
    (magic, version, flags, _r0,
     embedding_offset, embedding_length,
     metadata_offset, metadata_length,
     _r1) = struct.unpack(ASTRAPE_HEADER_FMT, buffer[:ASTRAPE_HEADER_SIZE])
    if version != VOICEBANK_FORMAT_VERSION:
        raise ValueError(f"unsupported .astrape version: {version}")
    if flags & ASTRAPE_FLAG_HAS_EMBEDDING == 0:
        raise ValueError(".astrape file does not contain an embedding")
    if embedding_length != ASTRAPE_EMBEDDING_BYTES:
        raise ValueError(
            f"unexpected embedding byte length: {embedding_length}"
        )
    if embedding_offset < ASTRAPE_HEADER_SIZE:
        raise ValueError("embedding overlaps .astrape header")
    if metadata_offset < embedding_offset + embedding_length:
        raise ValueError("metadata overlaps .astrape embedding")
    return {
        "flags": flags,
        "embedding_offset": embedding_offset,
        "embedding_length": embedding_length,
        "metadata_offset": metadata_offset,
        "metadata_length": metadata_length,
    }


def _build_from_astrape(embedding_blob: bytes, metadata_blob: bytes) -> VoiceBank:
    array = np.frombuffer(embedding_blob, dtype=ASTRAPE_EMBEDDING_DTYPE)
    if array.shape != (ASTRAPE_EMBEDDING_DIM,):
        raise ValueError(
            f"unexpected embedding shape: {array.shape!r}"
        )
    array = array.astype(np.float32, copy=False)
    payload = json.loads(metadata_blob.decode("utf-8"))
    return VoiceBank(
        global_embedding=torch.from_numpy(array.copy()).float(),
        duration_seconds=float(payload["duration_seconds"]),
        source_sample_rate=int(payload["source_sample_rate"]),
        source_path=str(payload["source_path"]),
        embedding_model=str(payload.get("embedding_model", MIO_GLOBAL_MODEL)),
        reference_sha256=str(payload.get("reference_sha256", "")),
        created_utc=str(payload.get("created_utc", "")),
        peak_amplitude=_decode_nan(payload.get("peak_amplitude")),
        rms_dbfs=_decode_nan(payload.get("rms_dbfs")),
        clipping_fraction=_decode_nan(payload.get("clipping_fraction")),
        active_speech_ratio=_decode_nan(payload.get("active_speech_ratio")),
        dc_offset=_decode_nan(payload.get("dc_offset")),
        quality_warnings=tuple(payload.get("quality_warnings") or ()),
    )


# Fast peek / mmap view ------------------------------------------------------------

def header_peek(path: str | Path) -> dict[str, Any]:
    """Return a small dict with format/version/embedding-length metadata.

    For `.astrape` files, the prefix read is bounded to the fixed 48-byte
    header. For legacy `.npz`, this falls back to a full ``np.load`` decode,
    so `.npz` callers receive a meaningful latency regression signal.
    """
    path = Path(path)
    fmt = detect_format(path)
    if fmt == "astrape":
        with path.open("rb") as handle:
            raw = handle.read(ASTRAPE_HEADER_SIZE)
        header = parse_astrape_header(raw)
        return {
            "format": "astrape",
            "version": VOICEBANK_FORMAT_VERSION,
            "embedding_length_bytes": header["embedding_length"],
            "embedding_dim": header["embedding_length"] // 4,
            "metadata_length_bytes": header["metadata_length"],
            "file_size_bytes": path.stat().st_size,
        }
    with np.load(path) as data:
        return {
            "format": "npz",
            "version": int(data["format_version"]),
            "embedding_length_bytes": int(data[_EMBEDDING_KEY].nbytes),
            "embedding_dim": int(data[_EMBEDDING_KEY].shape[0]),
            "metadata_length_bytes": int(data["duration_seconds"].nbytes)
                + int(data["source_sample_rate"].nbytes),
            "file_size_bytes": path.stat().st_size,
        }


def open_embedding_mmap(path: str | Path) -> "_MmapEmbeddingHandle":
    """Open the embedding as a read-only array handle.

    ``.astrape`` files yield a real ``mmap.mmap``-backed NumPy view. Keep the
    handle open while reading ``array``; use ``tensor()`` for an owning copy
    that may outlive the handle. Legacy
    ``.npz`` files fall back to ``np.load`` (no zero-copy possible because
    NumPy cannot memory-map a ZIP container), returning an owning array copy.
    """
    path = Path(path)
    fmt = detect_format(path)
    if fmt != "astrape":
        bank = VoiceBank.load(path)
        return _MmapEmbeddingHandle(
            bank.global_embedding.detach().cpu().numpy().copy(),
            own_buffer=True,
            mmap=None,
            file=None,
        )
    file_handle = path.open("rb")
    try:
        raw_header = file_handle.read(ASTRAPE_HEADER_SIZE)
        header = parse_astrape_header(raw_header)
        file_size = path.stat().st_size
        embedding_end = header["embedding_offset"] + header["embedding_length"]
        metadata_end = header["metadata_offset"] + header["metadata_length"]
        if embedding_end > file_size or metadata_end > file_size:
            raise ValueError(".astrape header points past end of file")
        mm = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        file_handle.close()
        raise
    buffer = np.ndarray(
        shape=(ASTRAPE_EMBEDDING_DIM,),
        dtype=ASTRAPE_EMBEDDING_DTYPE,
        buffer=mm,
        offset=header["embedding_offset"],
    )
    return _MmapEmbeddingHandle(buffer, own_buffer=False, mmap=mm, file=file_handle)


@dataclass
class _MmapEmbeddingHandle:
    array: np.ndarray
    own_buffer: bool
    mmap: "mmap.mmap | None"
    file: "Any"

    def tensor(self) -> torch.Tensor:
        # Always clone so the caller may outlive this handle safely.
        return torch.from_numpy(self.array.astype(np.float32, copy=True))

    def close(self) -> None:
        self.array = np.asarray((), dtype=ASTRAPE_EMBEDDING_DTYPE)
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self) -> "_MmapEmbeddingHandle":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# Legacy `.npz` writer/reader ------------------------------------------------------

def _write_npz(path: Path, bank: VoiceBank) -> Path:
    """Legacy `.npz` writer.

    ``np.savez_compressed`` appends ``.npz`` when the supplied path does not
    already end in ``.npz``. We honour the caller's intended final path by
    rewriting the target before delegating to NumPy.
    """
    target = path if path.suffix == VOICEBANK_LEGACY_EXTENSION else path.with_suffix(
        VOICEBANK_LEGACY_EXTENSION
    )
    np.savez_compressed(
        target,
        format_version=np.asarray(VOICEBANK_LEGACY_NPZ_FORMAT_VERSION, dtype=np.int64),
        global_embedding=bank.global_embedding.detach().cpu().numpy().astype(
            np.float32, copy=False
        ),
        duration_seconds=np.asarray(bank.duration_seconds, dtype=np.float32),
        source_sample_rate=np.asarray(bank.source_sample_rate, dtype=np.int64),
        source_path=np.asarray(bank.source_path),
        embedding_model=np.asarray(bank.embedding_model),
        reference_sha256=np.asarray(bank.reference_sha256),
        created_utc=np.asarray(bank.created_utc),
        peak_amplitude=np.asarray(bank.peak_amplitude, dtype=np.float32),
        rms_dbfs=np.asarray(bank.rms_dbfs, dtype=np.float32),
        clipping_fraction=np.asarray(bank.clipping_fraction, dtype=np.float32),
        active_speech_ratio=np.asarray(bank.active_speech_ratio, dtype=np.float32),
        dc_offset=np.asarray(bank.dc_offset, dtype=np.float32),
        quality_warnings=_string_array(bank.quality_warnings),
    )
    return target


def _read_npz(path: Path) -> VoiceBank:
    with np.load(path) as data:
        version = int(data["format_version"])
        if version not in (
            VOICEBANK_LEGACY_NPZ_V1_FORMAT_VERSION,
            VOICEBANK_LEGACY_NPZ_FORMAT_VERSION,
        ):
            raise ValueError(f"Unsupported VoiceBank version: {version}")
        bank = VoiceBank(
            global_embedding=torch.from_numpy(data[_EMBEDDING_KEY]).float(),
            duration_seconds=float(data["duration_seconds"]),
            source_sample_rate=int(data["source_sample_rate"]),
            source_path=str(data["source_path"].item()),
            embedding_model=_optional_string(
                data, "embedding_model", MIO_GLOBAL_MODEL,
            ),
            reference_sha256=_optional_string(data, "reference_sha256", ""),
            created_utc=_optional_string(data, "created_utc", ""),
            peak_amplitude=_optional_scalar(data, "peak_amplitude", _NAN),
            rms_dbfs=_optional_scalar(data, "rms_dbfs", _NAN),
            clipping_fraction=_optional_scalar(data, "clipping_fraction", _NAN),
            active_speech_ratio=_optional_scalar(data, "active_speech_ratio", _NAN),
            dc_offset=_optional_scalar(data, "dc_offset", _NAN),
            quality_warnings=(
                tuple(map(str, data["quality_warnings"].tolist()))
                if "quality_warnings" in data
                else ()
            ),
        )
    bank.validate()
    return bank
