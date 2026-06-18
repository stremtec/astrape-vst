"""Tests for the VoiceBank .astrape format and legacy .npz round-trip."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from astrape.voicebank import (
    ASTRAPE_EMBEDDING_BYTES,
    ASTRAPE_EMBEDDING_DIM,
    ASTRAPE_EMBEDDING_DTYPE,
    ASTRAPE_FLAG_HAS_EMBEDDING,
    ASTRAPE_HEADER_FMT,
    ASTRAPE_HEADER_SIZE,
    ASTRAPE_MAGIC,
    MIO_GLOBAL_MODEL,
    VOICEBANK_FORMAT_VERSION,
    VoiceBank,
    open_embedding_mmap,
)


class VoiceBankFormatTests(unittest.TestCase):
    def _make_bank(self, seed: int = 1) -> VoiceBank:
        generator = torch.Generator().manual_seed(seed)
        return VoiceBank(
            global_embedding=torch.randn(128, generator=generator),
            duration_seconds=6.0 + seed / 10.0,
            source_sample_rate=44100,
            source_path=f"/tmp/refs/speaker_{seed}.wav",
            embedding_model="Aratako/MioCodec-25Hz-44.1kHz-v2",
            reference_sha256=("%02x" % seed) * 32,
            created_utc="2026-06-15T12:00:00+00:00",
            peak_amplitude=0.8,
            rms_dbfs=-18.0,
            clipping_fraction=0.0,
            active_speech_ratio=0.9,
            dc_offset=0.001,
            quality_warnings=("test",),
        )

    def _write_raw_astrape(
        self,
        path: Path,
        embedding: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        embedding_bytes = np.asarray(
            embedding, dtype=ASTRAPE_EMBEDDING_DTYPE
        ).tobytes()
        metadata_bytes = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        header = struct.pack(
            ASTRAPE_HEADER_FMT,
            ASTRAPE_MAGIC,
            VOICEBANK_FORMAT_VERSION,
            ASTRAPE_FLAG_HAS_EMBEDDING,
            0,
            ASTRAPE_HEADER_SIZE,
            ASTRAPE_EMBEDDING_BYTES,
            ASTRAPE_HEADER_SIZE + ASTRAPE_EMBEDDING_BYTES,
            len(metadata_bytes),
            0,
        )
        path.write_bytes(header + embedding_bytes + metadata_bytes)

    def test_astrape_round_trip_is_bitexact(self):
        bank = self._make_bank(seed=42)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "speaker.astrape"
            bank.save(path)
            loaded = VoiceBank.load(path)
        torch.testing.assert_close(
            loaded.global_embedding, bank.global_embedding, atol=0, rtol=0
        )
        self.assertEqual(loaded.duration_seconds, bank.duration_seconds)
        self.assertEqual(loaded.source_sample_rate, bank.source_sample_rate)
        self.assertEqual(loaded.reference_sha256, bank.reference_sha256)
        self.assertEqual(loaded.quality_warnings, bank.quality_warnings)

    def test_npz_round_trip_preserves_embedding(self):
        bank = self._make_bank(seed=7)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "speaker.npz"
            bank.save(path)
            loaded = VoiceBank.load(path)
        torch.testing.assert_close(
            loaded.global_embedding, bank.global_embedding, atol=0, rtol=0
        )

    def test_npz_and_astrape_produce_identical_embeddings(self):
        bank = self._make_bank(seed=13)
        with tempfile.TemporaryDirectory() as d:
            npz_path = Path(d) / "speaker.npz"
            astrape_path = Path(d) / "speaker.astrape"
            bank.save(npz_path)
            bank.save(astrape_path)
            from_npz = VoiceBank.load(npz_path)
            from_astrape = VoiceBank.load(astrape_path)
        torch.testing.assert_close(
            from_npz.global_embedding, from_astrape.global_embedding, atol=0, rtol=0
        )

    def test_astrape_embedding_bytes_are_little_endian_float32(self):
        embedding = torch.linspace(-1.0, 1.0, ASTRAPE_EMBEDDING_DIM)
        bank = VoiceBank(
            global_embedding=embedding,
            duration_seconds=6.0,
            source_sample_rate=44100,
            source_path="/tmp/refs/little-endian.wav",
            embedding_model=MIO_GLOBAL_MODEL,
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "speaker.astrape"
            bank.save(path)
            raw = path.read_bytes()
            embedding_blob = raw[
                ASTRAPE_HEADER_SIZE : ASTRAPE_HEADER_SIZE + ASTRAPE_EMBEDDING_BYTES
            ]
            loaded = VoiceBank.load(path)

        self.assertEqual(
            embedding_blob,
            embedding.numpy().astype(ASTRAPE_EMBEDDING_DTYPE).tobytes(),
        )
        torch.testing.assert_close(loaded.global_embedding, embedding, atol=0, rtol=0)

    def test_astrape_load_runs_voicebank_validation(self):
        metadata = {
            "duration_seconds": 1.0,
            "source_sample_rate": 44100,
            "source_path": "/tmp/refs/too-short.wav",
            "embedding_model": MIO_GLOBAL_MODEL,
            "reference_sha256": "",
            "created_utc": "",
            "peak_amplitude": None,
            "rms_dbfs": None,
            "clipping_fraction": None,
            "active_speech_ratio": None,
            "dc_offset": None,
            "quality_warnings": [],
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "short.astrape"
            self._write_raw_astrape(
                path,
                np.zeros(ASTRAPE_EMBEDDING_DIM, dtype=np.float32),
                metadata,
            )
            with self.assertRaisesRegex(ValueError, "at least"):
                VoiceBank.load(path)

    def test_open_embedding_mmap_returns_readonly_astrape_view(self):
        bank = self._make_bank(seed=21)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "speaker.astrape"
            bank.save(path)
            with open_embedding_mmap(path) as handle:
                self.assertFalse(handle.own_buffer)
                self.assertEqual(handle.array.shape, (ASTRAPE_EMBEDDING_DIM,))
                self.assertFalse(handle.array.flags.writeable)
                np.testing.assert_array_equal(
                    handle.array, bank.global_embedding.numpy()
                )
                tensor = handle.tensor()

        torch.testing.assert_close(
            tensor, bank.global_embedding, atol=0, rtol=0
        )

    def test_validation_rejects_short_reference(self):
        bank = VoiceBank(
            global_embedding=torch.randn(128),
            duration_seconds=2.0,
            source_sample_rate=44100,
            source_path="/tmp/short.wav",
        )
        with self.assertRaises(ValueError):
            bank.validate()

    def test_validation_rejects_wrong_embedding_dim(self):
        bank = VoiceBank(
            global_embedding=torch.randn(64),
            duration_seconds=6.0,
            source_sample_rate=44100,
            source_path="/tmp/test.wav",
        )
        with self.assertRaises(ValueError):
            bank.validate()


if __name__ == "__main__":
    unittest.main()
