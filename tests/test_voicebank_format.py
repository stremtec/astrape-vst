"""Tests for the VoiceBank .astrape format and legacy .npz round-trip."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from astrape.voicebank import VoiceBank


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
