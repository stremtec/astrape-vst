from __future__ import annotations

import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import torch
from fastapi import HTTPException
from pydantic import ValidationError

from astrape.wave_decoder import WaveDecoderConfig
from webui import server


class WebUiServerTests(unittest.TestCase):
    def test_slug_normalizes_profile_names(self):
        self.assertEqual(server._slug(" Korean target 01 "), "Korean_target_01")
        self.assertEqual(server._slug("voice.name-v2"), "voice.name-v2")
        with self.assertRaises(HTTPException):
            server._slug("...")

    def test_runtime_settings_reject_invalid_ranges(self):
        settings = server.RuntimeSettings(
            chunk_ms=2.5,
            pitch_semitones=24,
            formant_semitones=-12,
            f0_min=50,
            f0_max=1100,
        )
        self.assertEqual(settings.f0_engine, "fcpe")
        with self.assertRaises(ValidationError):
            server.RuntimeSettings(chunk_ms=1)
        with self.assertRaises(ValidationError):
            server.RuntimeSettings(wet=1.1)

    def test_mio_python_prefers_configured_runtime(self):
        with tempfile.NamedTemporaryFile() as runtime:
            with mock.patch.dict(
                server.os.environ,
                {"MIO_PYTHON": runtime.name},
            ):
                self.assertEqual(server._mio_python(), runtime.name)

    def test_training_status_parses_progress_and_validation_cosine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            log_path = logs / "curriculum.log"
            log_path.write_text(
                "E010 blend step=1000/1000 loss=1.0 frame_cos=0.9912\n",
                encoding="utf-8",
            )
            (logs / "content_curriculum.latest").write_text(
                str(log_path),
                encoding="utf-8",
            )
            process_result = subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="",
            )
            with (
                mock.patch.object(server, "ROOT", root),
                mock.patch.object(
                    server.subprocess,
                    "run",
                    return_value=process_result,
                ),
            ):
                status = server._training_status()
        self.assertFalse(status["running"])
        self.assertEqual(status["epoch"], 10)
        self.assertEqual(status["phase"], "blend")
        self.assertEqual(status["step"], 1000)
        self.assertEqual(status["steps"], 1000)
        self.assertAlmostEqual(status["frame_cosine"], 0.9912)

    def test_decoder_capabilities_read_checkpoint_without_loading_model(self):
        config = WaveDecoderConfig(
            sample_rate=600,
            content_rate=50,
            content_dim=8,
            condition_dim=4,
            initial_channels=16,
            stage_channels=(12, 8, 4),
            upsample_factors=(3, 2, 2),
            supports_f0_conditioning=True,
            supports_formant_conditioning=True,
            f0_model="fcpe",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "direct_wave_decoder.best.pt"
            torch.save(
                {
                    "model_type": "direct_wave_decoder",
                    "config": asdict(config),
                    "state_dict": {
                        "weight": torch.zeros(3, 4),
                        "bias": torch.zeros(3),
                    },
                },
                checkpoint,
            )
            with mock.patch.object(
                server,
                "_wave_checkpoint",
                return_value=checkpoint,
            ):
                capabilities = server._decoder_capabilities()
        self.assertTrue(capabilities["ready"])
        self.assertTrue(capabilities["supports_f0_conditioning"])
        self.assertTrue(capabilities["supports_formant_conditioning"])
        self.assertEqual(capabilities["f0_model"], "fcpe")
        self.assertEqual(capabilities["sample_rate"], 600)
        self.assertEqual(capabilities["parameters"], 15)

    def test_decoder_capabilities_explain_missing_checkpoint(self):
        with mock.patch.object(server, "_wave_checkpoint", return_value=None):
            capabilities = server._decoder_capabilities()
        self.assertFalse(capabilities["ready"])
        self.assertIn("not trained", capabilities["reason"])


if __name__ == "__main__":
    unittest.main()
