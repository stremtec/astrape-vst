from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import extract_wavlm_targets
from astrape.data import masked_content_loss
from astrape.voicebank import MIO_GLOBAL_MODEL, VoiceBank
from train_decoder_phase0 import (
    DecoderPretrainDataset,
    collate_skip_none,
    load_voicebank_dir,
    validate_voicebank_coverage,
)
from train_encoder_ssl import (
    SSLTargetDataset,
    collate_fn,
    feature_target_loss,
    make_frame_mask,
)


class Phase0VoicebankHelperTests(unittest.TestCase):
    def _bank(self, model: str = MIO_GLOBAL_MODEL) -> VoiceBank:
        return VoiceBank(
            global_embedding=torch.randn(128),
            duration_seconds=6.0,
            source_sample_rate=44100,
            source_path="/tmp/reference.wav",
            embedding_model=model,
        )

    def test_voicebank_coverage_requires_matching_speaker_stems(self):
        embeddings = {"p225": torch.randn(128)}
        with self.assertRaisesRegex(ValueError, "p226"):
            validate_voicebank_coverage(np.asarray(["p225", "p226"]), embeddings)

    def test_voicebank_loader_rejects_condition_model_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p225.astrape"
            self._bank(model="other/model").save(path)
            with self.assertRaisesRegex(ValueError, "condition_model"):
                load_voicebank_dir(
                    Path(d),
                    expected_model=MIO_GLOBAL_MODEL,
                    expected_dim=128,
                )

    def test_phase0_collate_rejects_partial_embedding_coverage(self):
        batch = [
            {
                "content": torch.zeros(2, 768),
                "waveform": torch.zeros(2 * 1764),
                "speaker": "p225",
                "speaker_embedding": torch.randn(128),
            },
            {
                "content": torch.zeros(2, 768),
                "waveform": torch.zeros(2 * 1764),
                "speaker": "p226",
                "speaker_embedding": None,
            },
        ]
        with self.assertRaisesRegex(ValueError, "partial speaker embedding"):
            collate_skip_none(batch)

    def test_phase0_dataset_does_not_swallow_corrupt_sample_cache(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            np.savez(
                data_dir / "meta.npz",
                n_samples=1,
                source_files=np.asarray(["missing.wav"]),
                spk_names=np.asarray(["p225"]),
            )
            (data_dir / "s_00000.npz").write_bytes(b"not an npz")

            dataset = DecoderPretrainDataset(
                data_dir,
                data_dir,
                speaker_embeddings={},
            )
            with self.assertRaises(Exception):
                _ = dataset[0]

    def test_phase0_dataset_skips_missing_audio(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            np.savez(
                data_dir / "meta.npz",
                n_samples=1,
                source_files=np.asarray(["missing.wav"]),
                spk_names=np.asarray(["p225"]),
            )
            np.savez(
                data_dir / "s_00000.npz",
                ce_768=np.zeros((2, 768), dtype=np.float32),
            )

            dataset = DecoderPretrainDataset(
                data_dir,
                data_dir,
                speaker_embeddings={},
            )
            self.assertIsNone(dataset[0])


class EncoderSSLHelperTests(unittest.TestCase):
    def test_dataset_raises_on_corrupt_ssl_cache(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            np.savez(
                data_dir / "s_00000.npz",
                logmel=np.zeros((80, 4), dtype=np.float32),
                ce_768=np.zeros((2, 768), dtype=np.float32),
            )
            (data_dir / "ssl_00000.npz").write_bytes(b"not an npz")

            dataset = SSLTargetDataset(data_dir, np.asarray([0]))
            with self.assertRaises(Exception):
                _ = dataset[0]

    def test_collate_pads_content_ssl_targets_and_masks_valid_frames(self):
        batch = [
            {
                "mel": torch.zeros(80, 5),
                "content_target": torch.ones(2, 768),
                "ssl_target": torch.ones(3, 768),
            },
            {
                "mel": torch.zeros(80, 6),
                "content_target": torch.ones(4, 768),
                "ssl_target": torch.ones(4, 768),
            },
        ]
        out = collate_fn(batch)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(tuple(out["mel"].shape), (2, 80, 6))
        self.assertEqual(tuple(out["target_content"].shape), (2, 4, 768))
        self.assertEqual(tuple(out["ssl_target"].shape), (2, 4, 768))
        torch.testing.assert_close(out["content_lens"], torch.tensor([2, 4]))
        torch.testing.assert_close(out["ssl_lens"], torch.tensor([3, 4]))

        mask = make_frame_mask(out["mel_lens"], out["content_lens"], 4)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
            ]
        )
        self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_feature_target_loss_empty_mask_is_finite_and_backwardable(self):
        pred = torch.randn(1, 768, 2, requires_grad=True)
        target = torch.randn(1, 2, 768)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        loss, metrics = feature_target_loss(pred, target, mask)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["frames"], 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)


class WavLMExtractionHelperTests(unittest.TestCase):
    def test_align_target_frames_trims_and_pads_by_repeating_last_frame(self):
        target = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        trimmed = extract_wavlm_targets.align_target_frames(target, 2)
        self.assertEqual(trimmed.shape, (2, 3))
        np.testing.assert_array_equal(trimmed, target[:2])

        short = target[:2]
        padded = extract_wavlm_targets.align_target_frames(short, 4)
        self.assertEqual(padded.shape, (4, 3))
        np.testing.assert_array_equal(padded[:2], short)
        np.testing.assert_array_equal(padded[2:], np.repeat(short[-1:], 2, axis=0))

    def test_ssl_cache_validation_rejects_corrupt_and_misaligned_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ssl_00000.npz"
            self.assertFalse(extract_wavlm_targets.is_valid_ssl_cache(path, 2))

            path.write_bytes(b"not an npz")
            self.assertFalse(extract_wavlm_targets.is_valid_ssl_cache(path, 2))

            extract_wavlm_targets.save_npz_atomic(
                path,
                wavlm_25hz=np.ones((1, 768), dtype=np.float32),
            )
            self.assertFalse(extract_wavlm_targets.is_valid_ssl_cache(path, 2))
            self.assertTrue(extract_wavlm_targets.is_valid_ssl_cache(path, 1))

    def test_atomic_npz_write_cleans_temp_file_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ssl_00000.npz"
            original = extract_wavlm_targets.np.savez_compressed

            def boom(*args, **kwargs):
                raise RuntimeError("interrupted")

            extract_wavlm_targets.np.savez_compressed = boom
            try:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    extract_wavlm_targets.save_npz_atomic(
                        path, wavlm_25hz=np.ones((2, 3), dtype=np.float32)
                    )
            finally:
                extract_wavlm_targets.np.savez_compressed = original

            self.assertFalse(path.exists())
            self.assertEqual(list(Path(d).glob(".ssl_00000.npz.*.tmp")), [])


class SharedDataHelperTests(unittest.TestCase):
    def test_masked_content_loss_empty_mask_is_finite_and_backwardable(self):
        pred = torch.randn(1, 768, 2, requires_grad=True)
        target = torch.randn(1, 2, 768)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        loss, cosine = masked_content_loss(pred, target, mask)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(cosine.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)


if __name__ == "__main__":
    unittest.main()
