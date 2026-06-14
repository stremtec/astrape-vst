import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from astrape.audio import StreamingLogMel
from astrape.checkpoint import load_content_checkpoint, save_checkpoint
from astrape.curriculum import (
    CurriculumConfig,
    original_loss,
    phase_weights,
    validate_curriculum,
)
from astrape.data import (
    ContentSample,
    crop_aligned,
    masked_content_loss,
    speaker_disjoint_split,
)
from astrape.mel_decoder import CausalMelDecoder, MelDecoderConfig, load_mel_decoder
from astrape.model import ContentStudent, ContentStudentConfig
from astrape.fsq import (
    fit_fsq_projection,
    indices_to_codes,
    indices_to_level_indices,
)
from astrape.original_data import OriginalBatch, minimum_ctc_frames
from astrape.streaming_pipeline import OutputRingBuffer, StreamingVoiceConverter
from astrape.text import VOCAB_SIZE
from astrape.voicebank import (
    MIN_REFERENCE_SECONDS,
    MIO_GLOBAL_MODEL,
    VoiceBank,
    analyze_reference,
)
from astrape.wave_decoder import (
    DirectWaveDecoder,
    WaveDecoderConfig,
    load_wave_decoder,
    save_wave_decoder_checkpoint,
)
from tiers import TIERS, get_tier
from train_wave_decoder import WaveDataset


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def small_student(self) -> ContentStudent:
        return ContentStudent(
            ContentStudentConfig(
                hidden=64,
                n_layers=2,
                n_heads=4,
                content_dim=32,
            )
        ).eval()

    def test_content_model_has_no_future_leakage(self):
        model = self.small_student()
        prefix = torch.randn(1, 80, 12)
        suffix = torch.randn(1, 80, 8)
        prefix_output = model(prefix).content
        full_output = model(torch.cat((prefix, suffix), dim=-1)).content
        torch.testing.assert_close(prefix_output, full_output[:, :, :6])

    def test_content_streaming_matches_full_sequence(self):
        model = self.small_student()
        x = torch.randn(1, 80, 20)
        expected = model(x).content
        state = None
        chunks = []
        for start in range(0, x.shape[-1], 2):
            output, state = model.forward_stream(x[:, :, start : start + 2], state)
            chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_content_streaming_buffers_odd_chunks_and_flushes(self):
        model = self.small_student()
        x = torch.randn(1, 80, 21)
        expected = model(x).content
        state = None
        chunks = []
        for start, length in ((0, 3), (3, 4), (7, 1), (8, 7), (15, 6)):
            output, state = model.forward_stream(
                x[:, :, start : start + length], state
            )
            if output.content.shape[-1]:
                chunks.append(output.content)
        output, state = model.forward_stream(x[:, :, :0], state, flush=True)
        chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_content_streaming_emits_first_frame_without_pair_buffer(self):
        model = self.small_student()
        output, state = model.forward_stream(torch.randn(1, 80, 1))
        self.assertEqual(output.content.shape[-1], 1)
        output, state = model.forward_stream(torch.randn(1, 80, 1), state)
        self.assertEqual(output.content.shape[-1], 0)
        output, _ = model.forward_stream(torch.randn(1, 80, 1), state)
        self.assertEqual(output.content.shape[-1], 1)

    def test_limited_attention_streaming_matches_full_sequence(self):
        model = ContentStudent(
            ContentStudentConfig(
                hidden=64,
                n_layers=2,
                n_heads=4,
                content_dim=32,
                max_attention_context=4,
            )
        ).eval()
        x = torch.randn(1, 80, 20)
        expected = model(x).content
        state = None
        chunks = []
        for start in range(0, x.shape[-1], 2):
            output, state = model.forward_stream(x[:, :, start : start + 2], state)
            chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_mel_decoder_streaming_matches_full_sequence(self):
        decoder = CausalMelDecoder(
            MelDecoderConfig(hidden=64, n_layers=2, n_heads=4, dropout=0.0)
        ).eval()
        content = torch.randn(1, 10, 768)
        global_embedding = torch.randn(1, 128)
        expected = decoder(content, global_embedding)
        state = None
        chunks = []
        for index in range(content.shape[1]):
            output, state = decoder.forward_stream(
                content[:, index : index + 1], global_embedding, state
            )
            chunks.append(output)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def small_wave_decoder(self) -> DirectWaveDecoder:
        return DirectWaveDecoder(
            WaveDecoderConfig(
                content_dim=8,
                condition_dim=4,
                sample_rate=600,
                content_rate=100,
                initial_channels=16,
                stage_channels=(12, 8),
                upsample_factors=(2, 3),
                mrf_kernel_sizes=(3,),
                mrf_dilations=((1, 2),),
                output_kernel_size=3,
            )
        ).eval()

    def test_wave_decoder_output_length_and_no_future_leakage(self):
        decoder = self.small_wave_decoder()
        prefix = torch.randn(1, 3, 8)
        suffix = torch.randn(1, 2, 8)
        global_embedding = torch.randn(1, 4)
        prefix_audio = decoder(prefix, global_embedding)
        full_audio = decoder(
            torch.cat((prefix, suffix), dim=1),
            global_embedding,
        )
        self.assertEqual(prefix_audio.shape[-1], 3 * 6)
        torch.testing.assert_close(
            prefix_audio,
            full_audio[:, : prefix_audio.shape[-1]],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_wave_decoder_streaming_matches_irregular_chunks(self):
        decoder = self.small_wave_decoder()
        content = torch.randn(1, 7, 8)
        global_embedding = torch.randn(1, 4)
        expected = decoder(content, global_embedding)
        state = None
        chunks = []
        for start, length in ((0, 1), (1, 3), (4, 2), (6, 1)):
            output, state = decoder.forward_stream(
                content[:, start : start + length],
                global_embedding,
                state,
            )
            chunks.append(output)
        actual = torch.cat(chunks, dim=-1)
        self.assertEqual(state.content_frames, content.shape[1])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_wave_decoder_checkpoint_roundtrip(self):
        decoder = self.small_wave_decoder()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.pt"
            save_wave_decoder_checkpoint(
                path,
                decoder,
                step=12,
                metrics={"loss": 0.5},
            )
            loaded = load_wave_decoder(path)
            self.assertEqual(loaded.config, decoder.config)
            content = torch.randn(1, 2, 8)
            condition = torch.randn(1, 4)
            torch.testing.assert_close(
                loaded(content, condition),
                decoder(content, condition),
            )

    def test_voicebank_requires_one_reference_of_at_least_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voicebank.npz"
            bank = VoiceBank(
                global_embedding=torch.randn(128),
                duration_seconds=MIN_REFERENCE_SECONDS,
                source_sample_rate=44100,
                source_path="/tmp/reference.wav",
            )
            bank.save(path)
            loaded = VoiceBank.load(path)
            torch.testing.assert_close(
                loaded.global_embedding,
                bank.global_embedding,
            )
            self.assertEqual(loaded.duration_seconds, MIN_REFERENCE_SECONDS)
            with self.assertRaises(ValueError):
                VoiceBank(
                    global_embedding=torch.randn(128),
                    duration_seconds=MIN_REFERENCE_SECONDS - 0.01,
                    source_sample_rate=44100,
                    source_path="/tmp/short.wav",
                ).validate()

    def test_voicebank_v2_metadata_and_v1_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            versioned = directory / "v2.npz"
            legacy = directory / "v1.npz"
            bank = VoiceBank(
                global_embedding=torch.randn(128),
                duration_seconds=6.0,
                source_sample_rate=44100,
                source_path="/tmp/reference.wav",
                reference_sha256="abc",
                created_utc="2026-06-14T00:00:00+00:00",
                peak_amplitude=0.8,
                rms_dbfs=-20.0,
                clipping_fraction=0.0,
                active_speech_ratio=0.9,
                dc_offset=0.001,
                quality_warnings=("test_warning",),
            )
            bank.save(versioned)
            loaded = VoiceBank.load(versioned)
            self.assertEqual(loaded.reference_sha256, "abc")
            self.assertEqual(loaded.quality_warnings, ("test_warning",))
            np.savez_compressed(
                legacy,
                format_version=np.asarray(1, dtype=np.int64),
                global_embedding=bank.global_embedding.numpy(),
                duration_seconds=np.asarray(6.0, dtype=np.float32),
                source_sample_rate=np.asarray(44100, dtype=np.int64),
                source_path=np.asarray("/tmp/legacy.wav"),
            )
            migrated = VoiceBank.load(legacy)
            self.assertEqual(migrated.embedding_model, MIO_GLOBAL_MODEL)
            self.assertTrue(np.isnan(migrated.rms_dbfs))

    def test_reference_quality_detects_clipping_and_low_activity(self):
        audio = np.zeros(16000 * 6, dtype=np.float32)
        audio[:12] = 1.0
        quality = analyze_reference(audio, 16000)
        self.assertIn("clipping_detected", quality.warnings)
        self.assertIn("reference_too_quiet", quality.warnings)
        self.assertIn("low_active_speech_ratio", quality.warnings)

    def test_e2e_streaming_pipeline_matches_full_models(self):
        content_model = ContentStudent(
            ContentStudentConfig(
                hidden=32,
                n_layers=2,
                n_heads=4,
                content_dim=8,
            )
        ).eval()
        wave_model = DirectWaveDecoder(
            WaveDecoderConfig(
                content_dim=8,
                condition_dim=4,
                sample_rate=150,
                content_rate=25,
                initial_channels=16,
                stage_channels=(12, 8),
                upsample_factors=(2, 3),
                mrf_kernel_sizes=(3,),
                mrf_dilations=((1, 2),),
                output_kernel_size=3,
            )
        ).eval()
        condition = torch.randn(4)
        frontend = StreamingLogMel()
        waveform = torch.randn(6031)
        expected_mel = frontend(waveform)
        expected_content = content_model(expected_mel).content
        expected = wave_model(
            expected_content.transpose(1, 2),
            condition.unsqueeze(0),
        )
        pipeline = StreamingVoiceConverter(
            content_model,
            wave_model,
            condition,
        )
        output_chunks = []
        start = 0
        chunk_sizes = (157, 641, 83, 1000, 319)
        chunk_index = 0
        while start < waveform.numel():
            size = chunk_sizes[chunk_index % len(chunk_sizes)]
            chunk = pipeline.process(waveform[start : start + size])
            if chunk.output_samples:
                output_chunks.append(chunk.audio)
            start += size
            chunk_index += 1
        final = pipeline.flush()
        if final.output_samples:
            output_chunks.append(final.audio)
        actual = torch.cat(output_chunks, dim=-1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(
            pipeline.counters.output_samples,
            expected.shape[-1],
        )
        with self.assertRaises(RuntimeError):
            pipeline.process(torch.zeros(1))

    def test_output_ring_buffer_tracks_underruns(self):
        buffer = OutputRingBuffer(capacity_samples=4)
        buffer.write(torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(
            buffer.read(2),
            torch.tensor([1.0, 2.0]),
        )
        torch.testing.assert_close(
            buffer.read(3),
            torch.tensor([3.0, 0.0, 0.0]),
        )
        self.assertEqual(buffer.buffered_samples, 0)
        self.assertEqual(buffer.underrun_samples, 2)
        buffer.write(torch.tensor([1.0, 2.0, 3.0]))
        buffer.write(torch.tensor([4.0, 5.0, 6.0]))
        torch.testing.assert_close(
            buffer.read(4),
            torch.tensor([3.0, 4.0, 5.0, 6.0]),
        )
        self.assertEqual(buffer.overrun_samples, 2)

    def test_wave_validation_crop_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            np.savez(
                data_dir / "s_00000.npz",
                ce_768=np.arange(12 * 768, dtype=np.float32).reshape(12, 768),
                ge_128=np.zeros(128, dtype=np.float32),
                audio=np.arange(12 * 1764, dtype=np.float32),
            )
            dataset = WaveDataset(
                data_dir,
                np.asarray([0]),
                crop_frames=4,
                target_dir=None,
                seed=123,
                random_crops=False,
            )
            first = dataset[0]
            second = dataset[0]
            torch.testing.assert_close(first.content, second.content)
            torch.testing.assert_close(first.waveform, second.waveform)

    def test_streaming_logmel_matches_full_sequence(self):
        extractor = StreamingLogMel()
        waveform = torch.randn(1, 16000)
        expected = extractor(waveform)
        state = None
        chunks = []
        for start in range(0, waveform.shape[-1], 777):
            output, state = extractor.forward_stream(
                waveform[:, start : start + 777], state
            )
            chunks.append(output)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=1e-6, rtol=1e-6
        )

    def test_speaker_split_is_disjoint_and_deterministic(self):
        speakers = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 5 + ["d"] * 2)
        train_a, validation_a = speaker_disjoint_split(speakers, 0.25, 42)
        train_b, validation_b = speaker_disjoint_split(speakers, 0.25, 42)
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(validation_a, validation_b))
        self.assertFalse(set(speakers[train_a]) & set(speakers[validation_a]))

    def test_crop_uses_even_grid_and_includes_last_window(self):
        mel = torch.arange(20).repeat(80, 1).float()
        content = torch.arange(10).unsqueeze(1).repeat(1, 4).float()
        sample = ContentSample(mel, content, content.clone(), None, "p001", 0)

        class LastChoice(random.Random):
            def choice(self, sequence):
                return sequence[-1]

        cropped = crop_aligned(sample, 8, LastChoice())
        self.assertEqual(cropped.mel[0, 0].item(), 12)
        self.assertEqual(cropped.content[0, 0].item(), 6)
        self.assertEqual(cropped.mel.shape[1], 8)
        self.assertEqual(cropped.content.shape[0], 4)

    def test_masked_loss_ignores_padding(self):
        prediction = torch.zeros(2, 3, 4)
        target = torch.zeros(2, 4, 3)
        target[1, 2:] = 1000
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        loss, cosine = masked_content_loss(prediction, target, mask)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(cosine.item(), 0.0)

    def test_versioned_checkpoint_roundtrip_and_legacy_gate(self):
        model = self.small_student()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            versioned = directory / "model.pt"
            legacy = directory / "legacy.pt"
            save_checkpoint(versioned, model, epoch=3, metrics={"val_cosine": 0.5})
            loaded, metadata = load_content_checkpoint(versioned)
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(loaded.config, model.config)
            torch.save(model.state_dict(), legacy)
            with self.assertRaises(ValueError):
                load_content_checkpoint(legacy)
            loaded_legacy, metadata = load_content_checkpoint(
                legacy, allow_legacy=True
            )
            self.assertEqual(metadata["format_version"], 1)
            self.assertEqual(loaded_legacy.config.hidden, model.config.hidden)

    def test_mel_decoder_checkpoint_roundtrip(self):
        model = CausalMelDecoder(
            MelDecoderConfig(hidden=64, n_layers=2, n_heads=4, dropout=0.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.pt"
            torch.save(
                {
                    "model_type": "causal_mel_decoder",
                    "config": {
                        "hidden": 64,
                        "n_layers": 2,
                        "n_heads": 4,
                        "dropout": 0.0,
                    },
                    "state_dict": model.state_dict(),
                },
                path,
            )
            loaded = load_mel_decoder(path)
            self.assertEqual(loaded.config.hidden, 64)

    def test_structured_fsq_output_matches_frozen_projection(self):
        config = ContentStudentConfig(
            hidden=64,
            n_layers=2,
            n_heads=4,
            content_dim=16,
            structured_fsq=True,
            text_vocab_size=VOCAB_SIZE,
        )
        model = ContentStudent(config).eval()
        projection = {
            "weight": torch.randn(16, 5),
            "bias": torch.randn(16),
        }
        model.load_fsq_projection(projection)
        output = model(torch.randn(1, 80, 20))
        self.assertIsNotNone(output.fsq_codes)
        codes = []
        for axis, levels in enumerate(config.fsq_levels):
            values = (
                output.fsq_codes[:, :, axis].float() - levels // 2
            ) / (levels // 2)
            codes.append(values)
        codes = torch.stack(codes, dim=-1)
        expected = torch.nn.functional.linear(
            codes, projection["weight"], projection["bias"]
        ).transpose(1, 2)
        torch.testing.assert_close(output.content, expected)

    def test_fsq_index_roundtrip_targets(self):
        indices = torch.tensor([[0, 930, 12799]])
        levels = indices_to_level_indices(indices)
        codes = indices_to_codes(indices)
        self.assertEqual(tuple(levels.shape), (1, 3, 5))
        self.assertEqual(tuple(codes.shape), (1, 3, 5))
        self.assertTrue(torch.equal(levels[0, 0], torch.zeros(5, dtype=torch.long)))

    def test_fsq_projection_fit_recovers_affine_teacher(self):
        indices = torch.arange(12800)
        codes = indices_to_codes(indices)
        weight = torch.randn(16, 5)
        bias = torch.randn(16)
        embeddings = torch.nn.functional.linear(codes, weight, bias)
        fitted = fit_fsq_projection(indices, embeddings)
        torch.testing.assert_close(fitted["weight"], weight, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(fitted["bias"], bias, atol=2e-6, rtol=2e-6)

    def test_structured_fsq_streaming_matches_full_sequence(self):
        model = ContentStudent(
            ContentStudentConfig(
                hidden=64,
                n_layers=2,
                n_heads=4,
                content_dim=16,
                structured_fsq=True,
                max_attention_context=8,
            )
        ).eval()
        model.load_fsq_projection(
            {"weight": torch.randn(16, 5), "bias": torch.randn(16)}
        )
        x = torch.randn(1, 80, 20)
        expected = model(x).content
        state = None
        chunks = []
        for start in range(0, x.shape[-1], 2):
            output, state = model.forward_stream(x[:, :, start : start + 2], state)
            chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_ctc_minimum_frames_accounts_for_repeated_labels(self):
        self.assertEqual(minimum_ctc_frames(torch.tensor([1, 1, 2, 2, 2])), 8)

    def test_curriculum_phase_schedule(self):
        config = CurriculumConfig(
            data_dir=Path("."),
            mel_dir=Path("."),
            audio_root=Path("."),
            transcript_root=Path("."),
            fsq_projection=Path("."),
            output_dir=Path("."),
            original_epochs=2,
            blend_epochs=2,
            teacher_epochs=2,
        )
        self.assertEqual(phase_weights(0, config)[0], "original")
        self.assertEqual(phase_weights(2, config)[0], "blend")
        self.assertEqual(phase_weights(4, config)[0], "teacher")
        validate_curriculum(config)

    def test_curriculum_rejects_odd_teacher_crop(self):
        config = CurriculumConfig(
            data_dir=Path("."),
            mel_dir=Path("."),
            audio_root=Path("."),
            transcript_root=Path("."),
            fsq_projection=Path("."),
            output_dir=Path("."),
            max_teacher_mel_frames=79,
        )
        with self.assertRaises(ValueError):
            validate_curriculum(config)

    def test_original_ctc_loss_backpropagates(self):
        model = ContentStudent(
            ContentStudentConfig(
                hidden=32,
                n_layers=1,
                n_heads=4,
                content_dim=16,
                structured_fsq=True,
                text_vocab_size=VOCAB_SIZE,
            )
        )
        batch = OriginalBatch(
            mel=torch.randn(1, 80, 12),
            input_lengths=torch.tensor([12]),
            transcripts=torch.tensor([1, 2, 3]),
            transcript_lengths=torch.tensor([3]),
        )
        loss = original_loss(
            model,
            batch,
            torch.device("cpu"),
            torch.nn.CTCLoss(blank=0, zero_infinity=True),
        )
        loss.backward()
        self.assertIsNotNone(model.text_head.weight.grad)
        self.assertGreater(model.text_head.weight.grad.abs().sum().item(), 0.0)

    def test_all_tiers_construct(self):
        for name in TIERS:
            tier = get_tier(name)
            model = ContentStudent(tier.model)
            self.assertEqual(model.config, tier.model)


if __name__ == "__main__":
    unittest.main()
