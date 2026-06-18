"""Streaming equivalence tests for encoder, decoder, and full pipeline."""

from __future__ import annotations

import unittest

import torch

from astrape.encoder import CausalContentEncoder, EncoderConfig
from astrape.decoder import CausalSynthesisDecoder, SynthesisDecoderConfig
from astrape.streaming_pipeline import StreamingVoiceConverter


def tiny_encoder_config() -> EncoderConfig:
    return EncoderConfig(
        mel_dim=4,
        content_dim=8,
        frontend_dim=4,
        frontend_kernel=1,
        convnext_kernel=1,
        n_convnext_blocks=0,
        transformer_dim=4,
        transformer_heads=1,
        transformer_layers=1,
        transformer_ff_mult=1,
        transformer_window=2,
        fsq_levels=(3, 3),
    )


def tiny_decoder_config() -> SynthesisDecoderConfig:
    return SynthesisDecoderConfig(
        content_dim=4,
        condition_dim=3,
        sample_rate=4,
        content_rate=1,
        transformer_dim=4,
        transformer_heads=1,
        transformer_layers=1,
        transformer_ff_mult=1,
        transformer_window=2,
        resnet_blocks=1,
        resnet_kernel=1,
        resnet_dilations=(1,),
        stage_channels=(4,),
        upsample_factors=(2,),
        mrf_kernel_sizes=(1,),
        mrf_dilations=((1,),),
        output_kernel_size=1,
    )


class EncoderStreamingTests(unittest.TestCase):
    def setUp(self):
        self.config = EncoderConfig()
        self.model = CausalContentEncoder(self.config)
        self.model.eval()

    def test_streaming_matches_full(self):
        mel = torch.randn(1, 80, 20)
        with torch.no_grad():
            full = self.model(mel)

        state = None
        chunks = []
        for t in range(0, 20, 2):
            chunk = mel[:, :, t:t+2]
            out, state = self.model.forward_stream(chunk, state)
            if out.content.shape[-1] > 0:
                chunks.append(out.content)

        streamed = torch.cat(chunks, dim=-1)
        diff = (full.content[:, :, :streamed.shape[-1]] - streamed).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_flush_with_empty_mel(self):
        mel = torch.randn(1, 80, 5)
        state = None
        out, state = self.model.forward_stream(mel, state)
        # Flush with empty mel — should not crash
        empty = torch.empty(1, 80, 0)
        out2, state2 = self.model.forward_stream(empty, state, flush=True)
        self.assertIsNotNone(out2.content)

    def test_flush_emits_pending_frame(self):
        # 11 mel frames → 5 pairs + 1 pending
        mel = torch.randn(1, 80, 11)
        state = None
        out, state = self.model.forward_stream(mel, state)
        self.assertEqual(out.content.shape[-1], 5)
        self.assertIsNotNone(state.pending_frame)

        empty = torch.empty(1, 80, 0)
        out2, state2 = self.model.forward_stream(empty, state, flush=True)
        self.assertEqual(out2.content.shape[-1], 1)
        self.assertIsNone(state2.pending_frame)

    def test_flush_with_no_pending_emits_nothing(self):
        mel = torch.randn(1, 80, 10)
        state = None
        out, state = self.model.forward_stream(mel, state)
        self.assertIsNone(state.pending_frame)

        empty = torch.empty(1, 80, 0)
        out2, _ = self.model.forward_stream(empty, state, flush=True)
        self.assertEqual(out2.content.shape[-1], 0)

    def test_streaming_past_rope_initial_table_length(self):
        config = tiny_encoder_config()
        model = CausalContentEncoder(config)
        model.eval()
        old_table_len = config.transformer_window * 4
        mel = torch.randn(1, config.mel_dim, (old_table_len + 3) * 2)

        state = None
        chunks = []
        for t in range(mel.shape[-1]):
            out, state = model.forward_stream(mel[:, :, t:t+1], state)
            if out.content.shape[-1] > 0:
                chunks.append(out.content)

        streamed = torch.cat(chunks, dim=-1)
        self.assertEqual(state.cache_len, old_table_len + 3)
        self.assertEqual(streamed.shape[-1], old_table_len + 3)

    def test_full_forward_past_rope_initial_table_length(self):
        config = tiny_encoder_config()
        model = CausalContentEncoder(config)
        model.eval()
        old_table_len = config.transformer_window * 4
        mel = torch.randn(1, config.mel_dim, (old_table_len + 3) * 2)

        with torch.no_grad():
            out = model(mel)

        self.assertEqual(out.content.shape[-1], old_table_len + 3)

    def test_full_forward_pads_odd_mel_like_streaming_flush(self):
        config = tiny_encoder_config()
        model = CausalContentEncoder(config)
        model.eval()
        mel = torch.randn(1, config.mel_dim, 5)

        with torch.no_grad():
            full = model(mel)

        state = None
        streamed, state = model.forward_stream(mel, state)
        flushed, _ = model.forward_stream(
            torch.empty(1, config.mel_dim, 0), state, flush=True
        )
        combined = torch.cat((streamed.content, flushed.content), dim=-1)

        self.assertEqual(full.content.shape[-1], 3)
        torch.testing.assert_close(full.content, combined, atol=1e-5, rtol=1e-5)


class DecoderStreamingTests(unittest.TestCase):
    def setUp(self):
        self.config = SynthesisDecoderConfig()
        self.model = CausalSynthesisDecoder(self.config)
        self.model.eval()
        self.spk = torch.randn(1, 128)

    def test_streaming_matches_full(self):
        content = torch.randn(1, 6, 768)
        with torch.no_grad():
            full_audio = self.model(content, self.spk)

        state = None
        chunks = []
        for t in range(6):
            frame = content[:, t:t+1, :]
            with torch.no_grad():
                audio_chunk, state = self.model.forward_stream(frame, self.spk, state)
            chunks.append(audio_chunk)

        streamed = torch.cat(chunks, dim=-1)
        diff = (full_audio[:, :streamed.shape[-1]] - streamed).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_streaming_past_rope_initial_table_length(self):
        config = tiny_decoder_config()
        model = CausalSynthesisDecoder(config)
        model.eval()
        old_table_len = config.transformer_window * 4
        content = torch.randn(1, old_table_len + 3, config.content_dim)
        spk = torch.randn(1, config.condition_dim)

        state = None
        chunks = []
        for t in range(content.shape[1]):
            with torch.no_grad():
                audio_chunk, state = model.forward_stream(
                    content[:, t:t+1, :], spk, state,
                )
            chunks.append(audio_chunk)

        streamed = torch.cat(chunks, dim=-1)
        self.assertEqual(state.cache_len, old_table_len + 3)
        self.assertEqual(
            streamed.shape[-1],
            (old_table_len + 3) * config.samples_per_frame,
        )

    def test_full_forward_past_rope_initial_table_length(self):
        config = tiny_decoder_config()
        model = CausalSynthesisDecoder(config)
        model.eval()
        old_table_len = config.transformer_window * 4
        content = torch.randn(1, old_table_len + 3, config.content_dim)
        spk = torch.randn(1, config.condition_dim)

        with torch.no_grad():
            audio = model(content, spk)

        self.assertEqual(
            audio.shape[-1],
            (old_table_len + 3) * config.samples_per_frame,
        )

    def test_forward_stream_rejects_multi_frame_content(self):
        config = tiny_decoder_config()
        model = CausalSynthesisDecoder(config)
        model.eval()
        content = torch.randn(1, 2, config.content_dim)
        spk = torch.randn(1, config.condition_dim)

        with self.assertRaisesRegex(ValueError, "at most one content frame"):
            model.forward_stream(content, spk)

    def test_config_rejects_sample_rate_not_divisible_by_internal_rate(self):
        with self.assertRaisesRegex(ValueError, "divisible by internal_rate"):
            SynthesisDecoderConfig(sample_rate=44101)

    def test_config_rejects_resnet_block_dilation_mismatch(self):
        with self.assertRaisesRegex(ValueError, "resnet_blocks"):
            SynthesisDecoderConfig(resnet_blocks=2, resnet_dilations=(1,))

    def test_config_rejects_invalid_dropout(self):
        with self.assertRaisesRegex(ValueError, "dropout"):
            SynthesisDecoderConfig(dropout=1.0)

    def test_decoder_dropout_is_wired_to_transformer_layers(self):
        config = tiny_decoder_config()
        config = SynthesisDecoderConfig(
            **{**config.__dict__, "dropout": 0.25}
        )
        model = CausalSynthesisDecoder(config)
        self.assertEqual(model.transformer.layers[0].attn_drop.p, 0.25)
        self.assertEqual(model.transformer.layers[0].ffn_drop.p, 0.25)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.encoder = CausalContentEncoder()
        self.decoder = CausalSynthesisDecoder()
        self.encoder.eval()
        self.decoder.eval()
        self.embedding = torch.randn(128)

    def test_construction(self):
        vc = StreamingVoiceConverter(self.encoder, self.decoder, self.embedding)
        self.assertEqual(vc.input_sample_rate, 16000)
        self.assertEqual(vc.output_sample_rate, 44100)

    def test_process_and_flush(self):
        vc = StreamingVoiceConverter(self.encoder, self.decoder, self.embedding)
        # Feed enough audio for at least one content frame (need 2 mel frames = 832 samples min)
        waveform = torch.randn(1600)
        chunk = vc.process(waveform)
        self.assertGreaterEqual(chunk.output_samples, 0)

        # Flush should not crash
        flush_chunk = vc.flush()
        self.assertIsNotNone(flush_chunk.audio)

    def test_reset_allows_reuse(self):
        vc = StreamingVoiceConverter(self.encoder, self.decoder, self.embedding)
        waveform = torch.randn(1600)
        vc.process(waveform)
        vc.flush()

        vc.reset()
        chunk = vc.process(waveform)
        self.assertIsNotNone(chunk.audio)

    def test_dimension_mismatch_raises(self):
        bad_encoder = CausalContentEncoder(EncoderConfig(content_dim=256))
        bad_encoder.eval()
        with self.assertRaises(ValueError):
            StreamingVoiceConverter(bad_encoder, self.decoder, self.embedding)

    def test_mel_dimension_mismatch_raises(self):
        bad_encoder = CausalContentEncoder(EncoderConfig(mel_dim=64))
        bad_encoder.eval()
        with self.assertRaisesRegex(ValueError, "n_mels.*mel_dim"):
            StreamingVoiceConverter(bad_encoder, self.decoder, self.embedding)


if __name__ == "__main__":
    unittest.main()
