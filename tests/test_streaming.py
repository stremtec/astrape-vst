"""Streaming equivalence tests for encoder, decoder, and full pipeline."""

from __future__ import annotations

import unittest

import torch

from astrape.encoder import CausalContentEncoder, EncoderConfig
from astrape.decoder import CausalSynthesisDecoder, SynthesisDecoderConfig
from astrape.streaming_pipeline import StreamingVoiceConverter


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


if __name__ == "__main__":
    unittest.main()
