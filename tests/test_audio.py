from __future__ import annotations

import unittest

import torch

from astrape.audio import StreamingLogMel


class StreamingLogMelTests(unittest.TestCase):
    def test_forward_stream_rejects_invalid_waveform_ndim(self):
        frontend = StreamingLogMel()
        with self.assertRaisesRegex(ValueError, "waveform must have shape"):
            frontend.forward_stream(torch.zeros(1, 1, 512))


if __name__ == "__main__":
    unittest.main()
