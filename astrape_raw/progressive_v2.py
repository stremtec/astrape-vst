"""Progressive Raw Frontend V2 — simple, trainable, testable.

EnCodec-style stacked strided convs with ReLU (magnitude-like nonlinearity).
No mel init for now — just focused on getting the architecture right.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mcs_common import CausalConv1d, ResidualConvBlock
from train_mcs_q2d2 import MCSTransQ2D2Config


class ProgressiveRawFrontendV2(nn.Module):
    """Stacked causal strided convs → ReLU → residual blocks → stride-2 → 50Hz.

    PCM (B,1,T) → 3× CausalConv1d(stride=3) → 4×4×2 downsample
                  → ResidualBlocks → stride-2 → 50Hz → proj_in → transformer.
    """

    def __init__(self, config: MCSTransQ2D2Config):
        super().__init__()
        dim = config.conv_dim  # 320

        # Stage 1: 1→64, stride=3 (44100→14700Hz)
        self.s1 = nn.Sequential(
            CausalConv1d(1, 64, kernel_size=7, stride=3), nn.ReLU())
        # Stage 2: 64→128, stride=3 (14700→4900Hz)
        self.s2 = nn.Sequential(
            CausalConv1d(64, 128, kernel_size=7, stride=3), nn.ReLU())
        # Stage 3: 128→256, stride=7 (4900→700Hz)
        self.s3 = nn.Sequential(
            CausalConv1d(128, 256, kernel_size=7, stride=7), nn.ReLU())
        # Stage 4: 256→dim, stride=7 (700→100Hz)
        self.s4 = nn.Sequential(
            CausalConv1d(256, dim, kernel_size=7, stride=7), nn.ReLU())
        # Stage 5: dim→dim, stride=2 (100→50Hz)
        self.s5 = nn.Sequential(
            CausalConv1d(dim, dim, kernel_size=3, stride=2), nn.ReLU())

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualConvBlock(dim, config.conv_kernel, d, config.dropout)
            for d in config.stem_dilations
        ])

        # Skip connections from raw audio (stride to match main path)
        total_stride = 3 * 3 * 7 * 7  # = 441, matches pre-s5
        self.skips = nn.ModuleList([
            CausalConv1d(1, dim, kernel_size=2048, stride=total_stride, dilation=d)
            for d in config.skip_dilations
        ])
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.full((1, dim, 1), -2.0))
            for _ in config.skip_dilations
        ])

        # s5 already added above, remove old s5 and downsample
        # No additional s5/downsample needed — s5 already does stride-2

        # Proj to transformer dim
        self.proj_in = (
            nn.Linear(dim, config.trans_dim, bias=False)
            if dim != config.trans_dim else nn.Identity()
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        h = self.s1(waveform)
        h = self.s2(h)
        h = self.s3(h)
        h = self.s4(h)

        for block in self.blocks:
            h = block(h)

        for skip, gate in zip(self.skips, self.skip_gates):
            s = F.silu(skip(waveform))
            if s.shape[2] != h.shape[2]:
                s = F.interpolate(s, size=h.shape[2], mode='linear')
            h = h + torch.sigmoid(gate) * s

        h = self.s5(h)
        h = h.transpose(1, 2)
        return self.proj_in(h)


# ── Smoke test ──
if __name__ == "__main__":
    config = MCSTransQ2D2Config(
        n_layers=2, trans_dim=256, n_heads=4, ffn_dim=512, window=64,
    )
    frontend = ProgressiveRawFrontendV2(config)
    # Test with 3s of audio
    x = torch.randn(2, 1, 132300)  # 3s @44.1kHz
    out = frontend(x)
    params = sum(p.numel() for p in frontend.parameters())
    # Expected: 44100 / (96*4*2) = 57.4Hz → 3s * 57.4 = 172 frames before stride-2
    # After stride-2: 86 frames at ~28.7Hz... hmm
    # Let me just check:
    print(f"Input: {x.shape} (3s) → Output: {out.shape}")
    print(f"Expected ~150 frames at 50Hz, ~75 at 25Hz")
    print(f"Got {out.shape[1]} frames → {out.shape[1] / 3:.1f}Hz")
    print(f"Frontend params: {params:,}")
