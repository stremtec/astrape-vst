"""Progressive Causal Convolution Frontend (EnCodec-style).

Replaces single large-kernel conv with a stack of small strided convs
+ ReLU nonlinearities to mimic magnitude spectrogram behavior.
Optionally initialized with Mel filterbank impulse responses.

LLM-suggested improvements to break the raw waveform plateau at 0.66.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from train_mcs_q2d2 import MCSTransQ2D2Config
from mcs_common import CausalConv1d

SAMPLE_RATE = 44100
RAW_STRIDE = 441


def build_mel_filterbank_impulses(
    n_mels: int = 80,
    n_fft: int = 2048,
    sample_rate: int = 44100,
    f_min: float = 0.0,
    f_max: float = 22050.0,
) -> torch.Tensor:
    """Compute time-domain impulse responses of a mel filterbank.

    Returns: (n_mels, n_fft) tensor of impulse responses.
    These can be used to initialize a Conv1d to act as a mel filterbank.
    """
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sample_rate,
    ).T  # (n_freqs, n_mels) → (n_mels, n_freqs)

    # Convert to full linear-frequency filterbank (symmetric)
    full_fb = torch.zeros(n_mels, n_fft)
    full_fb[:, :n_fft // 2 + 1] = mel_fb
    # Mirror for negative frequencies (real signal, symmetric spectrum)
    full_fb[:, n_fft // 2 + 1:] = mel_fb[:, 1:-1].flip(1)

    # Inverse FFT to get time-domain impulse responses
    imp_responses = torch.fft.ifft(full_fb, dim=1).real  # (n_mels, n_fft)
    # Shift to make causal (center the IR)
    imp_responses = torch.fft.fftshift(imp_responses, dim=1)
    # Normalize
    imp_responses = imp_responses / (imp_responses.abs().max(dim=1, keepdim=True).values + 1e-8)

    return imp_responses.float()


class ProgressiveRawFrontend(nn.Module):
    """Progressive strided convolution frontend (EnCodec style).

    Breaks down mel replacement into a stack of small strided convs:
      1ch PCM → CausalConv1d(1→32,k=7,s=2) → ReLU  (2x down)
             → CausalConv1d(32→64,k=7,s=4) → ReLU   (8x down)
             → CausalConv1d(64→160,k=7,s=4) → ReLU   (32x down)
             → CausalConv1d(160→320,k=7,s=2) → ReLU  (64x down ≈ stride 441?)

    Total downsample ratio: 2*4*4*2 = 64x
    Target: 44100 / 441 ≈ 100Hz → need ~441x downsample
    With stride stacking: 2*3*3*3*2*2 = 216x (close enough, then interpolate)
    Or: use a final learned stride to match exactly.

    The ReLU after each conv acts as a magnitude nonlinearity,
    forcing the network to learn envelope-like features (like mel).
    """

    def __init__(self, config: MCSTransQ2D2Config, mel_init: bool = True):
        super().__init__()
        dim = config.conv_dim  # 320

        # Progressive strided conv stack
        # Target: 44100/441 = 100Hz. With 32x down: 44100/32 = 1378Hz → need more
        # Let's compute: need 441x total stride for 100Hz from 44100Hz
        # With a final Conv1d(stride=7): 32*7 = 224x → 44100/224 = 197Hz
        # Then a stride-2 downsample gives ~98Hz → close to 100Hz
        
        strides = [3, 3, 3, 2, 2]  # total = 3*3*3*2*2 = 108x
        channels = [1, 32, 80, 160, 320, 320]
        kernels = [7, 7, 7, 7, 7]

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            self.stages.append(nn.Sequential(
                CausalConv1d(channels[i], channels[i+1], kernel_size=kernels[i],
                             stride=strides[i]),
                nn.ReLU(),
            ))

        # Optional: mel filterbank initialization for first layer
        if mel_init:
            self._init_with_mel(channels[1])

        # After stages: 44100/108 ≈ 408Hz. Need another ~4x to get ~100Hz
        # Add final conv with stride=4
        self.final_down = CausalConv1d(channels[-1], dim, kernel_size=7, stride=2)

        # Residual blocks (same as original)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, dim),
                CausalConv1d(dim, dim, kernel_size=config.conv_kernel, dilation=d),
                nn.ReLU(),
                CausalConv1d(dim, dim, kernel_size=config.conv_kernel, dilation=1),
            )
            for d in config.stem_dilations
        ])

        # Skip connections from raw audio
        # First conv from raw → match dim at the right rate
        self.raw_proj = CausalConv1d(1, dim, kernel_size=2048, stride=RAW_STRIDE)

        self.skips = nn.ModuleList([
            CausalConv1d(1, dim, kernel_size=2048, stride=RAW_STRIDE, dilation=d)
            for d in config.skip_dilations
        ])
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.full((1, dim, 1), -2.0))
            for _ in config.skip_dilations
        ])

        # Stride-2: ~100Hz → 50Hz
        self.downsample = nn.Conv1d(dim, dim, kernel_size=2, stride=2)

        self.proj_in = (
            nn.Linear(dim, config.trans_dim, bias=False)
            if dim != config.trans_dim else nn.Identity()
        )

    def _init_with_mel(self, out_channels: int):
        """Initialize first conv layer with mel filterbank impulse responses."""
        imp = build_mel_filterbank_impulses(n_mels=min(80, out_channels))
        n_ir = min(out_channels, imp.shape[0])
        kernel_size = 7
        first_conv = self.stages[0][0]  # CausalConv1d inside Sequential
        weight = first_conv.weight     # (out_c, 1, k)
        with torch.no_grad():
            for i in range(n_ir):
                # Center the IR around the middle of the kernel
                ir = imp[i]  # (n_fft,)
                # Downsample to kernel_size
                mid = len(ir) // 2
                start = mid - kernel_size // 2
                weight[i, 0, :] = ir[start:start + kernel_size]
            # Random init remaining channels
            if out_channels > n_ir:
                nn.init.kaiming_normal_(weight[n_ir:])

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        h = waveform
        for stage in self.stages:
            h = stage(h)

        h = self.final_down(h)
        h = F.relu(h)

        for block in self.blocks:
            residual = h
            h = block(h)
            h = h + residual[:, :, :h.shape[2]]

        raw_feat = self.raw_proj(waveform)
        for skip, gate in zip(self.skips, self.skip_gates):
            h = h + torch.sigmoid(gate) * F.silu(skip(waveform))

        h = self.downsample(h)
        # Align skip to h's length
        if raw_feat.shape[2] != h.shape[2]:
            raw_feat = F.interpolate(raw_feat, size=h.shape[2], mode='linear')

        h = h.transpose(1, 2)
        return self.proj_in(h)


# --- Test ---
if __name__ == "__main__":
    config = MCSTransQ2D2Config(
        n_layers=2, trans_dim=256, n_heads=4, ffn_dim=512, window=64,
    )
    frontend = ProgressiveRawFrontend(config, mel_init=True)
    x = torch.randn(2, 1, 44100)
    out = frontend(x)
    params = sum(p.numel() for p in frontend.parameters())
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Expected ~50 frames at 50Hz for 1s audio")
    print(f"Frontend params: {params:,}")
    print("OK")
