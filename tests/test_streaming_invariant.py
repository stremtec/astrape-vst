"""Streaming invariance tests for CausalConv1d and the full MCSTransQ2D2 encoder.

Verifies the fundamental causal property: output at time t must depend only on
input up to ≤ t.  Adding (or removing) future samples must not change earlier
output frames.  This catches causality bugs before they reach training.

Usage:
    python tests/test_streaming_invariant.py          # CPU, fast
    python tests/test_streaming_invariant.py --mps    # Apple Silicon
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Prevent pytest from collecting these — they're standalone test functions
# that take a torch.device argument, not pytest fixtures. Run with:
#   python tests/test_streaming_invariant.py
__test__ = False

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "MioCodec" / "src"))

from astrape.nn import CausalConv1d
from astrape.encoder import MCSTransQ2D2Config, MCSTransQ2D2

B, C_IN, C_OUT, T = 2, 80, 320, 100
TOL = 1e-4  # fp32 tolerance


# ─────────────────────────────────────────────
# 1. CausalConv1d unit test
# ─────────────────────────────────────────────

def test_causal_conv1d_future_noise(device: torch.device) -> None:
    """Adding noise beyond time t must not change earlier outputs."""
    for kernel, dilation, groups in [(5, 1, 1), (7, 3, 1), (7, 8, C_OUT)]:
        in_ch = C_OUT if groups > 1 else C_IN
        conv = CausalConv1d(in_ch, C_OUT, kernel, dilation=dilation, groups=groups).to(device)
        x = torch.randn(B, in_ch, T, device=device)
        x_noised = x.clone()
        # Corrupt the last 30 % of time steps
        x_noised[..., int(T * 0.7):] += 100.0 * torch.randn_like(x_noised[..., int(T * 0.7):])

        with torch.no_grad():
            y_clean = conv(x)
            y_noised = conv(x_noised)

        # Every time step up to 70 % must be unchanged (plus left_context for the conv)
        lc = conv.left_context
        check_up_to = int(T * 0.7) - lc - 1
        diff = (y_clean - y_noised)[..., :check_up_to].abs().max().item()
        assert diff < TOL, (
            f"CausalConv1d(k={kernel}, dil={dilation}, g={groups}): "
            f"future noise leaked backward, max |diff|={diff:.6f}"
        )
    print(f"  ✓ CausalConv1d future-noise invariance (k={kernel}, dil={dilation}, g={groups})",
          flush=True)


# ─────────────────────────────────────────────
# 2. Full encoder: truncation invariance
# ─────────────────────────────────────────────

def test_encoder_truncation_invariance(device: torch.device) -> None:
    """Adding mel frames beyond time T must not change content at earlier frames.

    Feeds the full mel sequence, then feeds a truncated sequence (first 60 %),
    and asserts identity (within fp tol) on the overlapping region.
    """
    config = MCSTransQ2D2Config(
        n_layers=2,            # small for test speed
        trans_dim=256,
        n_heads=4,
        ffn_dim=512,
        window=64,
        use_rope=False,
        use_swiglu=False,
        q2d2_dim=6,
        q2d2_levels=(7, 7, 7, 7, 7, 7),
    )
    model = MCSTransQ2D2(config).to(device).eval()
    mel_frames = 120
    mel = torch.randn(B, 80, mel_frames, device=device)

    # Padding mask: all valid (no padding)
    mask_full = torch.ones(B, mel_frames // 2, dtype=torch.bool, device=device)

    with torch.no_grad():
        out_full = model(mel, padding_mask=mask_full)

    # Truncated: only first 60 % of mel frames + corresponding mask
    trunc = int(mel_frames * 0.6)
    mel_trunc = mel[:, :, :trunc]
    mask_trunc = torch.ones(B, (trunc + 1) // 2, dtype=torch.bool, device=device)

    with torch.no_grad():
        out_trunc = model(mel_trunc, padding_mask=mask_trunc)

    # Compare on the overlapping content frames (~trunc/2 - a few frames of conv latency)
    # The encoder downsamples 2×, so content frames = trunc // 2.
    content_overlap = min(out_full["projected"].shape[-1],
                          out_trunc["projected"].shape[-1])
    # Allow 1 frame of edge debounce (first/last frame can differ from conv edge effects)
    compare = content_overlap - 1
    diff = (out_full["projected"][:, :, :compare] -
            out_trunc["projected"][:, :, :compare]).abs().max().item()
    assert diff < TOL, (
        f"Encoder truncation invariance failed: max |diff|={diff:.6f} "
        f"(trunc={trunc} mel frames → {content_overlap} content frames)"
    )
    print(f"  ✓ Encoder truncation invariance (compare {compare} content frames)",
          flush=True)


# ─────────────────────────────────────────────
# 3. Encoder: future padding invariance
# ─────────────────────────────────────────────

def test_encoder_future_padding_invariance(device: torch.device) -> None:
    """Appending zeros to mel must not change content at earlier frames."""
    config = MCSTransQ2D2Config(
        n_layers=2, trans_dim=256, n_heads=4, ffn_dim=512, window=64,
        use_rope=False, use_swiglu=False,
        q2d2_dim=6, q2d2_levels=(7, 7, 7, 7, 7, 7),
    )
    model = MCSTransQ2D2(config).to(device).eval()

    mel_frames = 100
    pad_frames = 40
    mel = torch.randn(B, 80, mel_frames, device=device)
    mel_padded = F.pad(mel, (0, pad_frames))  # pad future
    mask = torch.ones(B, mel_frames // 2, dtype=torch.bool, device=device)
    mask_padded = F.pad(mask, (0, pad_frames // 2 + (pad_frames % 2)), value=False)

    with torch.no_grad():
        out = model(mel, padding_mask=mask)
        out_padded = model(mel_padded, padding_mask=mask_padded)

    compare = out["projected"].shape[-1] - 1
    diff = (out["projected"][:, :, :compare] -
            out_padded["projected"][:, :, :compare]).abs().max().item()
    assert diff < TOL, (
        f"Encoder future-padding invariance failed: max |diff|={diff:.6f}"
    )
    print(f"  ✓ Encoder future-padding invariance ({pad_frames} extra frames)",
          flush=True)


# ─────────────────────────────────────────────
# 4. Conv stem: causality across dilations
# ─────────────────────────────────────────────

def test_stem_causality(device: torch.device) -> None:
    """The full conv stem (no transformer) must be strictly causal."""
    from astrape.encoder import DepthwiseResidualBlock, CellDownsample

    config = MCSTransQ2D2Config(
        n_layers=2, stem_block_type="depthwise",
        conv_dim=128,  # small for test
    )
    dim = config.conv_dim
    block_class = (DepthwiseResidualBlock
                   if config.stem_block_type == "depthwise"
                   else None)

    # Build stem in isolation (same as MCSTransQ2D2.__init__)
    input_conv = CausalConv1d(config.in_dim, dim, config.conv_kernel).to(device)
    blocks = torch.nn.ModuleList([
        block_class(dim, config.conv_kernel, d, config.dropout)
        for d in config.stem_dilations
    ]).to(device)
    skips = torch.nn.ModuleList([
        CausalConv1d(config.in_dim, dim, config.conv_kernel, dilation=d)
        for d in config.skip_dilations
    ]).to(device)
    skip_gates = torch.nn.ParameterList([
        torch.nn.Parameter(torch.full((1, dim, 1), -2.0, device=device))
        for _ in config.skip_dilations
    ])
    downsample = CellDownsample(dim).to(device)

    x = torch.randn(B, config.in_dim, 200, device=device)
    x_future = x.clone()
    x_future[..., 150:] = 0.0  # zero out future

    with torch.no_grad():
        def stem_forward(inp):
            h = F.silu(input_conv(inp))
            for b in blocks:
                h = b(h)
            for sk, g in zip(skips, skip_gates):
                h = h + torch.sigmoid(g) * F.silu(sk(inp))
            return downsample(h)

        y = stem_forward(x)
        y_future_zeroed = stem_forward(x_future)

    # Zeroing input at T>=150 affects outputs where the receptive window
    # extends past T=150.  After the 8-block stem (~212 frame left-context)
    # + stride-2 downsample, outputs j where 2j+1 >= 150 may differ.
    # Compare only early frames (j < 60, safe margin).
    compare = min(60, y.shape[-1], y_future_zeroed.shape[-1])
    diff = (y[:, :, :compare] - y_future_zeroed[:, :, :compare]).abs().max().item()
    assert diff < TOL, (
        f"Stem causality failed: future info leaked, max |diff|={diff:.6f}"
    )
    print(f"  ✓ Stem causality (depthwise, 8 blocks, compare {compare} frames)",
          flush=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mps", action="store_true")
    args = p.parse_args()

    device = torch.device("mps" if args.mps and torch.backends.mps.is_available() else "cpu")
    print(f"Streaming invariance tests  (device={device}, tol={TOL})", flush=True)

    test_causal_conv1d_future_noise(device)
    test_stem_causality(device)
    test_encoder_truncation_invariance(device)
    test_encoder_future_padding_invariance(device)

    print("All streaming invariance tests passed.", flush=True)


if __name__ == "__main__":
    main()
