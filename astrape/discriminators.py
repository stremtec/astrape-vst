"""Adversarial discriminators for Decoder v5 (HiFi-GAN MPD + MSD).

Both operate purely in the time domain — NO torch.stft — so they run on MPS
(where stft is unstable; the repo already computes MR-STFT *loss* on CPU). A
spectral Multi-Resolution-STFT discriminator can be added when/if training moves
to CUDA, for extra phase/harmonic fidelity.

LSGAN objective + feature matching (the feature-matching term is what carries
most of the perceptual signal in HiFi-GAN-style vocoders).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

wn = nn.utils.parametrizations.weight_norm


class PeriodDiscriminator(nn.Module):
    """Reshape 1D audio to (T/p, p) and run 2D convs — captures periodic structure."""

    def __init__(self, period: int, kernel: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        pad = (kernel - 1) // 2
        ch = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList([
            wn(nn.Conv2d(ch[i], ch[i + 1], (kernel, 1), (stride, 1), (pad, 0)))
            for i in range(4)
        ] + [wn(nn.Conv2d(1024, 1024, (kernel, 1), 1, (pad, 0)))])
        self.post = wn(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor):
        b, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
            t = x.shape[1]
        x = x.view(b, 1, t // self.period, self.period)
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class ScaleDiscriminator(nn.Module):
    """Raw-waveform 1D conv discriminator (one scale)."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            wn(nn.Conv1d(1, 128, 15, 1, padding=7)),
            wn(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            wn(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            wn(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            wn(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            wn(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            wn(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = wn(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class CombinedDiscriminator(nn.Module):
    """MPD (periods 2,3,5,7,11) + MSD (3 scales: raw, /2, /4)."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), n_scales: int = 3):
        super().__init__()
        self.mpd = nn.ModuleList([PeriodDiscriminator(p) for p in periods])
        self.msd = nn.ModuleList([ScaleDiscriminator() for _ in range(n_scales)])

    def forward(self, x: torch.Tensor):
        logits, fmaps = [], []
        for d in self.mpd:
            lg, fm = d(x)
            logits.append(lg)
            fmaps.append(fm)
        y = x
        for i, d in enumerate(self.msd):
            if i > 0:
                y = F.avg_pool1d(y.unsqueeze(1), 4, 2, padding=2).squeeze(1)
            lg, fm = d(y)
            logits.append(lg)
            fmaps.append(fm)
        return logits, fmaps


# ── LSGAN losses ──────────────────────────────────────────────────

def discriminator_loss(real_logits, fake_logits) -> torch.Tensor:
    loss = 0.0
    for r, f in zip(real_logits, fake_logits):
        loss = loss + ((r - 1.0).pow(2)).mean() + (f.pow(2)).mean()
    return loss


def generator_adv_loss(fake_logits) -> torch.Tensor:
    loss = 0.0
    for f in fake_logits:
        loss = loss + ((f - 1.0).pow(2)).mean()
    return loss


def feature_matching_loss(real_fmaps, fake_fmaps) -> torch.Tensor:
    loss = 0.0
    for rfm, ffm in zip(real_fmaps, fake_fmaps):
        for r, f in zip(rfm, ffm):
            loss = loss + F.l1_loss(f, r.detach())
    return loss


if __name__ == "__main__":
    disc = CombinedDiscriminator()
    n = sum(p.numel() for p in disc.parameters())
    real = torch.randn(2, 44100)
    fake = torch.randn(2, 44100, requires_grad=True)
    rl, rf = disc(real)
    fl, ff = disc(fake)
    d_loss = discriminator_loss(rl, fl)
    g_loss = generator_adv_loss(fl) + 2.0 * feature_matching_loss(rf, ff)
    g_loss.backward()
    print(f"discriminator params: {n/1e6:.2f}M (training-only, 0 inference cost)")
    print(f"sub-discriminators: {len(rl)}  d_loss={d_loss.item():.3f}  g_loss={g_loss.item():.3f}")
    print("OK: forward + backward through generator path works.")
