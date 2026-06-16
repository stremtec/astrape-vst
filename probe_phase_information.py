"""Information-theoretic test of the phase hypothesis.

User's claim: "waveform conv stem fundamentally changes the game by preserving phase".
Test design:
  1. From raw waveform @ 16kHz, derive three 50Hz representations:
        a) |STFT|           = magnitude only (phase-discarding)
        b) cos(∠STFT)       = phase cosine (analytic-signal cos)
        c) sin(∠STFT)       = phase sine
  2. Compute canonical correlation / mutual information between
     each 50Hz stream and the teacher 5D FSQ target, conditioned on mel.

If phase carries EXTRA information beyond magnitude/mel, then
  P(target | mel, phase) > P(target | mel)
  => correlation of phase residual with target should be NON-zero.

If phase carries NO extra info for the 5D target, then
  cond-CC(phase ; target | mel) ~ 0.

We use two cheap estimators:
  (A) Ridge regression: mel -> residual of (mel+phase) predictions
  (B) Mutual info proxy: KSG k-NN MI between phase and target given mel
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly, stft

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from astrape.data import speaker_disjoint_split
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.fsq import indices_to_codes


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_MAX = 200
MAX_SECONDS = 2.0
N_MEL = 80


def _coerce(p):
    s = str(p)
    return s[len("np.str_("):-1] if s.startswith("np.str_(") else s


def load_wave(path, max_sec):
    audio, sr = sf.read(_coerce(path), always_2d=False, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g)
    return torch.from_numpy(audio[: round(max_sec * 16000)].astype(np.float32))


def features_50hz(wav_16k: torch.Tensor, mel_module):
    """Return three 50Hz streams aligned with mel frames.
        wav_16k: [T] @16kHz
        Returns: (mag, cos, sin) each [F, N_FRAMES] @ 50Hz
    """
    n_fft = 1024
    hop = 320  # 16kHz / 50Hz = 320
    win = 800
    arr = wav_16k.numpy()
    f, t, Z = stft(arr, fs=16000, nperseg=n_fft, noverlap=n_fft - hop,
                   nfft=n_fft, window="hann", boundary=None, padded=False)
    # Z shape: [n_freqs, n_frames]
    mag = np.abs(Z)
    phs = np.angle(Z)
    cos_p = np.cos(phs)
    sin_p = np.sin(phs)
    # aggregate |STFT| into 80 mel bins same as torchaudio MelSpectrogram
    import torchaudio
    m = mel_module
    mel_t = m(torch.from_numpy(arr).unsqueeze(0)).squeeze(0)  # [80, T_mel]
    T_mel = mel_t.shape[-1]
    # crop / interpolate magnitude to T_mel @ 50Hz
    # mag is at hop=320 -> exactly 50Hz
    T_stft = mag.shape[-1]
    L = min(T_stft, T_mel)
    mag = mag[:, :L]
    cos_p = cos_p[:, :L]
    sin_p = sin_p[:, :L]
    # aggregate |STFT| into freq bins per mel bin (use same filterbank)
    mel_fb = mel_module.mel_scale.fb  # [n_mels, n_freqs]
    mel_fb_np = mel_fb.numpy()
    mel_amp = (mel_fb_np @ mag) + 1e-8
    mel_log = (mel_amp + 1e-8).log()
    # Aggregate cos/sin into per-mel complex sum, then take cos/sin of mean phase
    mag_w = mag + 1e-8
    cos_w = (mel_fb_np @ (cos_p * mag_w)) / (mel_fb_np @ mag_w + 1e-8)
    sin_w = (mel_fb_np @ (sin_p * mag_w)) / (mel_fb_np @ mag_w + 1e-8)
    # normalize
    norm = (cos_w ** 2 + sin_w ** 2 + 1e-8).sqrt()
    cos_w = cos_w / norm
    sin_w = sin_w / norm
    return mel_log.numpy().T, cos_w.T, sin_w.T  # [T, 80]


def main():
    data_dir = ROOT / "data" / "mio_vctk_full_compact"
    with np.load(data_dir / "meta.npz") as m:
        spk = m["spk_names"][: int(m["n_samples"])].astype(str)
        src = m["source_files"][: int(m["n_samples"])].astype(str)
    train_idx, val_idx = speaker_disjoint_split(spk, 0.15, 42)
    val_idx = speaker_balanced_subset(val_idx, spk, N_MAX, 42)

    import torchaudio.transforms as T
    # Build a mel module that matches MioCodec defaults: 80 bins, 16kHz,
    # n_fft=1024, hop=320, win=800, fmin=80, fmax=7600, log-mel with natural log.
    mel = T.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, win_length=800,
        n_mels=80, f_min=80.0, f_max=7600.0, power=2.0, normalized=False,
        mel_scale="htk",
    ).to(DEVICE)

    X_mel, X_cos, X_sin, Y = [], [], [], []
    print(f"[probe] loading {len(val_idx)} samples ...")
    for j, i in enumerate(val_idx):
        idx = int(i)
        try:
            wav = load_wave(src[idx], MAX_SECONDS)
        except Exception as e:
            continue
        if wav.numel() < 1600: continue
        wav = wav.to(DEVICE)
        mel_log, cos_p, sin_p = features_50hz(wav.cpu(), mel)
        with np.load(data_dir / f"s_{idx:05d}.npz") as c:
            codes = indices_to_codes(torch.from_numpy(c["ct"].astype(np.int64))).numpy()
        # Use SAME mel from cache for ground-truth comparison, but our mel should be identical
        L = min(mel_log.shape[0], codes.shape[0])
        X_mel.append(mel_log[:L])
        X_cos.append(cos_p[:L])
        X_sin.append(sin_p[:L])
        Y.append(codes[:L])
        if len(X_mel) >= N_MAX: break
    Xm = np.concatenate(X_mel, axis=0)
    Xc = np.concatenate(X_cos, axis=0)
    Xs = np.concatenate(X_sin, axis=0)
    Y = np.concatenate(Y, axis=0)
    print(f"[probe] total frames: mel={Xm.shape}, cos={Xc.shape}, sin={Xs.shape}, Y={Y.shape}")

    # ---------- (A) Ridge regression residue test ----------
    # Predict Y[:,k] from mel, then measure whether mel+phase reduces squared error.
    Xmel = torch.from_numpy(Xm).float()
    Xcos = torch.from_numpy(Xc).float()
    Xsin = torch.from_numpy(Xs).float()
    Y = torch.from_numpy(Y).float()
    # Standardize features
    for X in (Xmel, Xcos, Xsin):
        X -= X.mean(0, keepdim=True)
        X /= X.std(0, keepdim=True).clamp(min=1e-6)

    lam = 1e-3
    def ridge_resid(Xa, Xb, Yk):
        # Solve Yk = Xa w_a + Xb w_b by closed form and report residual variance reduction.
        # Stack: X = [Xa, Xb]
        X = torch.cat([Xa, Xb], dim=1)
        XtX = X.T @ X + lam * torch.eye(X.shape[1])
        Xty = X.T @ Yk
        w = torch.linalg.solve(XtX, Xty)
        resid_full = (Yk - X @ w).pow(2).mean().item()
        # baseline: mel only
        X = Xa
        XtX = X.T @ X + lam * torch.eye(X.shape[1])
        Xty = X.T @ Yk
        w = torch.linalg.solve(XtX, Xty)
        resid_mel = (Yk - X @ w).pow(2).mean().item()
        var_target = Yk.var().item()
        r2_mel = 1 - resid_mel / var_target
        r2_full = 1 - resid_full / var_target
        return r2_mel, r2_full, r2_full - r2_mel

    print("\n=== (A) Ridge R^2: mel only vs mel+phase ===")
    print("  axis | r2_mel | r2_full | delta")
    for k in range(5):
        r2m, r2f, d = ridge_resid(Xmel, torch.cat([Xcos, Xsin], dim=1), Y[:, k])
        print(f"   {k}   | {r2m:+.4f} | {r2f:+.4f} | {d:+.4f}")

    # ---------- (B) Per-axis correlation of phase residual after mel-regression ----------
    # Regress Y on mel, take residuals, then correlate residual with phase features.
    print("\n=== (B) Phase residual correlation AFTER mel-removal ===")
    # Y on mel
    X = Xmel
    H = X.T @ X + lam * torch.eye(X.shape[1])
    W = torch.linalg.solve(H, X.T @ Y)  # [80, 5]
    Yhat = X @ W
    Yres = Y - Yhat
    # Now correlate per-mel-bin phase with target residual
    # Xcos shape [N, 80], target_residual [N, 5]
    Xp = torch.cat([Xcos, Xsin], dim=1)
    # |CC_k| = max cosine over (cos, sin) features per axis after normalization
    Yres_n = (Yres - Yres.mean(0)) / (Yres.std(0).clamp(min=1e-6) * (Yres.shape[0] ** 0.5))
    Xp_n = (Xp - Xp.mean(0)) / (Xp.std(0).clamp(min=1e-6) * (Xp.shape[0] ** 0.5))
    CC = Xp_n.T @ Yres_n  # [160, 5]
    print(f"  per-axis max |CC| over 160 phase features (after mel-removal):")
    print(f"  axis | max|CC| | top-bin source")
    # report strongest bin
    for k in range(5):
        ab = CC[:, k].abs()
        v, j = ab.max(0)
        src = "cos" if j.item() < 80 else "sin"
        bin_idx = (j.item() % 80)
        print(f"   {k}   | {v.item():+.4f} | {src}-{bin_idx}")

    # ---------- (C) Global CCA between phase and mel-on-target residual ----------
    # PCA-rank-5 phase, PCA-rank-5 mel-on-target residual, compute cosine of canonical vecs.
    print("\n=== (C) Top-k cosine between phase PCs and target residual PCs ===")
    P = Xp  # [N, 160]
    U, S, Vh = torch.linalg.svd(P - P.mean(0), full_matrices=False)
    P5 = U[:, :5] * S[:5]  # [N, 5]
    U2, S2, Vh2 = torch.linalg.svd(Yres - Yres.mean(0), full_matrices=False)
    R5 = U2[:, :5] * S2[:5]
    # covariance & SVD
    Pc = P5 - P5.mean(0); Rc = R5 - R5.mean(0)
    C = Pc.T @ Rc / P5.shape[0]
    cos_mat = F.cosine_similarity(C.unsqueeze(1), C.unsqueeze(0), dim=-1)
    u, sv, vh = torch.linalg.svd(C)
    canon = sv
    # Normalize by sqrt of var on each side
    varP = (Pc ** 2).sum(0) / P5.shape[0]
    varR = (Rc ** 2).sum(0) / R5.shape[0]
    canon_n = canon / (varP.clamp(min=1e-6).sqrt() * varR.clamp(min=1e-6).sqrt())
    print(f"  raw canon-corr (phase5, target_resid5): {canon.tolist()}")
    print(f"  normalized canon-corr:                  {canon_n.tolist()}")

    # ---------- (D) Sanity: how much info is *in phase*, period (not conditioned) ----
    # Predict Y from phase only.
    print("\n=== (D) Ridge R^2: phase only vs mel only ===")
    print("  axis | r2_phase | r2_mel |")
    for k in range(5):
        X = Xmel
        XtX = X.T @ X + lam * torch.eye(X.shape[1]); w = torch.linalg.solve(XtX, X.T @ Y[:, k])
        r2mel = 1 - (Y[:, k] - X @ w).pow(2).mean().item() / Y[:, k].var().item()
        X = Xp
        XtX = X.T @ X + lam * torch.eye(X.shape[1]); w = torch.linalg.solve(XtX, X.T @ Y[:, k])
        r2ph = 1 - (Y[:, k] - X @ w).pow(2).mean().item() / Y[:, k].var().item()
        print(f"   {k}   | {r2ph:+.4f}  | {r2mel:+.4f}")


if __name__ == "__main__":
    main()
