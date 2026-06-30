"""Encoder content distribution diagnostics.

Measures how our Q2D2 encoder's output distribution differs from the MioCodec
teacher's FSQ content embedding — per-dimension statistics, CKA, Procrustes,
and spectral analysis of the residual.

Usage (after encoder training finishes):
  .venv/bin/python -m astrape.encoding_gap \
      --encoder-ckpt /Volumes/UNTITLED/btrv5_checkpoints/striding_8l_200hz/striding_8l_200hz.best.pt \
      --data-dir data/mio_vctk_full_compact \
      --limit 500 --device mps
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

import numpy as np
import torch
import torch.nn.functional as F

from .data import MioCompactDataset, ContentCollator, split_by_speaker, move_batch
from .encoder import MCSTransQ2D2Config, MCSTransQ2D2
from .miocodec import load_mio, SAMPLE_RATE


def _load_encoder(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scfg = ck.get("config", {})
    known = set(MCSTransQ2D2Config.__dataclass_fields__.keys())
    scfg2 = {k: tuple(v) if isinstance(v, list) else v
             for k, v in scfg.items() if k in known}
    config = MCSTransQ2D2Config(**scfg2)
    model = MCSTransQ2D2(config).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    return model.eval(), config


@torch.no_grad()
def collect_representations(encoder, loader, device, limit):
    """Collect paired (teacher_content, student_content) over the dataset."""
    teacher_all, student_all = [], []
    count = 0
    for batch in loader:
        if count >= limit:
            break
        batch = move_batch(batch, device)
        output = encoder(batch.mel, padding_mask=batch.mask)
        student = output["projected"].permute(0, 2, 1)  # (B, T, 768)
        teacher = batch.content                                # (B, L, 768)
        # align lengths
        T = min(student.shape[1], teacher.shape[1], batch.mask.shape[1])
        mask = batch.mask[:, :T]
        # only keep valid (non-padded) frames
        for i in range(student.shape[0]):
            valid = mask[i]
            student_all.append(student[i, :T][valid].cpu())
            teacher_all.append(teacher[i, :T][valid].cpu())
        count += student.shape[0]
    return torch.cat(teacher_all, dim=0), torch.cat(student_all, dim=0)


# ── Metric 1: Per-dimension statistics ──

def per_dim_stats(teacher: torch.Tensor, student: torch.Tensor):
    """Mean, std, skew per dimension for both distributions."""
    T, S = teacher.float(), student.float()
    rows = []
    for name, X in [("teacher", T), ("student", S)]:
        mu = X.mean(0)
        sigma = X.std(0)
        skew = ((X - mu).pow(3).mean(0)) / (sigma.pow(3) + 1e-8)
        rows.append({"name": name, "mu": mu, "sigma": sigma, "skew": skew})
    return rows


# ── Metric 2: Linear CKA (Centered Kernel Alignment) ──

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """CKA measures representational similarity invariant to orthogonal transform."""
    X, Y = X.float(), Y.float()
    X_c = X - X.mean(0, keepdim=True)
    Y_c = Y - Y.mean(0, keepdim=True)
    # HSIC via Gram matrices: tr(K_X K_Y) where K = XX^T
    Kx = X_c @ X_c.T  # (N, N)
    Ky = Y_c @ Y_c.T  # (N, N)
    hsic = lambda K1, K2: (K1 * K2).sum()
    return (hsic(Kx, Ky) / (hsic(Kx, Kx).sqrt() * hsic(Ky, Ky).sqrt())).item()


# ── Metric 3: Procrustes alignment ──

def procrustes(X: torch.Tensor, Y: torch.Tensor):
    """Best orthogonal R, scale s: min ||Y - s·X·R||  → R, s, residual."""
    X, Y = X.float(), Y.float()
    # center
    X_c = X - X.mean(0, keepdim=True)
    Y_c = Y - Y.mean(0, keepdim=True)
    # SVD of X^T Y
    U, S_diag, Vh = torch.linalg.svd(X_c.T @ Y_c, full_matrices=False)
    R = U @ Vh          # optimal rotation
    s = S_diag.sum() / (X_c.pow(2).sum() + 1e-8)
    aligned = s * (X_c @ R)
    residual = (Y_c - aligned).norm() / (Y_c.norm() + 1e-8)
    cos_after = F.cosine_similarity(aligned, Y_c, dim=-1).mean().item()
    return {"scale": s.item(), "residual_rel": residual.item(), "cos_after": cos_after}


# ── Metric 4: Covariance eigenvalue spectrum ──

def cov_spectrum(X: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """Top-k eigenvalues of the covariance matrix."""
    X_c = X.float() - X.float().mean(0, keepdim=True)
    cov = (X_c.T @ X_c) / (X_c.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    return eigvals.flip(0)[:top_k]


# ── Report ──

def print_report(teacher: torch.Tensor, student: torch.Tensor):
    N, D = teacher.shape
    print(f"\n{'='*60}")
    print(f"  ENCODING GAP ANALYSIS  ({N:,} frames × {D}d)")
    print(f"{'='*60}")

    # 0. Global cosine
    cos = F.cosine_similarity(teacher.float(), student.float(), dim=-1).mean().item()
    print(f"\n── Global cosine similarity: {cos:.4f}")

    # 1. Per-dimension stats
    stats = per_dim_stats(teacher, student)
    t, s = stats[0], stats[1]
    print(f"\n── Per-dimension statistics (averaged over {D} dims)")
    print(f"  {'':12s} {'mean':>8s} {'std':>8s} {'skew':>8s}")
    print(f"  {'teacher':12s} {t['mu'].mean():8.4f} {t['sigma'].mean():8.4f} {t['skew'].mean():8.4f}")
    print(f"  {'student':12s} {s['mu'].mean():8.4f} {s['sigma'].mean():8.4f} {s['skew'].mean():8.4f}")
    # std ratio per dim
    std_ratio = s['sigma'] / (t['sigma'] + 1e-8)
    print(f"  std_ratio (student/teacher): mean={std_ratio.mean():.3f}  "
          f"min={std_ratio.min():.3f}  max={std_ratio.max():.3f}")

    # 2. CKA
    cka = linear_cka(teacher, student)
    print(f"\n── Linear CKA: {cka:.4f}  (1.0 = identical subspaces)")

    # 3. Procrustes
    pro = procrustes(teacher, student)
    print(f"\n── Procrustes alignment")
    print(f"  optimal scale s    = {pro['scale']:.4f}  (1.0 = same variance)")
    print(f"  cos after align    = {pro['cos_after']:.4f}  (1.0 = perfect)")
    print(f"  relative residual  = {pro['residual_rel']:.4f}  (0.0 = perfect)")

    # 4. Covariance spectrum
    top_k = 20
    eig_t = cov_spectrum(teacher, top_k)
    eig_s = cov_spectrum(student, top_k)
    print(f"\n── Covariance eigenvalue spectrum (top {top_k})")
    print(f"  {'k':>3s} {'teacher':>10s} {'student':>10s} {'ratio':>8s}")
    for k in range(top_k):
        r = eig_s[k].item() / (eig_t[k].item() + 1e-8)
        print(f"  {k+1:3d} {eig_t[k].item():10.2f} {eig_s[k].item():10.2f} {r:8.3f}")

    # 5. Per-dim mean/std shift distribution
    mu_shift = (s['mu'] - t['mu']).abs()
    sigma_shift = (s['sigma'] - t['sigma']).abs()
    print(f"\n── Per-dimension shift magnitudes")
    print(f"  |Δmean|  : mean={mu_shift.mean():.5f}  p50={mu_shift.median():.5f}  "
          f"p95={mu_shift.quantile(0.95):.5f}  max={mu_shift.max():.5f}")
    print(f"  |Δstd|   : mean={sigma_shift.mean():.5f}  p50={sigma_shift.median():.5f}  "
          f"p95={sigma_shift.quantile(0.95):.5f}  max={sigma_shift.max():.5f}")

    print(f"\n{'='*60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=500,
                    help="Max batches to sample (more = more accurate)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mel-frames", type=int, default=200)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading encoder from {args.encoder_ckpt} ...", flush=True)
    encoder, config = _load_encoder(str(args.encoder_ckpt), args.device)

    print(f"Loading data from {args.data_dir} ...", flush=True)
    with np.load(args.data_dir / "meta.npz", allow_pickle=False) as meta:
        n_samples = int(meta["n_samples"])
        speakers = meta["spk_names"][:n_samples].astype(str)

    _, val_idx = split_by_speaker(speakers, args.val_fraction, args.seed)
    ds = MioCompactDataset(args.data_dir, val_idx, speakers)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=ContentCollator(args.mel_frames, args.seed))

    print(f"Collecting representations (limit={args.limit} batches) ...", flush=True)
    teacher, student = collect_representations(encoder, loader, device, args.limit)
    print(f"  collected {teacher.shape[0]:,} frames × {teacher.shape[1]}d", flush=True)

    print_report(teacher, student)


if __name__ == "__main__":
    main()
