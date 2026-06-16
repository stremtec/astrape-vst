#!/usr/bin/env python3
"""Linear-ceiling probe for the 5D-FSQ content student.

Question: with NO temporal modelling (just one ridge per axis), how well
can mel@50Hz -> 5D FSQ code@25Hz be fitted on cached VCTK?

If linear ceiling ~= current model val_5d_cosine (~0.83):
   -> capacity is NOT the bottleneck; training (pretraining, axis-weighting)
      is what raises the score.
If linear ceiling << current model val_5d_cosine:
   -> we've captured most of the temporally-available information already.

We also run a "rank-augmented" linear probe (PCA-augmented design matrix of
mel) to chart the upper envelope as we add linear capacity r = 5..200.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from astrape.data import speaker_disjoint_split
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.fsq import indices_to_codes

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FSQ = (8, 8, 8, 5, 5)
DATA = ROOT / "data" / "mio_vctk_full_compact"
PROJ = ROOT / "checkpoints" / "teacher_fsq_proj_out.pt"


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--val-samples",   type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ridge-lam", type=float, default=1e-3)
    return p.parse_args()


def load_meta():
    with np.load(DATA / "meta.npz") as m:
        spk = m["spk_names"][: int(m["n_samples"])].astype(str)
    return spk


def load_concat(idx_list):
    Xs, Ycodes25, Yembs = [], [], []
    n_skipped = 0
    for i in idx_list:
        try:
            with np.load(DATA / f"s_{int(i):05d}.npz") as c:
                mel = c["logmel"].astype(np.float32)             # [80, T50]
                ct  = c["ct"].astype(np.int64)                    # [T25]
                ce  = c["ce_768"].astype(np.float32)              # [T25, 768]
            T25 = min(ct.shape[0], ce.shape[0])
            T50 = min(mel.shape[1], T25 * 2)
            if T25 < 4: n_skipped += 1; continue
            Xs.append(torch.from_numpy(mel[:, :T50]).T)             # [T50, 80]
            Ycodes25.append(indices_to_codes(torch.from_numpy(ct[:T25]), FSQ).float())  # [T25, 5]
            Yembs.append(torch.from_numpy(ce[:T25]))                                      # [T25, 768]
        except Exception:
            n_skipped += 1
    return Xs, Ycodes25, Yembs, n_skipped


def main():
    args = parse_args()
    proj = torch.load(PROJ, map_location="cpu")
    W = proj["weight"].cpu()  # [5, 768]
    b = proj["bias"].cpu()    # [768]

    spk = load_meta()
    train_idx, val_idx = speaker_disjoint_split(spk, 0.5, args.seed)
    train_idx = speaker_balanced_subset(train_idx, spk, args.train_samples, args.seed)
    val_idx   = speaker_balanced_subset(val_idx,   spk, args.val_samples,   args.seed)
    print(f"[probe] train={len(train_idx)} val={len(val_idx)}")

    print("[probe] loading train cache ...")
    Xtr_list, Ytr_code25, _, sk = load_concat(train_idx)
    print(f"[probe] skipped {sk}; train n_files={len(Xtr_list)}")
    print("[probe] loading val cache ...")
    Xv_list, Yv_code25, Yv_emb768, sk2 = load_concat(val_idx)
    print(f"[probe] skipped {sk2}; val n_files={len(Xv_list)}")

    # 25Hz -> 50Hz by repeating: each 25Hz target aligns to two consecutive mel frames.
    def repeat25to50(T25_5d):
        return T25_5d.repeat_interleave(2, dim=0)

    def trim(X50, Y50):
        L = min(X50.shape[0], Y50.shape[0])
        return X50[:L], Y50[:L]

    Xtr_g = []; Ytr_g = []
    for X50, Y25 in zip(Xtr_list, Ytr_code25):
        Y50 = repeat25to50(Y25)
        x_, y_ = trim(X50, Y50)
        Xtr_g.append(x_); Ytr_g.append(y_)
    Xtr_g = torch.cat(Xtr_g, dim=0)
    Ytr_g = torch.cat(Ytr_g, dim=0)
    print(f"[probe] train tensors X={tuple(Xtr_g.shape)} Y={tuple(Ytr_g.shape)}")

    mu = Xtr_g.mean(0, keepdim=True)
    sd = Xtr_g.std(0, keepdim=True).clamp(min=1e-3)
    Xtr_n = (Xtr_g - mu) / sd

    pairs_v = []
    for X50, Y25 in zip(Xv_list, Yv_code25):
        Y50 = repeat25to50(Y25)
        x_, y_ = trim(X50, Y50)
        pairs_v.append((x_, y_, Yv_emb768[Xv_list.index(X50)]) if False else None)  # placeholder
    # rebuild properly:
    pairs_v = []
    Yv_emb768_norm = []
    for X50, Y25, emb in zip(Xv_list, Yv_code25, Yv_emb768):
        Y50 = repeat25to50(Y25)
        x_, y_ = trim(X50, Y50)
        L25 = emb.shape[0]
        L50_keep = min(x_.shape[0], y_.shape[0], L25 * 2)
        pairs_v.append((x_, y_))
        Yv_emb768_norm.append(emb[: L50_keep // 2])

    # Joint ridge regression: mel(80) -> 5d FSQ
    lam = args.ridge_lam
    XtX = Xtr_n.T @ Xtr_n + lam * torch.eye(80)
    Xty = Xtr_n.T @ Ytr_g
    Beta = torch.linalg.solve(XtX, Xty)  # [80, 5]
    print("[probe] fitted joint ridge Beta.")

    def eval_design(X_test, Y_target_50, W_5to768, b_5to768, emb25_list):
        # group into utterances, predict 25Hz
        all_p, all_t = [], []
        for (X50, Yt50), emb25 in zip(X_test, Y_target_50):
            Xn = (X50 - mu) / sd
            pred50 = Xn @ Beta
            L = min(pred50.shape[0], Yt50.shape[0], emb25.shape[0] * 2)
            all_p.append(pred50[:L:2]); all_t.append(Yt50[:L:2])
        P25 = torch.cat(all_p, dim=0); T25 = torch.cat(all_t, dim=0)
        c5 = F.cosine_similarity(P25, T25, dim=-1)
        P768 = P25 @ W_5to768.T + b_5to768
        T768 = T25 @ W_5to768.T + b_5to768
        c768 = F.cosine_similarity(P768, T768, dim=-1)
        return c5, c768

    print()
    print("="*70)
    print("PER-AXIS JOINT RIDGE (mel50 -> axis_k), val-set 5d cos")
    print("="*70)
    per_axis_cos = [[] for _ in range(5)]
    per_axis_mae = [[] for _ in range(5)]
    with torch.inference_mode():
        for (X50, Yt50) in pairs_v:
            Xn = (X50 - mu) / sd
            pred50 = Xn @ Beta
            L = min(pred50.shape[0], Yt50.shape[0])
            P50 = pred50[:L]; T50 = Yt50[:L]
            for k in range(5):
                cos_k = F.cosine_similarity(P50[:, k:k+1], T50[:, k:k+1], dim=-1)
                per_axis_cos[k].append(cos_k.cpu())
                per_axis_mae[k].append((P50[:, k] - T50[:, k]).abs().mean().item())
    for k in range(5):
        v = torch.cat(per_axis_cos[k])
        print(f"  axis {k} (L={FSQ[k]}): cos_mean={v.mean():+.4f}  p05={v.quantile(0.05):+.4f}  mae={np.mean(per_axis_mae[k]):+.4f}")

    c5, c768 = eval_design(pairs_v, [y for _, y in pairs_v], W, b, Yv_emb768_norm)
    print(f"\n  joint 5d cos  on val: mean={c5.mean():+.4f}  p05={c5.quantile(0.05):+.4f}")
    print(f"  frozen 5d->768d cos:  mean={c768.mean():+.4f}  p05={c768.quantile(0.05):+.4f}")
    # oracle consistency check
    oracle_vals = []
    for (X50, Yt50), emb25 in zip(pairs_v, Yv_emb768_norm):
        L = min(Yt50.shape[0], emb25.shape[0]*2)
        T25 = Yt50[:L:2]
        oracle = T25 @ W.T + b
        target = emb25[:T25.shape[0]]
        oracle_vals.append(F.cosine_similarity(oracle, target, dim=-1))
    oc = torch.cat(oracle_vals)
    print(f"  oracle 5d->768 from teacher ct vs ce_768 ground truth: {oc.mean():+.4f}  (should be ~1.0)")

    # ---------------------- rank-augmented upper bound ----------------------
    print()
    print("="*70)
    print("RANK ABLATION: ridge + PC head (capacity upper envelope)")
    print("="*70)
    Xc = Xtr_n - Xtr_n.mean(0, keepdim=True)
    Ux, Sx, Vxh = torch.linalg.svd(Xc, full_matrices=False)
    rank_grid = [5, 10, 25, 50, 100, 200]
    print(f"  {'rank':>4} | {'train MSE':>10} | {'val 5d cos':>12} | {'val 768d cos':>12}")
    print(f"  {'----':>4}-+-{'-'*10:>10}-+-{'-'*12:>12}-+-{'-'*12:>12}")
    mu_train = Xtr_n.mean(0)
    for r in rank_grid:
        # augmented design: [mel_norm_scaled, top-r PC scores (Sx-scaled)]
        Z_tr = Ux[:, :r] * Sx[:r] if r > 0 else torch.zeros((Ux.shape[0], 0))
        Xa = torch.cat([Xtr_n, Z_tr], dim=1)
        XtX = Xa.T @ Xa + 1e-3 * torch.eye(Xa.shape[1])
        Xty = Xa.T @ Ytr_g
        Beta2 = torch.linalg.solve(XtX, Xty)
        tr_pred = Xa @ Beta2
        tr_mse = (tr_pred - Ytr_g).pow(2).mean().item()
        all_p25, all_t25 = [], []
        with torch.inference_mode():
            for (X50, Yt50) in pairs_v:
                Xn = (X50 - mu) / sd
                Xcc = Xn - mu_train
                Zv = (Xcc @ Vxh[:, :r]) if r > 0 else torch.zeros((Xcc.shape[0], 0))
                Zv = Zv * Sx[:r].unsqueeze(0)
                Xa_v = torch.cat([Xn, Zv], dim=1)
                pred50 = Xa_v @ Beta2
                L = min(pred50.shape[0], Yt50.shape[0])
                all_p25.append(pred50[:L:2]); all_t25.append(Yt50[:L:2])
        P25 = torch.cat(all_p25, dim=0); T25 = torch.cat(all_t25, dim=0)
        c5v = F.cosine_similarity(P25, T25, dim=-1).mean().item()
        P768 = P25 @ W.T + b; T768 = T25 @ W.T + b
        c768v = F.cosine_similarity(P768, T768, dim=-1).mean().item()
        print(f"  {r:>4} | {tr_mse:>10.5f} | {c5v:>+12.4f} | {c768v:>+12.4f}")

    # Reference: existing best 5d_cos reported 0.835 (mcs_base 5 epoch).
    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print("If 'val 5d cos' @ r>=50 is roughly equal to, say, 0.85+ then")
    print("there is room to push past 0.82 with stronger models.")
    print("If 'val 5d cos' saturates < 0.5 even at r=200, the basic")
    print("single-frame mel->5d mapping is not the data, and the rest")
    print("of the 5d cos must come from temporal context (RNN/attention).")


if __name__ == "__main__":
    main()
