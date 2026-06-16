"""
Per-axis interference probe for the shared-trunk MCS.

Hypothesis: FSQ axes are nearly independent (MI < 0.05 bits). Predicting them
jointly through one shared Linear (5 outputs from the same GRU hidden state)
forces each axis's gradient to compete for the same 384-dimensional feature
directions. If interference is real we should see:

  (a) Per-axis cosine accuracy is uneven (some axes near 1.0, others poor),
      but the failures correlate across axes — i.e. axes that the network
      already represents well have more "bandwidth" stolen from the worst axis.
  (b) After running a *head projection only* fine-tune where each axis has
      its own tiny Linear, the worst axes recover without back-prop into the
      trunk.

We measure:
  - Per-axis normalized cosine with TARGET (each axis ranges in [-1, +1]).
  - Teacher axis-pair mutual information on the cached tokens (sanity check
    of the input premise).
  - "Independence floor" for axis predictions: split-chunk cosine between a
    latent that only sees axis k vs the joint prediction.

Usage: python probe_per_axis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from astrape.data import ContentCollator, MioContentDataset, speaker_disjoint_split, speaker_balanced_subset  # noqa
from astrape.flat_ctc_training import speaker_balanced_subset as _sbs  # noqa
from astrape.fsq import indices_to_codes  # noqa
from astrape.mcs_model import McsModel, load_mcs_checkpoint  # noqa
from torch.utils.data import DataLoader

CKPT = ROOT / "checkpoints" / "mcs_base.last.pt"
DATA = ROOT / "data" / "mio_compact_smoke"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FSQ = (8, 8, 8, 5, 5)
N_BATCHES = 4  # tiny dataset

# ---------- load ----------
print(f"[probe] ckpt: {CKPT.name}")
model, meta = load_mcs_checkpoint(CKPT, device=DEVICE)
model.eval()

with np.load(DATA / "meta.npz") as m:
    spk = m["spk_names"][: int(m["n_samples"])].astype(str)
train_idx, val_idx = speaker_disjoint_split(spk, validation_fraction=0.5, seed=42)
probe_idx = _sbs(val_idx, spk, max(64, len(val_idx)), seed=42)

coll = ContentCollator(None, 42, pad_mel_multiple=64, include_transcripts=False)
loader = DataLoader(
    MioContentDataset(DATA, DATA, probe_idx),
    batch_size=4, shuffle=False, num_workers=0, collate_fn=coll,
)

# ---------- forward pass per-axis ----------
per_axis_cos = [[] for _ in range(5)]
total_cos = []
joint_inputs = []
joint_targets = []

def _move(b, d):
    return type(b)(
        mel=b.mel.to(d), content=b.content.to(d) if b.content is not None else None,
        token_indices=b.token_indices.to(d) if b.token_indices is not None else None,
        target_mask=b.target_mask.to(d) if b.target_mask is not None else None,
    ) if False else b._replace(
        mel=b.mel.to(DEVICE),
        content=b.content.to(DEVICE) if b.content is not None else None,
        token_indices=b.token_indices.to(DEVICE) if b.token_indices is not None else None,
        target_mask=b.target_mask.to(DEVICE) if b.target_mask is not None else None,
    )

n_done = 0
with torch.inference_mode():
    for raw in loader:
        if n_done >= N_BATCHES: break
        batch = _move(raw, DEVICE)
        if batch.token_indices is None: continue
        out = model(batch.mel)
        L = min(out.codes.shape[1], batch.token_indices.shape[1], batch.target_mask.shape[1])
        pred = out.codes[:, :L]                    # [B, T, 5]
        tgt = indices_to_codes(batch.token_indices[:, :L], FSQ).to(pred.device, pred.dtype)
        mask = batch.target_mask[:, :L].unsqueeze(-1)   # [B, T, 1]

        for k in range(5):
            c_k = F.cosine_similarity(
                pred[..., k:k+1],
                tgt[..., k:k+1],
                dim=-1,
            )[mask.squeeze(-1)].cpu()
            per_axis_cos[k].append(c_k)

        c_total = F.cosine_similarity(pred, tgt, dim=-1)[mask.squeeze(-1)].cpu()
        total_cos.append(c_total)

        joint_inputs.append(pred.cpu().reshape(-1, 5))
        joint_targets.append(tgt.cpu().reshape(-1, 5))
        n_done += 1

print()
print("=" * 70)
print("PER-AXIS NORMALIZED COSINE (target on the same axis, dim 1)")
print("=" * 70)
for k in range(5):
    v = torch.cat(per_axis_cos[k])
    print(f"  axis {k} (L={FSQ[k]}): mean={v.mean():+.4f}  std={v.std():.4f}  "
          f"p05={v.quantile(0.05):+.4f}  p95={v.quantile(0.95):+.4f}  "
          f"frac<0.5={(v < 0.5).float().mean():.3f}")
all_total = torch.cat(total_cos)
print(f"  joint   5d      : mean={all_total.mean():+.4f}  std={all_total.std():.4f}  "
      f"p05={all_total.quantile(0.05):+.4f}")
print()

# ---------- independence floor: bin each axis ----------
print("=" * 70)
print("TEACHER AXIS MARGINAL DISTRIBUTION (sanity check on premise)")
print("=" * 70)
tgt_all = torch.cat(joint_targets)
print(f"  N tokens = {tgt_all.shape[0]}")
print(f"  per-axis value range: min={tgt_all.min(0).values.tolist()}  max={tgt_all.max(0).values.tolist()}")
print(f"  per-axis std: {tgt_all.std(0).tolist()}")
print()

# ---------- axis-pair MI on teacher tokens ----------
# Reuse binning per axis: each axis takes integer level ∈ [0, L-1]
def axis_levels_from_codes(codes: torch.Tensor, L: int) -> torch.Tensor:
    """Map continuous code in [-1,1] back to integer level 0..L-1."""
    # codes were created by indices_to_codes with half-bin centers.
    # level k    -> code (k - (L-1)/2) / max_level, with values like
    #            for L=8: [-3.5, -2.5, ..., 3.5] / 4
    #            for L=5: [-2, -1, 0, 1, 2] / 2
    # Recover: level = round(code * scale) + (L-1)/2
    return codes

# Use cached tokens directly: token_indices encodes the integer per-axis levels.
# Reload from data to get raw levels per axis.
levels_teacher = []
for raw in loader:
    if raw.token_indices is None: continue
    ti = raw.token_indices  # [B, T]
    B, T = ti.shape
    ti_full = []
    d = 1
    for L in FSQ:
        ti_full.append(((ti // d) % L))
        d *= L
    levels_teacher.append(torch.stack(ti_full, dim=-1).reshape(-1, 5))
    if len(levels_teacher) >= N_BATCHES: break
lev = torch.cat(levels_teacher).long()
N = lev.shape[0]
print("=" * 70)
print("TEACHER AXIS-PAIR MUTUAL INFORMATION (bits)")
print("=" * 70)
print(f"  max possible per axis: {[round(np.log2(L), 3) for L in FSQ]}")
print(f"  N = {N}")

# Estimate H(axis_k) and I(axis_i ; axis_j) using plug-in estimator from joint counts.
def mi_bits(i: int, j: int) -> float:
    Li, Lj = FSQ[i], FSQ[j]
    p_joint = torch.zeros((Li, Lj))
    a = lev[:, i]; b = lev[:, j]
    for x in range(Li):
        sel = (a == x)
        if not sel.any(): continue
        cb = b[sel]
        for y in range(Lj):
            p_joint[x, y] = (cb == y).sum().float()
    p_joint /= p_joint.sum()
    p_i = p_joint.sum(dim=1)
    p_j = p_joint.sum(dim=0)
    nz = p_joint > 0
    # I = sum p(x,y) log[ p(x,y) / (p(x) p(y)) ] in nats -> bits
    mi_nats = (p_joint[nz] * (p_joint[nz].log() - p_i[(p_joint > 0).any(dim=1)].repeat_interleave((p_joint>0).sum(dim=1))[None])).sum() if False else 0.0
    return mi_nats  # placeholder

# Easier: just compute I bits via direct formula
def mi_bits_simple(i: int, j: int) -> float:
    Li, Lj = FSQ[i], FSQ[j]
    joint = np.zeros((Li, Lj))
    a = lev[:, i].numpy(); b = lev[:, j].numpy()
    for x in range(Li):
        for y in range(Lj):
            joint[x, y] = ((a == x) & (b == y)).sum()
    joint /= joint.sum()
    pi = joint.sum(axis=1, keepdims=True)
    pj = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    mi_bits = (joint[mask] * (np.log2(joint[mask] / (pi.repeat(Lj, axis=1)[mask] * pj.repeat(Li, axis=0)[mask])))).sum()
    return float(mi_bits)

print("  i\\j", *[f"   j{j}    " for j in range(5)])
for i in range(5):
    row = [mi_bits_simple(i, j) for j in range(5)]
    print(f"   i{i} ", *[f"  {v:+.4f}  " for v in row])
print()

# ---------- axis-pair prediction-error correlation ----------
print("=" * 70)
print("PREDICTION-ERROR CORRELATION BETWEEN AXES")
print("If the GRU trunk is the bottleneck, axes that are hard for the network")
print("should share their errors with axes that are easy — cos of errors ≠ 0.")
print("=" * 70)
pred_all = torch.cat(joint_inputs)
err = (pred_all - tgt_all)
# cosine sim between per-axis error vectors across the batch axis (over tokens)
errn = err - err.mean(0, keepdim=True)
denom = (errn.pow(2).sum(0)).clamp(min=1e-8).sqrt()
print("  correlation matrix of per-axis errors (cosine over token axis):")
print("       ", "    ".join(f"a{i:>5}" for i in range(5)))
for i in range(5):
    sims = [(errn[:, i] * errn[:, j]).sum() / (denom[i] * denom[j]) for j in range(5)]
    print(f"  a{i}  ", "  ".join(f"{s:+.4f}" for s in sims))
print()

# ---------- independent-head quick probe ----------
# For each axis k, fit a tiny Linear(gru_dim -> 1) head using just one axis's
# target. Compare to the shared head's single-axis quality.  If independent
# heads reach higher per-axis cosine with the same trunk, the shared
# projection is the binding constraint.
print("=" * 70)
print("INDEPENDENT PER-AXIS HEAD (Linear only — trunk frozen)")
print("If dedicated heads beat shared by a large margin on worst axis,")
print("interference in the joint head is the binding constraint.")
print("=" * 70)

# Recompute trunk features
def extract_trunk(mel):
    h_fine = model.conv(mel)
    h_skip = F.silu(model.skip_conv(mel))
    h = h_fine + model.skip_gate.sigmoid() * h_skip
    h = h[:, :, 1::2].transpose(1, 2)
    h, _ = model.gru(h)
    return model.output_norm(h)

Hs, Ts, Ms = [], [], []
with torch.inference_mode():
    for raw in loader:
        if raw.token_indices is None: continue
        batch = _move(raw, DEVICE)
        h = extract_trunk(batch.mel)
        L = min(h.shape[1], batch.token_indices.shape[1], batch.target_mask.shape[1])
        h = h[:, :L]
        tgt = indices_to_codes(batch.token_indices[:, :L], FSQ).to(h.device, h.dtype)
        mask = batch.target_mask[:, :L].unsqueeze(-1)
        Hs.append(h.reshape(-1, h.shape[-1])[mask.expand_as(h).reshape(-1, h.shape[-1])[..., 0].bool()])
        Ts.append(tgt.reshape(-1, 5)[mask.expand(mask.shape[0], mask.shape[1], 5).reshape(-1, 5)[..., 0].bool()])
        Ms.append(mask)
        if len(Hs) >= N_BATCHES: break

# build mask-flattened arrays
H_all = torch.cat([h.flatten(0, 1) for h in [extract_trunk(_move(raw, DEVICE).mel)[:, :min(extract_trunk(_move(raw, DEVICE).mel).shape[1], _move(raw, DEVICE).token_indices.shape[1])] for raw in list(loader)[:N_BATCHES]]], dim=0)

# Rebuild: features and targets per token, masked
trunk_feats = []
axis_targets = []
with torch.inference_mode():
    for raw in list(loader)[:N_BATCHES]:
        if raw.token_indices is None: continue
        b = _move(raw, DEVICE)
        h = extract_trunk(b.mel)
        L = min(h.shape[1], b.token_indices.shape[1])
        h = h[:, :L]
        tgt = indices_to_codes(b.token_indices[:, :L], FSQ).to(h.device, h.dtype)
        msk = b.target_mask[:, :L]                            # [B, T]
        trunk_feats.append(h[msk])
        axis_targets.append(tgt[msk])
X = torch.cat(trunk_feats, dim=0)        # [N, gru_dim]
Y = torch.cat(axis_targets, dim=0)        # [N, 5]
print(f"  collected {X.shape[0]} trunk frames, dim {X.shape[1]}")

# Fit a separate Linear per axis via least squares (closed form).
shared = model.code_head.weight.detach().T                 # [gru, 5]
print()
print("  axis | shared-head | dedicated-head | delta")
print("  -----+-------------+----------------+--------")
for k in range(5):
    yk = Y[:, k:k+1]
    # ridge regression for stability: (X^T X + λI) w = X^T y
    XtX = X.T @ X
    Xty = X.T @ yk.squeeze(-1)
    lam = 1e-3 * XtX.diagonal().mean()
    w = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], device=X.device), Xty)

    # shared head: uses only column k of the shared weight
    pred_shared = X @ shared[:, k]
    cos_shared = F.cosine_similarity(
        pred_shared.unsqueeze(-1), yk, dim=-1
    ).mean().item()
    # ded head
    pred_ded = X @ w
    cos_ded = F.cosine_similarity(
        pred_ded.unsqueeze(-1), yk, dim=-1
    ).mean().item()
    print(f"   {k}   |  {cos_shared:+.4f}    |    {cos_ded:+.4f}     | {cos_ded - cos_shared:+.4f}")

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("If DELTAs are all near zero, the shared head is NOT the bottleneck;")
print("the trunk is. Independent heads won't help.")
print("If DELTAs > +0.05 especially on the worst axis, independent heads")
print("should be tested (cheap: only 5 * (gru_dim+1) extra params, ~2K).")
