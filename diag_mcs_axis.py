"""Per-axis diagnostic on mcs_skip.best.pt.

Goal: identify which FSQ axis is the bottleneck for val_768_cosine reaching 0.9.
For each of the 5 axes and each candidate model (best / probe-best), report:
  - mean predicted value (z-score space → FSQ level bin)
  - mean absolute error vs target FSQ bin (z-score space)
  - per-axis cosine in code space
  - per-axis variance
Then rank axes by per-axis cosine: lowest is bottleneck.

Outputs JSON + printed summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from astrape.data import ContentCollator, MioContentDataset, speaker_disjoint_split
from astrape.fsq import indices_to_codes, indices_to_level_indices
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.mcs_model import McsConfig, load_mcs_checkpoint
from astrape.mcs_training import _move


@torch.inference_mode()
def per_axis_diag(model, loader, device, max_batches: int = 0):
    model.eval()
    cos5_acc = []
    abs_err = []
    pred_vals = []
    tgt_vals = []
    bin_correct = []
    n = 0
    started = time.perf_counter()
    for i, raw_batch in enumerate(loader):
        if 0 < max_batches <= i:
            break
        batch = _move(raw_batch, device)
        out = model(batch.mel)
        L = min(out.codes.shape[1], batch.token_indices.shape[1], batch.target_mask.shape[1])
        mask = batch.target_mask[:, :L]
        pred = out.codes[:, :L]
        # scalar maps per problem with `indices_to_codes` (already used in train)
        tgt = indices_to_codes(batch.token_indices[:, :L], (8, 8, 8, 5, 5)).to(device=pred.device, dtype=pred.dtype)
        cos_all = F.cosine_similarity(pred, tgt, dim=-1)
        cos5_acc.append(cos_all[mask].cpu())
        abs_err.append((pred - tgt).abs()[mask].cpu())
        pred_vals.append(pred[mask].cpu())
        tgt_vals.append(tgt[mask].cpu())
        # ordinal correctness per axis
        # output.ordinal_logits: shape [B, total_levels, T] where total_levels = sum(8,8,8,5,5)
        splits = list((8, 8, 8, 5, 5))
        ord_logits = torch.split(out.ordinal_logits[:, :, :L], splits, dim=1)
        pred_idx = torch.stack([lg.argmax(1) for lg in ord_logits], dim=-1)
        # tokens → per-axis bin
        n_samples = pred_idx.numel() // pred_idx.shape[-1]
        per_ax = []
        for ax in range(5):
            per_ax.append((batch.token_indices[:, :L] // (splits[0]*splits[1]*splits[2]*splits[3]*splits[4])) % splits[ax] if False else None)
        # Simpler: use index_to_axes using fsq
        bin_correct_via_tokens = []
        # Compare pred_idx [B,T,5] vs per axis values from batch.token_indices
        # token_indices is a flat index encoding 5 axes via base-encoding.
        # Use indices_to_codes inverse with fsq from astrape.fsq if available
        from astrape.fsq import inverse_indices_to_codes
        try:
            tgt_codes = batch.token_indices[:, :L].clone()
            tgt_axes = inverse_indices_to_codes(tgt_codes, splits).to(device=pred.device)
        except Exception:
            # fallback: derive via index math
            # level product base:
            bases = [1]
            for L_ in splits:
                bases.append(bases[-1] * L_)
            bases = bases[1:]
            tgt_axes = torch.stack([((batch.token_indices[:, :L] // bases[ax]) % splits[ax]) for ax in range(5)], dim=-1).to(device=pred.device)
        bin_correct.append((pred_idx == tgt_axes)[mask].cpu())
        n += i + 1
    cos5 = torch.cat(cos5_acc)
    err = torch.cat(abs_err)
    pv = torch.cat(pred_vals)
    tv = torch.cat(tgt_vals)
    bin_acc = torch.cat(bin_correct).reshape(-1, 5)

    # per-axis summary
    out = {
        "overall_5d_cos": cos5.mean().item(),
        "overall_5d_p05": cos5.quantile(0.05).item(),
        "per_axis": {},
    }
    # cosine per axis ≈ 1 - <err_axis²> / 2 / (<pred²>+<tgt²>) since cos = <p·t>/(|p||t|)
    for ax in range(5):
        pa, ta = pv[:, ax], tv[:, ax]
        em = (pa - ta).abs().mean().item()
        # var of pred / tgt
        vp, vt = pa.var().item(), ta.var().item()
        bin_acc_ax = bin_acc[:, ax].mean().item()
        # 1d cosine
        c1 = (pa * ta).sum() / ((pa.pow(2).sum().sqrt()) * (ta.pow(2).sum().sqrt())).item()
        out["per_axis"][f"axis_{ax}"] = {
            "mae": em,
            "pred_var": vp,
            "tgt_var": vt,
            "cos_1d": c1,
            "bin_accuracy": bin_acc_ax,
            "level": [8, 8, 8, 5, 5][ax],
        }
    out["elapsed_s"] = time.perf_counter() - started
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/mcs_skip.best.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    p.add_argument("--device", default="mps")
    p.add_argument("--probe-samples", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("docs/research/mcs_skip_axis_diag.json"))
    args = p.parse_args()
    device = torch.device(args.device)
    with np.load(args.data_dir / "meta.npz") as meta:
        speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    tr_idx, val_idx = speaker_disjoint_split(speakers, 0.15, 42)
    probe_idx = speaker_balanced_subset(val_idx, speakers, args.probe_samples, 42)
    probe_coll = ContentCollator(None, 42, pad_mel_multiple=64, include_transcripts=False)
    loader = DataLoader(
        MioContentDataset(args.data_dir, args.data_dir, probe_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=probe_coll,
    )
    model, _ = load_mcs_checkpoint(args.ckpt, device=device)
    r = per_axis_diag(model, loader, device, max_batches=args.max_batches)
    print(json.dumps(r, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
