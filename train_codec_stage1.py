#!/usr/bin/env python3
"""Stage 1: Train the MioStudentCodec encoder (FSQ distillation)."""

import argparse, time, random
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from astrape.causal_codec import (
    CausalCodecConfig, MioStudentCodec,
    save_codec_checkpoint, load_codec_checkpoint,
)
from astrape.data import (
    ContentBatch, ContentCollator, MioContentDataset, speaker_disjoint_split,
)
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.fsq import indices_to_codes, masked_fsq_cross_entropy
from astrape.training import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    p.add_argument("--projection-path", type=Path, default=Path("checkpoints/teacher_fsq_proj_out.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--run-name", default="codec_stage1")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--probe-samples", type=int, default=1024)
    p.add_argument("--full-val-every", type=int, default=5)
    p.add_argument("--target-cosine", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def _move(batch: ContentBatch, device: torch.device) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device), content=batch.content.to(device), pre_fsq=None,
        token_indices=batch.token_indices.to(device) if batch.token_indices is not None else None,
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
        transcripts=None, transcript_lengths=None,
    )


def encoder_loss(model, batch, axis_weights):
    if batch.token_indices is None:
        raise RuntimeError("FSQ tokens required")
    codes, content, logits = model.encode(batch.mel)
    L = min(codes.shape[1], batch.token_indices.shape[1], batch.target_mask.shape[1])
    pred = codes[:, :L]; mask = batch.target_mask[:, :L]
    tgt = indices_to_codes(batch.token_indices[:, :L], model.config.fsq_levels).to(
        device=pred.device, dtype=pred.dtype)

    # 5d cosine loss
    cos = F.cosine_similarity(pred, tgt, dim=-1)
    cos_loss = (1.0 - cos[mask]).mean()
    # Axis-weighted L1
    diff = (pred - tgt).abs()
    code_l1 = (diff * axis_weights.to(pred.device))[mask].mean()
    # Ordinal CE
    ord_loss, ord_acc, exact_acc = masked_fsq_cross_entropy(
        tuple(lg[:, :L, :].transpose(1,2) for lg in logits), batch.token_indices[:, :L],
        mask, model.config.fsq_levels,
    )
    loss = cos_loss + code_l1 + 0.1 * ord_loss
    return loss, {"cos_loss": cos_loss.item(), "code_l1": code_l1.item(),
                  "frame_cos": cos[mask].mean().item(), "ord_loss": ord_loss.item(),
                  "ord_acc": ord_acc.item(), "exact_acc": exact_acc.item()}


@torch.inference_mode()
def eval_encoder(model, loader, device):
    model.eval()
    cos5, cos768, ord_correct, exact_correct = [], [], [], []
    for raw in loader:
        b = _move(raw, device)
        if b.token_indices is None: continue
        codes, content, logits = model.encode(b.mel)
        L = min(codes.shape[1], b.token_indices.shape[1], b.target_mask.shape[1])
        mask = b.target_mask[:, :L]; pred = codes[:, :L]
        tgt = indices_to_codes(b.token_indices[:, :L], model.config.fsq_levels).to(
            device=pred.device, dtype=pred.dtype)
        cos5.append(F.cosine_similarity(pred, tgt, dim=-1)[mask].cpu())
        cos768.append(F.cosine_similarity(
            content[:,:,:L].transpose(1,2), b.content[:,:L], dim=-1)[mask].cpu())
        tgt_lvls = _token_to_levels(b.token_indices[:,:L], model.config.fsq_levels, device)
        pred_lvls = torch.stack([lg[:,:L,:].transpose(1,2).argmax(1) for lg in logits], dim=-1)
        correct = pred_lvls == tgt_lvls
        ord_correct.append(correct[mask].float().cpu())
        exact_correct.append(correct.all(dim=-1)[mask].float().cpu())
    c5 = torch.cat(cos5); c768 = torch.cat(cos768)
    ac = torch.cat(ord_correct); ec = torch.cat(exact_correct)
    return {"val_5d": c5.mean().item(), "val_5d_p05": c5.quantile(0.05).item(),
            "val_768": c768.mean().item(), "val_ord": ac.mean().item(),
            "val_exact": ec.mean().item()}


def _token_to_levels(tokens, levels, device):
    out, d = [], 1
    for L in levels: out.append(((tokens // d) % L).to(device)); d *= L
    return torch.stack(out, dim=-1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    with np.load(args.data_dir / "meta.npz") as m:
        if str(m["cache_format"].item()) != "compact-fp16-ctc-v2":
            raise SystemExit("requires compact-fp16-ctc-v2 cache")
        speakers = m["spk_names"][:int(m["n_samples"])].astype(str)
    tr_idx, vl_idx = speaker_disjoint_split(speakers, 0.15, args.seed)
    pr_idx = speaker_balanced_subset(vl_idx, speakers, args.probe_samples, args.seed)

    coll = ContentCollator(300, args.seed, history_mel_frames=100, pad_mel_multiple=64,
                           include_transcripts=False)
    tr_loader = DataLoader(
        MioContentDataset(args.data_dir, args.data_dir, tr_idx),
        batch_size=args.batch_size, shuffle=True, collate_fn=coll,
        generator=torch.Generator().manual_seed(args.seed),
    )
    pr_coll = ContentCollator(None, args.seed, pad_mel_multiple=64, include_transcripts=False)
    pr_loader = DataLoader(
        MioContentDataset(args.data_dir, args.data_dir, pr_idx),
        batch_size=args.batch_size, shuffle=False, collate_fn=pr_coll,
    )
    vl_loader = DataLoader(
        MioContentDataset(args.data_dir, args.data_dir, vl_idx),
        batch_size=args.batch_size, shuffle=False, collate_fn=pr_coll,
    )

    if args.resume:
        model, meta = load_codec_checkpoint(args.resume, device=device)
        start_epoch = int(meta["epoch"]) + 1
        best_probe = float(meta["metrics"].get("best_probe_5d", -1))
    else:
        model = MioStudentCodec(CausalCodecConfig(
        enc_dim=256, enc_dual_blocks=6, enc_ff_mult=2, enc_conv_blocks=3,
        dec_dim=384, dec_dilated_blocks=6, dec_prenet_blocks=3, dec_ff_mult=2,
    )).to(device)
        proj = torch.load(args.projection_path, map_location=device)
        model.load_fsq_projection(proj)
        start_epoch, best_probe = 0, -1.0

    # Stage 1: ONLY train encoder + FSQ. Decoder is frozen.
    for name, param in model.named_parameters():
        if name.startswith("dec_") or name.startswith("upsampler") or name.startswith("wave_head"):
            param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    enc_params = sum(p.numel() for p in trainable)
    print(f"Encoder+FSQ trainable: {enc_params:,} / {sum(p.numel() for p in model.parameters()):,} total",
          flush=True)
    opt = AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    if args.resume:
        p = torch.load(args.resume, map_location=device)
        opt.load_state_dict(p["optimizer_state_dict"])
        sched.load_state_dict(p["scheduler_state_dict"])
    aw = torch.tensor([1.0, 1.0, 1.0, 1.4, 1.5])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_p = args.output_dir / f"{args.run_name}.last.pt"
    best_p = args.output_dir / f"{args.run_name}.best.pt"
    pbest_p = args.output_dir / f"{args.run_name}.probe-best.pt"
    steps = min(args.steps_per_epoch, len(tr_loader))

    print(f"Train={len(tr_idx)} Probe={len(pr_idx)} Val={len(vl_idx)}", flush=True)
    print(f"Params={sum(p.numel() for p in model.parameters()):,} Device={device} "
          f"epochs={args.epochs}x{steps}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = {}
        started = time.perf_counter()
        for step, raw in enumerate(tr_loader, start=1):
            if step > steps: break
            b = _move(raw, device)
            loss, parts = encoder_loss(model, b, aw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            for k, v in parts.items(): totals[k] = totals.get(k, 0.0) + v
            if step % args.log_every == 0 or step == steps:
                n = step
                print(f"E{epoch:03d} step={step}/{steps} loss={totals.get('cos_loss',0)/n + totals.get('code_l1',0)/n:.4f} "
                      f"cos5={totals.get('frame_cos',0)/n:.4f} ord={totals.get('ord_acc',0)/n:.4f} "
                      f"exact={totals.get('exact_acc',0)/n:.4f} {(time.perf_counter()-started)/step:.3f}s/step",
                      flush=True)
        sched.step()

        probe = eval_encoder(model, pr_loader, device)
        metrics = {f"probe_{k}": v for k, v in probe.items()}
        metrics["train_5d"] = totals.get("frame_cos", 0) / steps
        metrics["best_probe_5d"] = max(best_probe, probe["val_5d"])

        full_due = (epoch+1) % args.full_val_every == 0 or epoch+1 == args.epochs
        if full_due:
            full = eval_encoder(model, vl_loader, device)
            metrics.update(full)
            metrics["best_full_5d"] = max(metrics.get("best_full_5d", -1), full["val_5d"])

        better = probe["val_5d"] > best_probe
        if better: best_probe = probe["val_5d"]
        save_codec_checkpoint(last_p, model, epoch=epoch, metrics=metrics, optimizer=opt, scheduler=sched)
        if better: save_codec_checkpoint(pbest_p, model, epoch=epoch, metrics=metrics)
        if full_due and metrics.get("val_5d", -1) > metrics.get("best_full_5d", -1):
            save_codec_checkpoint(best_p, model, epoch=epoch, metrics=metrics)

        print(f"E{epoch:03d} probe_5d={probe['val_5d']:.4f} p05={probe['val_5d_p05']:.4f} "
              f"768={probe['val_768']:.4f} ord={probe['val_ord']:.4f} "
              f"gap={max(0, args.target_cosine - probe['val_5d']):.4f}", flush=True)
        if full_due:
            print(f"E{epoch:03d} full_5d={full['val_5d']:.4f} 768={full['val_768']:.4f}", flush=True)


if __name__ == "__main__":
    main()
