"""Training loop for Minimal Causal Student – Scaled (MCSS)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .data import ContentBatch, ContentCollator, MioContentDataset, speaker_disjoint_split
from .flat_ctc_training import speaker_balanced_subset
from .fsq import indices_to_codes, masked_fsq_cross_entropy
from .mcss_model import McssConfig, McssModel, load_mcss_checkpoint, save_mcss_checkpoint
from .training import seed_everything


@dataclass(frozen=True)
class McssTrainingConfig:
    data_dir: Path
    projection_path: Path
    output_dir: Path
    run_name: str = "mcss"
    device: str = "mps"
    batch_size: int = 2
    epochs: int = 40
    steps_per_epoch: int = 1000
    learning_rate: float = 2e-4       # lower than MCS — attention sensitive
    scheduler_t_max: int = 40
    warm_restart_t0: int = 10        # shorter cycle, more frequent resets
    min_lr: float = 2e-5             # never decay to 0
    weight_decay: float = 5e-5      # light regularisation
    validation_fraction: float = 0.15
    supervised_mel_frames: int = 300
    history_mel_frames: int = 120     # wider history for attention
    pad_mel_multiple: int = 64
    ordinal_weight: float = 0.25
    delta_weight: float = 0.06
    gate_floor_weight: float = 0.1  # encourage residual gate ≥ gate_floor
    initial_axis_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 2.5, 3.0)  # focus on 5-level axes
    seed: int = 42
    num_workers: int = 0
    probe_samples: int = 1024
    full_validation_every: int = 5
    target_cosine: float = 0.90       # higher target than MCS (0.85)
    log_every: int = 50
    resume: Optional[Path] = None


def _move(batch: ContentBatch, device: torch.device) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device), content=batch.content.to(device), pre_fsq=None,
        token_indices=batch.token_indices.to(device) if batch.token_indices is not None else None,
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
        transcripts=None, transcript_lengths=None,
    )


def mcss_loss(output, batch, config, axis_weights, fsq_levels,
              gate_floor: float = 0.03, epoch: int = 0, total_epochs: int = 40):
    if batch.token_indices is None:
        raise RuntimeError("MCSS training requires cached FSQ tokens")
    L = min(output.codes.shape[1], batch.token_indices.shape[1], batch.target_mask.shape[1])
    pred = output.codes[:, :L]
    mask = batch.target_mask[:, :L]
    tgt = indices_to_codes(batch.token_indices[:, :L], fsq_levels).to(
        device=pred.device, dtype=pred.dtype)

    # 5d cosine loss per axis + mean
    cos = F.cosine_similarity(pred, tgt, dim=-1)
    per_axis_cos = F.cosine_similarity(
        pred[mask], tgt[mask], dim=0)  # [5] per-axis cosine
    cos_loss = (1.0 - cos[mask]).mean()

    # Weighted L1 per axis
    diff = (pred - tgt).abs()
    code_l1 = (diff * axis_weights.to(pred.device))[mask].mean()

    # Ordinal CE
    ord_loss, ord_acc, exact_acc = masked_fsq_cross_entropy(
        tuple(torch.split(output.ordinal_logits[:, :, :L], list(fsq_levels), dim=1)),
        batch.token_indices[:, :L], mask, fsq_levels,
    )

    # Delta (phonetic smoothness)
    if L >= 2:
        pd = pred[:, 1:] - pred[:, :-1]
        td = tgt[:, 1:] - tgt[:, :-1]
        pm = mask[:, 1:] & mask[:, :-1]
        delta = F.smooth_l1_loss(pd[pm], td[pm]) if pm.any() else pred.sum() * 0
    else:
        delta = pred.sum() * 0

    loss = cos_loss + code_l1 + config.ordinal_weight * ord_loss + config.delta_weight * delta

    # 768d auxiliary — decay across training: scaffold early, 5d focus late
    aux_weight = 0.10 * max(0.1, 1.0 - epoch / total_epochs)
    proj_768 = output.projected[:, :, :L].transpose(1, 2)
    tgt_768 = batch.content[:, :L]
    cos_768 = F.cosine_similarity(proj_768[mask], tgt_768[mask], dim=-1)
    loss = loss + (1.0 - cos_768).mean() * aux_weight

    # Residual gate: floor penalty below gate_floor, light L2 above
    below = F.relu(gate_floor - output.residual_gate)
    above = F.relu(output.residual_gate - gate_floor)
    loss = loss + below.pow(2) * config.gate_floor_weight + above.pow(2) * 0.001

    per_axis_dict = {f"cos_ax{i}": per_axis_cos[i].item() for i in range(len(per_axis_cos))}
    return loss, {
        "cos_loss": cos_loss.item(), "code_l1": code_l1.item(),
        "frame_cosine": cos[mask].mean().item(), "cos_768": cos_768.mean().item(),
        "ord_loss": ord_loss.item(), "ord_accuracy": ord_acc.item(),
        "exact_accuracy": exact_acc.item(), "delta_loss": delta.item(),
        "residual_gate": output.residual_gate.item(),
        "aux_weight": aux_weight,
        **per_axis_dict,
    }


@torch.inference_mode()
def evaluate_mcss(model, loader, device):
    model.eval()
    cos5, cos768, ord_correct, exact_correct = [], [], [], []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        if batch.token_indices is None:
            continue
        out = model(batch.mel)
        L = min(out.codes.shape[1], batch.token_indices.shape[1], batch.target_mask.shape[1])
        mask = batch.target_mask[:, :L]
        pred = out.codes[:, :L]
        tgt = indices_to_codes(batch.token_indices[:, :L], (8, 8, 8, 5, 5)).to(
            device=pred.device, dtype=pred.dtype)
        cos5.append(F.cosine_similarity(pred, tgt, dim=-1)[mask].cpu())
        cos768.append(F.cosine_similarity(
            out.projected[:, :, :L].transpose(1, 2), batch.content[:, :L], dim=-1,
        )[mask].cpu())
        tgt_lvls = _token_to_levels(batch.token_indices[:, :L], (8, 8, 8, 5, 5), device)
        ord_split = torch.split(out.ordinal_logits[:, :, :L], list((8, 8, 8, 5, 5)), dim=1)
        pred_lvls = torch.stack([lg.argmax(dim=1) for lg in ord_split], dim=-1)
        correct = pred_lvls == tgt_lvls
        ord_correct.append(correct[mask].float().cpu())
        exact_correct.append(correct.all(dim=-1)[mask].float().cpu())
    c5 = torch.cat(cos5)
    c768 = torch.cat(cos768)
    ac = torch.cat(ord_correct)
    ec = torch.cat(exact_correct)
    return {
        "val_5d_cosine": c5.mean().item(), "val_5d_p05": c5.quantile(0.05).item(),
        "val_768_cosine": c768.mean().item(), "val_768_p05": c768.quantile(0.05).item(),
        "val_ordinal_accuracy": ac.mean().item(), "val_exact_accuracy": ec.mean().item(),
    }


def _token_to_levels(tokens, levels, device):
    out, d = [], 1
    for L in levels:
        out.append(((tokens // d) % L).to(device))
        d *= L
    return torch.stack(out, dim=-1)


def train_mcss(model_config: McssConfig, train_config: McssTrainingConfig) -> None:
    seed_everything(train_config.seed)
    device = torch.device(train_config.device)

    with np.load(train_config.data_dir / "meta.npz") as meta:
        if str(meta["cache_format"].item()) != "compact-fp16-ctc-v2":
            raise ValueError("MCSS requires compact-fp16-ctc-v2 cache")
        speakers = meta["spk_names"][:int(meta["n_samples"])].astype(str)
    train_idx, val_idx = speaker_disjoint_split(
        speakers, train_config.validation_fraction, train_config.seed)
    probe_idx = speaker_balanced_subset(
        val_idx, speakers, train_config.probe_samples, train_config.seed)

    coll = ContentCollator(
        train_config.supervised_mel_frames, train_config.seed,
        history_mel_frames=train_config.history_mel_frames,
        pad_mel_multiple=train_config.pad_mel_multiple,
        include_transcripts=False,
    )
    train_loader = DataLoader(
        MioContentDataset(train_config.data_dir, train_config.data_dir, train_idx),
        batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, collate_fn=coll,
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    probe_coll = ContentCollator(
        None, train_config.seed, pad_mel_multiple=train_config.pad_mel_multiple,
        include_transcripts=False,
    )
    probe_loader = DataLoader(
        MioContentDataset(train_config.data_dir, train_config.data_dir, probe_idx),
        batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, collate_fn=probe_coll,
    )
    full_loader = DataLoader(
        MioContentDataset(train_config.data_dir, train_config.data_dir, val_idx),
        batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, collate_fn=probe_coll,
    )

    if train_config.resume is not None:
        model, metadata = load_mcss_checkpoint(train_config.resume, device=device)
        start_epoch = int(metadata["epoch"]) + 1
        best_probe = float(metadata["metrics"].get("best_probe_5d_cosine", -1))
        best_full = float(metadata["metrics"].get("best_full_5d_cosine", -1))
    else:
        model = McssModel(model_config).to(device)
        proj = torch.load(train_config.projection_path, map_location=device)
        model.load_fsq_projection(proj)
        start_epoch = 0
        best_probe = -1.0
        best_full = -1.0

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    sched = CosineAnnealingWarmRestarts(
        opt, T_0=train_config.warm_restart_t0, T_mult=1,
        eta_min=train_config.min_lr,
    )
    if train_config.resume is not None:
        p = torch.load(train_config.resume, map_location=device)
        opt.load_state_dict(p["optimizer_state_dict"])
        # Warm restart: fresh scheduler from cycle 0 at full lr

    axis_weights = torch.tensor(train_config.initial_axis_weights, dtype=torch.float32)
    train_config.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = train_config.output_dir / f"{train_config.run_name}.last.pt"
    best_path = train_config.output_dir / f"{train_config.run_name}.best.pt"
    pbest_path = train_config.output_dir / f"{train_config.run_name}.probe-best.pt"
    steps = min(train_config.steps_per_epoch, len(train_loader))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Train={len(train_idx)} Probe={len(probe_idx)} FullVal={len(val_idx)}", flush=True)
    print(f"Params={n_params:,} Device={device} "
          f"epochs={train_config.epochs}x{steps} target_5d={train_config.target_cosine}",
          flush=True)

    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        totals: dict[str, float] = {}
        started = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            if step > steps:
                break
            batch = _move(raw_batch, device)
            out = model(batch.mel)
            loss, parts = mcss_loss(out, batch, train_config, axis_weights,
                                    model.config.fsq_levels, model.config.gate_floor,
                                    epoch=epoch, total_epochs=train_config.epochs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            if step % train_config.log_every == 0 or step == steps:
                n = step
                ax_str = " ".join(f"{totals.get(f'cos_ax{i}',0)/n:.3f}" for i in range(5))
                print(f"E{epoch:03d} step={step}/{steps} "
                      f"loss={totals.get('cos_loss',0)/n + totals.get('code_l1',0)/n:.4f} "
                      f"cos5={totals.get('frame_cosine',0)/n:.4f} "
                      f"ax=[{ax_str}] "
                      f"cos768={totals.get('cos_768',0)/n:.4f} "
                      f"ord={totals.get('ord_accuracy',0)/n:.4f} "
                      f"gate={totals.get('residual_gate',0)/n:.4f} "
                      f"auxw={totals.get('aux_weight',0)/n:.3f} "
                      f"{(time.perf_counter()-started)/step:.3f}s/step", flush=True)
        sched.step()

        probe = evaluate_mcss(model, probe_loader, device)
        metrics = {f"probe_{k}": v for k, v in probe.items()}
        metrics.update({
            "train_5d_cosine": totals.get("frame_cosine", 0) / steps,
            "best_probe_5d_cosine": max(best_probe, probe.get("val_5d_cosine", -1)),
        })
        full_due = ((epoch + 1) % train_config.full_validation_every == 0
                    or epoch + 1 == train_config.epochs)
        if full_due:
            full = evaluate_mcss(model, full_loader, device)
            metrics.update(full)
            metrics["best_full_5d_cosine"] = max(best_full, full.get("val_5d_cosine", -1))

        probe_better = probe.get("val_5d_cosine", -1) > best_probe
        full_better = full_due and metrics.get("val_5d_cosine", -1) > best_full
        if probe_better:
            best_probe = probe["val_5d_cosine"]
        if full_better:
            best_full = metrics["val_5d_cosine"]

        save_mcss_checkpoint(last_path, model, epoch=epoch, metrics=metrics,
                             optimizer=opt, scheduler=sched)
        if probe_better:
            save_mcss_checkpoint(pbest_path, model, epoch=epoch, metrics=metrics)
        if full_better:
            save_mcss_checkpoint(best_path, model, epoch=epoch, metrics=metrics)

        print(f"E{epoch:03d} probe_5d={probe['val_5d_cosine']:.4f} "
              f"p05={probe['val_5d_p05']:.4f} "
              f"768_ref={probe['val_768_cosine']:.4f} "
              f"ord={probe.get('val_ordinal_accuracy',0):.4f} "
              f"gap={max(0, train_config.target_cosine - probe['val_5d_cosine']):.4f}",
              flush=True)
        if full_due:
            print(f"E{epoch:03d} full_5d={metrics['val_5d_cosine']:.4f} "
                  f"768={metrics.get('val_768_cosine',0):.4f}", flush=True)
