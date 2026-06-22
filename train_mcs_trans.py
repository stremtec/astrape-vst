"""MCS-Trans: causal local-window transformer replacing the GRU in MCS-v2.

Same conv frontend (CausalConv + ResidualBlocks + CellDownsample).
GRU replaced with N layers of causal multi-head self-attention (window=256).
Reuses data loading, collation, and loss infrastructure from mcs_common.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

from mcs_common import (
    Batch, MioCompactDataset, ContentCollator,
    split_by_speaker, speaker_balanced_subset,
    quantize_normalized_codes, masked_losses,
    move_batch, parse_axis_weights, save_checkpoint,
    CausalConv1d, ResidualConvBlock, CellDownsample,
    DEFAULT_DATA_DIR, DEFAULT_PROJECTION, FSQ_LEVELS,
)

DEFAULT_OUT_DIR = Path("checkpoints/mcs_trans")


@dataclass(frozen=True)
class MCSTransConfig:
    in_dim: int = 80
    conv_dim: int = 320
    trans_dim: int = 512
    n_heads: int = 8
    n_layers: int = 2
    ffn_dim: int = 1024
    window: int = 256
    conv_kernel: int = 5
    stem_dilations: tuple[int, ...] = (1, 2, 4, 8)
    skip_dilations: tuple[int, ...] = (16, 32)
    head_hidden: int = 128
    content_dim: int = 768
    levels: tuple[int, ...] = FSQ_LEVELS
    dropout: float = 0.0


def _causal_window_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """Additive float mask: 0=attend, -1e4=block. Causal + local window."""
    idx = torch.arange(seq_len, device=device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)   # diff[i,j] = i-j
    can_attend = (diff >= 0) & (diff < window)
    return torch.zeros(seq_len, seq_len, device=device).masked_fill(~can_attend, -1e4)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h,
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + F.dropout(h, self.dropout, self.training)
        h = self.norm2(x)
        x = x + F.dropout(self.ff(h), self.dropout, self.training)
        return x


class MCSTrans(nn.Module):
    def __init__(self, config: MCSTransConfig):
        super().__init__()
        self.config = config
        dim = config.conv_dim
        self.input_conv = CausalConv1d(config.in_dim, dim, config.conv_kernel)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(dim, config.conv_kernel, d, config.dropout)
             for d in config.stem_dilations]
        )
        self.skips = nn.ModuleList(
            [CausalConv1d(config.in_dim, dim, config.conv_kernel, dilation=d)
             for d in config.skip_dilations]
        )
        self.skip_gates = nn.ParameterList(
            [nn.Parameter(torch.full((1, dim, 1), -2.0)) for _ in config.skip_dilations]
        )
        self.downsample = CellDownsample(dim)
        self.proj_in = (nn.Linear(dim, config.trans_dim, bias=False)
                        if dim != config.trans_dim else nn.Identity())
        self.trans_layers = nn.ModuleList(
            [TransformerBlock(config.trans_dim, config.n_heads, config.ffn_dim, config.dropout)
             for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.trans_dim)
        self.smooth = CausalConv1d(
            config.trans_dim, config.trans_dim, kernel_size=3, groups=config.trans_dim
        )
        self.axis_heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(config.trans_dim, config.head_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(config.head_hidden, 1),
             ) for _ in config.levels]
        )
        self.ordinal_heads = nn.ModuleList(
            [nn.Linear(config.trans_dim, lv) for lv in config.levels]
        )
        self.proj_out = nn.Linear(len(config.levels), config.content_dim)
        for p in self.proj_out.parameters():
            p.requires_grad_(False)

    def load_projection(self, state: dict[str, torch.Tensor]) -> None:
        self.proj_out.load_state_dict(state, strict=True)
        for p in self.proj_out.parameters():
            p.requires_grad_(False)

    def forward(
        self, mel: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> dict:
        h = F.silu(self.input_conv(mel))
        for block in self.blocks:
            h = block(h)
        for skip, gate in zip(self.skips, self.skip_gates):
            h = h + torch.sigmoid(gate) * F.silu(skip(mel))
        h = self.downsample(h).transpose(1, 2)       # (B, T25, conv_dim)
        h = self.proj_in(h)                            # (B, T25, trans_dim)
        T = h.shape[1]
        attn_mask = _causal_window_mask(T, self.config.window, h.device)
        # key_padding_mask as float to match float attn_mask (avoids deprecation warning)
        kpm = (~padding_mask[:, :T]).float() * -1e4 if padding_mask is not None else None
        for layer in self.trans_layers:
            h = layer(h, attn_mask, kpm)
        h = self.norm(h)
        h = h + self.smooth(h.transpose(1, 2)).transpose(1, 2)
        codes = torch.cat([torch.tanh(head(h)) for head in self.axis_heads], dim=-1)
        quantized = quantize_normalized_codes(codes, self.config.levels)
        projected = self.proj_out(quantized).transpose(1, 2)
        ordinal = tuple(head(h).transpose(1, 2) for head in self.ordinal_heads)
        return {"codes": codes, "quantized": quantized, "projected": projected, "ordinal": ordinal}


@torch.inference_mode()
def evaluate(
    model: MCSTrans, loader: DataLoader, device: torch.device, args: argparse.Namespace
) -> dict[str, float]:
    model.eval()
    buckets: dict[str, list[float]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch.mel, padding_mask=batch.mask)
        _, metrics = masked_losses(output, batch, model.config.levels, args)
        for key, value in metrics.items():
            buckets.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(vs)) if vs else 0.0 for key, vs in buckets.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mel-frames", type=int, default=200)
    p.add_argument("--eval-mel-frames", type=int, default=300)
    p.add_argument("--probe-samples", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Initialize model weights/config from a checkpoint but start a fresh optimizer/scheduler run.",
    )
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--run-name", default="mcs_trans")
    # Transformer architecture
    p.add_argument("--conv-dim", type=int, default=320)
    p.add_argument("--trans-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--window", type=int, default=256)
    # Loss
    p.add_argument("--code-l1-weight", type=float, default=1.0)
    p.add_argument("--ordinal-weight", type=float, default=0.2)
    p.add_argument("--ordinal-weight-start", type=float, default=0.0)
    p.add_argument("--delta-weight", type=float, default=0.04)
    p.add_argument("--delta2-weight", type=float, default=0.0)
    p.add_argument("--content-delta-weight", type=float, default=0.0)
    p.add_argument("--content-delta2-weight", type=float, default=0.0)
    p.add_argument("--content-aux-weight", type=float, default=0.0)
    p.add_argument("--axis-weights", type=parse_axis_weights, default=(1.0, 1.0, 1.0, 1.6, 1.8))
    p.add_argument("--voiced-boost", type=float, default=1.0)
    p.add_argument("--ord-mode", default="ce",
                   choices=["ce", "none", "adjacent", "unified"])
    # Original-audio decoder-in-loop
    p.add_argument("--decoder-wave-weight", type=float, default=0.0,
                   help="Weight for MR-STFT loss against original wav via frozen Mio decoder. "
                        "0 = disabled (default). ~0.3 is a good starting point.")
    p.add_argument("--decoder-wave-prob", type=float, default=0.5,
                   help="Fraction of steps that compute the decoder wave loss (reduces cost).")
    p.add_argument("--decoder-n-ffts", default="512,1024,2048")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_from is not None and args.init_from is not None:
        raise ValueError("--resume-from and --init-from are mutually exclusive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with np.load(args.data_dir / "meta.npz", allow_pickle=False) as meta:
        n_samples = int(meta["n_samples"])
        speakers = meta["spk_names"][:n_samples].astype(str)
        source_files = meta["source_files"][:n_samples].astype(str)
        cache_format = str(meta["cache_format"].item())
    if cache_format != "compact-fp16-ctc-v2":
        raise ValueError(f"unexpected cache format: {cache_format}")

    # Decoder-in-loop setup
    mio = None
    decoder_n_ffts: tuple[int, ...] = ()
    if args.decoder_wave_weight > 0:
        from eval_mcs_trans_audio import load_mio, load_wave, SAMPLE_RATE  # noqa: E402
        from train_mcs_original_calibrator import multi_resolution_stft_loss  # noqa: E402
        print("Loading frozen MioCodec for decoder-in-loop ...", flush=True)
        mio = load_mio(device).eval()
        for p in mio.parameters():
            p.requires_grad_(False)
        decoder_n_ffts = tuple(int(v) for v in args.decoder_n_ffts.split(",") if v.strip())
        print(f"  decoder_wave_weight={args.decoder_wave_weight} "
              f"prob={args.decoder_wave_prob} n_ffts={decoder_n_ffts}", flush=True)

    train_idx, val_idx = split_by_speaker(speakers, args.val_fraction, args.seed)
    probe_idx = speaker_balanced_subset(val_idx, speakers, args.probe_samples, args.seed)

    train_loader = DataLoader(
        MioCompactDataset(args.data_dir, train_idx, speakers),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ContentCollator(args.mel_frames, args.seed),
        generator=torch.Generator().manual_seed(args.seed),
    )
    probe_loader = DataLoader(
        MioCompactDataset(args.data_dir, probe_idx, speakers),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ContentCollator(args.eval_mel_frames, args.seed + 999),
    )

    config = MCSTransConfig(
        conv_dim=args.conv_dim,
        trans_dim=args.trans_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        window=args.window,
    )
    checkpoint = None
    checkpoint_mode = None
    start_epoch, best_cos, global_step = 0, -1.0, 0
    checkpoint_path = args.resume_from or args.init_from
    if checkpoint_path is not None:
        checkpoint_mode = "resume" if args.resume_from is not None else "init"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_cfg = checkpoint["config"]
        config = MCSTransConfig(**{k: tuple(v) if isinstance(v, list) else v
                                   for k, v in saved_cfg.items()})

    model = MCSTrans(config).to(device)
    model.load_projection(torch.load(args.projection_path, map_location="cpu"))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if checkpoint_mode == "resume" and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint_mode == "resume" and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint_mode == "resume":
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            metrics = checkpoint.get("metrics", {})
            global_step = int(metrics.get("global_step", 0))
            current_cos = float(metrics.get("probe", {}).get("cos5", -1.0))
            best_cos = max(float(checkpoint.get("best_probe_cos5", -1.0)), current_cos)
            print(f"Resumed from {checkpoint_path} at epoch={start_epoch} "
                  f"global_step={global_step} best_probe_cos5={best_cos:.4f}", flush=True)
        else:
            print(f"Initialized weights/config from {checkpoint_path}; fresh optimizer/scheduler", flush=True)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ow_desc = (f"{args.ordinal_weight_start:.2f}→{args.ordinal_weight:.2f}"
               if args.ordinal_weight_start > 0 else f"{args.ordinal_weight:.2f}")
    print(f"Train={len(train_idx)} Probe={len(probe_idx)} speakers={len(set(speakers))} "
          f"params={params:,} trainable={trainable:,} device={device}", flush=True)
    print(f"Arch: trans_dim={config.trans_dim} n_heads={config.n_heads} "
          f"n_layers={config.n_layers} ffn_dim={config.ffn_dim} window={config.window}", flush=True)
    print(f"Objective: FSQ-5d primary, ordinal={ow_desc} [{args.ord_mode}], "
          f"voiced_boost={args.voiced_boost}, content_aux={args.content_aux_weight}, "
          f"delta={args.delta_weight}, delta2={args.delta2_weight}, "
          f"content_delta={args.content_delta_weight}, "
          f"content_delta2={args.content_delta2_weight}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()

    if args.eval_only:
        probe = evaluate(model, probe_loader, device, args)
        print(f"eval probe cos5={probe['cos5']:.4f} cos768={probe['cos768']:.4f} "
              f"ord={probe['ord_acc']:.4f} exact={probe['exact_acc']:.4f} "
              f"loss={probe['loss']:.4f}", flush=True)
        return

    total_epochs = max(1, args.epochs - start_epoch)
    for epoch in range(start_epoch, args.epochs):
        if args.ordinal_weight_start > 0:
            t = (epoch - start_epoch) / total_epochs
            ow = args.ordinal_weight_start * (1 - t) + args.ordinal_weight * t
        else:
            ow = None

        model.train()
        totals: dict[str, float] = {}
        step_started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = move_batch(batch, device)
            output = model(batch.mel, padding_mask=batch.mask)
            loss, metrics = masked_losses(output, batch, model.config.levels, args, ordinal_weight=ow)

            # Decoder-in-loop: MR-STFT vs original wav on one batch item
            if mio is not None and random.random() < args.decoder_wave_prob:
                # pick one item from the batch that has a valid source file
                idx_in_batch = random.randrange(len(batch.speakers))
                # recover dataset row via batch speakers mapping — use source_files directly
                # find a matching row for this speaker (approximate: use any row for the speaker)
                spk = batch.speakers[idx_in_batch]
                spk_rows = np.where(speakers == spk)[0]
                if len(spk_rows) > 0:
                    row = int(random.choice(spk_rows))
                    src_path = Path(str(source_files[row]))
                    if src_path.exists():
                        try:
                            orig_wave = load_wave(src_path, SAMPLE_RATE, max_seconds=3.0).to(device)
                            with torch.no_grad():
                                feats = mio.encode(orig_wave, return_content=True, return_global=True)
                                global_emb = feats.global_embedding.unsqueeze(0)
                                stft_len = mio._calculate_target_stft_length(orig_wave.numel())
                            # forward through mcs for this one item
                            mel_i = batch.mel[idx_in_batch].unsqueeze(0)
                            mask_i = batch.mask[idx_in_batch].unsqueeze(0)
                            out_i = model(mel_i, padding_mask=mask_i)
                            content_i = model.proj_out(out_i["codes"])
                            n_frames = min(content_i.shape[1], feats.content_embedding.shape[0])
                            pred_wave = mio.forward_wave(
                                content_i[:, :n_frames], global_emb, stft_length=stft_len
                            ).squeeze(0)
                            tgt_len = min(pred_wave.shape[-1], orig_wave.shape[-1])
                            wave_loss = multi_resolution_stft_loss(
                                pred_wave[:tgt_len], orig_wave[:tgt_len], decoder_n_ffts
                            )
                            loss = loss + args.decoder_wave_weight * wave_loss
                            metrics["decoder_wave"] = float(wave_loss.detach().cpu())
                        except Exception:
                            pass  # skip on any IO/shape error; don't break training

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
            optimizer.step()
            global_step += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            if step % args.log_every == 0 or step == args.steps_per_epoch:
                denom = max(step, 1)
                elapsed = time.time() - step_started
                print(
                    f"E{epoch:03d} step={step:04d}/{args.steps_per_epoch} "
                    f"loss={totals['loss']/denom:.4f} cos5={totals['cos5']/denom:.4f} "
                    f"cos768={totals['cos768']/denom:.4f} ord={totals['ord_acc']/denom:.4f} "
                    f"exact={totals['exact_acc']/denom:.4f} {elapsed/max(step,1):.3f}s/step",
                    flush=True,
                )
        scheduler.step()
        probe = evaluate(model, probe_loader, device, args)
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "probe": probe,
            "elapsed_seconds": time.time() - run_started,
        }
        print(
            f"E{epoch:03d} probe cos5={probe['cos5']:.4f} cos768={probe['cos768']:.4f} "
            f"ord={probe['ord_acc']:.4f} exact={probe['exact_acc']:.4f} "
            f"loss={probe['loss']:.4f}",
            flush=True,
        )
        save_checkpoint(
            args.out_dir / f"{args.run_name}.last.pt",
            model, optimizer, scheduler, epoch, metrics, args, best_cos,
        )
        if probe["cos5"] > best_cos:
            best_cos = probe["cos5"]
            save_checkpoint(
                args.out_dir / f"{args.run_name}.best.pt",
                model, optimizer, scheduler, epoch, metrics, args, best_cos,
            )
        if args.save_every_epoch:
            save_checkpoint(
                args.out_dir / f"{args.run_name}.epoch{epoch:03d}.pt",
                model, optimizer, scheduler, epoch, metrics, args, best_cos,
            )
        (args.out_dir / f"{args.run_name}.summary.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )

    print(f"done best_probe_cos5={best_cos:.4f} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
