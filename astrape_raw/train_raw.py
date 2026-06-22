"""Astrape Raw — Mel-free Waveform Frontend Training.

Replaces mel-spectrogram frontend with direct CausalConv1d on raw PCM.
Same Q2D2 + RoPE + SwiGLU + GRL + decoder-loop as train_mcs_q2d2.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

from mcs_common import (
    Batch, split_by_speaker, speaker_balanced_subset,
    move_batch, save_checkpoint,
    CausalConv1d, ResidualConvBlock, CellDownsample,
    DEFAULT_DATA_DIR,
    _voiced_weights, multi_resolution_stft_loss,
)
from mcs_q2d2 import Q2D2Projection, Q2D2Quantizer, compute_q2d2_perplexity

# Import from Q2D2 training script
from train_mcs_q2d2 import (
    MCSTransQ2D2Config,
    GradientReversal, grad_reverse, SpeakerClassifier,
    _precompute_rope_freqs, _apply_rope, _rotate_half,
    _causal_window_mask, TransformerBlock,
)

SAMPLE_RATE = 44100
RAW_STRIDE = 441  # 100Hz frame rate (44100/441=100)
CONTENT_STRIDE = 882  # 50Hz mel rate (used for teacher content alignment)
DEFAULT_OUT_DIR = Path("astrape_raw/checkpoints")


# ─────────────────────────────────────────────
# Raw Waveform Frontend
# ─────────────────────────────────────────────

class RawWaveformFrontend(nn.Module):
    """CausalConv1d on raw PCM, replacing mel frontend.

    PCM (B, 1, T) → CausalConv1d(k=2048, stride=441) → (B, 320, T100)
    → ResidualBlocks → skip → stride-2 downsample → (B, 320, T50)
    → proj_in → (B, T50, trans_dim)
    """

    def __init__(self, config: MCSTransQ2D2Config):
        super().__init__()
        dim = config.conv_dim  # 320

        self.raw_conv = CausalConv1d(1, dim, kernel_size=2048, stride=RAW_STRIDE)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(dim, config.conv_kernel, d, config.dropout)
            for d in config.stem_dilations
        ])
        self.skips = nn.ModuleList([
            CausalConv1d(1, dim, kernel_size=2048, stride=RAW_STRIDE, dilation=d)
            for d in config.skip_dilations
        ])
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.full((1, dim, 1), -2.0))
            for _ in config.skip_dilations
        ])
        self.downsample = CellDownsample(dim)
        self.proj_in = (
            nn.Linear(dim, config.trans_dim, bias=False)
            if dim != config.trans_dim else nn.Identity()
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.raw_conv(waveform))
        for block in self.blocks:
            h = block(h)
        for skip, gate in zip(self.skips, self.skip_gates):
            h = h + torch.sigmoid(gate) * F.silu(skip(waveform))
        h = self.downsample(h).transpose(1, 2)
        return self.proj_in(h)


# ─────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────

class MCSTransRaw(nn.Module):
    def __init__(self, config: MCSTransQ2D2Config):
        super().__init__()
        self.config = config
        self.frontend = RawWaveformFrontend(config)

        self.trans_layers = nn.ModuleList([
            TransformerBlock(config.trans_dim, config.n_heads,
                             config.ffn_dim, config.dropout,
                             use_rope=config.use_rope,
                             use_swiglu=config.use_swiglu)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.trans_dim)
        self.smooth = CausalConv1d(
            config.trans_dim, config.trans_dim, kernel_size=3,
            groups=config.trans_dim,
        )
        self.q2d2 = Q2D2Projection(
            encoder_dim=config.trans_dim,
            q2d2_dim=config.q2d2_dim,
            content_dim=config.content_dim,
            levels=list(config.q2d2_levels),
            vq_type=config.q2d2_grid,
        )
        self.speaker_classifier: SpeakerClassifier | None = None
        if config.grl_weight > 0 and config.grl_num_speakers > 0:
            self.speaker_classifier = SpeakerClassifier(
                dim=config.content_dim, num_speakers=config.grl_num_speakers,
            )

    def forward(self, waveform: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> dict:
        h = self.frontend(waveform)
        T = h.shape[1]
        attn_mask = _causal_window_mask(T, self.config.window, h.device)
        kpm = (~padding_mask[:, :T]).float() * -1e4 if padding_mask is not None else None
        for layer in self.trans_layers:
            h = layer(h, attn_mask, kpm)
        h = self.norm(h)
        h = h + self.smooth(h.transpose(1, 2)).transpose(1, 2)
        content, q2d2_codes = self.q2d2(h, return_codes=True)
        return {
            "projected": content.transpose(1, 2),
            "q2d2_codes": q2d2_codes,
            "ordinal": None,
        }


# ─────────────────────────────────────────────
# Waveform Dataset
# ─────────────────────────────────────────────

class WaveformDataset(Dataset):
    """Load raw waveform + teacher content from compact cache."""

    def __init__(self, root: Path, indices: np.ndarray, speakers: np.ndarray,
                 source_files: np.ndarray, max_seconds: float = 3.0):
        self.root = root
        self.indices = [int(i) for i in indices.tolist()]
        self.speakers = speakers
        self.source_files = source_files
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        import soundfile as sf
        idx = self.indices[item]
        src = Path(str(self.source_files[idx]))

        # Load raw audio
        wav, sr = sf.read(str(src), dtype="float32")
        wav = torch.from_numpy(np.asarray(wav))
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)

        if wav.shape[0] > self.max_samples:
            start = self.rng.randint(0, wav.shape[0] - self.max_samples)
            wav = wav[start:start + self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = F.pad(wav, (0, self.max_samples - wav.shape[0]))

        # Load teacher content from cache
        npz = np.load(self.root / f"s_{idx:05d}.npz", allow_pickle=False)
        teacher_content = torch.from_numpy(npz["ce_768"].astype(np.float32))

        return {
            "waveform": wav.float(),
            "content": teacher_content.float(),
            "speaker": str(self.speakers[idx]),
            "idx": idx,
        }


class WaveformCollator:
    """Collate waveform batches with content frame alignment."""

    def __init__(self, max_seconds: float):
        # Content frames for max_seconds at 25Hz
        self.max_content_frames = int(max_seconds * 25)

    def __call__(self, samples: list[dict]) -> Batch:
        wavs = torch.stack([s["waveform"] for s in samples]).unsqueeze(1)  # (B, 1, T)
        contents = []
        masks = []
        speakers = []
        indices = []

        for s in samples:
            c = s["content"]
            n = min(c.shape[0], self.max_content_frames)
            if c.shape[0] < self.max_content_frames:
                c = F.pad(c, (0, 0, 0, self.max_content_frames - c.shape[0]))
            contents.append(c[:self.max_content_frames])
            masks.append(torch.cat([
                torch.ones(n, dtype=torch.bool),
                torch.zeros(self.max_content_frames - n, dtype=torch.bool),
            ]))
            speakers.append(s["speaker"])
            indices.append(s["idx"])

        return Batch(
            mel=wavs,  # reusing mel field for waveform (shape differs)
            content=torch.stack(contents),
            tokens=torch.zeros(len(samples), self.max_content_frames, dtype=torch.long),
            mask=torch.stack(masks),
            speakers=speakers,
            indices=torch.tensor(indices, dtype=torch.long),
            crop_starts=torch.zeros(len(samples), dtype=torch.long),
        )


# ─────────────────────────────────────────────
# Loss (same as train_mcs_q2d2.py)
# ─────────────────────────────────────────────

def q2d2_losses(
    output: dict, batch: Batch, args: argparse.Namespace,
    quantizer: Q2D2Quantizer | None = None,
    speaker_classifier: nn.Module | None = None,
    speaker_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    projected = output["projected"]
    q2d2_codes = output.get("q2d2_codes")

    length = min(projected.shape[2], batch.content.shape[1], batch.mask.shape[1])
    mask = batch.mask[:, :length]

    pred_768 = projected[:, :, :length]
    tgt_768 = batch.content[:, :length]

    voiced_boost = getattr(args, "voiced_boost", 1.0)
    # For waveform input, compute voiced weights from energy (not mel RMS)
    if voiced_boost > 1.0 and batch.mel.ndim == 3 and batch.mel.shape[1] == 1:
        # Waveform: compute energy per content frame
        wav = batch.mel  # (B, 1, T_audio) — mel field reused for waveform
        T_content = length
        # Reshape to content frame resolution (~882 samples per frame at 50Hz→25Hz)
        frame_len = 882 * 2  # 25Hz = 1764 samples
        wav_frames = wav[:, 0, :T_content * frame_len].reshape(wav.shape[0], T_content, frame_len)
        rms = wav_frames.pow(2).mean(dim=-1).sqrt()  # (B, T_content)
        threshold = rms.mean(dim=1, keepdim=True).clamp(min=1e-5)
        voiced = (rms > threshold * 0.5).float()
        vw = 1.0 + (voiced_boost - 1.0) * voiced
    else:
        vw = _voiced_weights(batch.mel, length, voiced_boost)
    weighted_mask_sum = (vw * mask.float()).sum().clamp(min=1)

    pred_masked = pred_768.permute(0, 2, 1)[mask]
    tgt_masked = tgt_768[mask]
    cos768 = F.cosine_similarity(pred_masked, tgt_masked, dim=-1).mean()
    cos768_loss = 1.0 - cos768

    pred_flat = pred_768.permute(0, 2, 1)
    l1_per_frame = (pred_flat - tgt_768).abs().mean(dim=-1)
    content_l1 = ((l1_per_frame * vw * mask.float()).sum() / weighted_mask_sum)

    if length >= 2:
        delta_mask = mask[:, 1:] & mask[:, :-1]
        pred_delta = pred_flat[:, 1:] - pred_flat[:, :-1]
        tgt_delta = tgt_768[:, 1:] - tgt_768[:, :-1]
        delta_weights = 0.5 * (vw[:, 1:] + vw[:, :-1])
        delta = F.smooth_l1_loss(pred_delta[delta_mask], tgt_delta[delta_mask], reduction="mean")
    else:
        delta = projected.sum() * 0.0

    loss = (args.content_cos_weight * cos768_loss +
            args.content_l1_weight * content_l1 +
            args.delta_weight * delta)

    grl_loss_val: float = 0.0
    grl_acc_val: float = 0.0
    if speaker_classifier is not None and speaker_ids is not None:
        grl_weight = getattr(args, "grl_weight", 0.0)
        if grl_weight > 0:
            grl_content = grad_reverse(projected, grl_weight)
            speaker_logits = speaker_classifier(grl_content)
            grl_loss = F.cross_entropy(speaker_logits, speaker_ids)
            loss = loss + grl_loss
            grl_loss_val = float(grl_loss.detach().cpu())
            grl_acc_val = float((speaker_logits.argmax(dim=-1) == speaker_ids).float().mean().cpu())

    metrics: dict[str, float] = {
        "loss": float(loss.detach().cpu()),
        "cos768": float(cos768.detach().cpu()),
        "content_l1": float(content_l1.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "grl_loss": grl_loss_val,
        "grl_acc": grl_acc_val,
    }

    if quantizer is not None and q2d2_codes is not None:
        with torch.no_grad():
            stats = compute_q2d2_perplexity(quantizer, q2d2_codes)
            metrics["q2d2_usage"] = stats["overall_usage"]
            for i in range(quantizer.num_pairs):
                metrics[f"q2d2_pair{i}_usage"] = stats[f"pair_{i}_usage"]

    return loss, metrics


@torch.inference_mode()
def evaluate(model, loader, device, args, quantizer=None, speaker_to_id=None):
    model.eval()
    buckets: dict[str, list[float]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        speaker_ids = None
        if model.speaker_classifier is not None and speaker_to_id is not None:
            speaker_ids = torch.tensor(
                [speaker_to_id[s] for s in batch.speakers],
                device=device, dtype=torch.long,
            )
        output = model(batch.mel, padding_mask=batch.mask)
        _, metrics = q2d2_losses(output, batch, args, quantizer,
                                 model.speaker_classifier, speaker_ids)
        for key, value in metrics.items():
            buckets.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(vs)) if vs else 0.0 for key, vs in buckets.items()}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--probe-samples", type=int, default=256)
    p.add_argument("--max-seconds", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--run-name", default="astrape_raw")
    p.add_argument("--conv-dim", type=int, default=320)
    p.add_argument("--trans-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--rope", action="store_true")
    p.add_argument("--swiglu", action="store_true")
    p.add_argument("--q2d2-dim", type=int, default=6)
    p.add_argument("--q2d2-levels", type=str, default="7,7,7,7,7,7")
    p.add_argument("--q2d2-grid", default="rhombic")
    p.add_argument("--content-cos-weight", type=float, default=1.0)
    p.add_argument("--content-l1-weight", type=float, default=0.5)
    p.add_argument("--delta-weight", type=float, default=0.04)
    p.add_argument("--grl-weight", type=float, default=0.0)
    p.add_argument("--voiced-boost", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Data
    with np.load(args.data_dir / "meta.npz", allow_pickle=False) as meta:
        n_samples = int(meta["n_samples"])
        speakers = meta["spk_names"][:n_samples].astype(str)
        source_files = meta["source_files"][:n_samples].astype(str)

    unique_speakers = sorted(set(speakers))
    speaker_to_id = {s: i for i, s in enumerate(unique_speakers)}
    print(f"Speakers: {len(unique_speakers)}, GRL={args.grl_weight > 0}", flush=True)

    train_idx, val_idx = split_by_speaker(speakers, args.val_fraction, args.seed)
    probe_idx = speaker_balanced_subset(val_idx, speakers, args.probe_samples, args.seed)

    train_ds = WaveformDataset(args.data_dir, train_idx, speakers, source_files, args.max_seconds)
    probe_ds = WaveformDataset(args.data_dir, probe_idx, speakers, source_files, args.max_seconds)
    collator = WaveformCollator(args.max_seconds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    probe_loader = DataLoader(probe_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=0)

    # Config
    q2d2_levels = tuple(int(v) for v in args.q2d2_levels.split(","))
    config = MCSTransQ2D2Config(
        conv_dim=args.conv_dim, trans_dim=args.trans_dim,
        n_heads=args.n_heads, n_layers=args.n_layers, ffn_dim=args.ffn_dim,
        window=args.window, use_rope=args.rope, use_swiglu=args.swiglu,
        q2d2_dim=args.q2d2_dim, q2d2_levels=q2d2_levels, q2d2_grid=args.q2d2_grid,
        grl_weight=args.grl_weight, grl_num_speakers=len(unique_speakers),
    )

    # Model
    model = MCSTransRaw(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {params:,} params ({trainable:,} trainable), device={device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    quantizer = model.q2d2.quantizer
    best_cos = -1.0
    run_started = time.time()

    for epoch in range(args.epochs):
        model.train()
        totals: dict[str, float] = {}
        step_started = time.time()

        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = move_batch(batch, device)

            speaker_ids = None
            if model.speaker_classifier is not None:
                speaker_ids = torch.tensor(
                    [speaker_to_id[s] for s in batch.speakers],
                    device=device, dtype=torch.long,
                )

            output = model(batch.mel, padding_mask=batch.mask)
            loss, metrics = q2d2_losses(output, batch, args, quantizer,
                                        model.speaker_classifier, speaker_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value

            if step % args.log_every == 0 or step == args.steps_per_epoch:
                denom = max(step, 1)
                elapsed = time.time() - step_started
                print(
                    f"E{epoch:03d} step={step:04d}/{args.steps_per_epoch} "
                    f"loss={totals['loss']/denom:.4f} cos768={totals['cos768']/denom:.4f} "
                    f"l1={totals.get('content_l1',0)/denom:.4f} "
                    f"usage={totals.get('q2d2_usage',0)/denom:.3f} "
                    f"{elapsed/max(step,1):.3f}s/step",
                    flush=True,
                )

        scheduler.step()
        probe = evaluate(model, probe_loader, device, args, quantizer, speaker_to_id)
        metrics_full = {
            "epoch": epoch, "global_step": (epoch + 1) * args.steps_per_epoch,
            "probe": probe, "elapsed_seconds": time.time() - run_started,
        }

        current_cos = probe.get("cos768", 0.0)
        print(
            f"E{epoch:03d} probe cos768={current_cos:.4f} loss={probe.get('loss',0):.4f} "
            f"usage={probe.get('q2d2_usage',0):.3f}",
            flush=True,
        )

        save_checkpoint(
            args.out_dir / f"{args.run_name}.last.pt",
            model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
        )
        if current_cos > best_cos:
            best_cos = current_cos
            save_checkpoint(
                args.out_dir / f"{args.run_name}.best.pt",
                model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
            )
        (args.out_dir / f"{args.run_name}.summary.json").write_text(
            json.dumps(metrics_full, indent=2, sort_keys=True) + "\n"
        )

    print(f"done best_cos768={best_cos:.4f} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
