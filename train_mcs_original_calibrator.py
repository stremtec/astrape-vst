"""Original-audio decoder-in-loop training for the MCS content calibrator.

This treats Mio teacher content as an anchor, not the final objective. The
frozen MCS encoder emits continuous pre-round content, a small causal calibrator
adjusts it, and the frozen Mio wave decoder is trained-through against the
original waveform with multi-resolution STFT loss.
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
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

from eval_mcs_trans_audio import (  # noqa: E402
    DEFAULT_DATA_DIR,
    SAMPLE_RATE,
    load_mcs_trans,
    load_mio,
    load_wave,
)
from mcs_calibrator import (  # noqa: E402
    ContentCalibrator,
    ContentCalibratorConfig,
    load_calibrator,
    save_calibrator,
)
from mcs_common import Batch, move_batch, speaker_balanced_subset, split_by_speaker  # noqa: E402


DEFAULT_MCS_CHECKPOINT = Path("checkpoints/mcs_trans_t6_causal_scratch/mcs_trans_t6_causal_scratch.best.pt")
DEFAULT_INIT_CALIBRATOR = Path("checkpoints/mcs_content_calibrator_t6/mcs_content_calibrator_t6.best.pt")
DEFAULT_OUT_DIR = Path("checkpoints/mcs_original_calibrator_t6")


@dataclass
class OriginalExample:
    row: int
    speaker: str
    batch: Batch
    teacher_content: torch.Tensor
    source_global: torch.Tensor
    original_wave: torch.Tensor
    target_audio_length: int
    stft_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcs-checkpoint", type=Path, default=DEFAULT_MCS_CHECKPOINT)
    parser.add_argument("--init-calibrator", type=Path, default=DEFAULT_INIT_CALIBRATOR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="mcs_original_calibrator_t6")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mrstft-weight", type=float, default=1.0)
    parser.add_argument("--wave-l1-weight", type=float, default=0.05)
    parser.add_argument("--teacher-content-weight", type=float, default=0.2)
    parser.add_argument("--base-content-weight", type=float, default=0.05)
    parser.add_argument("--delta-weight", type=float, default=0.02)
    parser.add_argument("--n-ffts", default="512,1024,2048")
    return parser.parse_args()


def match_length(wave: torch.Tensor, length: int) -> torch.Tensor:
    if wave.shape[-1] > length:
        return wave[..., :length]
    if wave.shape[-1] < length:
        return F.pad(wave, (0, length - wave.shape[-1]))
    return wave


def stft_mag(wave: torch.Tensor, n_fft: int) -> torch.Tensor:
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    return spec.abs().clamp_min(1e-7)


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_ffts: tuple[int, ...],
) -> torch.Tensor:
    pred = pred.squeeze(0) if pred.dim() == 2 else pred
    target = target.squeeze(0) if target.dim() == 2 else target
    length = min(pred.shape[-1], target.shape[-1])
    pred = pred[:length]
    target = target[:length]

    losses = []
    for n_fft in n_ffts:
        pred_mag = stft_mag(pred, n_fft)
        target_mag = stft_mag(target, n_fft)
        spectral_convergence = torch.linalg.vector_norm(pred_mag - target_mag) / (
            torch.linalg.vector_norm(target_mag).clamp_min(1e-7)
        )
        log_mag = F.l1_loss(pred_mag.log(), target_mag.log())
        losses.append(spectral_convergence + log_mag)
    return torch.stack(losses).mean()


def content_regularizer(
    pred_content: torch.Tensor,
    teacher_content: torch.Tensor,
    base_content: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    length = min(pred_content.shape[1], teacher_content.shape[1], base_content.shape[1], mask.shape[1])
    pred = pred_content[:, :length]
    teacher = teacher_content[:, :length]
    base = base_content[:, :length]
    mask = mask[:, :length]

    teacher_cos = F.cosine_similarity(pred[mask], teacher[mask], dim=-1).mean()
    base_cos = F.cosine_similarity(pred[mask], base[mask], dim=-1).mean()
    loss = args.teacher_content_weight * (1.0 - teacher_cos)
    loss = loss + args.base_content_weight * (1.0 - base_cos)

    if length >= 2 and args.delta_weight > 0:
        dmask = mask[:, 1:] & mask[:, :-1]
        pred_delta = pred[:, 1:] - pred[:, :-1]
        teacher_delta = teacher[:, 1:] - teacher[:, :-1]
        delta = F.smooth_l1_loss(pred_delta[dmask], teacher_delta[dmask]) if dmask.any() else pred.sum() * 0
        loss = loss + args.delta_weight * delta
    else:
        delta = pred.sum() * 0

    return loss, {
        "teacher_cos": float(teacher_cos.detach().cpu()),
        "base_cos": float(base_cos.detach().cpu()),
        "content_delta": float(delta.detach().cpu()),
    }


def decode_wave(mio, content: torch.Tensor, global_embedding: torch.Tensor, example: OriginalExample) -> torch.Tensor:
    wave = mio.forward_wave(
        content,
        global_embedding.unsqueeze(0),
        stft_length=example.stft_length,
    )
    return match_length(wave.squeeze(0), example.target_audio_length)


@torch.no_grad()
def build_examples(
    args: argparse.Namespace,
    rows: np.ndarray,
    speakers: np.ndarray,
    source_files: np.ndarray,
    mcs,
    mio,
    device: torch.device,
    count: int,
    split_name: str,
) -> list[OriginalExample]:
    examples: list[OriginalExample] = []
    for row in rows.tolist():
        if len(examples) >= count:
            break
        source_path = Path(str(source_files[row]))
        if not source_path.exists():
            continue

        source_wave = load_wave(source_path, SAMPLE_RATE, max_seconds=args.max_seconds)
        if source_wave.numel() < int(args.min_seconds * SAMPLE_RATE):
            continue
        source_wave = source_wave.to(device)

        with np.load(args.data_dir / f"s_{row:05d}.npz", allow_pickle=False) as data:
            mel_full = torch.from_numpy(data["logmel"].astype(np.float32))
            cached_content = torch.from_numpy(data["ce_768"].astype(np.float32))
            tokens = torch.from_numpy(data["ct"].astype(np.int64))

        mel_frames = max(1, int(round(source_wave.numel() / SAMPLE_RATE * 50.0)))
        mel_frames = min(mel_frames, mel_full.shape[-1])
        content_frames = min(cached_content.shape[0], tokens.shape[0], (mel_frames + 1) // 2)
        if content_frames < 4:
            continue

        features = mio.encode(source_wave, return_content=True, return_global=True)
        teacher_content = features.content_embedding[:content_frames].detach().clone()
        source_global = features.global_embedding.detach().clone()
        content_frames = min(content_frames, teacher_content.shape[0])

        mel = mel_full[:, :mel_frames]
        cached_content = cached_content[:content_frames]
        tokens = tokens[:content_frames]
        mask = torch.ones(content_frames, dtype=torch.bool)
        batch = move_batch(
            Batch(
                mel=mel.unsqueeze(0),
                content=cached_content.unsqueeze(0),
                tokens=tokens.unsqueeze(0),
                mask=mask.unsqueeze(0),
                speakers=[str(speakers[row])],
            ),
            device,
        )

        target_audio_length = int(source_wave.numel())
        examples.append(
            OriginalExample(
                row=int(row),
                speaker=str(speakers[row]),
                batch=batch,
                teacher_content=teacher_content.unsqueeze(0),
                source_global=source_global,
                original_wave=source_wave.detach().clone(),
                target_audio_length=target_audio_length,
                stft_length=mio._calculate_target_stft_length(target_audio_length),
            )
        )
        print(
            f"{split_name} [{row:05d}] {speakers[row]} "
            f"{target_audio_length / SAMPLE_RATE:.2f}s content={content_frames}",
            flush=True,
        )

    if len(examples) < count:
        print(f"{split_name}: prepared {len(examples)}/{count} examples", flush=True)
    return examples


@torch.no_grad()
def base_content(mcs, batch: Batch) -> torch.Tensor:
    output = mcs(batch.mel, padding_mask=batch.mask)
    return mcs.proj_out(output["codes"])


def train_step(
    example: OriginalExample,
    mcs,
    calibrator: ContentCalibrator,
    mio,
    n_ffts: tuple[int, ...],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = base_content(mcs, example.batch)
    pred_content = calibrator(base)
    length = min(pred_content.shape[1], example.teacher_content.shape[1], example.batch.mask.shape[1])
    pred_content = pred_content[:, :length]

    pred_wave = decode_wave(mio, pred_content, example.source_global, example)
    mrstft = multi_resolution_stft_loss(pred_wave, example.original_wave, n_ffts)
    wave_l1 = F.smooth_l1_loss(pred_wave, example.original_wave)
    content_loss, content_metrics = content_regularizer(
        pred_content,
        example.teacher_content,
        base,
        example.batch.mask,
        args,
    )
    loss = args.mrstft_weight * mrstft + args.wave_l1_weight * wave_l1 + content_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "mrstft": float(mrstft.detach().cpu()),
        "wave_l1": float(wave_l1.detach().cpu()),
        **content_metrics,
    }
    return loss, metrics


@torch.no_grad()
def evaluate(
    examples: list[OriginalExample],
    mcs,
    calibrator: ContentCalibrator,
    mio,
    n_ffts: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, float]:
    calibrator.eval()
    buckets: dict[str, list[float]] = {}
    for example in examples:
        loss, metrics = train_step(example, mcs, calibrator, mio, n_ffts, args)
        for key, value in metrics.items():
            buckets.setdefault(key, []).append(value)
    calibrator.train()
    return {key: float(np.mean(values)) if values else 0.0 for key, values in buckets.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    n_ffts = tuple(int(value.strip()) for value in args.n_ffts.split(",") if value.strip())

    with np.load(args.data_dir / "meta.npz", allow_pickle=False) as meta:
        n_samples = int(meta["n_samples"])
        speakers = meta["spk_names"][:n_samples].astype(str)
        source_files = meta["source_files"][:n_samples].astype(str)

    train_idx, val_idx = split_by_speaker(speakers, 0.15, args.seed)
    train_rows = speaker_balanced_subset(train_idx, speakers, args.train_samples * 4, args.seed)
    eval_rows = speaker_balanced_subset(val_idx, speakers, args.eval_samples * 4, args.seed + 1)

    print(f"Loading frozen MCS from {args.mcs_checkpoint}", flush=True)
    mcs = load_mcs_trans(args.mcs_checkpoint, device).eval()
    for parameter in mcs.parameters():
        parameter.requires_grad_(False)

    print("Loading frozen MioCodec", flush=True)
    mio = load_mio(device).eval()
    for parameter in mio.parameters():
        parameter.requires_grad_(False)

    if args.init_calibrator is not None and args.init_calibrator.exists():
        print(f"Loading calibrator init from {args.init_calibrator}", flush=True)
        calibrator = load_calibrator(args.init_calibrator, device).train()
    else:
        config = ContentCalibratorConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
        calibrator = ContentCalibrator(config).to(device).train()

    print("Preparing original-audio examples", flush=True)
    train_examples = build_examples(
        args, train_rows, speakers, source_files, mcs, mio, device, args.train_samples, "train"
    )
    eval_examples = build_examples(
        args, eval_rows, speakers, source_files, mcs, mio, device, args.eval_samples, "eval"
    )
    if not train_examples:
        raise RuntimeError("No train examples prepared.")
    if not eval_examples:
        raise RuntimeError("No eval examples prepared.")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in calibrator.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_eval = float("inf")
    started = time.time()
    last_eval: dict[str, float] = {}

    initial_eval = evaluate(eval_examples, mcs, calibrator, mio, n_ffts, args)
    print(
        f"initial eval loss={initial_eval['loss']:.4f} mrstft={initial_eval['mrstft']:.4f} "
        f"teacher_cos={initial_eval['teacher_cos']:.4f}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        example = random.choice(train_examples)
        loss, metrics = train_step(example, mcs, calibrator, mio, n_ffts, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d}/{args.steps} row={example.row:05d} "
                f"loss={metrics['loss']:.4f} mrstft={metrics['mrstft']:.4f} "
                f"wave={metrics['wave_l1']:.5f} teacher_cos={metrics['teacher_cos']:.4f}",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            last_eval = evaluate(eval_examples, mcs, calibrator, mio, n_ffts, args)
            elapsed = time.time() - started
            print(
                f"eval step={step:04d} loss={last_eval['loss']:.4f} "
                f"mrstft={last_eval['mrstft']:.4f} wave={last_eval['wave_l1']:.5f} "
                f"teacher_cos={last_eval['teacher_cos']:.4f}",
                flush=True,
            )
            payload_metrics = {
                "epoch": 0,
                "global_step": step,
                "probe": last_eval,
                "elapsed_seconds": elapsed,
                "initial_eval": initial_eval,
            }
            save_calibrator(
                args.out_dir / f"{args.run_name}.last.pt",
                calibrator,
                optimizer,
                0,
                payload_metrics,
                args,
            )
            if last_eval["loss"] < best_eval:
                best_eval = last_eval["loss"]
                save_calibrator(
                    args.out_dir / f"{args.run_name}.best.pt",
                    calibrator,
                    optimizer,
                    0,
                    payload_metrics,
                    args,
                )
            (args.out_dir / f"{args.run_name}.summary.json").write_text(
                json.dumps(payload_metrics, indent=2, sort_keys=True) + "\n"
            )

    print(f"done best_eval_loss={best_eval:.4f} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
