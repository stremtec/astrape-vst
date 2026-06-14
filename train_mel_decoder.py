#!/usr/bin/env python3
"""Train the source-restored causal AdaLN-Zero mel decoder."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from astrape.data import speaker_disjoint_split
from astrape.mel_decoder import CausalMelDecoder, MelDecoderConfig


SAMPLE_RATE = 44100
CONTENT_RATE = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument(
        "--target-cache", type=Path, default=Path("data/mio_4k_decoder_mel")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="causal_mel_decoder_v2")
    parser.add_argument("--target-mode", choices=("original", "teacher"), default="teacher")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_decoder_checkpoint(
    path: Path,
    model: CausalMelDecoder,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2,
            "model_type": "causal_mel_decoder",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def build_target_cache(args: argparse.Namespace, n_samples: int) -> None:
    args.target_cache.mkdir(parents=True, exist_ok=True)
    cache_meta = args.target_cache / "meta.npz"
    if cache_meta.exists():
        with np.load(cache_meta) as metadata:
            cached_mode = str(metadata["target_mode"].item())
        if cached_mode != args.target_mode:
            raise SystemExit(
                f"Target cache was built in {cached_mode!r} mode, not "
                f"{args.target_mode!r}: {args.target_cache}"
            )
    teacher = None
    if args.target_mode == "teacher":
        try:
            from miocodec.model import MioCodecModel
        except ModuleNotFoundError as error:
            raise SystemExit(
                "MioCodec is required for --target-mode teacher."
            ) from error
        teacher = MioCodecModel.from_pretrained(
            "Aratako/MioCodec-25Hz-44.1kHz-v2"
        ).eval()
    extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=SAMPLE_RATE // CONTENT_RATE,
        n_mels=80,
        f_min=80,
        f_max=14000,
        center=False,
        power=1,
    )
    for index in range(n_samples):
        target_path = args.target_cache / f"m_{index:05d}.npz"
        if target_path.exists():
            continue
        with np.load(args.data_dir / f"s_{index:05d}.npz") as data:
            audio = torch.from_numpy(data["audio"]).float()
            if teacher is not None:
                with torch.inference_mode():
                    audio = teacher.decode(
                        global_embedding=torch.from_numpy(data["ge_128"]).float(),
                        content_token_indices=torch.from_numpy(data["ct"]).long(),
                        target_audio_length=len(audio),
                    ).cpu()
        mel = extractor(audio.view(1, 1, -1))
        logmel = torch.log(mel.squeeze(0).squeeze(0).clamp_min(1e-5))
        np.savez_compressed(target_path, logmel=logmel.numpy())
        if (index + 1) % 100 == 0:
            print(f"Target cache: {index + 1}/{n_samples}")
    np.savez_compressed(
        cache_meta,
        target_mode=np.asarray(args.target_mode),
        n_samples=np.asarray(n_samples, dtype=np.int64),
    )


def load_batch(
    data_dir: Path,
    target_cache: Path,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    contents = []
    globals_ = []
    targets = []
    lengths = []
    for index in indices:
        with np.load(data_dir / f"s_{int(index):05d}.npz") as data:
            content = torch.from_numpy(data["ce_768"]).float()
            global_embedding = torch.from_numpy(data["ge_128"]).float()
        with np.load(target_cache / f"m_{int(index):05d}.npz") as data:
            target = torch.from_numpy(data["logmel"]).float()
        length = min(content.shape[0], target.shape[1])
        contents.append(content[:length])
        globals_.append(global_embedding)
        targets.append(target[:, :length])
        lengths.append(length)
    max_length = max(lengths)
    content_batch = torch.stack(
        [F.pad(content, (0, 0, 0, max_length - len(content))) for content in contents]
    )
    target_batch = torch.stack(
        [F.pad(target, (0, max_length - target.shape[1])) for target in targets]
    )
    return (
        content_batch.to(device),
        torch.stack(globals_).to(device),
        target_batch.to(device),
        torch.tensor(lengths, device=device),
    )


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    positions = torch.arange(prediction.shape[-1], device=prediction.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).expand_as(prediction)
    return F.l1_loss(prediction.masked_select(mask), target.masked_select(mask))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    meta = np.load(args.data_dir / "meta.npz")
    n_samples = int(meta["n_samples"])
    speakers = meta["spk_names"][:n_samples].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(speakers, 0.15, args.seed)
    build_target_cache(args, n_samples)
    device = torch.device(args.device)
    model = CausalMelDecoder(MelDecoderConfig()).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = float("inf")
    best_path = args.output_dir / f"{args.run_name}.best.pt"
    last_path = args.output_dir / f"{args.run_name}.last.pt"

    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.permutation(train_indices)
        total = 0.0
        batches = 0
        for start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[start : start + args.batch_size]
            content, global_embedding, target, lengths = load_batch(
                args.data_dir, args.target_cache, batch_indices, device
            )
            prediction = model(content, global_embedding, lengths)
            loss = masked_l1(prediction, target, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        scheduler.step()

        model.eval()
        validation_sum = 0.0
        validation_frames = 0
        with torch.inference_mode():
            for start in range(0, len(validation_indices), args.batch_size):
                batch_indices = validation_indices[start : start + args.batch_size]
                content, global_embedding, target, lengths = load_batch(
                    args.data_dir, args.target_cache, batch_indices, device
                )
                prediction = model(content, global_embedding, lengths)
                loss = masked_l1(prediction, target, lengths)
                frames = int(lengths.sum().item())
                validation_sum += loss.item() * frames
                validation_frames += frames
        validation_loss = validation_sum / max(validation_frames, 1)
        metrics = {
            "train_l1": total / max(batches, 1),
            "val_l1": validation_loss,
        }
        save_decoder_checkpoint(last_path, model, epoch, metrics)
        if validation_loss < best:
            best = validation_loss
            save_decoder_checkpoint(best_path, model, epoch, metrics)
        print(
            f"E{epoch:03d} train={metrics['train_l1']:.4f} "
            f"val={validation_loss:.4f} best={best:.4f}"
        )


if __name__ == "__main__":
    main()
