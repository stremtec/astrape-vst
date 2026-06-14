#!/usr/bin/env python3
"""Train the direct causal content-to-waveform decoder."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from astrape.data import speaker_disjoint_split
from astrape.wave_decoder import (
    DirectWaveDecoder,
    WaveDecoderConfig,
    load_wave_decoder,
    save_wave_decoder_checkpoint,
)


@dataclass
class WaveSample:
    content: torch.Tensor
    global_embedding: torch.Tensor
    waveform: torch.Tensor


class WaveDataset(Dataset[WaveSample]):
    def __init__(
        self,
        data_dir: Path,
        indices: np.ndarray,
        crop_frames: int,
        target_dir: Path | None,
        seed: int,
        random_crops: bool = True,
    ):
        self.data_dir = data_dir
        self.indices = indices
        self.crop_frames = crop_frames
        self.target_dir = target_dir
        self.seed = seed
        self.random_crops = random_crops

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> WaveSample:
        index = int(self.indices[item])
        with np.load(self.data_dir / f"s_{index:05d}.npz") as data:
            content = torch.from_numpy(data["ce_768"]).float()
            global_embedding = torch.from_numpy(data["ge_128"]).float()
            waveform = torch.from_numpy(data["audio"]).float().reshape(-1)
        if self.target_dir is not None:
            with np.load(self.target_dir / f"w_{index:05d}.npz") as data:
                waveform = torch.from_numpy(
                    data["waveform"]
                ).float().reshape(-1)
        samples_per_frame = 1764
        valid_frames = min(
            content.shape[0],
            waveform.shape[0] // samples_per_frame,
        )
        if valid_frames <= 0:
            raise ValueError(f"No aligned waveform frames in sample {index}")
        crop_frames = min(self.crop_frames, valid_frames)
        if self.random_crops:
            start = random.randint(0, valid_frames - crop_frames)
        else:
            start = random.Random(self.seed + index).randint(
                0,
                valid_frames - crop_frames,
            )
        sample_start = start * samples_per_frame
        return WaveSample(
            content=content[start : start + crop_frames],
            global_embedding=global_embedding,
            waveform=waveform[
                sample_start : sample_start + crop_frames * samples_per_frame
            ],
        )


def collate_wave(samples: list[WaveSample]) -> WaveSample:
    length = min(sample.content.shape[0] for sample in samples)
    sample_length = length * 1764
    return WaveSample(
        content=torch.stack([sample.content[:length] for sample in samples]),
        global_embedding=torch.stack(
            [sample.global_embedding for sample in samples]
        ),
        waveform=torch.stack(
            [sample.waveform[:sample_length] for sample in samples]
        ),
    )


def multires_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    device = prediction.device
    if device.type == "mps":
        prediction = prediction.cpu()
        target = target.cpu()
    loss = prediction.new_zeros(())
    resolutions = 0
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        if prediction.shape[-1] < n_fft:
            continue
        window = torch.hann_window(
            n_fft,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        predicted = torch.stft(
            prediction,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        expected = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        spectral_convergence = (
            (expected - predicted).norm()
            / expected.norm().clamp_min(1e-6)
        )
        log_magnitude = F.l1_loss(
            torch.log1p(predicted),
            torch.log1p(expected),
        )
        loss = loss + spectral_convergence + log_magnitude
        resolutions += 1
    if resolutions == 0:
        return loss.to(device)
    return (loss / resolutions).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="direct_wave_decoder")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-frames", type=int, default=25)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--spectral-every", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quality-profile", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.crop_frames <= 0 or args.steps <= 0:
        raise SystemExit("--crop-frames and --steps must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    with np.load(args.data_dir / "meta.npz") as metadata:
        count = int(metadata["n_samples"])
        speakers = metadata["spk_names"][:count].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers,
        0.15,
        args.seed,
    )
    train_loader = DataLoader(
        WaveDataset(
            args.data_dir,
            train_indices,
            args.crop_frames,
            args.target_dir,
            args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_wave,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_dataset = WaveDataset(
        args.data_dir,
        validation_indices,
        args.crop_frames,
        args.target_dir,
        args.seed + 1,
        random_crops=False,
    )
    resume_payload = None
    if args.resume is not None:
        model = load_wave_decoder(args.resume, device)
        resume_payload = torch.load(args.resume, map_location=device)
        step = int(resume_payload.get("step", 0))
    else:
        config = (
            WaveDecoderConfig.quality_profile()
            if args.quality_profile
            else WaveDecoderConfig()
        )
        model = DirectWaveDecoder(config).to(device)
        step = 0
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
    if resume_payload is not None and "optimizer_state_dict" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    resume_metrics = resume_payload.get("metrics", {}) if resume_payload else {}
    best_validation = float(
        resume_metrics.get(
            "best_val_waveform_l1",
            resume_metrics.get("val_waveform_l1", float("inf")),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{args.run_name}.best.pt"
    last_path = args.output_dir / f"{args.run_name}.last.pt"
    print(
        f"train={len(train_indices)} val={len(validation_indices)} "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"target={'teacher-cache' if args.target_dir else 'original'}"
    )

    iterator = iter(train_loader)
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        step += 1
        content = batch.content.to(device)
        global_embedding = batch.global_embedding.to(device)
        target = batch.waveform.to(device)
        prediction = model(content, global_embedding)
        waveform_loss = F.l1_loss(prediction, target)
        spectral_active = args.spectral_every > 0 and step % args.spectral_every == 0
        spectral_loss = (
            multires_stft_loss(prediction, target)
            if spectral_active
            else prediction.new_zeros(())
        )
        loss = waveform_loss + (
            spectral_loss * args.spectral_every if spectral_active else 0.0
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step % args.log_every == 0:
            print(
                f"step={step}/{args.steps} loss={loss.item():.4f} "
                f"wave={waveform_loss.item():.4f} "
                f"stft={spectral_loss.item():.4f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            model.eval()
            validation_losses = []
            with torch.inference_mode():
                for index in range(min(64, len(validation_dataset))):
                    sample = collate_wave([validation_dataset[index]])
                    prediction = model(
                        sample.content.to(device),
                        sample.global_embedding.to(device),
                    )
                    validation_losses.append(
                        F.l1_loss(
                            prediction,
                            sample.waveform.to(device),
                        ).item()
                    )
            model.train()
            validation_l1 = float(np.mean(validation_losses))
            improved = validation_l1 < best_validation
            best_validation = min(best_validation, validation_l1)
            metrics = {
                "val_waveform_l1": validation_l1,
                "best_val_waveform_l1": best_validation,
            }
            save_wave_decoder_checkpoint(
                last_path,
                model,
                step=step,
                metrics=metrics,
                optimizer=optimizer,
            )
            if improved:
                save_wave_decoder_checkpoint(
                    best_path,
                    model,
                    step=step,
                    metrics=metrics,
                )
            print(
                f"validation step={step} l1={validation_l1:.4f} "
                f"best={best_validation:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
