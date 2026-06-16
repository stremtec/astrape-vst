#!/usr/bin/env python3
"""Compare log-mel and causalized WavLM-conv inputs with the same small probe."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

from astrape.data import speaker_disjoint_split
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.fsq import indices_to_codes


@dataclass
class ProbeSample:
    features: torch.Tensor
    codes: torch.Tensor


class CausalWavlmConv(nn.Module):
    """Reuse WavLM filters while replacing time-wide GroupNorm."""

    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.layers = feature_extractor.conv_layers

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        hidden = waveform.unsqueeze(1)
        for index, layer in enumerate(self.layers):
            hidden = layer.conv(hidden)
            if index == 0:
                hidden = F.layer_norm(
                    hidden.transpose(1, 2),
                    (hidden.shape[1],),
                ).transpose(1, 2)
            hidden = F.gelu(hidden)
        return hidden.transpose(1, 2)


class CausalProbe(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden)
        self.encoder = nn.GRU(
            hidden,
            hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.output = nn.Linear(hidden, 5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input_projection(self.input_norm(features)))
        hidden, _ = self.encoder(hidden)
        return self.output(hidden[:, ::2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument(
        "--fsq-projection",
        type=Path,
        default=Path("checkpoints/teacher_fsq_proj_out.pt"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _coerce_path(path):
    raw = str(path)
    if raw.startswith("np.str_(") and raw.endswith(")"):
        return raw[len("np.str_("):-1]
    return raw


def load_audio(path: str, max_seconds: float) -> torch.Tensor:
    path = _coerce_path(path)
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor,
        )
    return torch.from_numpy(
        np.asarray(audio[: round(max_seconds * 16000)], dtype=np.float32)
    )


def collate(
    samples: list[ProbeSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_lengths = torch.tensor(
        [sample.features.shape[0] for sample in samples],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [
            min((sample.features.shape[0] + 1) // 2, sample.codes.shape[0])
            for sample in samples
        ],
        dtype=torch.long,
    )
    max_features = int(feature_lengths.max())
    max_targets = int(target_lengths.max())
    features = torch.stack(
        [
            F.pad(sample.features, (0, 0, 0, max_features - len(sample.features)))
            for sample in samples
        ]
    )
    targets = torch.stack(
        [
            F.pad(
                sample.codes[:length],
                (0, 0, 0, max_targets - length),
            )
            for sample, length in zip(samples, target_lengths.tolist())
        ]
    )
    mask = torch.arange(max_targets).unsqueeze(0) < target_lengths.unsqueeze(1)
    return features, targets, mask


def projected_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    predicted_content = prediction @ weight.T + bias
    target_content = target @ weight.T + bias
    return F.cosine_similarity(
        predicted_content,
        target_content,
        dim=-1,
    )[mask].mean()


def train_probe(
    name: str,
    train_samples: list[ProbeSample],
    validation_samples: list[ProbeSample],
    *,
    hidden: int,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CausalProbe(train_samples[0].features.shape[-1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    axis_weights = torch.tensor(
        [1.0, 0.8, 0.8, 1.7, 2.0],
        device=device,
    )
    model.train()
    for step in range(1, steps + 1):
        selected = rng.sample(train_samples, batch_size)
        features, target, mask = collate(selected)
        features = features.to(device)
        target = target.to(device)
        mask = mask.to(device)
        prediction = model(features)
        length = min(prediction.shape[1], target.shape[1])
        prediction = prediction[:, :length]
        target = target[:, :length]
        mask = mask[:, :length]
        code_loss = (
            F.smooth_l1_loss(
                prediction[mask],
                target[mask],
                reduction="none",
            )
            * axis_weights
        ).mean()
        cosine = projected_cosine(
            prediction,
            target,
            mask,
            weight,
            bias,
        )
        loss = code_loss + 1.0 - cosine
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0:
            print(
                f"{name} step={step}/{steps} loss={loss.item():.5f} "
                f"cos={cosine.item():.6f}"
            )

    model.eval()
    cosines = []
    axis_errors = []
    with torch.inference_mode():
        for start in range(0, len(validation_samples), batch_size):
            features, target, mask = collate(
                validation_samples[start : start + batch_size]
            )
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            prediction = model(features)
            length = min(prediction.shape[1], target.shape[1])
            prediction = prediction[:, :length]
            target = target[:, :length]
            mask = mask[:, :length]
            predicted_content = prediction @ weight.T + bias
            target_content = target @ weight.T + bias
            cosines.append(
                F.cosine_similarity(
                    predicted_content,
                    target_content,
                    dim=-1,
                )[mask].cpu()
            )
            axis_errors.append((prediction[mask] - target[mask]).abs().cpu())
    cosine = torch.cat(cosines)
    errors = torch.cat(axis_errors)
    result = {
        "cosine": cosine.mean().item(),
        "p05": torch.quantile(cosine, 0.05).item(),
    }
    print(
        f"{name} validation_cos={result['cosine']:.6f} "
        f"p05={result['p05']:.6f} "
        "axis_mae="
        + ",".join(f"{value:.4f}" for value in errors.mean(dim=0).tolist())
    )
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with np.load(args.data_dir / "meta.npz") as meta:
        count = int(meta["n_samples"])
        speakers = meta["spk_names"][:count].astype(str)
        source_files = meta["source_files"][:count].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers,
        0.15,
        args.seed,
    )
    train_indices = speaker_balanced_subset(
        train_indices,
        speakers,
        args.train_samples,
        args.seed,
    )
    validation_indices = speaker_balanced_subset(
        validation_indices,
        speakers,
        args.validation_samples,
        args.seed,
    )

    from miocodec.model import MioCodecModel

    teacher = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-44.1kHz-v2"
    ).eval()
    raw_frontend = CausalWavlmConv(
        teacher.ssl_feature_extractor.model.feature_extractor
    ).to(device).eval()
    del teacher

    logmel_samples: dict[int, ProbeSample] = {}
    raw_samples: dict[int, ProbeSample] = {}
    all_indices = np.concatenate((train_indices, validation_indices))
    with torch.inference_mode():
        for position, index in enumerate(all_indices.tolist(), start=1):
            with np.load(args.data_dir / f"s_{index:05d}.npz") as cached:
                logmel = torch.from_numpy(cached["logmel"]).float().T
                codes = indices_to_codes(
                    torch.from_numpy(cached["ct"].astype(np.int64))
                )
            max_frames = round(args.max_seconds * 50)
            logmel = logmel[:max_frames]
            waveform = load_audio(source_files[index], args.max_seconds)
            raw = raw_frontend(waveform.unsqueeze(0).to(device))
            raw = raw.squeeze(0).float().cpu()
            logmel_samples[index] = ProbeSample(logmel, codes)
            raw_samples[index] = ProbeSample(raw, codes)
            if position % 32 == 0 or position == len(all_indices):
                print(f"features={position}/{len(all_indices)}")

    projection = torch.load(args.fsq_projection, map_location=device)
    weight = projection["weight"].to(device)
    bias = projection["bias"].to(device)
    logmel_train = [logmel_samples[int(index)] for index in train_indices]
    logmel_validation = [
        logmel_samples[int(index)] for index in validation_indices
    ]
    raw_train = [raw_samples[int(index)] for index in train_indices]
    raw_validation = [raw_samples[int(index)] for index in validation_indices]
    logmel_result = train_probe(
        "logmel",
        logmel_train,
        logmel_validation,
        hidden=args.hidden,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        weight=weight,
        bias=bias,
    )
    raw_result = train_probe(
        "raw_conv",
        raw_train,
        raw_validation,
        hidden=args.hidden,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        weight=weight,
        bias=bias,
    )
    print(
        f"raw_minus_logmel_cosine="
        f"{raw_result['cosine'] - logmel_result['cosine']:+.6f}"
    )


if __name__ == "__main__":
    main()
