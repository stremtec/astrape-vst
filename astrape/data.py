from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class ContentSample:
    mel: torch.Tensor
    content: torch.Tensor
    pre_fsq: Optional[torch.Tensor]
    speaker: str
    index: int


@dataclass
class ContentBatch:
    mel: torch.Tensor
    content: torch.Tensor
    pre_fsq: Optional[torch.Tensor]
    input_lengths: torch.Tensor
    target_lengths: torch.Tensor
    target_mask: torch.Tensor


class MioContentDataset(Dataset[ContentSample]):
    def __init__(
        self,
        data_dir: str | Path,
        mel_dir: str | Path,
        indices: Optional[Sequence[int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.mel_dir = Path(mel_dir)
        meta = np.load(self.data_dir / "meta.npz")
        self.n_samples = int(meta["n_samples"])
        self.speakers = meta["spk_names"][: self.n_samples].astype(str)
        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(self.n_samples)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> ContentSample:
        index = int(self.indices[item])
        with np.load(self.mel_dir / f"m_{index:05d}.npz") as mel_data:
            mel = torch.from_numpy(mel_data["logmel"]).float()
        with np.load(self.data_dir / f"s_{index:05d}.npz") as output_data:
            content = torch.from_numpy(output_data["ce_768"]).float()
            pre_fsq = (
                torch.from_numpy(output_data["pre_fsq_768"]).float()
                if "pre_fsq_768" in output_data
                else None
            )
        return ContentSample(
            mel=mel,
            content=content,
            pre_fsq=pre_fsq,
            speaker=self.speakers[index],
            index=index,
        )


def speaker_disjoint_split(
    speakers: Sequence[str], validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(map(str, speakers))))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    validation_count = max(1, round(len(unique) * validation_fraction))
    validation_speakers = set(unique[:validation_count])
    train = np.array(
        [index for index, speaker in enumerate(speakers) if speaker not in validation_speakers]
    )
    validation = np.array(
        [index for index, speaker in enumerate(speakers) if speaker in validation_speakers]
    )
    return train, validation


def crop_aligned(
    sample: ContentSample, max_mel_frames: Optional[int], rng: random.Random
) -> ContentSample:
    if max_mel_frames is None or sample.mel.shape[1] <= max_mel_frames:
        return sample
    if max_mel_frames <= 0 or max_mel_frames % 2:
        raise ValueError("max_mel_frames must be a positive even number")
    max_start = sample.mel.shape[1] - max_mel_frames
    starts = range(0, max_start + 1, 2)
    start = rng.choice(starts)
    target_start = start // 2
    target_length = (max_mel_frames + 1) // 2
    return ContentSample(
        mel=sample.mel[:, start : start + max_mel_frames],
        content=sample.content[target_start : target_start + target_length],
        pre_fsq=(
            sample.pre_fsq[target_start : target_start + target_length]
            if sample.pre_fsq is not None
            else None
        ),
        speaker=sample.speaker,
        index=sample.index,
    )


class ContentCollator:
    def __init__(self, max_mel_frames: Optional[int], seed: int):
        self.max_mel_frames = max_mel_frames
        self.rng = random.Random(seed)

    def __call__(self, samples: list[ContentSample]) -> ContentBatch:
        samples = [
            crop_aligned(sample, self.max_mel_frames, self.rng) for sample in samples
        ]
        input_lengths = torch.tensor(
            [sample.mel.shape[1] for sample in samples], dtype=torch.long
        )
        target_lengths = torch.tensor(
            [
                min(
                    (sample.mel.shape[1] + 1) // 2,
                    sample.content.shape[0],
                    sample.pre_fsq.shape[0]
                    if sample.pre_fsq is not None
                    else sample.content.shape[0],
                )
                for sample in samples
            ],
            dtype=torch.long,
        )
        max_input = int(input_lengths.max())
        max_target = int(target_lengths.max())
        mel = torch.stack(
            [F.pad(sample.mel, (0, max_input - sample.mel.shape[1])) for sample in samples]
        )
        content = torch.stack(
            [
                F.pad(
                    sample.content[:length],
                    (0, 0, 0, max_target - length),
                )
                for sample, length in zip(samples, target_lengths.tolist())
            ]
        )
        has_pre_fsq = all(sample.pre_fsq is not None for sample in samples)
        pre_fsq = None
        if has_pre_fsq:
            pre_fsq = torch.stack(
                [
                    F.pad(
                        sample.pre_fsq[:length],
                        (0, 0, 0, max_target - length),
                    )
                    for sample, length in zip(samples, target_lengths.tolist())
                ]
            )
        positions = torch.arange(max_target)
        target_mask = positions.unsqueeze(0) < target_lengths.unsqueeze(1)
        return ContentBatch(
            mel=mel,
            content=content,
            pre_fsq=pre_fsq,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            target_mask=target_mask,
        )


def masked_content_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    l1_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = prediction.transpose(1, 2)
    length = min(prediction.shape[1], target.shape[1], mask.shape[1])
    prediction = prediction[:, :length]
    target = target[:, :length]
    mask = mask[:, :length]
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    cosine_mean = cosine.masked_select(mask).mean()
    expanded_mask = mask.unsqueeze(-1).expand_as(prediction)
    l1 = F.l1_loss(
        prediction.masked_select(expanded_mask),
        target.masked_select(expanded_mask),
    )
    return (1 - cosine_mean) + l1_weight * l1, cosine_mean
