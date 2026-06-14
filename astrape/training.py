from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .checkpoint import load_content_checkpoint, save_checkpoint
from .data import (
    ContentBatch,
    ContentCollator,
    MioContentDataset,
    masked_content_loss,
    speaker_disjoint_split,
)
from .model import ContentStudent, ContentStudentConfig


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    mel_dir: Path
    output_dir: Path
    run_name: str
    device: str = "cpu"
    batch_size: int = 4
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    max_mel_frames: int | None = 80
    seed: int = 42
    num_workers: int = 0
    resume: Path | None = None
    import_legacy: Path | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _move(batch: ContentBatch, device: torch.device) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        pre_fsq=batch.pre_fsq.to(device) if batch.pre_fsq is not None else None,
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
    )


def train_content_student(
    model_config: ContentStudentConfig, training: TrainingConfig
) -> None:
    seed_everything(training.seed)
    device = torch.device(training.device)
    meta = np.load(training.data_dir / "meta.npz")
    speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers, training.validation_fraction, training.seed
    )
    train_speakers = set(speakers[train_indices])
    validation_speakers = set(speakers[validation_indices])
    if train_speakers & validation_speakers:
        raise RuntimeError("Speaker-disjoint split failed")
    print(
        f"Train: {len(train_indices)} samples/{len(train_speakers)} speakers | "
        f"Val: {len(validation_indices)} samples/{len(validation_speakers)} speakers"
    )

    train_dataset = MioContentDataset(
        training.data_dir, training.mel_dir, train_indices
    )
    validation_dataset = MioContentDataset(
        training.data_dir, training.mel_dir, validation_indices
    )
    generator = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        collate_fn=ContentCollator(training.max_mel_frames, training.seed),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=ContentCollator(None, training.seed),
    )

    start_epoch = 0
    best_cosine = float("-inf")
    if training.resume is not None:
        model, metadata = load_content_checkpoint(
            training.resume,
            device=device,
            safe_convs=model_config.safe_convs,
        )
        if model.config != model_config:
            raise ValueError("Resume checkpoint config does not match requested config")
        start_epoch = int(metadata.get("epoch", -1)) + 1
        best_cosine = float(metadata.get("metrics", {}).get("val_cosine", best_cosine))
    elif training.import_legacy is not None:
        model, _ = load_content_checkpoint(
            training.import_legacy,
            device=device,
            allow_legacy=True,
            safe_convs=model_config.safe_convs,
        )
        if model.config != model_config:
            raise ValueError("Legacy checkpoint architecture does not match requested config")
    else:
        model = ContentStudent(model_config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=training.epochs)
    if training.resume is not None:
        payload = torch.load(training.resume, map_location=device)
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

    training.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = training.output_dir / f"{training.run_name}.best.pt"
    last_path = training.output_dir / f"{training.run_name}.last.pt"
    print(
        f"Params: {sum(parameter.numel() for parameter in model.parameters()):,} | "
        f"Device: {device}"
    )

    for epoch in range(start_epoch, training.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            output = model(batch.mel, batch.input_lengths)
            loss, _ = masked_content_loss(
                output.content, batch.content, batch.target_mask
            )
            if output.pre_fsq is not None and batch.pre_fsq is not None:
                auxiliary_loss, _ = masked_content_loss(
                    output.pre_fsq, batch.pre_fsq, batch.target_mask, l1_weight=0.0
                )
                loss = loss + 0.3 * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()

        model.eval()
        validation_cosine_sum = 0.0
        validation_frames = 0
        with torch.inference_mode():
            for raw_batch in validation_loader:
                batch = _move(raw_batch, device)
                output = model(batch.mel, batch.input_lengths)
                _, cosine = masked_content_loss(
                    output.content, batch.content, batch.target_mask
                )
                frame_count = int(batch.target_mask.sum().item())
                validation_cosine_sum += cosine.item() * frame_count
                validation_frames += frame_count
        validation_cosine = validation_cosine_sum / max(validation_frames, 1)
        train_loss = total_loss / max(batches, 1)
        metrics = {"train_loss": train_loss, "val_cosine": validation_cosine}
        save_checkpoint(
            last_path,
            model,
            epoch=epoch,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if validation_cosine > best_cosine:
            best_cosine = validation_cosine
            save_checkpoint(best_path, model, epoch=epoch, metrics=metrics)
        print(
            f"E{epoch:03d} loss={train_loss:.4f} "
            f"val_cos={validation_cosine:.4f} best={best_cosine:.4f}"
        )
