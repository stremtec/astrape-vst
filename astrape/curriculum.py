from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss
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
from .fsq import masked_fsq_cross_entropy
from .model import ContentStudent, ContentStudentConfig
from .original_data import (
    OriginalBatch,
    OriginalCollator,
    OriginalVCTKDataset,
    scan_vctk,
)
from .training import seed_everything


@dataclass(frozen=True)
class CurriculumConfig:
    data_dir: Path
    mel_dir: Path
    audio_root: Path
    transcript_root: Path
    fsq_projection: Path
    output_dir: Path
    run_name: str = "content_student_768x10_fsq"
    device: str = "mps"
    batch_size: int = 4
    original_epochs: int = 5
    blend_epochs: int = 10
    teacher_epochs: int = 30
    steps_per_epoch: int = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    max_teacher_mel_frames: int = 80
    validation_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 0
    resume: Path | None = None
    target_cosine: float = 0.99
    log_every: int = 50

    @property
    def epochs(self) -> int:
        return self.original_epochs + self.blend_epochs + self.teacher_epochs


def phase_weights(epoch: int, config: CurriculumConfig) -> tuple[str, float, float]:
    if epoch < config.original_epochs:
        return "original", 0.0, 1.0
    blend_end = config.original_epochs + config.blend_epochs
    if epoch < blend_end:
        progress = (epoch - config.original_epochs + 1) / max(config.blend_epochs, 1)
        return "blend", progress, 1.0 - 0.8 * progress
    return "teacher", 0.9, 0.1


def validate_curriculum(config: CurriculumConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("Curriculum must contain at least one epoch")
    if config.steps_per_epoch <= 0 or config.batch_size <= 0:
        raise ValueError("steps_per_epoch and batch_size must be positive")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if (
        config.max_teacher_mel_frames <= 0
        or config.max_teacher_mel_frames % 2
    ):
        raise ValueError("max_teacher_mel_frames must be a positive even number")
    if not 0.0 <= config.target_cosine <= 1.0:
        raise ValueError("target_cosine must be between 0 and 1")


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _move_teacher(batch: ContentBatch, device: torch.device) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        pre_fsq=batch.pre_fsq.to(device) if batch.pre_fsq is not None else None,
        token_indices=(
            batch.token_indices.to(device) if batch.token_indices is not None else None
        ),
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
    )


def teacher_loss(
    model: ContentStudent, batch: ContentBatch
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch.token_indices is None:
        raise RuntimeError("Teacher batch does not contain FSQ token indices")
    output = model(batch.mel, batch.input_lengths)
    if output.fsq_logits is None:
        raise RuntimeError("Model does not have a structured FSQ head")
    embedding_loss, soft_cosine = masked_content_loss(
        output.content, batch.content, batch.target_mask
    )
    fsq_loss, axis_accuracy, exact_accuracy = masked_fsq_cross_entropy(
        output.fsq_logits,
        batch.token_indices,
        batch.target_mask,
        model.config.fsq_levels,
    )
    loss = embedding_loss + fsq_loss
    if output.pre_fsq is not None and batch.pre_fsq is not None:
        pre_fsq_loss, _ = masked_content_loss(
            output.pre_fsq, batch.pre_fsq, batch.target_mask, l1_weight=0.0
        )
        loss = loss + 0.2 * pre_fsq_loss
    return loss, {
        "soft_cosine": soft_cosine.item(),
        "axis_accuracy": axis_accuracy.item(),
        "exact_accuracy": exact_accuracy.item(),
    }


def original_loss(
    model: ContentStudent, batch: OriginalBatch, device: torch.device, criterion: CTCLoss
) -> torch.Tensor:
    mel = batch.mel.to(device)
    input_lengths = batch.input_lengths.to(device)
    output = model(mel, input_lengths)
    if output.text_logits is None:
        raise RuntimeError("Model does not have a text CTC head")
    ctc_device = torch.device("cpu") if device.type == "mps" else device
    return criterion(
        output.text_logits.log_softmax(dim=-1)
        .transpose(0, 1)
        .to(ctc_device),
        batch.transcripts.to(ctc_device),
        input_lengths.to(ctc_device),
        batch.transcript_lengths.to(ctc_device),
    )


@torch.inference_mode()
def evaluate_teacher(
    model: ContentStudent,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    frame_cosines = []
    sequence_cosines = []
    axis_correct = 0
    axis_total = 0
    exact_correct = 0
    exact_total = 0
    for raw_batch in loader:
        batch = _move_teacher(raw_batch, device)
        output = model(batch.mel, batch.input_lengths)
        if output.fsq_logits is None or batch.token_indices is None:
            raise RuntimeError("Validation requires structured FSQ labels and logits")
        prediction = output.content.transpose(1, 2)
        length = min(prediction.shape[1], batch.content.shape[1])
        mask = batch.target_mask[:, :length]
        prediction = prediction[:, :length]
        target = batch.content[:, :length]
        frame_cosines.append(
            F.cosine_similarity(prediction, target, dim=-1)[mask].cpu()
        )
        for item in range(prediction.shape[0]):
            valid = mask[item]
            sequence_cosines.append(
                F.cosine_similarity(
                    prediction[item, valid].reshape(1, -1),
                    target[item, valid].reshape(1, -1),
                ).item()
            )
        _, axis_accuracy, exact_accuracy = masked_fsq_cross_entropy(
            output.fsq_logits,
            batch.token_indices,
            batch.target_mask,
            model.config.fsq_levels,
        )
        frames = int(batch.target_mask.sum())
        axes = frames * len(model.config.fsq_levels)
        axis_correct += axis_accuracy.item() * axes
        axis_total += axes
        exact_correct += exact_accuracy.item() * frames
        exact_total += frames
    frame_cosines_tensor = torch.cat(frame_cosines)
    return {
        "val_frame_cosine": frame_cosines_tensor.mean().item(),
        "val_frame_cosine_p05": torch.quantile(
            frame_cosines_tensor, 0.05
        ).item(),
        "val_sequence_cosine": float(np.mean(sequence_cosines)),
        "val_axis_accuracy": axis_correct / max(axis_total, 1),
        "val_exact_token_accuracy": exact_correct / max(exact_total, 1),
    }


def train_curriculum(
    model_config: ContentStudentConfig, curriculum: CurriculumConfig
) -> None:
    if not model_config.structured_fsq or model_config.text_vocab_size <= 0:
        raise ValueError("Curriculum requires structured_fsq and a text CTC head")
    validate_curriculum(curriculum)
    seed_everything(curriculum.seed)
    device = torch.device(curriculum.device)
    meta = np.load(curriculum.data_dir / "meta.npz")
    speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers, curriculum.validation_fraction, curriculum.seed
    )
    train_speakers = sorted(set(speakers[train_indices]))
    validation_speakers = sorted(set(speakers[validation_indices]))

    teacher_train = DataLoader(
        MioContentDataset(curriculum.data_dir, curriculum.mel_dir, train_indices),
        batch_size=curriculum.batch_size,
        shuffle=True,
        num_workers=curriculum.num_workers,
        collate_fn=ContentCollator(
            curriculum.max_teacher_mel_frames, curriculum.seed
        ),
        generator=torch.Generator().manual_seed(curriculum.seed),
    )
    teacher_validation = DataLoader(
        MioContentDataset(
            curriculum.data_dir, curriculum.mel_dir, validation_indices
        ),
        batch_size=curriculum.batch_size,
        shuffle=False,
        num_workers=curriculum.num_workers,
        collate_fn=ContentCollator(None, curriculum.seed),
    )
    original_records = scan_vctk(
        curriculum.audio_root,
        curriculum.transcript_root,
        allowed_speakers=train_speakers,
    )
    if not original_records:
        raise RuntimeError(
            "No full VCTK utterances matched the teacher training speakers"
        )
    original_train = DataLoader(
        OriginalVCTKDataset(original_records),
        batch_size=curriculum.batch_size,
        shuffle=True,
        num_workers=curriculum.num_workers,
        collate_fn=OriginalCollator(),
        generator=torch.Generator().manual_seed(curriculum.seed + 1),
    )
    print(
        f"Teacher train={len(train_indices)}/{len(train_speakers)} speakers, "
        f"val={len(validation_indices)}/{len(validation_speakers)} speakers | "
        f"original={len(original_records)} full utterances"
    )

    start_epoch = 0
    best_cosine = float("-inf")
    if curriculum.resume is not None:
        model, metadata = load_content_checkpoint(
            curriculum.resume, device=device
        )
        if model.config != model_config:
            raise ValueError("Resume checkpoint config does not match")
        start_epoch = int(metadata.get("epoch", -1)) + 1
        best_cosine = float(
            metadata.get("metrics", {}).get("val_frame_cosine", best_cosine)
        )
    else:
        model = ContentStudent(model_config).to(device)
        projection = torch.load(curriculum.fsq_projection, map_location=device)
        model.load_fsq_projection(projection)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=curriculum.learning_rate,
        weight_decay=curriculum.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=curriculum.epochs)
    if curriculum.resume is not None:
        payload = torch.load(curriculum.resume, map_location=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    ctc = CTCLoss(blank=0, zero_infinity=True)
    curriculum.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = curriculum.output_dir / f"{curriculum.run_name}.best.pt"
    last_path = curriculum.output_dir / f"{curriculum.run_name}.last.pt"

    for epoch in range(start_epoch, curriculum.epochs):
        phase, teacher_probability, ctc_weight = phase_weights(epoch, curriculum)
        model.train()
        teacher_iterator = iter(teacher_train)
        original_iterator = iter(original_train)
        total_loss = 0.0
        teacher_steps = 0
        original_steps = 0
        epoch_started = time.perf_counter()
        for step in range(1, curriculum.steps_per_epoch + 1):
            use_teacher = random.random() < teacher_probability
            if use_teacher:
                raw_batch, teacher_iterator = _next(
                    teacher_iterator, teacher_train
                )
                batch = _move_teacher(raw_batch, device)
                loss, _ = teacher_loss(model, batch)
                teacher_steps += 1
            else:
                batch, original_iterator = _next(
                    original_iterator, original_train
                )
                loss = ctc_weight * original_loss(model, batch, device, ctc)
                original_steps += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if step % curriculum.log_every == 0 or step == curriculum.steps_per_epoch:
                elapsed = time.perf_counter() - epoch_started
                print(
                    f"E{epoch:03d} {phase} step={step}/{curriculum.steps_per_epoch} "
                    f"loss={total_loss / step:.4f} "
                    f"teacher={teacher_steps} original={original_steps} "
                    f"{elapsed / step:.3f}s/step",
                    flush=True,
                )
        scheduler.step()
        metrics = evaluate_teacher(model, teacher_validation, device)
        metrics.update(
            {
                "train_loss": total_loss / curriculum.steps_per_epoch,
                "teacher_steps": float(teacher_steps),
                "original_steps": float(original_steps),
                "target_gap": max(
                    0.0,
                    curriculum.target_cosine - metrics["val_frame_cosine"],
                ),
            }
        )
        save_checkpoint(
            last_path,
            model,
            epoch=epoch,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if metrics["val_frame_cosine"] > best_cosine:
            best_cosine = metrics["val_frame_cosine"]
            save_checkpoint(best_path, model, epoch=epoch, metrics=metrics)
        print(
            f"E{epoch:03d} {phase} loss={metrics['train_loss']:.4f} "
            f"frame_cos={metrics['val_frame_cosine']:.4f} "
            f"p05={metrics['val_frame_cosine_p05']:.4f} "
            f"seq_cos={metrics['val_sequence_cosine']:.4f} "
            f"axis_acc={metrics['val_axis_accuracy']:.3f} "
            f"token_acc={metrics['val_exact_token_accuracy']:.3f} "
            f"gap_to_{curriculum.target_cosine:.2f}={metrics['target_gap']:.4f}"
        )
