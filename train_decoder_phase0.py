"""Phase 0: Pretrain CausalSynthesisDecoder on teacher ce_768 → waveform.

The decoder learns to synthesize 44.1kHz audio from ground-truth teacher
content embeddings before the encoder exists. This ensures the decoder
depends on content input from day one.

Requires: raw waveform files accessible from the data cache source_files paths.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from astrape.decoder import CausalSynthesisDecoder, SynthesisDecoderConfig, save_decoder_checkpoint
from astrape.voicebank import VoiceBank


def load_voicebank_dir(
    directory: Path,
    *,
    expected_model: str,
    expected_dim: int,
) -> dict[str, torch.Tensor]:
    """Load .astrape files keyed by stem, validating decoder compatibility."""
    if not directory.exists():
        raise FileNotFoundError(f"VoiceBank directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"VoiceBank path is not a directory: {directory}")

    embeddings = {}
    for f in sorted(directory.glob("*.astrape")):
        bank = VoiceBank.load(f)
        if bank.embedding_model != expected_model:
            raise ValueError(
                f"VoiceBank {f.name} uses embedding_model={bank.embedding_model!r}; "
                f"decoder expects condition_model={expected_model!r}"
            )
        embedding = bank.global_embedding.detach().float()
        if embedding.shape != (expected_dim,):
            raise ValueError(
                f"VoiceBank {f.name} embedding shape {tuple(embedding.shape)} "
                f"does not match decoder condition_dim={expected_dim}"
            )
        embeddings[f.stem] = embedding.clone()
    if not embeddings:
        raise ValueError(f"No .astrape VoiceBank files found in {directory}")
    return embeddings


def validate_voicebank_coverage(
    speakers: np.ndarray,
    speaker_embeddings: dict[str, torch.Tensor],
) -> None:
    """Require every dataset speaker to have a matching .astrape stem."""
    required = sorted(set(map(str, speakers)))
    missing = [speaker for speaker in required if speaker not in speaker_embeddings]
    if missing:
        shown = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... (+{len(missing) - 10} more)"
        raise ValueError(
            "VoiceBank speaker coverage mismatch: missing embeddings for "
            f"{len(missing)}/{len(required)} dataset speakers: {shown}{suffix}. "
            ".astrape file stems must match meta.npz spk_names exactly."
        )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# --- Multi-scale Mel Loss ---


class MultiScaleMelLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        n_ffts: tuple[int, ...] = (512, 1024, 2048),
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.n_ffts = n_ffts
        self.transforms = nn.ModuleList()
        for n_fft in n_ffts:
            self.transforms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    n_mels=n_mels,
                    f_min=f_min,
                    f_max=f_max or sample_rate // 2,
                    center=True,
                    power=1.0,
                )
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        for transform in self.transforms:
            pred_mel = torch.log(transform(pred).clamp_min(1e-5))
            target_mel = torch.log(transform(target).clamp_min(1e-5))
            loss = loss + F.l1_loss(pred_mel, target_mel)
        return loss / len(self.transforms)


# --- Dataset ---


class DecoderPretrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        audio_root: str | Path,
        speaker_embeddings: dict[str, torch.Tensor],
        target_sr: int = 44100,
        max_content_frames: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.audio_root = Path(audio_root)
        self.speaker_embeddings = speaker_embeddings
        self.target_sr = target_sr
        self.max_content_frames = max_content_frames

        meta = np.load(self.data_dir / "meta.npz")
        self.n_samples = int(meta["n_samples"])
        self.source_files = meta["source_files"][: self.n_samples]
        self.speakers = meta["spk_names"][: self.n_samples].astype(str)

    def __len__(self) -> int:
        return self.n_samples

    def _resolve_audio_path(self, source_file: str) -> Path:
        p = Path(str(source_file))
        if p.exists():
            return p
        # Try relative to audio_root
        name = p.name
        parts = p.parts
        # Look for speaker/filename pattern
        for i, part in enumerate(parts):
            candidate = self.audio_root / "/".join(parts[i:])
            if candidate.exists():
                return candidate
        # Fallback: search by filename
        matches = list(self.audio_root.rglob(name))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Cannot find audio: {source_file}")

    def __getitem__(self, index: int) -> Optional[dict]:
        sample_path = self.data_dir / f"s_{index:05d}.npz"
        try:
            with np.load(sample_path) as data:
                ce_768 = torch.from_numpy(data["ce_768"]).float()

            # Load and resample audio
            audio_path = self._resolve_audio_path(str(self.source_files[index]))
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            waveform = waveform.mean(dim=0)  # mono

            # Align: 1 content frame = 1764 samples @ 44.1kHz
            n_frames = ce_768.shape[0]
            expected_samples = n_frames * 1764
            if waveform.shape[0] < expected_samples:
                waveform = F.pad(waveform, (0, expected_samples - waveform.shape[0]))
            else:
                waveform = waveform[:expected_samples]

            # Crop to max_content_frames
            if n_frames > self.max_content_frames:
                start = torch.randint(0, n_frames - self.max_content_frames, (1,)).item()
                ce_768 = ce_768[start : start + self.max_content_frames]
                audio_start = start * 1764
                waveform = waveform[audio_start : audio_start + self.max_content_frames * 1764]

            # Speaker embedding: same-speaker conditioning (source == target in Phase 0)
            spk_name = self.speakers[index]
            spk_emb = self.speaker_embeddings.get(spk_name)

            return {
                "content": ce_768,
                "waveform": waveform,
                "speaker": spk_name,
                "speaker_embedding": spk_emb,
            }
        except (FileNotFoundError, Exception) as e:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_frames = max(b["content"].shape[0] for b in batch)
    max_samples = max_frames * 1764

    contents = []
    waveforms = []
    embeddings = []
    embedding_flags = [b["speaker_embedding"] is not None for b in batch]
    has_embeddings = all(embedding_flags)
    if any(embedding_flags) and not has_embeddings:
        missing = [str(b["speaker"]) for b in batch if b["speaker_embedding"] is None]
        raise ValueError(
            "Batch has partial speaker embedding coverage; missing "
            f"{', '.join(missing[:10])}"
        )
    for b in batch:
        c = b["content"]
        w = b["waveform"]
        if c.shape[0] < max_frames:
            c = F.pad(c, (0, 0, 0, max_frames - c.shape[0]))
        if w.shape[0] < max_samples:
            w = F.pad(w, (0, max_samples - w.shape[0]))
        contents.append(c)
        waveforms.append(w)
        if has_embeddings:
            embeddings.append(b["speaker_embedding"])

    result = {
        "content": torch.stack(contents),
        "waveform": torch.stack(waveforms),
    }
    if has_embeddings:
        result["speaker_embedding"] = torch.stack(embeddings)
    return result


# --- Training Loop ---


def train(args):
    device = torch.device(args.device)
    log.info(f"Device: {device}")

    # Model
    config = SynthesisDecoderConfig()
    model = CausalSynthesisDecoder(config).to(device)
    log.info(f"Decoder params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    mel_loss_fn = MultiScaleMelLoss(sample_rate=config.sample_rate).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    # Load voicebank for speaker embeddings
    speaker_embeddings: dict[str, torch.Tensor] = {}
    voicebank_dir = Path(args.voicebank_dir) if args.voicebank_dir else None
    if voicebank_dir is not None:
        speaker_embeddings = load_voicebank_dir(
            voicebank_dir,
            expected_model=config.condition_model,
            expected_dim=config.condition_dim,
        )
        log.info(f"Loaded {len(speaker_embeddings)} speaker embeddings from {voicebank_dir}")
    else:
        log.warning("No voicebank provided - using random speaker embeddings per batch")

    # Data
    dataset = DecoderPretrainDataset(
        args.data_dir, args.audio_root,
        speaker_embeddings=speaker_embeddings,
        target_sr=config.sample_rate,
        max_content_frames=args.max_frames,
    )
    if voicebank_dir is not None:
        validate_voicebank_coverage(dataset.speakers, speaker_embeddings)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_skip_none,
        drop_last=True,
        pin_memory=True,
    )


    # Training loop
    step = 0
    best_loss = float("inf")
    model.train()

    log.info(f"Starting Phase 0 decoder pretraining for {args.steps} steps")

    while step < args.steps:
        for batch in loader:
            if batch is None:
                continue
            if step >= args.steps:
                break

            content = batch["content"].to(device)
            waveform = batch["waveform"].to(device)

            # Real speaker embeddings so AdaLN gates learn conditioning from day one
            if "speaker_embedding" in batch:
                global_emb = batch["speaker_embedding"].to(device)
            elif voicebank_dir is not None:
                raise RuntimeError(
                    "VoiceBank was provided but this batch has no speaker embeddings"
                )
            else:
                # Fallback: random unit-norm embeddings (still non-zero, forces gates to activate)
                global_emb = torch.randn(content.shape[0], config.condition_dim, device=device)
                global_emb = F.normalize(global_emb, dim=-1)

            # Forward
            pred = model(content, global_emb)

            # Trim to match lengths
            min_len = min(pred.shape[-1], waveform.shape[-1])
            pred = pred[..., :min_len]
            waveform = waveform[..., :min_len]

            # Loss
            loss = mel_loss_fn(pred, waveform)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            if step % args.log_every == 0:
                log.info(
                    f"step={step:06d} loss={loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if step % args.save_every == 0:
                metrics = {"mel_loss": loss.item()}
                save_decoder_checkpoint(
                    Path(args.checkpoint_dir) / "decoder_phase0.last.pt",
                    model, step=step, metrics=metrics, optimizer=optimizer,
                )
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_decoder_checkpoint(
                        Path(args.checkpoint_dir) / "decoder_phase0.best.pt",
                        model, step=step, metrics=metrics,
                    )
                log.info(f"Saved checkpoint at step {step}")

    log.info("Phase 0 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Decoder pretraining")
    parser.add_argument("--data-dir", type=str, default="data/mio_vctk_full_compact")
    parser.add_argument("--audio-root", type=str, required=True, help="Root directory containing VCTK wav/flac files")
    parser.add_argument("--voicebank-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()
    train(args)
