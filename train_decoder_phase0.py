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
        target_sr: int = 44100,
        max_content_frames: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.audio_root = Path(audio_root)
        self.target_sr = target_sr
        self.max_content_frames = max_content_frames

        meta = np.load(self.data_dir / "meta.npz")
        self.n_samples = int(meta["n_samples"])
        self.source_files = meta["source_files"][: self.n_samples]
        self.speakers = meta["spk_names"][: self.n_samples].astype(str)

        # Build speaker → global embedding map (loaded lazily)
        self._speaker_embeddings: dict[str, torch.Tensor] = {}

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

            return {
                "content": ce_768,
                "waveform": waveform,
                "speaker": self.speakers[index],
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
    for b in batch:
        c = b["content"]
        w = b["waveform"]
        # Pad to max
        if c.shape[0] < max_frames:
            c = F.pad(c, (0, 0, 0, max_frames - c.shape[0]))
        if w.shape[0] < max_samples:
            w = F.pad(w, (0, max_samples - w.shape[0]))
        contents.append(c)
        waveforms.append(w)

    return {
        "content": torch.stack(contents),
        "waveform": torch.stack(waveforms),
    }


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

    # Data
    dataset = DecoderPretrainDataset(
        args.data_dir, args.audio_root,
        target_sr=config.sample_rate,
        max_content_frames=args.max_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_skip_none,
        drop_last=True,
        pin_memory=True,
    )

    # Load global embeddings for conditioning
    # For pretraining we use a fixed set or random conditioning
    # since we don't have per-utterance global embeddings in the cache.
    # We'll use the voicebank if available, otherwise random 128d.
    voicebank_dir = Path(args.voicebank_dir) if args.voicebank_dir else None

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

            # Use zero global embedding for now (decoder must still use content)
            # In practice, we'd load per-speaker embeddings
            global_emb = torch.zeros(content.shape[0], 128, device=device)

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
