"""Extract WavLM[6,9] features and cache as NPZ for fast training.

Reads audio from external drive, runs frozen WavLM, avg-pools to 25Hz,
and saves alongside the existing content cache.

Output: data/mio_vctk_full_compact/ssl_{index:05d}.npz with key 'wavlm_25hz' (T, 768)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.pipelines as pipelines
import torchaudio.transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_wavlm(device="cpu"):
    bundle = pipelines.WAVLM_BASE_PLUS
    model = bundle.get_model().to(device)
    model.eval()
    return model, bundle.sample_rate


def resolve_audio(source_file: str, audio_root: Path) -> Path:
    p = Path(str(source_file))
    if p.exists():
        return p
    parts = p.parts
    for i in range(len(parts)):
        candidate = audio_root / "/".join(parts[i:])
        if candidate.exists():
            return candidate
    matches = list(audio_root.rglob(p.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find: {source_file}")


def cached_content_length(data_dir: Path, index: int) -> int | None:
    sample_path = data_dir / f"s_{index:05d}.npz"
    if not sample_path.exists():
        return None
    with np.load(sample_path) as data:
        if "ce_768" not in data:
            return None
        return int(data["ce_768"].shape[0])


def align_target_frames(
    wavlm_25hz: np.ndarray,
    content_length: int | None,
) -> np.ndarray:
    """Align WavLM targets to cached content length when available."""
    if content_length is None or wavlm_25hz.shape[0] == content_length:
        return wavlm_25hz
    if wavlm_25hz.ndim != 2:
        raise ValueError(f"wavlm_25hz must be 2D, got shape {wavlm_25hz.shape}")
    if content_length < 0:
        raise ValueError("content_length must be non-negative")
    if content_length == 0:
        return wavlm_25hz[:0]
    if wavlm_25hz.shape[0] > content_length:
        return wavlm_25hz[:content_length]
    if wavlm_25hz.shape[0] == 0:
        raise ValueError("Cannot pad an empty WavLM target")
    pad_count = content_length - wavlm_25hz.shape[0]
    padding = np.repeat(wavlm_25hz[-1:], pad_count, axis=0)
    return np.concatenate([wavlm_25hz, padding], axis=0)


def is_valid_ssl_cache(path: Path, content_length: int | None = None) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            if "wavlm_25hz" not in data:
                return False
            target = data["wavlm_25hz"]
            if target.ndim != 2 or target.shape[1] != 768:
                return False
            if content_length is not None and target.shape[0] != content_length:
                return False
    except Exception:
        return False
    return True


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ through a same-directory temp file, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp_path.replace(path)
        try:
            with np.load(path) as data:
                for key, array in arrays.items():
                    if key not in data or data[key].shape != array.shape:
                        raise ValueError(f"atomic write validation failed for {key}")
        except Exception:
            path.unlink(missing_ok=True)
            raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main(args):
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    audio_root = Path(args.audio_root)

    meta = np.load(data_dir / "meta.npz")
    n_samples = int(meta["n_samples"])
    source_files = meta["source_files"][:n_samples]
    log.info(f"Processing {n_samples} samples")

    # Load WavLM
    wavlm, ssl_sr = load_wavlm(device)
    log.info(f"WavLM loaded on {device}, ssl_sr={ssl_sr}")

    # Check what's already done
    start_idx = 0
    if args.resume:
        for i in range(n_samples):
            content_length = cached_content_length(data_dir, i)
            if not is_valid_ssl_cache(
                data_dir / f"ssl_{i:05d}.npz", content_length
            ):
                start_idx = i
                break
        else:
            start_idx = n_samples
        log.info(f"Resuming from index {start_idx}")

    t0 = time.time()
    done = 0
    errors = 0

    for i in range(start_idx, n_samples):
        out_path = data_dir / f"ssl_{i:05d}.npz"
        content_length = cached_content_length(data_dir, i)
        if is_valid_ssl_cache(out_path, content_length):
            done += 1
            continue

        try:
            audio_path = resolve_audio(str(source_files[i]), audio_root)
            data, sr = sf.read(str(audio_path))
            wav = torch.from_numpy(data.astype(np.float32))
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)

            # Resample to 16kHz
            if sr != ssl_sr:
                resampler = torchaudio.transforms.Resample(sr, ssl_sr)
                wav = resampler(wav)

            wav = wav.unsqueeze(0).to(device)

            with torch.no_grad():
                features = wavlm.extract_features(wav, num_layers=9)[0]
                # Average layers 6 and 9 (0-indexed: 5 and 8)
                ssl_50hz = (features[5] + features[8]) / 2  # (1, T_50hz, 768)
                # Pool to 25Hz
                ssl_25hz = F.avg_pool1d(
                    ssl_50hz.transpose(1, 2), kernel_size=2, stride=2
                ).transpose(1, 2).squeeze(0).cpu().numpy()  # (T_25hz, 768)

            ssl_25hz = align_target_frames(ssl_25hz, content_length)
            save_npz_atomic(out_path, wavlm_25hz=ssl_25hz)
            done += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                log.warning(f"Error at index {i}: {e}")

        if (done + errors) % args.log_every == 0:
            elapsed = time.time() - t0
            rate = (done + errors) / elapsed if elapsed > 0 else 0
            eta = (n_samples - start_idx - done - errors) / rate if rate > 0 else 0
            log.info(
                f"Progress: {done}/{n_samples} done, {errors} errors, "
                f"{rate:.1f} samples/s, ETA {eta/60:.0f}min"
            )

    elapsed = time.time() - t0
    log.info(f"Complete: {done} extracted, {errors} errors in {elapsed/60:.1f}min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WavLM features to NPZ cache")
    parser.add_argument("--data-dir", type=str, default="data/mio_vctk_full_compact")
    parser.add_argument("--audio-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    main(args)
