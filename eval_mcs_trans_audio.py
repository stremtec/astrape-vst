"""Listening-set eval for MCS-Trans content through the MioCodec decoder.

For each source row this writes:

  source_ref.wav       original/source waveform used for global embedding
  teacher_recon.wav    cached teacher content + source global
  mcs_recon.wav        MCS-Trans content + source global
  target_ref.wav       optional target reference
  teacher_vc.wav       cached teacher content + target global
  mcs_vc.wav           MCS-Trans content + target global

The script can work without the original VCTK mount if source wavs exist under
refs/source_audio/{row}_{speaker}_input.wav.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

from train_mcs_trans import MCSTrans, MCSTransConfig  # noqa: E402
from mcs_calibrator import load_calibrator  # noqa: E402


MIO_CONFIG = Path(
    "/Users/asill/.cache/huggingface/hub/models--Aratako--MioCodec-25Hz-44.1kHz-v2/"
    "snapshots/67faba34153fe74e6665991c432a7327e23c5c1c/config.yaml"
)
MIO_WEIGHTS = Path(
    "/Users/asill/.cache/huggingface/hub/models--Aratako--MioCodec-25Hz-44.1kHz-v2/"
    "snapshots/67faba34153fe74e6665991c432a7327e23c5c1c/model.safetensors"
)
DEFAULT_DATA_DIR = Path("data/mio_vctk_full_compact")
DEFAULT_CHECKPOINT = Path("checkpoints/mcs_trans_t6_causal_ft/mcs_trans_t6_causal_ft.best.pt")
DEFAULT_SOURCE_AUDIO_DIR = Path("refs/source_audio")
DEFAULT_TARGET_AUDIO = Path("/Users/asill/Downloads/abcdefgh.mp3")
DEFAULT_OUT_DIR = Path("eval_out/mcs_trans_audio")
SAMPLE_RATE = 44100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-audio-dir", type=Path, default=DEFAULT_SOURCE_AUDIO_DIR)
    parser.add_argument("--source-row", type=int, action="append", default=None)
    parser.add_argument("--target-audio", type=Path, default=DEFAULT_TARGET_AUDIO)
    parser.add_argument("--target-voicebank", type=Path, default=None,
                        help="Path to .astrape voicebank; overrides --target-audio for VC.")
    parser.add_argument("--calibrator", type=Path, default=None)
    parser.add_argument(
        "--projection-mode",
        choices=["hard", "continuous"],
        default="hard",
        help="Use rounded FSQ codes or pre-round continuous codes before proj_out.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seconds", type=float, default=6.0)
    return parser.parse_args()


def load_wave(path: Path, sample_rate: int, max_seconds: float | None = None) -> torch.Tensor:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.asarray(data))
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
    except Exception:
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if max_seconds is not None:
        max_len = int(round(max_seconds * sample_rate))
        wav = wav[:max_len]
    return wav.contiguous().float()


def write_wave(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.detach().cpu().float().numpy(), sample_rate)


def load_mcs_trans(path: Path, device: torch.device) -> MCSTrans:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = MCSTransConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in checkpoint["config"].items()}
    )
    model = MCSTrans(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def load_mio(device: torch.device):
    import logging

    import miocodec

    logging.disable(logging.INFO)
    model = miocodec.MioCodecModel.from_pretrained(
        config_path=str(MIO_CONFIG),
        weights_path=str(MIO_WEIGHTS),
    )
    return model.to(device).eval()


def source_audio_path(source_audio_dir: Path, row: int, speaker: str) -> Path | None:
    direct = source_audio_dir / f"{row:05d}_{speaker}_input.wav"
    if direct.exists():
        return direct
    matches = sorted(source_audio_dir.glob(f"{row:05d}_{speaker}*input.wav"))
    return matches[0] if matches else None


def default_rows(source_audio_dir: Path, speakers: np.ndarray) -> list[int]:
    rows: list[int] = []
    for path in sorted(source_audio_dir.glob("*_input.wav")):
        parts = path.name.split("_")
        if not parts or not parts[0].isdigit():
            continue
        row = int(parts[0])
        if row < len(speakers):
            rows.append(row)
    return rows[:4]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.data_dir / "meta.npz", allow_pickle=False) as meta:
        n_samples = int(meta["n_samples"])
        speakers = meta["spk_names"][:n_samples].astype(str)

    rows = args.source_row or default_rows(args.source_audio_dir, speakers)
    if not rows:
        raise RuntimeError(
            "No source rows were supplied and no *_input.wav files were found in "
            f"{args.source_audio_dir}"
        )

    print(f"Loading MCS-Trans from {args.checkpoint}")
    mcs = load_mcs_trans(args.checkpoint, device)
    print(f"  params={sum(p.numel() for p in mcs.parameters()):,}")
    calibrator = None
    if args.calibrator is not None:
        print(f"Loading content calibrator from {args.calibrator}")
        calibrator = load_calibrator(args.calibrator, device)
        print(f"  params={sum(p.numel() for p in calibrator.parameters()):,}")
    print("Loading MioCodec")
    mio = load_mio(device)
    print("  OK")

    target_wave = None
    target_features = None
    if args.target_voicebank is not None and args.target_voicebank.exists():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from astrape.voicebank import VoiceBank
        vb = VoiceBank.load(args.target_voicebank)
        # Wrap into a lightweight namespace that has the .global_embedding attr MioCodec needs
        class _FakeFeatures:
            global_embedding = vb.global_embedding.to(device)
        target_features = _FakeFeatures()
        print(f"Target voicebank: {args.target_voicebank}  "
              f"({vb.duration_seconds:.1f}s, {vb.source_sample_rate}Hz)")
    elif args.target_audio.exists():
        target_wave = load_wave(args.target_audio, SAMPLE_RATE, max_seconds=args.max_seconds).to(device)
        target_features = mio.encode(target_wave, return_content=False, return_global=True)
        write_wave(args.out_dir / "target_ref.wav", target_wave, SAMPLE_RATE)
        print(f"Target: {args.target_audio}")
    else:
        print(f"Target missing, VC outputs skipped: {args.target_audio}")

    records = []
    for row in rows:
        speaker = str(speakers[row])
        src_path = source_audio_path(args.source_audio_dir, row, speaker)
        if src_path is None:
            print(f"[{row:05d}] {speaker}: source wav not found, skipping")
            continue

        source_wave = load_wave(src_path, SAMPLE_RATE, max_seconds=args.max_seconds).to(device)
        npz = np.load(args.data_dir / f"s_{row:05d}.npz", allow_pickle=False)
        mel_full = torch.from_numpy(npz["logmel"].astype(np.float32))
        cached_content = torch.from_numpy(npz["ce_768"].astype(np.float32)).to(device)

        # Keep the cached mel/content aligned to the exact waveform crop being
        # decoded. A mismatched content length can make even teacher recon sound
        # artificially bad.
        mel_frames = max(1, int(round(source_wave.numel() / SAMPLE_RATE * 50.0)))
        mel_frames = min(mel_frames, mel_full.shape[-1])
        content_frames = min(cached_content.shape[0], (mel_frames + 1) // 2)
        mel = mel_full[:, :mel_frames].unsqueeze(0).to(device)
        cached_content = cached_content[:content_frames]

        source_features = mio.encode(source_wave, return_content=True, return_global=True)
        teacher_content = source_features.content_embedding

        output = mcs(mel)
        if args.projection_mode == "continuous":
            pred_content = mcs.proj_out(output["codes"]).squeeze(0)
        else:
            pred_content = output["projected"].transpose(1, 2).squeeze(0)
        if calibrator is not None:
            pred_content = calibrator(pred_content.unsqueeze(0)).squeeze(0)
        length = min(pred_content.shape[0], teacher_content.shape[0], cached_content.shape[0])
        pred_content = pred_content[:length]
        teacher_content = teacher_content[:length]
        cached_content = cached_content[:length]
        cos768_fresh = F.cosine_similarity(pred_content, teacher_content, dim=-1).mean().item()
        cos768_cached = F.cosine_similarity(pred_content, cached_content, dim=-1).mean().item()

        row_dir = args.out_dir / f"{row:05d}_{speaker}"
        row_dir.mkdir(parents=True, exist_ok=True)
        write_wave(row_dir / "source_ref.wav", source_wave, SAMPLE_RATE)

        teacher_recon = mio.decode(
            global_embedding=source_features.global_embedding,
            content_embedding=teacher_content,
            target_audio_length=source_wave.numel(),
        )
        mcs_recon = mio.decode(
            global_embedding=source_features.global_embedding,
            content_embedding=pred_content,
            target_audio_length=source_wave.numel(),
        )
        write_wave(row_dir / "teacher_recon.wav", teacher_recon, SAMPLE_RATE)
        write_wave(row_dir / "mcs_recon.wav", mcs_recon, SAMPLE_RATE)

        record = {
            "row": row,
            "speaker": speaker,
            "source_path": str(src_path),
            "cos768_fresh_teacher": cos768_fresh,
            "cos768_cached": cos768_cached,
            "outputs": {
                "source_ref": str(row_dir / "source_ref.wav"),
                "teacher_recon": str(row_dir / "teacher_recon.wav"),
                "mcs_recon": str(row_dir / "mcs_recon.wav"),
            },
        }

        if target_features is not None:
            teacher_vc = mio.decode(
                global_embedding=target_features.global_embedding,
                content_embedding=teacher_content,
                target_audio_length=source_wave.numel(),
            )
            mcs_vc = mio.decode(
                global_embedding=target_features.global_embedding,
                content_embedding=pred_content,
                target_audio_length=source_wave.numel(),
            )
            write_wave(row_dir / "teacher_vc.wav", teacher_vc, SAMPLE_RATE)
            write_wave(row_dir / "mcs_vc.wav", mcs_vc, SAMPLE_RATE)
            record["outputs"]["target_ref"] = str(args.out_dir / "target_ref.wav")
            record["outputs"]["teacher_vc"] = str(row_dir / "teacher_vc.wav")
            record["outputs"]["mcs_vc"] = str(row_dir / "mcs_vc.wav")

        records.append(record)
        print(
            f"[{row:05d}] {speaker} "
            f"cos768_fresh={cos768_fresh:.4f} cos768_cached={cos768_cached:.4f} -> {row_dir}"
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "calibrator": str(args.calibrator) if args.calibrator is not None else None,
        "projection_mode": args.projection_mode,
        "target_audio": str(args.target_audio) if args.target_audio.exists() else None,
        "rows": records,
        "mean_cos768_fresh_teacher": (
            float(np.mean([r["cos768_fresh_teacher"] for r in records])) if records else 0.0
        ),
        "mean_cos768_cached": float(np.mean([r["cos768_cached"] for r in records])) if records else 0.0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"Mean cos768 fresh={summary['mean_cos768_fresh_teacher']:.4f} "
        f"cached={summary['mean_cos768_cached']:.4f}"
    )
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
