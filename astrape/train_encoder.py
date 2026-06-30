"""MCS-Trans with Q2D2 quantizer — drop-in FSQ replacement.

Clone of train_mcs_trans.py with the FSQ (5-axis independent scalar)
replaced by Q2D2 (3-pair 2D grid quantization from ICML 2026 paper
"Two-Dimensional Quantization for Geometry-Aware Audio Coding").

Same conv frontend + causal transformer.  Only the quantizer head
and losses change.

Reference:
  train_mcs_trans.py   — original MCS-Trans with FSQ
  mcs_q2d2.py          — Q2D2 quantizer implementation
  https://arxiv.org/abs/2512.01537
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

from .encoder import MCSTransQ2D2, MCSTransQ2D2Config, DEFAULT_Q2D2_LEVELS
from .losses import q2d2_losses, multi_resolution_stft_loss
from .data import (
    MioCompactDataset, ContentCollator,
    split_by_speaker, speaker_balanced_subset, move_batch,
    DEFAULT_DATA_DIR, DEFAULT_PROJECTION,
)
from .quantizer import Q2D2Quantizer

DEFAULT_OUT_DIR = Path("checkpoints/mcs_trans_q2d2")


# ── checkpoint I/O (moved from mcs_common.py) ──

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    best_cos: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model.__class__.__name__,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }
    if "Q2D2" in model.__class__.__name__:
        payload["best_probe_cos768"] = best_cos
    else:
        payload["best_probe_cos5"] = best_cos
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ── dataset wrappers ──

class CenterFalseMelWrapper(Dataset):
    """Wraps a base dataset, recomputing mel from raw audio with center=False."""

    SAMPLE_RATE = 44100

    def __init__(self, base, srcs):
        self.base = base
        self.src = srcs

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        import soundfile as sf
        import torchaudio

        sample = self.base[idx]
        si = int(sample['idx'])
        w, sr = sf.read(str(Path(self.src[si])), dtype='float32')
        w = torch.from_numpy(np.asarray(w))
        if w.ndim == 2:
            w = w.mean(1)
        if sr != self.SAMPLE_RATE:
            w = torchaudio.functional.resample(
                w.unsqueeze(0), sr, self.SAMPLE_RATE
            ).squeeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            self.SAMPLE_RATE, 2048, 882, n_mels=80, f_min=0.0,
            f_max=self.SAMPLE_RATE / 2.0, power=1, center=False,
        )(w.unsqueeze(0))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        sample['mel'] = mel[0]
        return sample


class WavLMFrontendDataset(Dataset):
    """Replaces 'mel' with cached WavLM CNN features (`astrape.cache --what wavlm`).

    The npz cache stores 'wavlm_cnn' as (T, 512) float32 at ~46 Hz.
    We transpose to (512, T) so ContentCollator treats the second dim
    (time) correctly for cropping/padding.

    ContentCollator's mel_frames then controls the number of WavLM
    output frames to keep, just like for mel.
    """

    def __init__(self, base: Dataset, data_dir: Path, wavlm_dir: str = "wavlm_16k"):
        self.base = base
        self.data_dir = data_dir
        self.wavlm_dir = wavlm_dir

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        si = int(sample['idx'])
        # Load cached WavLM CNN from .npy file
        cnn_path = Path(self.wavlm_dir) / f's_{si:05d}.npy' if Path(self.wavlm_dir).is_absolute() else self.data_dir / self.wavlm_dir / f's_{si:05d}.npy'
        if not cnn_path.exists():
            raise RuntimeError(
                f"WavLM frontend: sample s_{si:05d} has no wavlm_cnn cache. "
                f"Re-run `astrape.cache --what wavlm` or disable --wavlm-frontend."
            )
        cnn = np.load(cnn_path, allow_pickle=False)
        # wavlm_cnn: (T, 512) → transpose → (512, T)
        sample['mel'] = torch.from_numpy(cnn.astype(np.float32)).t()
        return sample


# ── evaluation ──

@torch.inference_mode()
def evaluate(
    model: MCSTransQ2D2,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    quantizer: Q2D2Quantizer | None = None,
    speaker_to_id: dict[str, int] | None = None,
) -> dict[str, float]:
    model.eval()
    buckets: dict[str, list[float]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        speaker_ids = None
        if model.speaker_classifier is not None and speaker_to_id is not None:
            speaker_ids = torch.tensor(
                [speaker_to_id[s] for s in batch.speakers],
                device=device, dtype=torch.long,
            )
        output = model(batch.mel, padding_mask=batch.mask)
        _, metrics = q2d2_losses(output, batch, args, quantizer,
                                 model.speaker_classifier, speaker_ids,
                                 time_shift=args.time_shift,
                                 ssl_heads=model.ssl_heads,
                                 ssl_layers=model.config.ssl_layers)
        for key, value in metrics.items():
            buckets.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(vs)) if vs else 0.0
            for key, vs in buckets.items()}


# ── argument parsing ──

def parse_q2d2_levels(s: str) -> tuple[int, ...]:
    """Parse comma-separated level string, e.g. '7,7,7,7,7,7'."""
    parts = [v.strip() for v in s.split(",") if v.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("empty level list")
    levels = tuple(int(p) for p in parts)
    if len(levels) % 2 != 0:
        raise argparse.ArgumentTypeError(
            f"Q2D2 requires even number of levels, got {len(levels)}"
        )
    return levels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCS-Trans with Q2D2 quantizer (replaces FSQ)",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mel-frames", type=int, default=200)
    p.add_argument("--eval-mel-frames", type=int, default=300)
    p.add_argument("--probe-samples", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--init-from", type=Path, default=None,
                   help="Initialize from FSQ checkpoint (proj_out skipped).")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--run-name", default="mcs_trans_q2d2")

    # Transformer architecture (unchanged)
    p.add_argument("--conv-dim", type=int, default=320)
    p.add_argument("--trans-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--window", type=int, default=256)

    # Transformer improvements
    p.add_argument("--rope", action="store_true",
                   help="Use Rotary Position Embedding in transformer attention.")
    p.add_argument("--swiglu", action="store_true",
                   help="Use SwiGLU FFN (gated SiLU) instead of plain SiLU.")

    # Q2D2-specific
    p.add_argument("--q2d2-dim", type=int, default=6,
                   help="Q2D2 latent dimension (must be even).")
    p.add_argument("--q2d2-levels", type=parse_q2d2_levels,
                   default=DEFAULT_Q2D2_LEVELS,
                   help="Per-dimension levels, comma-separated (even count). "
                        "Default: 7,7,7,7,7,7 → 3 rhombic pairs.")
    p.add_argument("--q2d2-grid", default="rhombic",
                   choices=["rhombic", "hexagon", "rectangle"],
                   help="2D grid geometry type.")
    p.add_argument("--q2d2-l2-norm", action="store_true",
                   help="L2-normalize Q2D2 feature pairs before grid snap.")

    # Loss weights
    p.add_argument("--content-cos-weight", type=float, default=1.0,
                   help="Weight on (1 - cos768) content fidelity loss.")
    p.add_argument("--content-l1-weight", type=float, default=0.5,
                   help="Weight on L1 content loss.")
    p.add_argument("--delta-weight", type=float, default=0.04,
                   help="Weight on temporal delta smoothness.")
    p.add_argument("--voiced-boost", type=float, default=1.0,
                   help="Voiced frame weight multiplier.")

    # GRL speaker disentanglement
    p.add_argument("--grl-weight", type=float, default=0.0,
                   help="GRL speaker disentanglement weight (0=disabled, ~0.1).")
    p.add_argument("--grl-num-speakers", type=int, default=0,
                   help="Number of speakers for GRL classifier (auto if 0).")
    p.add_argument("--time-shift", type=int, default=0,
                   help="Shift teacher target by Δ frames. 1 frame = 40ms.")
    p.add_argument("--forecast-weight", type=float, default=0.0,
                   help="Weight on forecast heads.")
    p.add_argument("--stem-block-type", default="depthwise",
                   choices=["standard","depthwise"],
                   help="Conv stem block type.")
    p.add_argument("--center-false", action="store_true",
                   help="Compute center=False mel on-the-fly from raw audio.")

    # Decoder-in-loop (original audio feedback)
    p.add_argument("--decoder-wave-weight", type=float, default=0.0,
                   help="Weight for MR-STFT loss against original wav "
                        "via frozen Mio decoder.")
    p.add_argument("--decoder-wave-prob", type=float, default=0.5,
                   help="Fraction of steps that compute decoder wave loss.")
    p.add_argument("--decoder-n-ffts", default="512,1024,2048")

    # ── recovered features ──
    p.add_argument("--delta2-weight", type=float, default=0.0,
                   help="Weight on 2nd-order temporal smoothness.")
    p.add_argument("--contrastive-weight", type=float, default=0.0,
                   help="Weight on InfoNCE contrastive loss.")
    p.add_argument("--contrastive-tau", type=float, default=0.1,
                   help="InfoNCE temperature τ.")
    p.add_argument("--ssl-weight", type=float, default=0.0,
                   help="Weight on WavLM multi-target distillation (L0,L4,L8).")
    p.add_argument("--ssl-layers", default="0,4,8",
                   help="Comma-separated WavLM layer targets for SSL distillation.")
    p.add_argument("--mamba-layers", type=int, default=0,
                   help="Number of trailing layers replaced by Mamba/SSM.")
    p.add_argument("--mamba-d-state", type=int, default=16,
                   help="Mamba SSM state dimension (d_state).")
    p.add_argument("--q2d2-gumbel-start", type=float, default=0.0,
                   help="Initial Gumbel temperature for Q2D2 relaxation.")
    p.add_argument("--q2d2-gumbel-end", type=float, default=0.0,
                   help="Final Gumbel temperature for Q2D2 relaxation (anneal).")
    p.add_argument("--wavlm-frontend", action="store_true",
                   help="Use cached WavLM CNN features instead of mel.")
    p.add_argument("--wavlm-dir", default="wavlm_16k",
                   help="Subdirectory for WavLM cache (default: wavlm_16k, use wavlm_L4 for L4)")
    p.add_argument("--wavlm-rate", type=int, default=50,
                   help="WavLM feature rate in Hz (50=default, 200=L4 raw)")

    return p.parse_args()


# ── main training loop ──

def main() -> None:
    args = parse_args()
    if args.resume_from is not None and args.init_from is not None:
        raise ValueError("--resume-from and --init-from are mutually exclusive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── data ──
    with np.load(args.data_dir / "meta.npz", allow_pickle=False) as meta:
        n_samples = int(meta["n_samples"])
        speakers = meta["spk_names"][:n_samples].astype(str)
        source_files = meta["source_files"][:n_samples].astype(str)
        cache_format = str(meta["cache_format"].item())
    if cache_format != "compact-fp16-ctc-v2":
        raise ValueError(f"unexpected cache format: {cache_format}")

    # Speaker ID mapping for GRL
    unique_speakers = sorted(set(speakers))
    speaker_to_id = {s: i for i, s in enumerate(unique_speakers)}
    print(f"Speakers: {len(unique_speakers)} unique, "
          f"GRL={args.grl_weight > 0}", flush=True)

    train_idx, val_idx = split_by_speaker(speakers, args.val_fraction, args.seed)
    probe_idx = speaker_balanced_subset(val_idx, speakers, args.probe_samples, args.seed)

    train_ds = MioCompactDataset(args.data_dir, train_idx, speakers)
    probe_ds = MioCompactDataset(args.data_dir, probe_idx, speakers)

    if args.center_false:
        train_ds = CenterFalseMelWrapper(train_ds, source_files)
        probe_ds = CenterFalseMelWrapper(probe_ds, source_files)
        print("center=False mel: computing on-the-fly from raw audio", flush=True)

    if args.wavlm_frontend:
        train_ds = WavLMFrontendDataset(train_ds, args.data_dir, args.wavlm_dir)
        probe_ds = WavLMFrontendDataset(probe_ds, args.data_dir, args.wavlm_dir)
        print(f"WavLM frontend: using cached {args.wavlm_dir} (512d) instead of mel",
              flush=True)

    # Frontend frames per 25Hz teacher token: Mel/50Hz-WavLM = 2, 200Hz L4 = 8.
    frames_per_token = (args.wavlm_rate // 25) if args.wavlm_frontend else 2
    # persistent_workers: keep workers alive across epochs instead of respawning
    # them each epoch (the respawn leaks ~17 fds/epoch and hits macOS's default
    # ulimit -n=256 after ~15 epochs). Requires num_workers > 0.
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        collate_fn=ContentCollator(args.mel_frames, args.seed,
                                   pad_mel_multiple=frames_per_token,
                                   frames_per_token=frames_per_token),
        generator=torch.Generator().manual_seed(args.seed),
    )
    probe_loader = DataLoader(
        probe_ds,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        collate_fn=ContentCollator(args.eval_mel_frames, args.seed + 999,
                                   pad_mel_multiple=frames_per_token,
                                   frames_per_token=frames_per_token),
    )

    # ── config ──
    ssl_layer_list = tuple(
        int(v.strip()) for v in args.ssl_layers.split(",") if v.strip()
    )
    config = MCSTransQ2D2Config(
        conv_dim=args.conv_dim,
        trans_dim=args.trans_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        window=args.window,
        use_rope=args.rope,
        use_swiglu=args.swiglu,
        q2d2_dim=args.q2d2_dim,
        q2d2_levels=args.q2d2_levels,
        q2d2_grid=args.q2d2_grid,
        grl_weight=args.grl_weight,
        grl_num_speakers=args.grl_num_speakers if args.grl_num_speakers > 0 else len(unique_speakers),
        stem_block_type=args.stem_block_type,
        use_wavlm_frontend=args.wavlm_frontend,
        wavlm_rate=args.wavlm_rate,
        delta2_weight=args.delta2_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_tau=args.contrastive_tau,
        ssl_weight=args.ssl_weight,
        ssl_layers=ssl_layer_list,
        mamba_layers=args.mamba_layers,
        mamba_d_state=args.mamba_d_state,
        q2d2_gumbel_start=args.q2d2_gumbel_start,
        q2d2_gumbel_end=args.q2d2_gumbel_end,
        q2d2_l2_norm=args.q2d2_l2_norm,
    )

    # ── model ──
    checkpoint = None
    checkpoint_mode = None
    start_epoch, best_cos, global_step = 0, -1.0, 0
    checkpoint_path = args.resume_from or args.init_from

    if checkpoint_path is not None:
        checkpoint_mode = "resume" if args.resume_from is not None else "init"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if checkpoint_mode == "init":
            # Loading from FSQ checkpoint: extract config and weights
            # but skip axis_heads, ordinal_heads, proj_out (replaced by Q2D2)
            saved_cfg = checkpoint.get("config", {})
            # Map old config to new Q2D2 config (only keep shared fields)
            config = MCSTransQ2D2Config(
                conv_dim=saved_cfg.get("conv_dim", config.conv_dim),
                trans_dim=saved_cfg.get("trans_dim", config.trans_dim),
                n_heads=saved_cfg.get("n_heads", config.n_heads),
                n_layers=saved_cfg.get("n_layers", config.n_layers),
                ffn_dim=saved_cfg.get("ffn_dim", config.ffn_dim),
                window=saved_cfg.get("window", config.window),
                use_rope=args.rope,
                use_swiglu=args.swiglu,
                q2d2_dim=args.q2d2_dim,
                q2d2_levels=args.q2d2_levels,
                q2d2_grid=args.q2d2_grid,
                grl_weight=args.grl_weight,
                grl_num_speakers=len(unique_speakers),
                use_wavlm_frontend=args.wavlm_frontend,
                delta2_weight=args.delta2_weight,
                contrastive_weight=args.contrastive_weight,
                contrastive_tau=args.contrastive_tau,
                ssl_weight=args.ssl_weight,
                ssl_layers=ssl_layer_list,
                mamba_layers=args.mamba_layers,
                mamba_d_state=args.mamba_d_state,
                q2d2_gumbel_start=args.q2d2_gumbel_start,
                q2d2_gumbel_end=args.q2d2_gumbel_end,
                q2d2_l2_norm=args.q2d2_l2_norm,
            )
            # Resume from Q2D2 checkpoint
            saved_cfg = checkpoint.get("config", {})
            known = {f.name for f in __import__("dataclasses").fields(MCSTransQ2D2Config)}
            cfg_filtered = {}
            for k, v in saved_cfg.items():
                if k in known:
                    cfg_filtered[k] = tuple(v) if isinstance(v, list) else v
            config = MCSTransQ2D2Config(**cfg_filtered)

    model = MCSTransQ2D2(config).to(device)

    # Load shared weights from FSQ checkpoint (conv + transformer)
    if checkpoint is not None and checkpoint_mode == "init":
        fsq_state = checkpoint.get("state_dict", checkpoint)
        # Load all matching keys. Skip only old FSQ-specific heads.
        # Q2D2 weights ARE loaded if present (Q2D2→Q2D2 transfer).
        shared_state = {}
        for k, v in fsq_state.items():
            if k.startswith("axis_heads."):
                continue  # old FSQ, removed in Q2D2
            if k.startswith("ordinal_heads."):
                continue  # old FSQ, removed in Q2D2
            if k.startswith("proj_out."):
                continue  # old FSQ proj_out, Q2D2 has its own
            shared_state[k] = v

        missing, unexpected = model.load_state_dict(shared_state, strict=False)
        if missing:
            print(f"Q2D2 init: missing keys (new Q2D2 head, expected): {len(missing)}")
            for m in missing[:5]:
                print(f"  {m}")
        if unexpected:
            print(f"Q2D2 init: unexpected keys (old FSQ): {len(unexpected)}")
        print(f"Initialized conv+transformer from {checkpoint_path}", flush=True)

    elif checkpoint is not None and checkpoint_mode == "resume":
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}", flush=True)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("metrics", {}).get("global_step", 0))
        current_cos = float(checkpoint.get("metrics", {}).get("probe", {}).get("cos768", -1.0))
        best_cos = max(float(checkpoint.get("best_probe_cos768", -1.0)), current_cos)
        print(f"Resumed from {checkpoint_path} at epoch={start_epoch} "
              f"global_step={global_step} best_cos768={best_cos:.4f}", flush=True)

    # ── optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs),
    )

    if checkpoint is not None and checkpoint_mode == "resume":
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                print("Optimizer mismatch, starting fresh")
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    # ── decoder-in-loop setup ──
    mio = None
    decoder_n_ffts: tuple[int, ...] = ()
    if args.decoder_wave_weight > 0:
        from .miocodec import load_mio, load_wave, SAMPLE_RATE  # noqa: E402
        print("Loading frozen MioCodec for decoder-in-loop ...", flush=True)
        mio = load_mio(device).eval()
        for p in mio.parameters():
            p.requires_grad_(False)
        decoder_n_ffts = tuple(
            int(v) for v in args.decoder_n_ffts.split(",") if v.strip()
        )
        print(f"  decoder_wave_weight={args.decoder_wave_weight} "
              f"prob={args.decoder_wave_prob} n_ffts={decoder_n_ffts}", flush=True)

    # ── info ──
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    q2d2_cbs = model.q2d2.codebook_size

    print(f"Train={len(train_idx)} Probe={len(probe_idx)} "
          f"speakers={len(set(speakers))} "
          f"params={params:,} trainable={trainable:,} device={device}", flush=True)
    print(f"Arch: trans_dim={config.trans_dim} n_heads={config.n_heads} "
          f"n_layers={config.n_layers} ffn_dim={config.ffn_dim} "
          f"window={config.window} "
          f"rope={config.use_rope} swiglu={config.use_swiglu}", flush=True)
    print(f"Q2D2: grid={config.q2d2_grid} dim={config.q2d2_dim} "
          f"levels={list(config.q2d2_levels)} "
          f"codebook={q2d2_cbs:,} (FSQ=12,800)", flush=True)
    print(f"Objective: content_cos={args.content_cos_weight} "
          f"content_l1={args.content_l1_weight} "
          f"delta={args.delta_weight} "
          f"delta2={args.delta2_weight} "
          f"decoder_wave={args.decoder_wave_weight}", flush=True)
    print(f"Extra: contrastive={args.contrastive_weight} "
          f"ssl={args.ssl_weight} "
          f"mamba_layers={config.mamba_layers}/{config.n_layers} "
          f"gumbel={args.q2d2_gumbel_start:.1f}→{args.q2d2_gumbel_end:.1f} "
          f"wavlm_frontend={args.wavlm_frontend} "
          f"stem={args.stem_block_type}"
          f"l2_norm={config.q2d2_l2_norm}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        probe = evaluate(model, probe_loader, device, args, model.q2d2.quantizer,
                          speaker_to_id)
        print(f"eval probe cos768={probe.get('cos768', 0):.4f} "
              f"loss={probe.get('loss', 0):.4f} "
              f"q2d2_usage={probe.get('q2d2_usage', 0):.3f}", flush=True)
        return

    # ── training loop ──
    quantizer = model.q2d2.quantizer  # for utilization stats
    run_started = time.time()

    def _gumbel_temp(epoch: int) -> float:
        """Linear annealing from q2d2_gumbel_start → q2d2_gumbel_end."""
        start = args.q2d2_gumbel_start
        end = args.q2d2_gumbel_end
        if start <= 0.0:
            return 0.0
        E = max(1, args.epochs - start_epoch)
        frac = min(1.0, epoch / max(1, E - 1))
        return start + (end - start) * frac

    # ── trend tracker ──
    prev_cos768 = -1.0
    prev_usage = 0.0
    total_steps_all = max(1, args.steps_per_epoch) * max(1, args.epochs - start_epoch)

    for epoch in range(start_epoch, args.epochs):
        # ── Gumbel temperature annealing ──
        tau = _gumbel_temp(epoch)
        if args.q2d2_gumbel_start > 0:
            quantizer.gumbel_temperature = tau

        model.train()
        totals: dict[str, float] = {}
        step_started = time.time()

        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = move_batch(batch, device)

            # Build speaker IDs for GRL
            speaker_ids = None
            if model.speaker_classifier is not None:
                speaker_ids = torch.tensor(
                    [speaker_to_id[s] for s in batch.speakers],
                    device=device, dtype=torch.long,
                )

            output = model(batch.mel, padding_mask=batch.mask)
            loss, metrics = q2d2_losses(output, batch, args, quantizer,
                                        model.speaker_classifier, speaker_ids,
                                        time_shift=args.time_shift,
                                        ssl_heads=model.ssl_heads,
                                        ssl_layers=config.ssl_layers)

            # Decoder-in-loop: MR-STFT vs original wav
            if (mio is not None and
                    args.decoder_wave_weight > 0 and
                    random.random() < args.decoder_wave_prob):
                if True:  # no try/except - let errors surface
                    idx_in_batch = random.randrange(len(batch.speakers))
                    row = int(batch.indices[idx_in_batch].item())
                    src_path = Path(str(source_files[row]))
                    if src_path.exists():
                        from .miocodec import load_wave, SAMPLE_RATE  # noqa: E402
                        orig_wave = load_wave(src_path, SAMPLE_RATE,
                                              max_seconds=10.0).to(device)
                        # Crop to match the EXACT mel window used in training
                        mel_crop_start = int(batch.crop_starts[idx_in_batch].item())
                        wav_start = mel_crop_start * 882
                        wav_len = int(args.mel_frames * 882)
                        orig_wave = orig_wave[wav_start:wav_start + wav_len]
                        with torch.no_grad():
                            feats = mio.encode(orig_wave.unsqueeze(0),
                                              return_content=True,
                                              return_global=True)
                            global_emb = feats.global_embedding.unsqueeze(0)
                            stft_len = mio._calculate_target_stft_length(
                                orig_wave.numel()
                            )
                        content_i = output["projected"][idx_in_batch].unsqueeze(0)
                        content_i_t = content_i.transpose(1, 2)
                        n_frames = min(content_i_t.shape[1],
                                       feats.content_embedding.shape[0])
                        pred_wave = mio.forward_wave(
                            content_i_t[:, :n_frames], global_emb,
                            stft_length=stft_len,
                        ).squeeze(0)
                        tgt_len = min(pred_wave.shape[-1],
                                      orig_wave.shape[-1])
                        wave_loss = multi_resolution_stft_loss(
                            pred_wave[:tgt_len], orig_wave[:tgt_len],
                            decoder_n_ffts,
                        )
                        loss = loss + args.decoder_wave_weight * wave_loss
                        metrics["decoder_wave"] = float(
                            wave_loss.detach().cpu()
                        )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            global_step += 1

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value

            if step % args.log_every == 0 or step == args.steps_per_epoch:
                s = max(step, 1)
                elapsed = time.time() - step_started
                # Progress + trend + ETA
                pct = step * 100 // args.steps_per_epoch
                bar = '\u2588' * (pct // 5) + '\u2591' * (20 - pct // 5)
                steps_done = (epoch - start_epoch) * args.steps_per_epoch + step
                sps = elapsed / s
                eta_h = sps * (total_steps_all - steps_done) / 3600

                # Combined trend: cos direction + usage direction
                avg_cos = totals['cos768'] / s
                avg_use = totals.get('q2d2_usage', 0) / s
                if prev_cos768 > 0:
                    d_cos = avg_cos - prev_cos768
                    d_use = avg_use - prev_usage

                    if d_cos > 0.001:
                        cd = '\u2191'  # cos↑
                    elif d_cos < -0.001:
                        cd = '\u2193'  # cos↓
                    else:
                        cd = '\u2192'  # cos→

                    if d_use < -0.003:
                        ud = '\u2699'   # usage↓ (compression)
                    elif d_use > 0.003:
                        ud = '\u2b06'   # usage↑ (expansion)
                    else:
                        ud = '\u2194'   # usage↔

                    trend = f'{ud}{cd}'
                else:
                    trend = '\u00b7'
                prev_cos768, prev_usage = avg_cos, avg_use

                print(
                    f"E{epoch:03d} [{bar}] {pct:3d}% "
                    f"| loss={totals['loss']/s:.4f} cos={avg_cos:.4f} "
                    f"l1={totals.get('content_l1',0)/s:.4f} "
                    f"use={avg_use:.3f} {trend} "
                    f"| {sps:.2f}s | ETA {eta_h:.1f}h",
                    flush=True,
                )

        scheduler.step()
        probe = evaluate(model, probe_loader, device, args, quantizer,
                          speaker_to_id)
        metrics_full = {
            "epoch": epoch,
            "global_step": global_step,
            "probe": probe,
            "elapsed_seconds": time.time() - run_started,
        }

        pc = probe.get('cos768', 0)
        ep_done = epoch - start_epoch + 1
        pct_done = ep_done * 100 // max(1, args.epochs - start_epoch)
        elapsed_h = (time.time() - run_started) / 3600
        print(
            f"E{epoch:03d} \u2713 probe cos={pc:.4f} "
            f"loss={probe.get('loss',0):.1f} l1={probe.get('content_l1',0):.4f} "
            f"use={probe.get('q2d2_usage',0):.3f} "
            f"| {pct_done}% done \u00b7 {elapsed_h:.1f}h",
            flush=True,
        )

        save_checkpoint(
            args.out_dir / f"{args.run_name}.last.pt",
            model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
        )

        current_cos = probe.get("cos768", 0.0)
        if current_cos > best_cos:
            best_cos = current_cos
            save_checkpoint(
                args.out_dir / f"{args.run_name}.best.pt",
                model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
            )

        if args.save_every_epoch:
            save_checkpoint(
                args.out_dir / f"{args.run_name}.epoch{epoch:03d}.pt",
                model, optimizer, scheduler, epoch, metrics_full, args, best_cos,
            )

        (args.out_dir / f"{args.run_name}.summary.json").write_text(
            json.dumps(metrics_full, indent=2, sort_keys=True) + "\n"
        )

    print(f"done best_cos768={best_cos:.4f} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
