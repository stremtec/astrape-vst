"""Phase 0: Train CausalDecoder with frozen Q2D2 encoder + MR-STFT loss.

Follows cdecoder.md training strategy:
  1. Teacher Forcing on STFT — decoder output vs original audio MR-STFT
  2. Gaussian-blurred teacher target (causal model이 따라가기 쉽게)
  3. Frozen encoder supplies Q2D2 content @ 25Hz
  4. Speaker embeddings from pre-extracted spk_1k.npy

Pipeline:
  VCTK audio 44.1kHz → WavLM CNN 16kHz cache → [frozen encoder] → content 768d @ 25Hz
  → CausalDecoder(content, speaker) → waveform 44.1kHz
  → MR-STFT(waveform, original_audio_blurred) → loss

Usage:
  .venv/bin/python train_decoder.py --device mps --epochs 30 --steps-per-epoch 1000
"""

import sys, warnings, logging, argparse, random, time, json, math
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
sys.path.insert(0, ".")

import torch, torchaudio, numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from mcs_common import multi_resolution_stft_loss
from astrape.causal_decoder import CausalDecoderConfig, CausalDecoder

S = 44100

# ═══════════════════════════════════════════════════════════════════
# Gaussian blur (cdecoder.md: "Teacher STFT에 시간축 가우시안 블러")
# ═══════════════════════════════════════════════════════════════════

def gaussian_blur_wave(wave: torch.Tensor, sigma_ms: float = 2.0) -> torch.Tensor:
    """Apply Gaussian blur in time domain to smooth teacher target.

    cdecoder.md: causal 모델이 미래 문맥 없이도 따라갈 수 있도록
    teacher 파형을 부드럽게 만든다.

    Args:
        wave: (B, T) waveform
        sigma_ms: Gaussian sigma in milliseconds (default 2ms = ~88 samples @ 44.1kHz)

    Returns:
        blurred: (B, T) smoothed waveform
    """
    if sigma_ms <= 0:
        return wave

    sigma_samples = int(sigma_ms / 1000 * S)
    if sigma_samples < 1:
        return wave

    # Kernel: Gaussian window, radius = 4*sigma
    radius = min(4 * sigma_samples, 512)
    kernel_size = 2 * radius + 1
    t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=wave.device)
    kernel = torch.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    # ── Apply along time dim (CAUSAL: left-only padding, no future leakage) ──
    wave_3d = wave.unsqueeze(1)  # (B, 1, T)
    # Left-only padding: only past samples influence current output
    padded = F.pad(wave_3d, (kernel_size - 1, 0), mode='reflect')
    blurred = F.conv1d(padded, kernel.expand(1, 1, -1))
    return blurred.squeeze(1)


# ═══════════════════════════════════════════════════════════════════
# Dataset: WavLM 16kHz cache + audio + speaker
# ═══════════════════════════════════════════════════════════════════

class Phase0Dataset(Dataset):
    """Loads WavLM 16kHz CNN features, original audio, and speaker embedding."""

    def __init__(self, indices, wavlm_dir, source_files, spk_embeds,
                 spk_names, max_content_frames=50, seed=42):
        self.indices = [int(i) for i in indices]
        self.wavlm_dir = Path(wavlm_dir)
        self.source_files = source_files
        self.spk_names = spk_names       # (N,) array of speaker IDs (e.g., 'p315')
        self.spk_embeds = spk_embeds     # (N_spk, 128) float32 tensor
        self.max_cf = max_content_frames
        self.max_wavlm = max_content_frames * 2
        self.max_samples = max_content_frames * 1764
        self.rng = random.Random(seed)

        # Build speaker ID → embedding index mapping from first 1K samples
        n_emb = len(self.spk_embeds)
        self.spk_to_emb = {}
        for i in range(min(n_emb, len(spk_names))):
            spk_id = str(spk_names[i])
            if spk_id not in self.spk_to_emb:
                self.spk_to_emb[spk_id] = i
        print(f"  Speaker mapping: {len(self.spk_to_emb)} unique speakers → "
              f"{n_emb} embeddings")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import soundfile as sf
        idx = self.indices[i]

        # ── WavLM 16k features ──
        wl_path = self.wavlm_dir / f"s_{idx:05d}.npy"
        wavlm = np.load(wl_path, allow_pickle=False).astype(np.float32)  # (T, 512)
        wavlm = torch.from_numpy(wavlm)

        # Crop: ensure enough frames
        if wavlm.shape[0] < self.max_wavlm:
            pad_len = self.max_wavlm - wavlm.shape[0]
            wavlm = F.pad(wavlm, (0, 0, 0, pad_len))
            wl_start = 0
        elif wavlm.shape[0] > self.max_wavlm:
            wl_start = self.rng.randint(0, wavlm.shape[0] - self.max_wavlm)
            wavlm = wavlm[wl_start:wl_start + self.max_wavlm]
        else:
            wl_start = 0

        # ── Original audio ──
        src_path = str(self.source_files[idx])
        wave, sr = sf.read(src_path, dtype="float32")
        wave = torch.from_numpy(np.asarray(wave))
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != S:
            wave = torchaudio.functional.resample(wave.unsqueeze(0), sr, S).squeeze(0)

        # Crop audio to match WavLM window
        audio_start = wl_start * 882  # each WavLM frame = 882 audio samples @ 44.1kHz
        audio_end = audio_start + self.max_samples
        if audio_end > wave.shape[0]:
            wave = F.pad(wave, (0, audio_end - wave.shape[0]))
            wave = wave[audio_start:audio_end]
        elif audio_start + self.max_samples > wave.shape[0]:
            wave = wave[audio_start:]
            wave = F.pad(wave, (0, self.max_samples - wave.shape[0]))
        else:
            wave = wave[audio_start:audio_end]

        # ── Speaker embedding (match actual speaker, not random) ──
        spk_id = str(self.spk_names[idx])
        spk_idx = self.spk_to_emb.get(spk_id, 0)  # fallback to first
        spk = self.spk_embeds[spk_idx].clone()

        return {"wavlm": wavlm, "audio": wave, "speaker": spk, "idx": idx}


def collate_phase0(batch):
    """Stack batch items."""
    wavlm = torch.stack([b["wavlm"] for b in batch])      # (B, T_wl, 512)
    audio = torch.stack([b["audio"] for b in batch])        # (B, samples)
    speaker = torch.stack([b["speaker"] for b in batch])    # (B, 128)
    indices = [b["idx"] for b in batch]
    return wavlm, audio, speaker, indices


# ═══════════════════════════════════════════════════════════════════
# Encoder loader (frozen, WavLM frontend)
# ═══════════════════════════════════════════════════════════════════

def load_encoder(checkpoint_path, device="cpu"):
    """Load frozen Q2D2 encoder from checkpoint."""
    from train_mcs_q2d2 import MCSTransQ2D2Config, MCSTransQ2D2

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scfg = ck.get("config", {})

    # Reconstruct config from checkpoint
    scfg2 = {k: tuple(v) if isinstance(v, list) else v
             for k, v in scfg.items() if not k.startswith("_")}
    scfg2["use_wavlm_frontend"] = True
    known = set(MCSTransQ2D2Config.__dataclass_fields__.keys())
    scfg2 = {k: v for k, v in scfg2.items() if k in known}
    config = MCSTransQ2D2Config(**scfg2)

    model = MCSTransQ2D2(config).to(device).eval()
    # Load shared weights (skip new heads not in checkpoint)
    model.load_state_dict(ck["state_dict"], strict=False)

    for p in model.parameters():
        p.requires_grad_(False)

    return model, config


# ═══════════════════════════════════════════════════════════════════
# Main training
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Phase 0: CausalDecoder training")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=50,
                    help="Content frames per sample (@25Hz, 50=2sec)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate (cdecoder recommends ≤1e-4 for decoder)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder-ckpt", type=Path,
                    default=Path("/Volumes/UNTITLED/btrv5_checkpoints/dws_7l_16k_grl/dws_7l_16k_grl.last.pt"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Volumes/UNTITLED/btrv5_checkpoints/decoder_phase0"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--wavlm-dir", type=str, default="wavlm_16k")

    # cdecoder.md: MR-STFT + teacher smoothing
    ap.add_argument("--nffts", type=int, nargs="+", default=[512, 1024, 2048],
                    help="FFT sizes for MR-STFT loss")
    ap.add_argument("--blur-sigma-ms", type=float, default=2.0,
                    help="Gaussian blur sigma (ms) on teacher waveform (cdecoder.md)")
    ap.add_argument("--mrstft-weight", type=float, default=1.0)
    ap.add_argument("--mel-l1-weight", type=float, default=0.3,
                    help="Auxiliary Mel L1 loss weight")
    ap.add_argument("--use-mamba", action="store_true",
                    help="Use Mamba SSM blocks instead of ShallowConv in Phase 3")

    # Optimizer
    ap.add_argument("--clip-grad", type=float, default=0.5)
    ap.add_argument("--warmup-steps", type=int, default=1000)

    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    data_dir = args.data_dir
    wavlm_dir = data_dir / args.wavlm_dir
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n_samples = int(meta["n_samples"])
    source_files = meta["source_files"][:n_samples].astype(str)

    # Speaker embeddings
    spk_path = data_dir / "spk_1k.npy"
    if spk_path.exists():
        spk_embeds = torch.from_numpy(
            np.load(spk_path, allow_pickle=False).astype(np.float32)
        )
        print(f"Loaded {len(spk_embeds)} speaker embeddings from {spk_path}")
    else:
        raise FileNotFoundError(f"Speaker embeddings not found: {spk_path}")

    # Train/val split
    indices = np.arange(min(n_samples, 10000))
    np.random.default_rng(args.seed).shuffle(indices)
    split = int(len(indices) * 0.95)
    train_idx, val_idx = indices[:split], indices[split:split + 200]

    # Build speaker mapping
    spk_names = meta["spk_names"]

    train_ds = Phase0Dataset(train_idx, wavlm_dir, source_files,
                             spk_embeds, spk_names, args.max_frames, args.seed)
    val_ds = Phase0Dataset(val_idx, wavlm_dir, source_files,
                           spk_embeds, spk_names, args.max_frames, args.seed)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              collate_fn=collate_phase0, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            collate_fn=collate_phase0, drop_last=False)

    # ── Encoder (frozen) ──
    print(f"Loading encoder from {args.encoder_ckpt} ...")
    encoder, enc_cfg = load_encoder(args.encoder_ckpt, device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder: {enc_params:,} params ({enc_params/1e6:.1f}M) [FROZEN]")

    # ── Decoder ──
    dec_cfg = CausalDecoderConfig(use_mamba=args.use_mamba)
    if args.use_mamba:
        print(f"Mamba config: d_state={dec_cfg.mamba_d_state}, expand={dec_cfg.mamba_expand}")
    decoder = CausalDecoder(dec_cfg).to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        decoder.load_state_dict(ck["state_dict"], strict=False)
        print(f"Resumed decoder from {args.resume}")

    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder: {dec_params:,} params ({dec_params/1e6:.1f}M)")

    # ── Optimizer ──
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ── Mel loss function ──
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=S, n_fft=2048, hop_length=512,
        n_mels=80, f_min=0, f_max=S/2, power=1,
    ).to(device)

    # ── Training config summary ──
    content_sec = args.max_frames / 25.0
    print(f"\nPhase 0 Config:")
    print(f"  Content: {args.max_frames} frames @ 25Hz = {content_sec:.1f}s")
    print(f"  Audio:   {args.max_frames * 1764} samples @ 44.1kHz = {content_sec:.1f}s")
    print(f"  Batch:   {args.batch_size} × {args.steps_per_epoch} steps/epoch")
    print(f"  LR:      {args.lr} (warmup {args.warmup_steps} steps, cosine to 0)")
    print(f"  MR-STFT: {args.nffts} (weight={args.mrstft_weight})")
    print(f"  Blur:    σ={args.blur_sigma_ms}ms (cdecoder.md teacher smoothing)")
    print(f"  Mel L1:  weight={args.mel_l1_weight}")
    print(f"  Encoder: {args.encoder_ckpt.name} ({enc_params/1e6:.1f}M, frozen)")
    print(f"  Decoder: {dec_params/1e6:.1f}M")
    print(f"  Total:   {(enc_params + dec_params)/1e6:.1f}M", flush=True)

    t0 = time.time()
    best_loss = float("inf")
    global_step = 0

    for ep in range(args.epochs):
        decoder.train()
        epoch_mrstft = 0.0
        epoch_mel = 0.0
        epoch_loss = 0.0
        epoch_steps = 0

        for wavlm, audio, speaker, indices in train_loader:
            epoch_steps += 1
            if epoch_steps > args.steps_per_epoch:
                break

            wavlm = wavlm.to(device)
            audio = audio.to(device)
            speaker = speaker.to(device)

            # ── Frozen encoder forward ──
            # Build padding mask: WavLM features → content frames
            T_wl = wavlm.shape[1]
            T_content = T_wl // 2
            mask = torch.ones(wavlm.shape[0], T_content, dtype=torch.bool, device=device)

            with torch.no_grad():
                enc_out = encoder(wavlm.transpose(1, 2), padding_mask=mask)
                content = enc_out["projected"].transpose(1, 2)  # (B, T_c, 768)

            # ── Decoder forward ──
            stft_len = decoder._compute_stft_length(content.shape[1])
            pred_wave = decoder(content, speaker, stft_length=stft_len)

            # ── Align lengths ──
            t_len = min(pred_wave.shape[1], audio.shape[1])
            pred_wave = pred_wave[:, :t_len]
            audio_aligned = audio[:, :t_len]

            # ── cdecoder.md: Gaussian blur teacher ──
            if args.blur_sigma_ms > 0:
                blurred_audio = gaussian_blur_wave(audio_aligned, args.blur_sigma_ms)
            else:
                blurred_audio = audio_aligned

            # ── MR-STFT loss (CPU STFT: MPS torch.stft is unstable) ──
            # Transfer batch ONCE to CPU, compute per-sample on CPU
            pred_cpu = pred_wave.cpu()
            tgt_cpu = blurred_audio.cpu()
            mrstft_cpu = 0.0
            for b in range(pred_cpu.shape[0]):
                mrstft_cpu += multi_resolution_stft_loss(
                    pred_cpu[b], tgt_cpu[b], tuple(args.nffts),
                ).item()
            mrstft = torch.tensor(mrstft_cpu / pred_cpu.shape[0], device=device)

            # ── Auxiliary Mel L1 loss (log domain, stable) ──
            if args.mel_l1_weight > 0:
                pred_mel = mel_fn(pred_wave)
                tgt_mel = mel_fn(blurred_audio)
                pred_mel = torch.log(torch.clamp(pred_mel, min=1e-5))
                tgt_mel = torch.log(torch.clamp(tgt_mel, min=1e-5))
                mel_l1 = F.l1_loss(pred_mel, tgt_mel)
            else:
                mel_l1 = torch.tensor(0.0, device=device)

            loss = args.mrstft_weight * mrstft + args.mel_l1_weight * mel_l1

            # ── Backward ──
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_grad)
            opt.step()

            # Linear LR warmup
            global_step += 1
            if global_step < args.warmup_steps:
                scale = global_step / args.warmup_steps
                for pg in opt.param_groups:
                    pg['lr'] = args.lr * scale

            epoch_mrstft += float(mrstft.cpu())
            epoch_mel += float(mel_l1.cpu())
            epoch_loss += float(loss.cpu())

            if epoch_steps % 100 == 0:
                avg_l = epoch_loss / epoch_steps
                avg_m = epoch_mrstft / epoch_steps
                avg_ml = epoch_mel / epoch_steps
                print(f"E{ep:02d}/{epoch_steps:05d} "
                      f"loss={avg_l:.4f} mrstft={avg_m:.4f} mel_l1={avg_ml:.4f}",
                      flush=True)

        sch.step()

        # ── Validation ──
        decoder.eval()
        val_loss = 0.0
        val_mrstft = 0.0
        val_mel = 0.0
        val_steps = 0
        for wavlm, audio, speaker, indices in val_loader:
            wavlm = wavlm.to(device); audio = audio.to(device)
            speaker = speaker.to(device)

            T_wl = wavlm.shape[1]
            mask = torch.ones(wavlm.shape[0], T_wl // 2, dtype=torch.bool, device=device)
            with torch.no_grad():
                enc_out = encoder(wavlm.transpose(1, 2), padding_mask=mask)
                content = enc_out["projected"].transpose(1, 2)
                stft_len = decoder._compute_stft_length(content.shape[1])
                pred_wave = decoder(content, speaker, stft_length=stft_len)

            t_len = min(pred_wave.shape[1], audio.shape[1])
            pred_wave = pred_wave[:, :t_len]
            audio_aligned = audio[:, :t_len]

            if args.blur_sigma_ms > 0:
                blurred_audio = gaussian_blur_wave(audio_aligned, args.blur_sigma_ms)
            else:
                blurred_audio = audio_aligned

            # MR-STFT on CPU
            pred_cpu = pred_wave.cpu()
            tgt_cpu = blurred_audio.cpu()
            mrstft_cpu = 0.0
            for b in range(pred_cpu.shape[0]):
                mrstft_cpu += multi_resolution_stft_loss(
                    pred_cpu[b], tgt_cpu[b], tuple(args.nffts),
                ).item()
            val_mrstft += mrstft_cpu / pred_cpu.shape[0]

            pred_mel = mel_fn(pred_wave)
            tgt_mel = mel_fn(blurred_audio)
            pred_mel = torch.log(torch.clamp(pred_mel, min=1e-5))
            tgt_mel = torch.log(torch.clamp(tgt_mel, min=1e-5))
            mel_l1 = F.l1_loss(pred_mel, tgt_mel)

            val_mel += float(mel_l1.cpu())
            val_loss += mrstft_cpu / pred_cpu.shape[0] + args.mel_l1_weight * float(mel_l1.cpu())
            val_steps += 1
            if val_steps >= 30:
                break

        decoder.train()
        nv = max(val_steps, 1)
        val_loss /= nv; val_mrstft /= nv; val_mel /= nv

        et = epoch_steps
        print(f"E{ep:02d} "
              f"train_loss={epoch_loss/et:.4f} mrstft={epoch_mrstft/et:.4f} mel={epoch_mel/et:.4f} "
              f"val_loss={val_loss:.4f} val_mrstft={val_mrstft:.4f} val_mel={val_mel:.4f}",
              flush=True)

        # ── Checkpoint ──
        ckpt = {
            "state_dict": decoder.state_dict(),
            "decoder_config": dec_cfg.__dict__,
            "enc_config": enc_cfg.__dict__ if hasattr(enc_cfg, '__dict__') else {},
            "epoch": ep,
            "global_step": global_step,
            "train_loss": epoch_loss / et,
            "val_loss": val_loss,
            "val_mrstft": val_mrstft,
            "val_mel_l1": val_mel,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(ckpt, args.out_dir / "best.pt")

        summary = {
            "epoch": ep,
            "global_step": global_step,
            "train_loss": epoch_loss / et,
            "train_mrstft": epoch_mrstft / et,
            "train_mel_l1": epoch_mel / et,
            "val_loss": val_loss,
            "val_mrstft": val_mrstft,
            "val_mel_l1": val_mel,
            "elapsed_seconds": time.time() - t0,
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    print(f"Done. Best val_loss={best_loss:.4f} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
