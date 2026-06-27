"""Decoder v5 training — 2-phase adversarial curriculum.

Curriculum
  Phase A  (epochs < --warmup-epochs):  RECONSTRUCTION ONLY
      MR-STFT (CPU, grad-preserving) + Mel-L1 on Gaussian-blurred target.
      Gets the generator to a stable spectral baseline before the GAN.
  Phase B  (epochs >= --warmup-epochs): ADVERSARIAL
      + MPD/MSD discriminators (LSGAN) + feature matching, recon kept as anchor.
      (Optionally enable NSF via --use-nsf for the harmonic source.)

Reuses the frozen encoder + data pipeline from train_decoder.py.

Usage:
  .venv/bin/python train_decoder_v5.py --device mps --epochs 60 --warmup-epochs 10 \
      --encoder-ckpt /Volumes/UNTITLED/btrv5_checkpoints/striding_8l_200hz/striding_8l_200hz.best.pt \
      --wavlm-dir wavlm_L4_200hz --num-workers 6
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from astrape.data import Phase0Dataset, collate_phase0, gaussian_blur_wave
from astrape.decoder import CausalDecoderV5, CausalDecoderV5Config
from astrape.discriminators import (
    CombinedDiscriminator, discriminator_loss, generator_adv_loss, feature_matching_loss,
)

S = 44100


def mrstft(pred: torch.Tensor, tgt: torch.Tensor, nffts) -> torch.Tensor:
    """Batched multi-resolution STFT loss, computed on the input's device.

    torch.stft works correctly on MPS in this torch build (verified: forward
    matches CPU to ~4e-5, backward gives finite grads), so this runs ON-DEVICE
    and BATCHED — no per-sample Python loop or MPS↔CPU transfer/sync that the old
    CPU path needed. Numerically identical objective; works on CPU too.
    """
    loss = pred.new_zeros(())
    for n_fft in nffts:
        win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        ps = torch.stft(pred, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                        window=win, return_complex=True).abs().clamp_min(1e-7)
        ts = torch.stft(tgt, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                        window=win, return_complex=True).abs().clamp_min(1e-7)
        sc = torch.linalg.vector_norm(ps - ts) / torch.linalg.vector_norm(ts).clamp_min(1e-7)
        loss = loss + sc + F.l1_loss(ps.log(), ts.log())
    return loss / len(nffts)


def load_encoder(checkpoint_path, device="cpu"):
    """Load the frozen Q2D2 encoder from a checkpoint (Phase 2: → astrape.encoder)."""
    from train_mcs_q2d2 import MCSTransQ2D2Config, MCSTransQ2D2
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scfg = ck.get("config", {})
    scfg2 = {k: tuple(v) if isinstance(v, list) else v
             for k, v in scfg.items() if not k.startswith("_")}
    scfg2["use_wavlm_frontend"] = True
    known = set(MCSTransQ2D2Config.__dataclass_fields__.keys())
    scfg2 = {k: v for k, v in scfg2.items() if k in known}
    config = MCSTransQ2D2Config(**scfg2)
    model = MCSTransQ2D2(config).to(device).eval()
    model.load_state_dict(ck["state_dict"], strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup-epochs", type=int, default=10,
                    help="Reconstruction-only epochs before the GAN turns on.")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--encoder-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Volumes/UNTITLED/btrv5_checkpoints/decoder_v5"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--wavlm-dir", type=str, default="wavlm_L4_200hz")
    ap.add_argument("--wavlm-rate", type=int, default=200,
                    help="WavLM cache rate (must match the encoder; 200 for wavlm_L4_200hz).")
    # losses / curriculum
    ap.add_argument("--nffts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--blur-sigma-ms", type=float, default=2.0)
    ap.add_argument("--mrstft-weight", type=float, default=1.0)
    ap.add_argument("--mel-l1-weight", type=float, default=1.0)
    ap.add_argument("--adv-weight", type=float, default=1.0)
    ap.add_argument("--fm-weight", type=float, default=2.0)
    ap.add_argument("--disc-window", type=int, default=16384,
                    help="Samples the discriminator sees per step (random crop, "
                         "HiFi-GAN-style ~0.37s). Reconstruction still uses full audio. "
                         "Cuts discriminator compute ~5x with no quality loss.")
    ap.add_argument("--use-nsf", action="store_true", help="Enable Phase 2b NSF harmonic source.")
    ap.add_argument("--n-fft", type=int, default=1512)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    data_dir = args.data_dir
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n_samples = int(meta["n_samples"])
    source_files = meta["source_files"][:n_samples].astype(str)
    spk_names = meta["spk_names"]
    # Per-speaker centroids (run cache_speaker_embeddings.py first) — covers all
    # speakers, unlike spk_1k.npy which only has the first ~4 (VCTK is grouped).
    cz_path = data_dir / "spk_centroids.npz"
    if not cz_path.exists():
        raise SystemExit(
            f"Missing {cz_path}.\n  Run first: .venv/bin/python cache_speaker_embeddings.py")
    cz = np.load(cz_path, allow_pickle=False)
    speaker_emb_map = {str(s): torch.from_numpy(e).float()
                       for s, e in zip(cz["speakers"], cz["embeddings"])}
    print(f"Loaded {len(speaker_emb_map)} speaker centroids")

    idx = np.arange(n_samples)              # ALL samples → all speakers (VCTK is grouped)
    np.random.default_rng(args.seed).shuffle(idx)
    split = int(len(idx) * 0.95)
    train_idx = idx[:split]
    train_ds = Phase0Dataset(train_idx, data_dir / args.wavlm_dir, source_files,
                             None, spk_names, args.max_frames, args.seed,
                             wavlm_rate=args.wavlm_rate, speaker_emb_map=speaker_emb_map)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0,
                              collate_fn=collate_phase0, drop_last=True)

    # ── encoder (frozen), generator (v5), discriminators ──
    encoder, _ = load_encoder(args.encoder_ckpt, device)
    dec_cfg = CausalDecoderV5Config(use_nsf=args.use_nsf, n_fft=args.n_fft)
    decoder = CausalDecoderV5(dec_cfg).to(device)
    disc = CombinedDiscriminator().to(device)

    opt_g = torch.optim.AdamW(decoder.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        decoder.load_state_dict(ck["state_dict"], strict=False)
        disc.load_state_dict(ck.get("disc", {}), strict=False)
        if "opt_g" in ck: opt_g.load_state_dict(ck["opt_g"])
        if "opt_d" in ck: opt_d.load_state_dict(ck["opt_d"])
        if "sch_g" in ck: sch_g.load_state_dict(ck["sch_g"])
        if "sch_d" in ck: sch_d.load_state_dict(ck["sch_d"])
        start_epoch = int(ck.get("epoch", -1)) + 1

    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=S, n_fft=2048, hop_length=512, n_mels=80, f_min=0, f_max=S / 2, power=1,
    ).to(device)

    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder v5: {dec_params/1e6:.2f}M  (n_fft={dec_cfg.n_fft}, nsf={dec_cfg.use_nsf}, "
          f"algo-latency={(dec_cfg.n_fft-dec_cfg.hop_length)/2/S*1000:.1f}ms)", flush=True)
    print(f"Curriculum: warmup(recon) epochs 0..{args.warmup_epochs-1}, "
          f"adversarial epochs {args.warmup_epochs}..{args.epochs-1}", flush=True)

    t0 = time.time()
    for ep in range(start_epoch, args.epochs):
        adversarial = ep >= args.warmup_epochs
        decoder.train(); disc.train()
        # On-device accumulators → no per-step MPS↔CPU sync (only .item() at log).
        acc = {k: torch.zeros((), device=device) for k in ("recon", "adv", "fm", "d")}
        steps = 0
        for wavlm, audio, speaker, _idx in train_loader:
            steps += 1
            if steps > args.steps_per_epoch:
                break
            wavlm, audio, speaker = wavlm.to(device), audio.to(device), speaker.to(device)

            with torch.no_grad():
                mask = torch.ones(wavlm.shape[0], wavlm.shape[1] // 2, dtype=torch.bool, device=device)
                content = encoder(wavlm.transpose(1, 2), padding_mask=mask)["projected"].transpose(1, 2)
            stft_len = decoder._compute_stft_length(content.shape[1])
            pred = decoder(content, speaker, stft_length=stft_len)

            t_len = min(pred.shape[1], audio.shape[1])
            pred, tgt = pred[:, :t_len], audio[:, :t_len]
            tgt_blur = gaussian_blur_wave(tgt, args.blur_sigma_ms) if args.blur_sigma_ms > 0 else tgt

            # ── reconstruction (always) — full audio, on-device batched MR-STFT ──
            recon = args.mrstft_weight * mrstft(pred, tgt_blur, args.nffts)
            recon = recon + args.mel_l1_weight * F.l1_loss(
                mel_fn(pred).clamp_min(1e-5).log(), mel_fn(tgt_blur).clamp_min(1e-5).log())

            if adversarial:
                # Discriminate on a random short window (HiFi-GAN segment style):
                # same crop for real+fake, full audio still used for reconstruction.
                # Discriminators are local/shift-invariant → quality-neutral, ~5x cheaper.
                W = min(args.disc_window, pred.shape[1])
                st = torch.randint(0, pred.shape[1] - W + 1, (1,)).item()
                pred_w, tgt_w = pred[:, st:st + W], tgt[:, st:st + W]

                # ── D step ──
                real_lg, _ = disc(tgt_w)
                fake_lg, _ = disc(pred_w.detach())
                d_loss = discriminator_loss(real_lg, fake_lg)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), args.clip_grad)
                opt_d.step()

                # ── G step ──
                fake_lg, fake_fm = disc(pred_w)
                real_lg, real_fm = disc(tgt_w)
                adv = generator_adv_loss(fake_lg)
                fm = feature_matching_loss(real_fm, fake_fm)
                g_loss = args.adv_weight * adv + args.fm_weight * fm + recon
                acc["adv"] += adv.detach(); acc["fm"] += fm.detach(); acc["d"] += d_loss.detach()
            else:
                g_loss = recon

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_grad)
            opt_g.step()
            acc["recon"] += recon.detach()

            if steps % 100 == 0:
                d = steps
                tag = "ADV" if adversarial else "REC"
                print(f"E{ep:02d}[{tag}] {steps:04d}/{args.steps_per_epoch} "
                      f"recon={acc['recon'].item()/d:.3f} adv={acc['adv'].item()/d:.3f} "
                      f"fm={acc['fm'].item()/d:.3f} d={acc['d'].item()/d:.3f} "
                      f"{(time.time()-t0)/((ep-start_epoch)*args.steps_per_epoch + steps):.3f}s/step",
                      flush=True)

        sch_g.step(); sch_d.step()
        tot = {k: v.item() for k, v in acc.items()}
        ckpt = {"state_dict": decoder.state_dict(), "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "sch_g": sch_g.state_dict(), "sch_d": sch_d.state_dict(),
                "decoder_config": dec_cfg.__dict__, "epoch": ep}
        torch.save(ckpt, args.out_dir / "last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ckpt, args.out_dir / f"epoch{ep:03d}.pt")
        (args.out_dir / "summary.json").write_text(json.dumps(
            {"epoch": ep, "phase": "adversarial" if adversarial else "warmup",
             **{k: v / max(steps, 1) for k, v in tot.items()}}, indent=2) + "\n")
        print(f"E{ep:02d} done ({'ADV' if adversarial else 'REC'})", flush=True)

    print(f"Done. elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
