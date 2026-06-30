"""Train CausalDecoderV6 by output distillation from the frozen MioCodec teacher.

Loss: MR-STFT (main) + Mel-L1 (aux) + small complex-STFT (phase nudge, 0.01).
Spectral losses computed on CPU (MPS torch.stft backward is non-finite).
No GAN needed — distillation from the teacher's correct output is sufficient
(and matches the proven recipe from train_mcs_decoder.py).

  .venv/bin/python -m astrape.train_decoder_v6 --device mps --epochs 80 \
      --content-dir /Users/asill/btrv5_content --num-workers 4 \
      --out-dir /Volumes/UNTITLED/btrv5_checkpoints/decoder_v6
"""
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from .data import Phase0Dataset, collate_phase0
from .train_decoder import mrstft, complex_stft_loss
from .decoder_v6 import CausalDecoderV6, CausalDecoderV6Config

S = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--wavlm-dir", type=str, default="wavlm_L4_200hz")
    ap.add_argument("--wavlm-rate", type=int, default=200)
    ap.add_argument("--content-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--nffts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--mel-weight", type=float, default=1.0)
    ap.add_argument("--cstft-weight", type=float, default=0.01)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    # v6 architecture overrides
    ap.add_argument("--fusion-layers", type=int, default=4,
                    help="AdaLN transformer layers at 50Hz in the fusion stage.")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Epochs of lower LR warmup (linear 0→lr).")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    meta = np.load(args.data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); src = meta["source_files"][:n].astype(str); spk_names = meta["spk_names"]
    cz_path = args.data_dir / "spk_centroids.npz"
    if not cz_path.exists():
        raise SystemExit(f"Missing {cz_path}.\n  Run first: .venv/bin/python -m astrape.cache --what speakers")
    cz = np.load(cz_path, allow_pickle=False)
    semap = {str(s): torch.from_numpy(e).float() for s, e in zip(cz["speakers"], cz["embeddings"])}
    idx = np.arange(n); np.random.default_rng(args.seed).shuffle(idx)
    split = int(len(idx) * 0.95)
    ds = Phase0Dataset(idx[:split], args.data_dir / args.wavlm_dir, src, None, spk_names,
                       args.max_frames, args.seed, wavlm_rate=args.wavlm_rate, speaker_emb_map=semap,
                       content_dir=args.data_dir / args.content_dir)
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0, collate_fn=collate_phase0, drop_last=True)

    # ── model ──
    from .miocodec import load_mio
    mio = load_mio(args.device).eval()
    dec_cfg = CausalDecoderV6Config(fusion_layers=args.fusion_layers)
    dec = CausalDecoderV6(dec_cfg).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mel_fn = torchaudio.transforms.MelSpectrogram(S, n_fft=2048, hop_length=512, n_mels=80, f_min=0, f_max=S/2, power=1)

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        dec.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
        start = int(ck.get("epoch", -1)) + 1

    dec_params = sum(p.numel() for p in dec.parameters())
    algo = (dec_cfg.n_fft - dec_cfg.hop_length) / 2 / S * 1000
    print(f"CausalDecoderV6: {dec_params/1e6:.2f}M  fusion={dec_cfg.fusion_layers}L  "
          f"algo-latency={algo:.1f}ms  "
          f"(output distill: MR-STFT + {args.mel_weight}*Mel-L1 + {args.cstft_weight}*cSTFT)", flush=True)

    t0 = time.time()
    for ep in range(start, args.epochs):
        dec.train()
        # warmup: linear 0→lr for first warmup_epochs
        if args.warmup_epochs > 0 and ep < args.warmup_epochs:
            for pg in opt.param_groups:
                pg['lr'] = args.lr * (ep + 1) / args.warmup_epochs

        acc = {k: torch.zeros((), device=device) for k in ("stft", "mel", "cstft")}
        steps = skipped = 0
        for content, audio, speaker, _i in loader:
            steps += 1
            if steps > args.steps_per_epoch:
                break
            content, audio, speaker = content.to(device), audio.to(device), speaker.to(device)
            with torch.no_grad():
                tch_len = mio._calculate_target_stft_length(audio.shape[1])
                teacher = mio.forward_wave(content, speaker, stft_length=tch_len)
            wav = dec(content, speaker)
            tl = min(wav.shape[1], teacher.shape[1])
            pc, tc = wav[:, :tl].float().cpu(), teacher[:, :tl].float().cpu()
            stft_l = mrstft(pc, tc, args.nffts)
            mel_l = F.l1_loss(mel_fn(pc).clamp_min(1e-5).log(), mel_fn(tc).clamp_min(1e-5).log())
            cstft_l = complex_stft_loss(pc, tc, args.nffts)
            loss = (stft_l + args.mel_weight * mel_l + args.cstft_weight * cstft_l).to(device)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), args.clip_grad)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; continue
            opt.step()
            acc["stft"] += stft_l.detach().to(device); acc["mel"] += mel_l.detach().to(device)
            acc["cstft"] += cstft_l.detach().to(device)
            if steps % 100 == 0:
                d = max(steps - skipped, 1)
                pct = steps * 100 // args.steps_per_epoch
                elapsed = time.time() - t0
                s_per = elapsed / ((ep - start) * args.steps_per_epoch + steps)
                print(f"E{ep:02d} [{pct:3d}%] stft={acc['stft'].item()/d:.3f} "
                      f"mel={acc['mel'].item()/d:.3f} cstft={acc['cstft'].item()/d:.3f} "
                      f"skip={skipped} {s_per:.2f}s/step", flush=True)
        sch.step()
        ckpt = {"state_dict": dec.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                "decoder_config": dec_cfg.__dict__, "epoch": ep}
        torch.save(ckpt, args.out_dir / "last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ckpt, args.out_dir / f"epoch{ep:03d}.pt")
        (args.out_dir / "summary.json").write_text(json.dumps(
            {"epoch": ep, "skipped": skipped,
             **{k: v.item() / max(steps - skipped, 1) for k, v in acc.items()}}, indent=2) + "\n")
        print(f"E{ep:02d} done", flush=True)
    print(f"Done. {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
