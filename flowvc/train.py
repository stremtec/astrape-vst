"""
FlowVC training pipeline with real data loader + checkpoint save/resume.
"""

from __future__ import annotations
import argparse, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

from .config import EncoderConfig, DecoderConfig, FlowConverterConfig
from .encoder import make_encoder
from .decoder import F3Decoder
from .converter import make_vector_field_net
from .speaker import make_speaker_encoder
from .prosody import make_prosody_extractor
from .cfm_loss import CFMLoss
from .dataset import VCTKDataset, create_dataloader


def save_checkpoint(models: dict, opt: torch.optim.Optimizer, step: int, path: str):
    ckpt = {"step": step, "opt_state": opt.state_dict()}
    for name, model in models.items():
        ckpt[name] = model.state_dict()
    torch.save(ckpt, path)

def load_checkpoint(models: dict, opt: torch.optim.Optimizer | None, path: str, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for name, model in models.items():
        if name in ckpt:
            model.load_state_dict(ckpt[name])
    if opt is not None and "opt_state" in ckpt:
        opt.load_state_dict(ckpt["opt_state"])
    return ckpt.get("step", 0)


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── models ──
    encoder = make_encoder().to(device)
    decoder = F3Decoder(DecoderConfig()).to(device)
    speaker_enc = make_speaker_encoder().to(device)
    prosody = make_prosody_extractor(device=str(device)).to(device)

    print(f"[Phase {args.phase}] device={device}, batch={args.batch_size}, steps={args.steps}")

    # ── optimizer ──
    if args.phase == 0:  # AE pretrain: encoder + decoder + prosody
        params = list(encoder.parameters()) + list(decoder.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.8, 0.9), weight_decay=0.01)
    elif args.phase == 1:  # CFM: only VFN
        vfn = make_vector_field_net().to(device)
        encoder.eval(); decoder.eval(); speaker_enc.eval(); prosody.eval()
        for m in [encoder, decoder, speaker_enc, prosody]:
            for p in m.parameters(): p.requires_grad = False
        params = list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.8, 0.9))
    else:  # Phase 2: E2E fine-tune
        vfn = make_vector_field_net().to(device)
        params = list(encoder.parameters()) + list(decoder.parameters()) + list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr * 0.1, betas=(0.8, 0.9))

    # ── resume ──
    start_step = 0
    if args.resume:
        models = {"encoder": encoder, "decoder": decoder, "speaker_enc": speaker_enc}
        if args.phase >= 1:
            models["vfn"] = vfn
        start_step = load_checkpoint(models, opt, args.resume, device)
        print(f"  Resumed from step {start_step}")

    # ── data ──
    if not args.data_dir:
        print("ERROR: --data-dir required")
        return
    ds = VCTKDataset(args.data_dir, crop_seconds=2.0)
    loader = create_dataloader(ds, batch_size=args.batch_size, shuffle=True)

    # ── CFM loss ──
    cfm_loss = CFMLoss(sigma_min=0.001) if args.phase >= 1 else None

    # ── training loop ──
    step = start_step
    running_loss = 0.0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            src = batch["src_wav"].to(device)
            tgt = batch["tgt_wav"].to(device)
            ref = batch["ref_wav"].to(device)

            if args.phase == 0:
                # ── Phase 0: AE reconstruction (no speaker conditioning) ──
                # Decoder receives zeros for speaker emb to learn speaker-agnostic decoding.
                # This prevents speaker info leakage into the latent space.
                z = encoder(src, training=True)
                recon = decoder(z, torch.zeros(src.size(0), 192, device=device))
                loss = F.l1_loss(recon, src) + 0.1 * F.mse_loss(z, z.detach())  # latent consistency

            elif args.phase == 1:
                # ── Phase 1: CFM ──
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    z_tgt = encoder.encode(tgt)
                    spk_emb, prompt = speaker_enc(ref)
                    pros = prosody(src)
                loss, logs = cfm_loss(vfn, z_src, z_tgt, spk_emb, prompt, pros)

            else:
                # ── Phase 2: E2E ──
                speaker_enc.eval(); prosody.eval()
                for m in [speaker_enc, prosody]:
                    for p in m.parameters(): p.requires_grad = False
                z_src = encoder.encode(src)
                with torch.no_grad():
                    spk_emb, prompt = speaker_enc(ref)
                    pros = prosody(src)

                from .converter import solve_cfm_euler
                z_tgt = solve_cfm_euler(vfn, z_src, spk_emb, prompt, pros, n_steps=4)
                out = decoder(z_tgt, spk_emb)
                loss = F.l1_loss(out, tgt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            running_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                elapsed = time.time()
                print(f"  step {step:>7d}/{args.steps}  loss={avg:.6f}")
                running_loss = 0.0

            if step % args.save_interval == 0:
                models = {"encoder": encoder, "decoder": decoder, "speaker_enc": speaker_enc, "prosody": prosody}
                if args.phase >= 1:
                    models["vfn"] = vfn
                ckpt_path = os.path.join(args.output_dir, f"step_{step:07d}.pt")
                save_checkpoint(models, opt, step, ckpt_path)
                print(f"  saved: {ckpt_path}")

    # Final save
    models = {"encoder": encoder, "decoder": decoder, "speaker_enc": speaker_enc, "prosody": prosody}
    if args.phase >= 1:
        models["vfn"] = vfn
    final_path = os.path.join(args.output_dir, "final.pt")
    save_checkpoint(models, opt, step, final_path)
    print(f"  final: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="FlowVC training")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
