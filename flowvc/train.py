"""
FlowVC 学習パイプライン — WavLM蒸留ベース。

Phase 0: Encoder学習 (WavLM蒸留, decoder不使用)
Phase 1: CFM学習 (VFNのみ, encoder凍結)
Phase 2: Decoder学習 + E2E微調整
"""

from __future__ import annotations
import argparse, os, sys, time, torch, torch.nn as nn, torch.nn.functional as F

from .config import DecoderConfig
from .encoder import make_encoder
from .decoder import F3Decoder
from .converter import make_vector_field_net, solve_cfm_euler
from .speaker import make_speaker_encoder
from .prosody import make_prosody_extractor
from .cfm_loss import CFMLoss
from .distiller import WavLMDistiller
from .dataset import VCTKDataset, create_dataloader


def save_checkpoint(models: dict, opt, step: int, path: str):
    ckpt = {"step": step, "opt_state": opt.state_dict() if opt else {}}
    for name, model in models.items():
        sd = model.state_dict()
        if name == "prosody":
            sd = {k: v for k, v in sd.items() if not k.startswith("_fcpe_model.")}
        ckpt[name] = sd
    torch.save(ckpt, path)

def load_checkpoint(models: dict, opt, path: str, device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for name, model in models.items():
        if name in ckpt:
            strict = name != "prosody"
            model.load_state_dict(ckpt[name], strict=strict)
    if opt and "opt_state" in ckpt:
        try: opt.load_state_dict(ckpt["opt_state"])
        except ValueError: print("  opt_state skipped (phase change)")
    return ckpt.get("step", 0)


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    encoder = make_encoder().to(device)
    decoder = None
    vfn = None
    speaker_enc = make_speaker_encoder().to(device)
    prosody = make_prosody_extractor(device=str(device)).to(device)

    print(f"[Phase {args.phase}] device={device}, batch={args.batch_size}, steps={args.steps}")

    # ── Phase 0: WavLM蒸留 (encoderのみ学習) ──
    if args.phase == 0:
        distiller = WavLMDistiller(encoder, device=str(device)).to(device)
        params = list(encoder.parameters()) + list(distiller.proj.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
        model_dict = {"encoder": encoder}

    # ── Phase 1: CFM (VFNのみ, encoder凍結) ──
    elif args.phase == 1:
        decoder = F3Decoder(DecoderConfig()).to(device).eval()
        vfn = make_vector_field_net().to(device)
        decoder.eval(); speaker_enc.eval(); prosody.eval()
        for m in [encoder, decoder, speaker_enc, prosody]:
            for p in m.parameters(): p.requires_grad = False
        params = list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98))
        model_dict = {"encoder": encoder, "vfn": vfn, "speaker_enc": speaker_enc, "prosody": prosody}

    # ── Phase 2: Decoder学習 + E2E ──
    else:
        decoder = F3Decoder(DecoderConfig()).to(device)
        vfn = make_vector_field_net().to(device)
        speaker_enc.eval(); prosody.eval()
        for m in [speaker_enc, prosody]:
            for p in m.parameters(): p.requires_grad = False
        params = list(decoder.parameters()) + list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr * 0.1, betas=(0.9, 0.98))
        model_dict = {"encoder": encoder, "decoder": decoder, "vfn": vfn,
                      "speaker_enc": speaker_enc, "prosody": prosody}

    # ── resume ──
    start_step = 0
    if args.resume:
        load_checkpoint(model_dict, opt, args.resume, device)
        print(f"  Resumed")

    # ── data ──
    ds = VCTKDataset(args.data_dir, crop_seconds=2.0)
    loader = create_dataloader(ds, batch_size=args.batch_size, shuffle=True)
    cfm_loss = CFMLoss(sigma_min=0.001)

    # ── training loop ──
    step = start_step
    running_loss = 0.0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps: break

            src = batch["src_wav"].to(device)
            tgt = batch["tgt_wav"].to(device)
            ref = batch["ref_wav"].to(device)

            if args.phase == 0:
                z, loss = distiller(src)

            elif args.phase == 1:
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    z_tgt = encoder.encode(tgt)
                    spk_emb, prompt = speaker_enc(ref)
                    pros = prosody(src)
                loss, _ = cfm_loss(vfn, z_src, z_tgt, spk_emb, prompt, pros)

            else:  # Phase 2
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    spk_emb, prompt = speaker_enc(ref)
                    pros = prosody(src)
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
                print(f"  step {step:>7d}/{args.steps}  loss={running_loss/args.log_interval:.6f}")
                running_loss = 0.0

            if step % args.save_interval == 0:
                save_checkpoint(model_dict, opt, step,
                              os.path.join(args.output_dir, f"step_{step:07d}.pt"))

    save_checkpoint(model_dict, opt, step, os.path.join(args.output_dir, "final.pt"))
    print(f"  final: {args.output_dir}/final.pt")


def main():
    p = argparse.ArgumentParser(description="FlowVC training")
    p.add_argument("--phase", type=int, default=0)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./runs")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default="")
    train(p.parse_args())

if __name__ == "__main__":
    main()
