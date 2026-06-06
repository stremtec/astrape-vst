"""
FlowVC training with AudioDec encoder (48kHz, pretrained, causal).

Phase 1: CFM training (VFN only, AudioDec encoder frozen)
Phase 2: Decoder + E2E fine-tuning
"""

from __future__ import annotations
import argparse, os, time, torch, torch.nn as nn, torch.nn.functional as F
from .audiodec import AudioDecEncoder, AudioDecDecoder, AUDIODEC_DIM, AUDIODEC_SR
from .converter import make_vector_field_net, solve_cfm_euler
from .ecapa import ECAPASpeakerEncoder  # pretrained ECAPA-TDNN
from .prosody import make_prosody_extractor
from .cfm_loss import CFMLoss
from .dataset import VCTKDataset, create_dataloader


def save_checkpoint(models: dict, opt, step: int, path: str):
    ckpt = {"step": step, "opt_state": opt.state_dict() if opt else {}}
    for name, model in models.items():
        sd = model.state_dict()
        if name == "prosody":
            sd = {k: v for k, v in sd.items() if not k.startswith("_fcpe_model.")}
        if name == "speaker_enc":
            sd = {k: v for k, v in sd.items() if not k.startswith("_model.")}
        ckpt[name] = sd
    torch.save(ckpt, path)


def load_checkpoint(models: dict, opt, path: str, device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for name, model in models.items():
        if name in ckpt:
            model.load_state_dict(ckpt[name], strict=(name != "prosody"))
    if opt and "opt_state" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt_state"])
        except ValueError:
            print("  opt_state skipped")
    return ckpt.get("step", 0)


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Phase {args.phase}] AudioDec 48kHz, device={device}, batch={args.batch_size}")

    # AudioDec encoder (pretrained, frozen)
    encoder = AudioDecEncoder(device=str(device)).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # AudioDec decoder
    decoder = AudioDecDecoder(device=str(device)).to(device)

    # VFN adapted for AudioDec latent
    vfn = make_vector_field_net(
        latent_dim=AUDIODEC_DIM,  # 64
        hidden_dim=128,            # scaled down from 512
        speaker_dim=192,
        prosody_dim=3,
        n_blocks=8,
        dilations=(1, 2, 4, 8, 1, 2, 4, 8),
    ).to(device)

    # ECAPA speaker encoder (pretrained, frozen)
    speaker_enc = ECAPASpeakerEncoder(device=str(device)).to(device)
    speaker_enc.eval()
    for p in speaker_enc.parameters():
        p.requires_grad = False
    prosody = make_prosody_extractor(device=str(device)).to(device)

    cfm_loss = CFMLoss(sigma_min=0.001)

    # ── Phase 1: CFM (VFN only) ──
    if args.phase == 1:
        decoder.eval()
        speaker_enc.eval()
        prosody.eval()
        for m in [decoder, speaker_enc, prosody]:
            for p in m.parameters():
                p.requires_grad = False
        params = list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98))
        model_dict = {"vfn": vfn, "speaker_enc": speaker_enc, "prosody": prosody}

    # ── Phase 2: E2E (VFN only, decoder frozen, audio-domain loss) ──
    else:
        decoder.eval()
        for m in [decoder, speaker_enc, prosody]:
            for p in m.parameters():
                p.requires_grad = False
        params = list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr * 0.1, betas=(0.9, 0.98))
        model_dict = {"vfn": vfn}

    # ── resume ──
    start_step = 0
    if args.resume:
        load_checkpoint(model_dict, opt, args.resume, device)

    # ── data ──
    ds = VCTKDataset(args.data_dir, crop_seconds=2.0, sample_rate=AUDIODEC_SR, sr_orig=48000)
    loader = create_dataloader(ds, batch_size=args.batch_size, shuffle=True)

    step = start_step
    running_loss = 0.0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            src = batch["src_wav"].to(device)
            tgt = batch["tgt_wav"].to(device)
            ref = batch["ref_wav"].to(device)

            if args.phase == 1:
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
                out = decoder(z_tgt)
                loss = F.l1_loss(out, tgt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            running_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"  step {step:>7d}/{args.steps}  loss={avg:.6f}")
                running_loss = 0.0

            if step % args.save_interval == 0:
                save_checkpoint(model_dict, opt, step,
                              os.path.join(args.output_dir, f"step_{step:07d}.pt"))

    save_checkpoint(model_dict, opt, step, os.path.join(args.output_dir, "final.pt"))
    print(f"  final: {args.output_dir}/final.pt")


def main():
    p = argparse.ArgumentParser(description="FlowVC AudioDec training")
    p.add_argument("--phase", type=int, default=1, help="1=CFM, 2=E2E")
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
