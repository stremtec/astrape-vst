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
from .disentangle import KanadeDisentangler, AUDIODEC_DIM, SPEAKER_DIM
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

    # Optional modules (used only in specific phases)
    disentangler = None
    vfn_small = None

    # VFN adapted for AudioDec latent
    vfn = make_vector_field_net(
        latent_dim=AUDIODEC_DIM,  # 64
        hidden_dim=256,            # increased for 160Hz latent
        speaker_dim=192,
        prosody_dim=3,
        n_blocks=12,
        dilations=(1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8),
    ).to(device)

    # ECAPA on CPU (too large for MPS alongside AudioDec)
    speaker_enc = ECAPASpeakerEncoder(device="cpu")
    # Don't .to(device) — keep on CPU to save MPS memory
    prosody = make_prosody_extractor(device=str(device)).to(device)

    cfm_loss = CFMLoss(sigma_min=0.001)

    # ── Phase 1: CFM (VFN only) ──
    if args.phase == 1:
        decoder.eval()
        prosody.eval()
        for m in [decoder, prosody]:
            for p in m.parameters():
                p.requires_grad = False
        params = list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98))
        model_dict = {"vfn": vfn, "prosody": prosody}
        # Progressive t sampling: start narrow, expand
        t_min = 0.0
        t_max = 1.0
        use_residual = True  # residual → CFM curriculum

    # ── Phase 1b: CFM curriculum (residual warmstart) ──
    elif args.phase == 11:
        decoder.eval()
        prosody.eval()
        for m in [decoder, prosody]:
            for p in m.parameters():
                p.requires_grad = False
        params = list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr * 0.1, betas=(0.9, 0.98))
        model_dict = {"vfn": vfn, "prosody": prosody}
        use_residual = False  # full CFM

    # ── Phase 20: Disentanglement training ──
    elif args.phase == 20:
        disentangler = KanadeDisentangler().to(device)
        decoder.eval(); prosody.eval()
        for m in [encoder, decoder, prosody]:
            for p in m.parameters():
                p.requires_grad = False
        params = list(disentangler.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98))
        model_dict = {"disentangler": disentangler}
        # Lightweight save: only disentangler (few KB), skip frozen encoder/decoder

    # ── Phase 21: Converter with disentangled latent ──
    elif args.phase == 21:
        disentangler = KanadeDisentangler().to(device)
        decoder.eval(); prosody.eval()
        for m in [encoder, decoder, prosody, speaker_enc]:
            for p in m.parameters():
                p.requires_grad = False
        # VFN for speaker-only prediction (32-dim input, 32-dim output)
        vfn_small = make_vector_field_net(
            latent_dim=SPEAKER_DIM, hidden_dim=128, speaker_dim=192,
            prosody_dim=3, n_blocks=6,
            dilations=(1,2,4,8,1,2),
        ).to(device)
        params = list(disentangler.parameters()) + list(vfn_small.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98))
        model_dict = {"disentangler": disentangler, "vfn_small": vfn_small}

    # ── Phase 12/13: Patch CFM variants ──
    elif args.phase in (12, 13):
        decoder.eval()
        prosody.eval()
        for m in [decoder, prosody]:
            for p in m.parameters():
                p.requires_grad = False
        params = list(vfn.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98))
        model_dict = {"vfn": vfn, "prosody": prosody}

    # ── Phase 2: E2E ──
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
                    spk_emb = spk_emb.to(device)
                    prompt = prompt.to(device)
                    pros = prosody(src)
                # Residual baseline
                z_out = vfn(z_src, torch.zeros(1, device=device), spk_emb, prompt, pros)
                loss = F.mse_loss(z_out, z_tgt) + 0.1 * F.mse_loss(z_out, z_src)

            elif args.phase == 13:
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    z_tgt = encoder.encode(tgt)
                    spk_emb, prompt = speaker_enc(ref)
                    spk_emb = spk_emb.to(device)
                    prompt = prompt.to(device)
                    pros = prosody(src)
                # Patch CFM v2: learnable projection + overlapping + multi-anchor
                B, T, D = z_src.shape
                patch_size, stride = 4, 2  # 50% overlap
                # Unfold into overlapping patches
                z_src_u = z_src.unfold(1, patch_size, stride)  # (B, n_patches, D, 4)
                z_tgt_u = z_tgt.unfold(1, patch_size, stride)
                n_patches = z_src_u.size(1)
                z_src_p = z_src_u.reshape(B, n_patches, D*patch_size)  # (B, n, 256)
                z_tgt_p = z_tgt_u.reshape(B, n_patches, D*patch_size)
                # Learnable projection 256→64
                if not hasattr(vfn, '_proj'):
                    vfn._proj = nn.Linear(256, 64).to(device)
                z_src_p = vfn._proj(z_src_p)
                z_tgt_p = vfn._proj(z_tgt_p)
                
                # CFM with 3-point auxiliary anchors (t=0, 0.5, 1.0)
                t_vals = torch.rand(B, device=device)
                z_t = (1 - t_vals.view(B,1,1)) * z_src_p + t_vals.view(B,1,1) * z_tgt_p + torch.randn_like(z_src_p) * 0.001
                v_target = z_tgt_p - z_src_p
                v_pred = vfn(z_t, t_vals, spk_emb, prompt, pros)
                cfm = F.mse_loss(v_pred, v_target)
                
                # Multi-point auxiliary anchors
                aux = 0.0
                for t_a in [0.0, 0.5, 1.0]:
                    v_a = vfn((1-t_a)*z_src_p + t_a*z_tgt_p,
                             torch.full((B,), t_a, device=device), spk_emb, prompt, pros)
                    aux = aux + F.mse_loss(v_a, v_target)
                aux = aux / 3.0
                
                loss = cfm + 0.5 * aux

            elif args.phase == 20:
                # Disentanglement v2: recon + ECAPA teacher
                z = encoder.encode(src)
                z_content, z_spk = disentangler(z)
                
                # ECAPA teacher: speaker vector should match ECAPA embedding
                ecapa_emb = speaker_enc(src)
                if isinstance(ecapa_emb, tuple):
                    ecapa_emb = ecapa_emb[0]
                ecapa_emb = ecapa_emb.to(device)
                # Cosine loss: speaker vector should align with ECAPA (cached proj)
                spk_vec = z_spk.squeeze(1)  # (B, 64)
                if not hasattr(disentangler, '_ecapa_proj'):
                    disentangler._ecapa_proj = torch.randn(64, 192, device=device) * 0.02
                spk_pred = spk_vec @ disentangler._ecapa_proj
                speaker_loss = 1 - F.cosine_similarity(spk_pred, ecapa_emb, dim=-1).mean()
                
                z_spk_exp = z_spk.expand(-1, z.size(1), -1)
                z_out = z_content + z_spk_exp
                out = decoder(z_out)
                loss = F.l1_loss(out, src) + 2.0 * speaker_loss

            elif args.phase == 21:
                # Converter with disentangled latent
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    z_tgt = encoder.encode(tgt)
                    ecapa_emb = speaker_enc(ref)
                    if isinstance(ecapa_emb, tuple):
                        ecapa_emb = ecapa_emb[0]
                    ecapa_emb = ecapa_emb.to(device)
                    pros = prosody(src)
                # Split source, keep content, predict target speaker
                c_src, _ = disentangler(z_src)
                s_pred = vfn_small(c_src, torch.zeros(1, device=device), ecapa_emb, None, pros)
                # Expand predicted speaker to match content frames
                s_pred_exp = s_pred.mean(dim=1, keepdim=True).expand(-1, c_src.size(1), -1)
                z_out = c_src + s_pred_exp
                out = decoder(z_out)
                loss = F.l1_loss(out, tgt)

            elif args.phase == 12:
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    z_tgt = encoder.encode(tgt)
                    spk_emb, prompt = speaker_enc(ref)
                    spk_emb = spk_emb.to(device)
                    prompt = prompt.to(device)
                    pros = prosody(src)
                # Patch Flow: group 4 frames → patch dim 256, project to 64
                B, T, D = z_src.shape
                pad = (4 - T % 4) % 4
                if pad > 0:
                    z_src = F.pad(z_src, (0, 0, 0, pad))
                    z_tgt = F.pad(z_tgt, (0, 0, 0, pad))
                z_src_p = z_src.reshape(B, -1, 4, D).flatten(2)  # (B, T/4, 256)
                z_tgt_p = z_tgt.reshape(B, -1, 4, D).flatten(2)
                # Fixed random projection 256→64
                proj = torch.randn(256, 64, device=device) * 0.02
                z_src_p = z_src_p @ proj
                z_tgt_p = z_tgt_p @ proj
                
                # CFM on patches
                t_vals = torch.rand(B, device=device)
                z_t = (1 - t_vals.view(B,1,1)) * z_src_p + t_vals.view(B,1,1) * z_tgt_p + torch.randn_like(z_src_p) * 0.001
                v_target = z_tgt_p - z_src_p
                v_pred = vfn(z_t, t_vals, spk_emb, prompt, pros)
                cfm = F.mse_loss(v_pred, v_target)
                
                # Auxiliary: t=0 anchor
                v0 = vfn(z_src_p, torch.zeros(B, device=device), spk_emb, prompt, pros)
                aux = F.mse_loss(v0, v_target)
                
                loss = cfm + 0.3 * aux

            elif args.phase == 11:
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    z_tgt = encoder.encode(tgt)
                    spk_emb, prompt = speaker_enc(ref)
                    spk_emb = spk_emb.to(device)
                    prompt = prompt.to(device)
                    pros = prosody(src)
                # Full CFM after residual warmstart
                loss, _ = cfm_loss(vfn, z_src, z_tgt, spk_emb, prompt, pros)

            else:  # Phase 2
                with torch.no_grad():
                    z_src = encoder.encode(src)
                    spk_emb, prompt = speaker_enc(ref)
                    spk_emb = spk_emb.to(device)
                    prompt = prompt.to(device)
                    pros = prosody(src)
                # Residual prediction + audio-domain loss
                z_out = vfn(z_src, torch.zeros(1, device=device), spk_emb, prompt, pros)
                out = decoder(z_out.cpu()).to(device)
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
                torch.mps.empty_cache()  # prevent OOM on save
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
