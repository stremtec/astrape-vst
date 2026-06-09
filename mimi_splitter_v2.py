"""
MimiSplitter v2.1 — working decoder path + latent-space training.

Key insight: train splitter on latent reconstruction, not audio waveform.
Decode path: z_vc → _to_encoder_framerate → decoder_transformer → decoder → audio.

Stages:
  2: Same-speaker latent reconstruction (freeze Mimi)
  3: Speaker adversarial (gradient reversal on content path)
  4: Target speaker VC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os, sys, glob, json, argparse
from typing import Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from scipy import signal

# ── Mimi helper ───────────────────────────────────────────────────────────
def load_mimi(device="cpu"):
    from moshi.models.loaders import get_mimi
    ckpt = os.path.expanduser(
        "~/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/"
        "snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/"
        "tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    mimi = get_mimi(ckpt, device=device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False
    return mimi


def mimi_decode_latent(mimi, z):
    """Decode a quantized latent z (B, D, T_frame) to audio."""
    with torch.no_grad():
        emb = mimi._to_encoder_framerate(z)       # (B, D, T_enc)
        (emb,) = mimi.decoder_transformer(emb)     # (B, D, T_enc)
        out = mimi.decoder(emb)                    # (B, 1, T_audio)
    return out


def mimi_encode(x, mimi):
    """Encode audio → quantized latent + codes."""
    with torch.no_grad():
        z_post = mimi.encode_to_latent(x, quantize=True)
        codes = mimi.encode(x)
    return z_post, codes


# ── Splitter components ───────────────────────────────────────────────────
MIMI_DIM = 512
STRIDE_SAMPLES = 1920


class ContentExtractor(nn.Module):
    """q0 latent → bottleneck → content-clean latent."""
    def __init__(self, dim=512, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, bottleneck, 1),
            nn.GELU(),
            nn.Conv1d(bottleneck, dim, 1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_q0):
        # z_q0: (B, D, T)
        h = self.net(z_q0).transpose(1, 2)
        h = self.norm(h)
        return (z_q0 + h.transpose(1, 2))


class SpeakerEncoder(nn.Module):
    """Global speaker embedding from acoustic codebooks (q1-q7)."""
    def __init__(self, dim=512, spk_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, spk_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(spk_dim, spk_dim, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(spk_dim, dim)

    def forward(self, z_acoustic):
        # z_acoustic: (B, D, T) — from q1-q7
        h = self.conv(z_acoustic)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)  # (B, D)


class AcousticAdapter(nn.Module):
    """Adapt q1-q7: speaker-conditioned modulation + content awareness."""
    def __init__(self, dim=512, hidden=256):
        super().__init__()
        # FiLM: speaker → scale + bias
        self.scale = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.bias = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv1d(dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, dim, 3, padding=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_acoustic, spk_emb, z_content):
        # FiLM
        s = self.scale(spk_emb).unsqueeze(-1)   # (B, D, 1)
        b = self.bias(spk_emb).unsqueeze(-1)    # (B, D, 1)
        z_mod = z_acoustic * (1 + torch.tanh(s)) + b

        # Content-aware refinement
        z_in = z_content + z_mod
        h = z_in.transpose(1, 2)
        h = self.norm(h)
        h = h.transpose(1, 2)
        z_ref = self.refine(h)

        return z_mod + z_ref


class MimiSplitterV2(nn.Module):
    """Factorizes Mimi latent: Content(q0..qNc) + Speaker + Acoustic detail."""

    def __init__(self, mimi, dim=512, content_bottleneck=64, spk_dim=256, n_content=1):
        super().__init__()
        self.mimi = mimi
        self.n_content = n_content  # number of codebooks for content (1=q0, 2=q0+q1)
        self.content_extractor = ContentExtractor(dim, content_bottleneck)
        self.speaker_encoder = SpeakerEncoder(dim, spk_dim)
        self.acoustic_adapter = AcousticAdapter(dim, spk_dim)

    def _get_content_acoustic(self, codes):
        """Extract content and acoustic latents from codes."""
        with torch.no_grad():
            nc = self.n_content
            self.mimi.set_num_codebooks(nc)
            z_content = self.mimi.decode_latent(codes[:, :nc, :])
            n_acoustic = codes.shape[1] - nc
            self.mimi.set_num_codebooks(n_acoustic)
            z_ac = self.mimi.decode_latent(codes[:, nc:, :])
            self.mimi.set_num_codebooks(8)
        return z_content, z_ac

    def forward(self, z_post, codes):
        z_content, z_ac = self._get_content_acoustic(codes)
        C = self.content_extractor(z_content)
        S = self.speaker_encoder(z_post)
        A = self.acoustic_adapter(z_ac, S, C)
        z_vc = C + A
        return z_vc, C, S, A

    def decode(self, z_vc):
        """z_vc → audio."""
        return mimi_decode_latent(self.mimi, z_vc)

    def infer_swap(self, x_src, x_tgt):
        """VC: source content + target speaker."""
        with torch.no_grad():
            z_src, codes_src = mimi_encode(x_src, self.mimi)
            z_tgt, codes_tgt = mimi_encode(x_tgt, self.mimi)

            z_q0_src, _ = self._get_q0_qacoustic(codes_src)
            _, z_ac_tgt = self._get_q0_qacoustic(codes_tgt)

            C = self.content_extractor(z_q0_src)
            S_tgt = self.speaker_encoder(z_tgt)
            A = self.acoustic_adapter(z_ac_tgt, S_tgt, C)

            z_vc = C + A
            x_vc = self.decode(z_vc)
        return x_vc


# ── Training ──────────────────────────────────────────────────────────────
def pad_to_stride(audio, stride=STRIDE_SAMPLES):
    L = audio.shape[-1]
    pad = (stride - L % stride) % stride
    if pad > 0:
        audio = F.pad(audio, (0, pad))
    return audio


class VCTKLatentDataset(torch.utils.data.Dataset):
    def __init__(self, root, spk_list, mimi, duration=2.0):
        self.mimi = mimi
        self.samples = []
        safe_len = int(duration * 24000) // STRIDE_SAMPLES * STRIDE_SAMPLES
        self.safe_len = safe_len
        self.spk_to_id = {s: i for i, s in enumerate(spk_list)}

        for spk in spk_list:
            files = sorted(glob.glob(f"{root}/{spk}/{spk}_*_mic1.flac"))
            for f in files:
                self.samples.append((f, spk))

        print(f"VCTKLatentDataset: {len(self.samples)} samples, {len(spk_list)} spks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, spk = self.samples[idx]
        d, sr = sf.read(path)
        if d.ndim > 1:
            d = d.mean(axis=1)
        if sr != 24000:
            d = signal.resample(d, int(len(d) * 24000 / sr))

        if len(d) > self.safe_len:
            start = np.random.randint(0, len(d) - self.safe_len)
            d = d[start:start + self.safe_len]
        elif len(d) < self.safe_len:
            d = np.pad(d, (0, self.safe_len - len(d)))

        x = torch.from_numpy(d).float().view(1, 1, -1)
        with torch.no_grad():
            z_post, codes = mimi_encode(x, self.mimi)
        return z_post.squeeze(0), codes.squeeze(0), self.spk_to_id[spk]


def train_stage2(args):
    """Stage 2: Same-speaker latent reconstruction."""
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    print("Loading Mimi...", flush=True)
    mimi = load_mimi(device).to(device)
    print(f"Mimi loaded", flush=True)
    splitter = MimiSplitterV2(mimi).to(device)
    print(f"Params: {sum(p.numel() for p in splitter.parameters() if p.requires_grad)}", flush=True)

    spk_list = ['p225','p226','p227','p228','p229',
                 'p230','p231','p232','p233','p234']
    print(f"Loading dataset...", flush=True)
    dataset = VCTKLatentDataset(
        root="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed",
        spk_list=spk_list,
        mimi=mimi,
        duration=2.0,
    )
    print(f"Dataset loaded: {len(dataset)} samples", flush=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    optimizer = AdamW(splitter.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse = nn.MSELoss()

    print(f"Training {args.epochs} epochs, {len(dataset)} samples, bs={args.batch_size}")
    os.makedirs("checkpoints", exist_ok=True)
    log_lines = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        splitter.train()

        for z_post, codes, _ in loader:
            z_post = z_post.to(device)      # (B, D, T)
            codes = codes.unsqueeze(0) if codes.dim() == 2 else codes
            codes = codes.to(device)

            z_vc, C, S, A = splitter(z_post, codes)
            loss = mse(z_vc, z_post)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        log = f"Epoch {epoch+1:3d}/{args.epochs} | MSE: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        log_lines.append(log)
        print(log)

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"checkpoints/mimi_splitter_v2_e{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": splitter.state_dict(),
                "opt": optimizer.state_dict(),
            }, ckpt_path)

            # Validation: decode a sample
            splitter.eval()
            with torch.no_grad():
                z_val, codes_val, _ = dataset[0]
                z_val = z_val.unsqueeze(0).to(device)
                codes_val = codes_val.unsqueeze(0).to(device)

                z_vc, _, _, _ = splitter(z_val, codes_val)
                x_recon = splitter.decode(z_vc)

                # Also decode ground truth for comparison
                x_gt = mimi_decode_latent(mimi, z_val)

                sf.write(f"checkpoints/val_gt_e{epoch+1}.wav",
                         x_gt[0, 0].cpu().numpy(), 24000)
                sf.write(f"checkpoints/val_splitter_e{epoch+1}.wav",
                         x_recon[0, 0].cpu().numpy(), 24000)
            splitter.train()
            print(f"  Saved ckpt + val audio")

    # Final save
    final_path = "checkpoints/mimi_splitter_v2_final.pt"
    torch.save({"model_state_dict": splitter.state_dict()}, final_path)
    print(f"Final: {final_path}")

    return splitter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.stage == 2:
        train_stage2(args)
    elif args.stage == 3:
        print("Stage 3: Speaker adversarial — TODO")
    elif args.stage == 4:
        print("Stage 4: Target speaker VC — TODO")
