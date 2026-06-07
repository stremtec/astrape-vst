"""
Mimi Splitter training pipeline.

Phase 0: Cache Mimi latents for VCTK (faster training)
Phase 1: Train splitter with reconstruction + content invariance losses
Phase 2: VC inference with trained splitter
"""

from __future__ import annotations
import os, sys, argparse, random, time, torch, torch.nn.functional as F
from pathlib import Path
import soundfile as sf
from scipy import signal
from collections import defaultdict

sys.path.insert(0, '/Users/asill/btrvrc0')  # for moshi

from moshi.models import loaders as moshi_loaders
from flowvc.mimi_splitter import MimiSplitterV2, MIMI_DIM

MIMI_SR = 24000


# ── Phase 0: Cache ─────────────────────────────────────────────

def cache_mimi_latents(data_dir: str, cache_dir: str, device: str = "cpu", max_per_spk: int = 30):
    """Pre-encode VCTK audio with Mimi to speed up splitter training."""
    os.makedirs(cache_dir, exist_ok=True)

    mimi_path = Path(os.path.expanduser(
        "~/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/"
        "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors"
    ))
    mimi = moshi_loaders.get_mimi(mimi_path)
    mimi.eval()
    for p in mimi.parameters(): p.requires_grad_(False)

    # Find all speakers
    speakers = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    print(f"Found {len(speakers)} speakers")

    index = []
    for spk_dir in speakers:
        spk_id = spk_dir.name
        files = sorted(spk_dir.glob("*.flac"))
        random.shuffle(files)
        files = files[:max_per_spk]

        os.makedirs(Path(cache_dir) / spk_id, exist_ok=True)
        for f in files:
            data, sr = sf.read(str(f))
            if sr != MIMI_SR:
                data = signal.resample(data, int(len(data) * MIMI_SR / sr), axis=0)
            if data.ndim > 1: data = data.mean(axis=1)
            data = data[:MIMI_SR * 2]  # 2 seconds

            x = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                z = mimi.encode_to_latent(x, quantize=False).cpu()  # (1, 512, T)

            # Extract utterance base from filename: p225_001_mic1 → p225_001
            utt_base = f.stem.rsplit("_", 1)[0] if "_mic" in f.stem else f.stem

            out_path = Path(cache_dir) / spk_id / f"{f.stem}.pt"
            torch.save({"z": z.squeeze(0), "speaker": spk_id, "utterance_id": utt_base}, out_path)
            index.append({"path": str(out_path.relative_to(cache_dir)), "speaker": spk_id, "utterance_id": utt_base})

        print(f"  {spk_id}: {len(files)} files")

    # Save index
    import json
    with open(Path(cache_dir) / "index.json", "w") as f:
        json.dump(index, f)

    print(f"Cached {len(index)} files to {cache_dir}")


# ── Phase 1: Train Splitter ────────────────────────────────────

class CachedMimiDataset(torch.utils.data.Dataset):
    """Loads pre-cached Mimi latents with speaker groups."""

    def __init__(self, cache_dir: str):
        import json
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "index.json") as f:
            self.index = json.load(f)

        self.speaker_files = defaultdict(list)
        for i, entry in enumerate(self.index):
            self.speaker_files[entry["speaker"]].append(i)

        self.speakers = list(self.speaker_files.keys())
        print(f"  CachedMimiDataset: {len(self.index)} files, {len(self.speakers)} speakers")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        data = torch.load(self.cache_dir / entry["path"], map_location="cpu", weights_only=True)
        return data["z"], entry["speaker"]


def collate_varlen(batch):
    """Collate variable-length Mimi latents by padding to max length."""
    zs, speakers = zip(*batch)
    max_T = max(z.shape[1] for z in zs)
    padded = []
    for z in zs:
        if z.shape[1] < max_T:
            z = F.pad(z, (0, max_T - z.shape[1]))
        padded.append(z)
    return torch.stack(padded, dim=0), list(speakers)


def train_splitter(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = CachedMimiDataset(args.cache_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_varlen
    )

    # Splitter
    splitter = MimiSplitterV2().to(device)
    
    # Resume
    start_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        splitter.load_state_dict(sd)
        print(f"Resumed from {args.resume}")

    opt = torch.optim.AdamW(splitter.parameters(), lr=args.lr, betas=(0.9, 0.98))

    print(f"Training splitter: {args.steps} steps, batch={args.batch_size}, lr={args.lr}")

    step = 0
    running_loss = 0.0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps: break

            z_batch, speakers = batch
            B = z_batch.size(0)

            # Create cross-speaker pairs within batch
            z_src, z_tgt = [], []
            for i in range(B):
                # Find a different speaker in the batch
                for j in range(B):
                    if speakers[j] != speakers[i]:
                        z_src.append(z_batch[i:i+1])
                        z_tgt.append(z_batch[j:j+1])
                        break

            if len(z_src) == 0: continue

            z_src = torch.cat(z_src, dim=0).to(device)
            z_tgt = torch.cat(z_tgt, dim=0).to(device)
            
            # Pad to max length in batch
            max_T = max(z_src.shape[2], z_tgt.shape[2])
            if z_src.shape[2] < max_T:
                z_src = F.pad(z_src, (0, max_T - z_src.shape[2]))
            if z_tgt.shape[2] < max_T:
                z_tgt = F.pad(z_tgt, (0, max_T - z_tgt.shape[2]))
            T = max_T

            # Split
            c_src, s_src = splitter(z_src)
            c_tgt, s_tgt = splitter(z_tgt)

            # Losses
            z_recon = c_src + s_src.expand(-1, -1, T)
            recon_loss = F.mse_loss(z_recon, z_src)  # latent MSE (no decoder — much faster!)

            c_cos = F.cosine_similarity(
                c_src.transpose(1, 2).reshape(-1, 512),
                c_tgt.transpose(1, 2).reshape(-1, 512), dim=-1
            ).mean()
            content_loss = (1 - c_cos) ** 2

            s_cos = F.cosine_similarity(s_src.flatten(), s_tgt.flatten(), dim=0)
            # Stronger separation: push apart + keep magnitude different
            sep_loss = torch.relu(s_cos - 0.3) + F.mse_loss(s_src, -s_tgt.detach()) * 0.5

            loss = recon_loss + 0.5 * content_loss + 0.5 * sep_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
            opt.step()

            running_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"  step {step:>6d}/{args.steps}: loss={avg:.4f} c_cos={c_cos.item():.4f} s_cos={s_cos.item():.4f}")
                running_loss = 0.0

            if step % args.save_interval == 0:
                torch.save(splitter.state_dict(), os.path.join(args.output_dir, f"splitter_step{step:07d}.pt"))

    torch.save(splitter.state_dict(), os.path.join(args.output_dir, "splitter_final.pt"))
    print(f"Saved: {args.output_dir}/splitter_final.pt")


# ── CLI ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Mimi Splitter training")
    sub = p.add_subparsers(dest="cmd")

    # Cache
    pc = sub.add_parser("cache")
    pc.add_argument("--data-dir", required=True)
    pc.add_argument("--cache-dir", default="./runs/mimi_cache")
    pc.add_argument("--max-per-spk", type=int, default=30)

    # Train
    pt = sub.add_parser("train")
    pt.add_argument("--cache-dir", required=True)
    pt.add_argument("--output-dir", default="./runs/mimi_splitter")
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--batch-size", type=int, default=4)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--steps", type=int, default=5000)
    pt.add_argument("--log-interval", type=int, default=200)
    pt.add_argument("--save-interval", type=int, default=1000)
    pt.add_argument("--resume", type=str, default="")

    args = p.parse_args()
    if args.cmd == "cache":
        cache_mimi_latents(args.data_dir, args.cache_dir, max_per_spk=args.max_per_spk)
    elif args.cmd == "train":
        train_splitter(args)


if __name__ == "__main__":
    main()
