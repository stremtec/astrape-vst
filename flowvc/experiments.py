"""
Mimi Splitter experiments — systematic comparison of disentanglement approaches.

Expt A: Tighter bottleneck (128→64→32→16)
Expt B: Mean pooling instead of attention pooling (no content leakage via attn)
Expt C: Gradient reversal — speaker classifier on content path (push speaker OUT)
Expt D: Cosine separation loss — direct push-apart for different speakers

Each experiment trains 1000 steps and reports metrics.
"""

import torch, torch.nn as nn, torch.nn.functional as F, argparse, os, time, json, random
from pathlib import Path
from collections import defaultdict
import soundfile as sf
from scipy import signal

sys.path.insert(0, '/Users/asill/btrvrc0')
from moshi.models import loaders as moshi_loaders
MIMI_DIM = 512


# ── Dataset (shared) ──────────────────────────────────────────

def collate_varlen(batch):
    zs, speakers = zip(*batch)
    max_T = max(z.shape[1] for z in zs)
    padded = []
    for z in zs:
        if z.shape[1] < max_T:
            z = F.pad(z, (0, max_T - z.shape[1]))
        padded.append(z)
    return torch.stack(padded, dim=0), list(speakers)


class CachedMimiDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str):
        with open(Path(cache_dir) / "index.json") as f:
            self.index = json.load(f)
        self.speaker_files = defaultdict(list)
        for i, entry in enumerate(self.index):
            self.speaker_files[entry["speaker"]].append(i)
        self.speakers = list(self.speaker_files.keys())
        self.cache_dir = Path(cache_dir)

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        data = torch.load(self.cache_dir / entry["path"], map_location="cpu", weights_only=True)
        return data["z"], entry["speaker"]


# ── Model Variants ────────────────────────────────────────────

class ContentBottleneck(nn.Module):
    """Variable bottleneck size."""
    def __init__(self, bottleneck: int):
        super().__init__()
        dim = MIMI_DIM
        self.compress = nn.Conv1d(dim, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.compress(x); h = F.gelu(h); h = self.expand(h)
        h = h.transpose(1,2); h = self.norm(h)
        return (h + x.transpose(1,2)).transpose(1,2)


class SpeakerAttnPool(nn.Module):
    """Attention pooling speaker encoder."""
    def __init__(self):
        super().__init__()
        dim = MIMI_DIM
        self.conv = nn.Sequential(nn.Conv1d(dim, 256, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(256, 256, 5, padding=2), nn.GELU())
        self.query = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.proj = nn.Linear(256, dim)

    def forward(self, x):
        h = self.conv(x).transpose(1,2)  # (B, T, 256)
        q = self.query.expand(h.size(0), -1, -1)
        pooled, _ = self.attn(q, h, h)
        return self.proj(pooled).transpose(1,2)  # (B, D, 1)


class SpeakerMeanPool(nn.Module):
    """Simple mean pooling speaker encoder (no content leakage via attention)."""
    def __init__(self):
        super().__init__()
        dim = MIMI_DIM
        self.conv = nn.Sequential(nn.Conv1d(dim, 256, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(256, 256, 5, padding=2), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, x):
        h = self.conv(x).mean(dim=2)  # (B, 256) — mean pool
        return self.proj(h).unsqueeze(-1)  # (B, D, 1)


class SpeakerClassifier(nn.Module):
    """Gradient reversal speaker classifier for adversarial disentanglement."""
    def __init__(self, n_speakers: int):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(MIMI_DIM, 256), nn.GELU(), nn.Linear(256, n_speakers))

    def forward(self, x):
        return self.clf(x.mean(dim=2))  # (B, n_speakers)


class SimpleSplitter(nn.Module):
    """Configurable splitter with bottleneck size + speaker encoder type."""
    def __init__(self, bottleneck: int = 128, speaker_type: str = "attn"):
        super().__init__()
        self.content = ContentBottleneck(bottleneck)
        self.speaker = SpeakerAttnPool() if speaker_type == "attn" else SpeakerMeanPool()

    def forward(self, z):
        return self.content(z), self.speaker(z)


# ── Training ──────────────────────────────────────────────────

def run_experiment(args, exp_name: str, bottleneck: int, speaker_type: str,
                   use_grl: bool = False, use_cos_sep: bool = False):
    device = torch.device(args.device)
    ds = CachedMimiDataset(args.cache_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_varlen)

    model = SimpleSplitter(bottleneck, speaker_type).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    if use_grl:
        n_spk = len(ds.speakers)
        spk_clf = SpeakerClassifier(n_spk).to(device)
        opt_clf = torch.optim.AdamW(spk_clf.parameters(), lr=args.lr)

    print("")
    print("="*60)
    print(f"Expt: {exp_name}")
    print(f"  bottleneck={bottleneck}, speaker={speaker_type}, grl={use_grl}, cos_sep={use_cos_sep}")
    print("="*60)

    step = 0
    results = []

    while step < args.steps:
        for batch in loader:
            if step >= args.steps: break
            z_batch, speakers = batch
            B = z_batch.size(0)

            # Cross-speaker pairs
            z_src_list, z_tgt_list = [], []
            for i in range(B):
                for j in range(B):
                    if speakers[j] != speakers[i]:
                        z_src_list.append(z_batch[i:i+1])
                        z_tgt_list.append(z_batch[j:j+1])
                        break
            if len(z_src_list) == 0: continue

            z_src = torch.cat(z_src_list, dim=0).to(device)
            z_tgt = torch.cat(z_tgt_list, dim=0).to(device)
            max_T = max(z_src.shape[2], z_tgt.shape[2])
            if z_src.shape[2] < max_T: z_src = F.pad(z_src, (0, max_T - z_src.shape[2]))
            if z_tgt.shape[2] < max_T: z_tgt = F.pad(z_tgt, (0, max_T - z_tgt.shape[2]))
            T = max_T

            c_src, s_src = model(z_src)
            c_tgt, s_tgt = model(z_tgt)

            # Reconstruction
            recon_loss = F.mse_loss(c_src + s_src.expand(-1, -1, T), z_src)

            # Content invariance
            c_cos = F.cosine_similarity(c_src.transpose(1,2).reshape(-1,512),
                                         c_tgt.transpose(1,2).reshape(-1,512), dim=-1).mean()
            content_loss = (1 - c_cos) ** 2

            # Speaker separation
            s_cos = F.cosine_similarity(s_src.flatten(), s_tgt.flatten(), dim=0)

            if use_cos_sep:
                # Direct push-apart: maximize negative cosine
                sep_loss = torch.relu(s_cos + 0.5)  # target: cos < -0.5
            else:
                # Anti-collapse: push toward opposite
                sep_loss = torch.relu(s_cos - 0.3) + F.mse_loss(s_src, -s_tgt.detach()) * 0.5

            loss = recon_loss + 0.5 * content_loss + 0.5 * sep_loss

            # GRL: adversarial speaker classifier on content
            if use_grl:
                spk_labels = torch.tensor([ds.speakers.index(s) for s in speakers[:len(z_src_list)]], device=device)
                spk_pred = spk_clf(c_src.detach())
                clf_loss = F.cross_entropy(spk_pred, spk_labels)
                opt_clf.zero_grad(); clf_loss.backward(); opt_clf.step()

                # Adversarial: content should NOT predict speaker
                spk_pred_adv = spk_clf(c_src)
                grl_loss = -F.cross_entropy(spk_pred_adv, spk_labels)
                loss = loss + 0.1 * grl_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            if step % 200 == 0:
                print(f"  step {step:>4d}: recon={recon_loss.item():.4f} c_cos={c_cos.item():.4f} s_cos={s_cos.item():.4f}")
                results.append({"step": step, "recon": recon_loss.item(),
                               "c_cos": c_cos.item(), "s_cos": s_cos.item()})

    return results


# ── CLI ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="runs/mimi_cache")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()

    experiments = [
        ("A_bn128_attn", 128, "attn", False, False),
        ("B_bn64_attn", 64, "attn", False, False),
        ("C_bn32_attn", 32, "attn", False, False),
        ("D_bn64_meanpool", 64, "mean", False, False),
        ("E_bn64_attn_cossep", 64, "attn", False, True),
        ("F_bn64_attn_grl", 64, "attn", True, False),
    ]

    all_results = {}
    for name, bn, spk_type, grl, cos_sep in experiments:
        res = run_experiment(args, name, bn, spk_type, grl, cos_sep)
        # Final metrics
        final = res[-1] if res else {}
        all_results[name] = final
        print(f"  => final: recon={final.get('recon',0):.4f} c_cos={final.get('c_cos',0):.4f} s_cos={final.get('s_cos',0):.4f}")

    # Summary
    print("")
    print("="*60)
    print("SUMMARY")
    print("="*60)
    for name, m in all_results.items():
        c = m.get('c_cos', 0); s = m.get('s_cos', 0)
        score = c * (1 - max(s, 0))  # higher = better
        print(f"  {name:25s}: c_cos={c:.4f} s_cos={s:.4f} score={score:.4f}")


import sys
if __name__ == "__main__":
    main()
