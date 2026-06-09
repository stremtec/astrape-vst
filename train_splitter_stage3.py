#!/usr/bin/env python3
"""Stage 3: Adversarial training to remove speaker from content path."""

import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, torch.nn as nn, time
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi).to(device)
print(f"Trainable params: {sum(p.numel() for p in splitter.parameters() if p.requires_grad)}")

# Train/val split: 5 speakers train, 5 val
TRAIN_SPKS = ['p225','p226','p227','p228','p229']
VAL_SPKS = ['p230','p231','p232','p233','p234']
root = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
safe_len = 48000

def encode_spk(spk_list, n_utt=3):
    samples = []
    for spk_idx, spk in enumerate(spk_list):
        import glob
        files = sorted(glob.glob(f"{root}/{spk}/{spk}_*_mic1.flac"))[:n_utt]
        for f in files:
            d, sr = sf.read(f)
            if d.ndim > 1: d = d.mean(axis=1)
            if sr != 24000: d = signal.resample(d, int(len(d)*24000/sr))
            if len(d) < safe_len: d = np.pad(d, (0, safe_len - len(d)))
            d = d[:safe_len]
            x = torch.from_numpy(d).float().view(1,1,-1).to(device)
            with torch.no_grad():
                z, codes = mimi_encode(x, mimi)
            samples.append((z.squeeze(0), codes.squeeze(0), spk_idx))
    return samples

print("Pre-encoding train...")
train_samples = encode_spk(TRAIN_SPKS, 3)
print("Pre-encoding val...")
val_samples = encode_spk(VAL_SPKS, 3)
print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

# ── Speaker classifier for adversarial ──────────────────────────────────
class SpeakerClassifier(nn.Module):
    def __init__(self, dim=512, n_spk=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, n_spk),
        )
    def forward(self, x):
        # x: (B, D) or (B, D, T) -> pool to (B, D)
        if x.dim() == 3:
            x = x.mean(dim=-1)
        return self.net(x)

# ── Gradient reversal layer ──────────────────────────────────────────────
from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

# ── Training ─────────────────────────────────────────────────────────────
optimizer = AdamW(splitter.parameters(), lr=1e-3, weight_decay=1e-5)
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

spk_clf_C = SpeakerClassifier(n_spk=len(TRAIN_SPKS)).to(device)
spk_clf_S = SpeakerClassifier(n_spk=len(TRAIN_SPKS)).to(device)
opt_clf = AdamW(list(spk_clf_C.parameters()) + list(spk_clf_S.parameters()), lr=1e-3)

EPOCHS = 300
BATCH_SIZE = 3

print(f"Training {EPOCHS} epochs (Stage 3: adversarial)...", flush=True)

def probe(splitter, samples, spk_list, tag=""):
    C_list, S_list, y_list = [], [], []
    for z, codes, spk_id in samples:
        z_in = z.unsqueeze(0)
        codes_in = codes.unsqueeze(0)
        with torch.no_grad():
            z_vc, C, S, A = splitter(z_in, codes_in)
        C_list.append(C.squeeze(0).mean(dim=1).detach().cpu().numpy())
        S_list.append(S.squeeze(0).detach().cpu().numpy())
        y_list.append(spk_id)
    X_C = np.stack(C_list); X_S = np.stack(S_list); y = np.array(y_list)
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    if len(set(y)) >= 3:
        scores_C = cross_val_score(clf, scaler.fit_transform(X_C), y, cv=min(3, len(y)//3 or 2))
        scores_S = cross_val_score(clf, scaler.fit_transform(X_S), y, cv=min(3, len(y)//3 or 2))
    else:
        scores_C = [0]; scores_S = [0]
    chance = 100/len(spk_list)
    return scores_C.mean()*100, scores_S.mean()*100, chance

for epoch in range(EPOCHS):
    idxs = torch.randperm(len(train_samples))
    total_loss_recon = 0.0
    total_loss_adv_C = 0.0
    total_loss_spk_S = 0.0
    n_batches = 0

    for i in range(0, len(train_samples), BATCH_SIZE):
        batch_idxs = idxs[i:i+BATCH_SIZE]
        z_batch = torch.stack([train_samples[j][0] for j in batch_idxs])
        code_batch = torch.stack([train_samples[j][1] for j in batch_idxs])
        spk_ids = torch.tensor([train_samples[j][2] for j in batch_idxs]).to(device)

        # Splitter forward
        z_vc, C, S, A = splitter(z_batch, code_batch)

        # 1. Reconstruction loss
        L_recon = mse(z_vc, z_batch)

        # 2. Adversarial: C should NOT predict speaker
        C_rev = grad_reverse(C, alpha=1.0)
        pred_C = spk_clf_C(C_rev.mean(dim=-1))
        L_adv_C = ce(pred_C, spk_ids)

        # 3. Speaker: S SHOULD predict speaker
        pred_S = spk_clf_S(S)
        L_spk_S = ce(pred_S, spk_ids)

        # Total
        loss = L_recon + 0.1 * L_adv_C + 0.1 * L_spk_S

        optimizer.zero_grad()
        opt_clf.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
        optimizer.step()
        opt_clf.step()

        total_loss_recon += L_recon.item()
        total_loss_adv_C += L_adv_C.item()
        total_loss_spk_S += L_spk_S.item()
        n_batches += 1

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        # Probe train and val
        train_C, train_S, train_chance = probe(splitter, train_samples, TRAIN_SPKS, "train")
        val_C, val_S, val_chance = probe(splitter, val_samples, VAL_SPKS, "val")

        print(f"  E{epoch:3d} | L_recon={total_loss_recon/max(n_batches,1):.4f} "
              f"L_adv={total_loss_adv_C/max(n_batches,1):.4f} "
              f"L_spk={total_loss_spk_S/max(n_batches,1):.4f} | "
              f"Train C→{train_C:.0f}% S→{train_S:.0f}% (chance={train_chance:.0f}) | "
              f"Val C→{val_C:.0f}% S→{val_S:.0f}% (chance={val_chance:.0f})", flush=True)

# Final save
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": splitter.state_dict()},
           "checkpoints/mimi_splitter_v2_adv.pt")

# Final probe
print()
print("=== Final Results ===")
train_C, train_S, tc = probe(splitter, train_samples, TRAIN_SPKS)
val_C, val_S, vc = probe(splitter, val_samples, VAL_SPKS)
print(f"Train: C={train_C:.1f}% (chance={tc:.0f}%), S={train_S:.1f}%")
print(f"Val:   C={val_C:.1f}% (chance={vc:.0f}%), S={val_S:.1f}%")

# Decode val sample
print()
print("Decoding val sample...")
splitter.eval()
with torch.no_grad():
    z_val = val_samples[0][0].unsqueeze(0)
    code_val = val_samples[0][1].unsqueeze(0)
    z_vc, C, S, A = splitter(z_val, code_val)
    x_gt = mimi_decode_latent(mimi, z_val)
    x_vc = mimi_decode_latent(mimi, z_vc)
    sf.write("checkpoints/val_s3_gt.wav", x_gt[0,0].cpu().numpy(), 24000)
    sf.write("checkpoints/val_s3_splitter.wav", x_vc[0,0].cpu().numpy(), 24000)
    print(f"  Saved val_s3_*.wav")

# LSD comparison
from scipy.signal import stft
def compute_lsd(a, b, sr=24000):
    f1,_,Z1=stft(a,fs=sr,nperseg=512,noverlap=256)
    f2,_,Z2=stft(b,fs=sr,nperseg=512,noverlap=256)
    T=min(Z1.shape[1],Z2.shape[1])
    return np.mean(np.sqrt(np.mean((np.log10(np.abs(Z1[:,:T])+1e-8)-np.log10(np.abs(Z2[:,:T])+1e-8))**2,axis=0)))*20

x_src = sf.read(f"{root}/p230/p230_001_mic1.flac")[0][:48000]
lsd_gt = compute_lsd(x_src, x_gt[0,0].cpu().numpy())
lsd_vc = compute_lsd(x_src, x_vc[0,0].cpu().numpy())
print(f"  GT LSD: {lsd_gt:.1f}dB")
print(f"  Splitter LSD: {lsd_vc:.1f}dB")

print()
print("Done!")
