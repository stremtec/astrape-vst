#!/usr/bin/env python3
"""Quick training harness for MimiSplitter v2 with small dataset."""

import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, torch.nn as nn, time
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Load Mimi (moved to MPS)
mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi).to(device)
print(f"Trainable params: {sum(p.numel() for p in splitter.parameters() if p.requires_grad)}")

# Small dataset: 5 spks × 3 utterances
spk_list = ['p225','p226','p227','p228','p229']
root = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
safe_len = 48000  # 2s at 24kHz == 25 frames latent

# Pre-encode all samples
print("Pre-encoding...")
samples = []
for spk_idx, spk in enumerate(spk_list):
    import glob
    files = sorted(glob.glob(f"{root}/{spk}/{spk}_*_mic1.flac"))[:3]
    for f in files:
        d, sr = sf.read(f)
        if d.ndim > 1: d = d.mean(axis=1)
        if sr != 24000: d = signal.resample(d, int(len(d)*24000/sr))
        if len(d) < safe_len: d = np.pad(d, (0, safe_len - len(d)))
        d = d[:safe_len]
        x = torch.from_numpy(d).float().view(1,1,-1).to(device)
        with torch.no_grad():
            z, codes = mimi_encode(x, mimi)
        samples.append((z.squeeze(0), codes.squeeze(0), spk_idx))  # keep on device
        print(f"  {spk} sample {len(samples)}: z={z.shape} codes={codes.shape}", flush=True)

print(f"Pre-encoded: {len(samples)} samples")

# Training
optimizer = AdamW(splitter.parameters(), lr=1e-3, weight_decay=1e-5)
mse = nn.MSELoss()

EPOCHS = 200
BATCH_SIZE = 3

print(f"Training {EPOCHS} epochs...", flush=True)
for epoch in range(EPOCHS):
    # Shuffle
    idxs = torch.randperm(len(samples))
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(samples), BATCH_SIZE):
        batch_idxs = idxs[i:i+BATCH_SIZE]
        
        # Stack batch (all should have same T due to safe_len=48000)
        z_batch = torch.stack([samples[j][0] for j in batch_idxs])  # (B, D, T)
        code_batch = torch.stack([samples[j][1] for j in batch_idxs])  # (B, Nq, T)
        
        z_vc, C, S, A = splitter(z_batch, code_batch)
        loss = mse(z_vc, z_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | MSE: {avg_loss:.6f}", flush=True)

# Save
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": splitter.state_dict()}, "checkpoints/mimi_splitter_v2_final.pt")

# Validation: decode first sample
print("Decoding validation...", flush=True)
splitter.eval()
with torch.no_grad():
    z_val = samples[0][0].unsqueeze(0)
    code_val = samples[0][1].unsqueeze(0)
    
    z_vc, C, S, A = splitter(z_val, code_val)
    
    # Ground truth audio
    x_gt = mimi_decode_latent(mimi, z_val)
    # Splitter audio
    x_vc = mimi_decode_latent(mimi, z_vc)
    
    sf.write("checkpoints/val_gt.wav", x_gt[0,0].cpu().numpy(), 24000)
    sf.write("checkpoints/val_splitter.wav", x_vc[0,0].cpu().numpy(), 24000)

# Also: speaker probe on C (content path should have low speaker info)
print("Speaker probe on content...", flush=True)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

C_all = []
S_all = []
y_all = []
for z, codes, spk_id in samples:
    z_in = z.unsqueeze(0)
    codes_in = codes.unsqueeze(0)
    _, C, S, _ = splitter(z_in, codes_in)
    C_all.append(C.squeeze(0).mean(dim=1).cpu().numpy())  # mean-pool over time
    S_all.append(S.squeeze(0).cpu().numpy())
    y_all.append(spk_id)

X_C = np.stack(C_all)
X_S = np.stack(S_all)
y = np.array(y_all)

scaler = StandardScaler()
if len(set(y)) > 2:
    clf = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    scores_C = cross_val_score(clf, scaler.fit_transform(X_C), y, cv=3)
    scores_S = cross_val_score(clf, scaler.fit_transform(X_S), y, cv=3)
    chance = 100/len(spk_list)
    print(f"  Content (C): speaker acc = {scores_C.mean()*100:.1f}% (±{scores_C.std()*100:.1f}) vs chance={chance:.0f}%")
    print(f"  Speaker (S): speaker acc = {scores_S.mean()*100:.1f}% (±{scores_S.std()*100:.1f}) vs chance={chance:.0f}%")

print("Done!")
