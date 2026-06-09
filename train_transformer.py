#!/usr/bin/env python3
"""
Train Mimi Splitter with Transformer acoustic adapter.
- 40 train / 10 val / 9 test speakers
- Adversarial training (C→chance, S→100%)
- Fixed evaluation metrics
"""

import sys, os, glob, time, warnings
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, mimi_encode, mimi_decode_latent, ContentExtractor, SpeakerEncoder
from transformer_adapter import TransformerAcousticAdapter
import torch, torch.nn as nn
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SR = 24000
SAFE_LEN = 48000
BATCH_SIZE = 8
EPOCHS = 300
ADV_WEIGHT = 0.5
SPK_WEIGHT = 0.5
NUM_LAYERS = 2  # Transformer layers (reduced from 4 to fight overfitting)

ALL_SPKS = [
    'p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266',
    'p267','p268','p269','p270','p271','p272','p273','p274','p275','p276',
    'p277','p278','p279','p280','p281','p282','p283','p284','p285',
]
TRAIN_SPKS = ALL_SPKS[:40]
VAL_SPKS = ALL_SPKS[40:50]
TEST_SPKS = ALL_SPKS[50:]

device = torch.device('cpu')
print("Device:", device, flush=True)
print("Train:", len(TRAIN_SPKS), "Val:", len(VAL_SPKS), "Test:", len(TEST_SPKS))

# Load Mimi
print("Loading Mimi...", flush=True)
mimi = load_mimi(device).to(device)

# Create splitter with transformer adapter
class MimiSplitterTransformer(nn.Module):
    def __init__(self, mimi, dim=512, spk_dim=256, num_layers=4):
        super().__init__()
        self.mimi = mimi
        self.content_extractor = ContentExtractor(dim, bottleneck=64)
        self.speaker_encoder = SpeakerEncoder(dim, spk_dim)
        self.acoustic_adapter = TransformerAcousticAdapter(
            dim=dim, spk_dim=spk_dim, num_layers=num_layers,
            num_heads=8, ff_mult=4, dropout=0.2
        )

    def _get_q0_qacoustic(self, codes):
        with torch.no_grad():
            self.mimi.set_num_codebooks(1)
            z_q0 = self.mimi.decode_latent(codes[:, :1, :])
            n_ac = codes.shape[1] - 1
            self.mimi.set_num_codebooks(n_ac)
            z_ac = self.mimi.decode_latent(codes[:, 1:, :])
            self.mimi.set_num_codebooks(8)
        return z_q0, z_ac

    def forward(self, z_post, codes):
        z_q0, z_ac = self._get_q0_qacoustic(codes)
        C = self.content_extractor(z_q0)
        S = self.speaker_encoder(z_post)
        A = self.acoustic_adapter(z_ac, S, C)
        z_vc = C + A
        return z_vc, C, S, A

splitter = MimiSplitterTransformer(mimi, num_layers=NUM_LAYERS).to(device)
print("Trainable params:", sum(p.numel() for p in splitter.parameters() if p.requires_grad))

# Pre-encode
ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def encode_speakers(spk_list, n_utt=5):
    samples = []
    for spk_idx, spk in enumerate(spk_list):
        files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:n_utt]
        for f in files:
            d, sr = sf.read(f)
            if d.ndim > 1:
                d = d.mean(axis=1)
            if sr != SR:
                d = signal.resample(d, int(len(d)*SR/sr))
            if len(d) < SAFE_LEN:
                d = np.pad(d, (0, SAFE_LEN - len(d)))
            d = d[:SAFE_LEN]
            x = torch.from_numpy(d).float().view(1,1,-1).to(device)
            with torch.no_grad():
                z, codes = mimi_encode(x, mimi)
            samples.append({
                'z': z.squeeze(0).cpu(),
                'codes': codes.squeeze(0).cpu(),
                'spk': spk_idx,
                'spk_name': spk,
            })
    return samples

print("Pre-encoding...")
train_data = encode_speakers(TRAIN_SPKS, 5)
val_data = encode_speakers(VAL_SPKS, 5)
test_data = encode_speakers(TEST_SPKS, 5)
print("Train:", len(train_data), "Val:", len(val_data), "Test:", len(test_data), flush=True)

# Classifiers
class SpeakerClassifier(nn.Module):
    def __init__(self, dim=512, n_spk=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_spk),
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=-1)
        return self.net(x)

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

mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()
optimizer = AdamW(splitter.parameters(), lr=1e-3, weight_decay=1e-5)
spk_clf_C = SpeakerClassifier(n_spk=len(TRAIN_SPKS)).to(device)
spk_clf_S = SpeakerClassifier(n_spk=len(TRAIN_SPKS)).to(device)
opt_clf = AdamW(list(spk_clf_C.parameters()) + list(spk_clf_S.parameters()), lr=1e-3)

def probe(splitter, data, n_spk):
    splitter.eval()
    C_list, S_list, y_list = [], [], []
    for s in data:
        z = s['z'].unsqueeze(0).to(device)
        codes = s['codes'].unsqueeze(0).to(device)
        with torch.no_grad():
            _, C, S, _ = splitter(z, codes)
        C_list.append(C.squeeze(0).mean(dim=1).cpu().numpy())
        S_list.append(S.squeeze(0).cpu().numpy())
        y_list.append(s['spk'])
    splitter.train()
    X_C = np.stack(C_list); X_S = np.stack(S_list); y = np.array(y_list)
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    n_classes = len(set(y))
    cv = min(3, n_classes) if n_classes >= 2 else 2
    try:
        scores_C = cross_val_score(clf, scaler.fit_transform(X_C), y, cv=cv)
        scores_S = cross_val_score(clf, scaler.fit_transform(X_S), y, cv=cv)
    except:
        return 0, 0
    return scores_C.mean()*100, scores_S.mean()*100

print()
print("Training", EPOCHS, "epochs (Transformer adapter)...", flush=True)
best_val_acc = 0

for epoch in range(EPOCHS):
    idxs = torch.randperm(len(train_data))
    total_recon, total_adv, total_spk = 0.0, 0.0, 0.0
    n_batches = 0

    for i in range(0, len(train_data), BATCH_SIZE):
        batch = [train_data[j] for j in idxs[i:i+BATCH_SIZE]]
        z_batch = torch.stack([s['z'] for s in batch]).to(device)
        code_batch = torch.stack([s['codes'] for s in batch]).to(device)
        spk_ids = torch.tensor([s['spk'] for s in batch]).to(device)

        z_vc, C, S, A = splitter(z_batch, code_batch)
        L_recon = mse(z_vc, z_batch)
        C_rev = grad_reverse(C.mean(dim=-1), alpha=1.0)
        L_adv_C = ce(spk_clf_C(C_rev), spk_ids)
        L_spk_S = ce(spk_clf_S(S), spk_ids)

        loss = L_recon + ADV_WEIGHT * L_adv_C + SPK_WEIGHT * L_spk_S
        optimizer.zero_grad()
        opt_clf.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
        optimizer.step()
        opt_clf.step()

        total_recon += L_recon.item()
        total_adv += L_adv_C.item()
        total_spk += L_spk_S.item()
        n_batches += 1

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        n = max(n_batches, 1)
        tr_C, tr_S = probe(splitter, train_data[:50], len(TRAIN_SPKS))
        val_C, val_S = probe(splitter, val_data, len(VAL_SPKS))
        test_C, test_S = probe(splitter, test_data, len(TEST_SPKS))
        tr_chance = 100/len(TRAIN_SPKS)
        val_chance = 100/len(VAL_SPKS)
        test_chance = 100/len(TEST_SPKS)

        print(f"  E{epoch:3d} | Recon={total_recon/n:.4f} Adv={total_adv/n:.3f} Spk={total_spk/n:.3f}", flush=True)
        print(f"    Train C={tr_C:.1f}% S={tr_S:.1f}% (chance={tr_chance:.1f}%)", flush=True)
        print(f"    Val   C={val_C:.1f}% S={val_S:.1f}% (chance={val_chance:.0f}%)", flush=True)
        print(f"    Test  C={test_C:.1f}% S={test_S:.1f}% (chance={test_chance:.0f}%)", flush=True)

        if val_S > best_val_acc:
            best_val_acc = val_S
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model_state_dict": splitter.state_dict()},
                       "checkpoints/mimi_transformer_adapter.pt")

print()
print("=" * 60)
print("FINAL RESULTS (Transformer Adapter)")
print("=" * 60)
splitter.eval()
tr_C, tr_S = probe(splitter, train_data[:50], len(TRAIN_SPKS))
val_C, val_S = probe(splitter, val_data, len(VAL_SPKS))
test_C, test_S = probe(splitter, test_data, len(TEST_SPKS))

print()
print("  Model                  C-acc  S-acc")
print(f"  Transformer (train)    {tr_C:>5.1f}% {tr_S:>5.1f}% (chance={100/len(TRAIN_SPKS):.1f}%)")
print(f"  Transformer (val)      {val_C:>5.1f}% {val_S:>5.1f}% (chance={100/len(VAL_SPKS):.0f}%)")
print(f"  Transformer (test)     {test_C:>5.1f}% {test_S:>5.1f}% (chance={100/len(TEST_SPKS):.0f}%)")
print()
print("Done!")
