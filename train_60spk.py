#!/usr/bin/env python3
"""
Mimi Splitter VC - Full training pipeline with 60 speakers.
- 40 train, 10 val, 10 holdout-test
- Adversarial training with gradient reversal
- WER evaluation with Whisper
"""

import sys, os, glob, time, warnings
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
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
EPOCHS = 200
ADV_WEIGHT = 0.5
SPK_WEIGHT = 0.5

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
print(f"Train: {len(TRAIN_SPKS)}, Val: {len(VAL_SPKS)}, Test: {len(TEST_SPKS)}")

device = torch.device('cpu')
print(f"Device: {device}", flush=True)

print("Loading Mimi...", flush=True)
mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi).to(device)
print(f"Trainable params: {sum(p.numel() for p in splitter.parameters() if p.requires_grad)}")

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
                'path': f,
            })
    return samples

print("Pre-encoding train...")
train_data = encode_speakers(TRAIN_SPKS, 5)
print("Pre-encoding val...")
val_data = encode_speakers(VAL_SPKS, 5)
print("Pre-encoding test...")
test_data = encode_speakers(TEST_SPKS, 5)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


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


def probe(splitter, data, n_spk, tag=""):
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
    X_C = np.stack(C_list)
    X_S = np.stack(S_list)
    y = np.array(y_list)
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    n_classes = len(set(y))
    cv = min(5, n_classes) if n_classes >= 2 else 2
    try:
        scores_C = cross_val_score(clf, scaler.fit_transform(X_C), y, cv=cv)
        scores_S = cross_val_score(clf, scaler.fit_transform(X_S), y, cv=cv)
    except:
        return 0, 0, 0
    chance = 100.0 / n_spk
    return scores_C.mean() * 100, scores_S.mean() * 100, chance


print()
print(f"Training {EPOCHS} epochs (Stage 3: adversarial)...", flush=True)
best_val_acc = 0

for epoch in range(EPOCHS):
    idxs = torch.randperm(len(train_data))
    total_recon = 0.0
    total_adv = 0.0
    total_spk = 0.0
    n_batches = 0

    for i in range(0, len(train_data), BATCH_SIZE):
        batch = [train_data[j] for j in idxs[i:i+BATCH_SIZE]]
        z_batch = torch.stack([s['z'] for s in batch]).to(device)
        code_batch = torch.stack([s['codes'] for s in batch]).to(device)
        spk_ids = torch.tensor([s['spk'] for s in batch]).to(device)

        z_vc, C, S, A = splitter(z_batch, code_batch)
        L_recon = mse(z_vc, z_batch)

        C_pool = C.mean(dim=-1)
        C_rev = grad_reverse(C_pool, alpha=1.0)
        pred_C = spk_clf_C(C_rev)
        L_adv_C = ce(pred_C, spk_ids)

        pred_S = spk_clf_S(S)
        L_spk_S = ce(pred_S, spk_ids)

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
        tr_C, tr_S, tr_chance = probe(splitter, train_data[:50], len(TRAIN_SPKS))
        val_C, val_S, val_chance = probe(splitter, val_data, len(VAL_SPKS))
        test_C, test_S, test_chance = probe(splitter, test_data, len(TEST_SPKS))

        print(f"  E{epoch:3d} | Recon={total_recon/n:.4f} Adv={total_adv/n:.3f} Spk={total_spk/n:.3f}", flush=True)
        print(f"    Train C={tr_C:.1f}% S={tr_S:.1f}% (chance={tr_chance:.0f}%)", flush=True)
        print(f"    Val   C={val_C:.1f}% S={val_S:.1f}% (chance={val_chance:.0f}%)", flush=True)
        print(f"    Test  C={test_C:.1f}% S={test_S:.1f}% (chance={test_chance:.0f}%)", flush=True)

        if val_S > best_val_acc:
            best_val_acc = val_S
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model_state_dict": splitter.state_dict()},
                       "checkpoints/mimi_splitter_v2_60spk.pt")


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

splitter.eval()
tr_C, tr_S, tr_ch = probe(splitter, train_data[:50], len(TRAIN_SPKS))
val_C, val_S, val_ch = probe(splitter, val_data, len(VAL_SPKS))
test_C, test_S, test_ch = probe(splitter, test_data, len(TEST_SPKS))

print()
print(f"{'':>8} {'C speaker':>12} {'S speaker':>12} {'Chance':>8}")
print(f"{'Train':>8} {tr_C:>11.1f}% {tr_S:>11.1f}% {tr_ch:>7.0f}%")
print(f"{'Val':>8} {val_C:>11.1f}% {val_S:>11.1f}% {val_ch:>7.0f}%")
print(f"{'Test':>8} {test_C:>11.1f}% {test_S:>11.1f}% {test_ch:>7.0f}%")


print()
print("--- WER Evaluation ---")
try:
    import whisper
    model_whisper = whisper.load_model("base")
    print("  Whisper loaded")
except Exception as e:
    print(f"  Whisper not available: {e}")
    model_whisper = None

if model_whisper is not None:
    src_sample = test_data[0]
    tgt_sample = test_data[-1]
    print(f"  Source: {src_sample['spk_name']}, Target: {tgt_sample['spk_name']}")

    z_src = src_sample['z'].unsqueeze(0).to(device)
    z_tgt = tgt_sample['z'].unsqueeze(0).to(device)
    codes_src = src_sample['codes'].unsqueeze(0).to(device)

    with torch.no_grad():
        mimi.set_num_codebooks(1)
        z_q0_src = mimi.decode_latent(codes_src[:, :1, :])
        mimi.set_num_codebooks(8)
        C_src = splitter.content_extractor(z_q0_src)
        S_tgt = splitter.speaker_encoder(z_tgt)
        n_ac = codes_src.shape[1] - 1
        mimi.set_num_codebooks(n_ac)
        z_ac = mimi.decode_latent(codes_src[:, 1:, :])
        mimi.set_num_codebooks(8)
        A_vc = splitter.acoustic_adapter(z_ac, S_tgt, C_src)
        z_vc = C_src + A_vc
        x_vc = mimi_decode_latent(mimi, z_vc)
        x_src_rt = mimi_decode_latent(mimi, z_src)

    sf.write("checkpoints/vc60_src_rt.wav", x_src_rt[0,0].cpu().numpy(), SR)
    sf.write("checkpoints/vc60_output.wav", x_vc[0,0].cpu().numpy(), SR)

    audio_np = x_vc[0,0].cpu().numpy()
    audio_np = audio_np / (np.abs(audio_np).max() + 1e-8)
    result = model_whisper.transcribe(audio_np, language="en")
    print(f"  VC transcript: \"{result['text']}\"")

    audio_src = x_src_rt[0,0].cpu().numpy()
    audio_src = audio_src / (np.abs(audio_src).max() + 1e-8)
    result_src = model_whisper.transcribe(audio_src, language="en")
    print(f"  Src transcript: \"{result_src['text']}\"")

print()
print("Done!")
