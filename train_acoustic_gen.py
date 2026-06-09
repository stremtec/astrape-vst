#!/usr/bin/env python3
"""
AcousticPredictor: q0 content + target speaker → acoustic residual generation.
Fixes the quality bottleneck while keeping content speaker-clean (n_content=1).

Stage 1: Same-speaker — predict q1-q7 from q0 + speaker
Stage 2: Cross-speaker VC with acoustic prediction
"""

import sys, os, glob, time, warnings
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, mimi_encode, mimi_decode_latent
import torch, torch.nn as nn
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW

warnings.filterwarnings('ignore')
SR = 24000; SAFE_LEN = 48000; STRIDE = 1920

device = torch.device('cpu')
print("Device:", device)

mimi = load_mimi(device).to(device)

# ── Acoustic Predictor ──────────────────────────────────────────────────
class AcousticPredictor(nn.Module):
    """Predict q1-q7 acoustic latent from q0 content + speaker embedding."""
    def __init__(self, dim=512, spk_dim=256, num_layers=2, dropout=0.15):
        super().__init__()
        self.dim = dim
        self.spk_proj = nn.Linear(dim, spk_dim)
        
        # Content processing
        self.content_proj = nn.Conv1d(dim, dim, 1)
        
        # Speaker FiLM
        self.film_scale = nn.Sequential(nn.Linear(spk_dim, dim), nn.Tanh())
        self.film_bias = nn.Linear(spk_dim, dim)
        
        # Convolutional blocks for temporal modeling
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(dim, dim, 5, padding=2, groups=dim),  # depthwise
                nn.Conv1d(dim, dim, 1),  # pointwise
                nn.GroupNorm(8, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        
        self.out_proj = nn.Conv1d(dim, dim, 1)
    
    def forward(self, C, S):
        """
        C: (B, D, T) — q0 content (speaker-clean)
        S: (B, D)   — target speaker embedding
        Returns: A_pred (B, D, T) — predicted acoustic residual
        """
        spk = self.spk_proj(S)  # (B, spk_dim)
        scale = self.film_scale(spk).unsqueeze(-1)  # (B, D, 1)
        bias = self.film_bias(spk).unsqueeze(-1)    # (B, D, 1)
        
        h = self.content_proj(C)
        h = h * (1 + scale) + bias  # speaker conditioning
        
        for block in self.conv_blocks:
            residual = h
            h = block(h)
            h = h + residual  # residual connection
        
        return self.out_proj(h)


# ── Splitter with AcousticPredictor ─────────────────────────────────────
class MimiSplitterGen(nn.Module):
    def __init__(self, mimi, dim=512, spk_dim=256):
        super().__init__()
        self.mimi = mimi
        from mimi_splitter_v2 import ContentExtractor, SpeakerEncoder
        self.content_extractor = ContentExtractor(dim, bottleneck=64)
        self.speaker_encoder = SpeakerEncoder(dim, spk_dim)
        self.acoustic_predictor = AcousticPredictor(dim, spk_dim, num_layers=2)
    
    def _get_q0(self, codes):
        with torch.no_grad():
            self.mimi.set_num_codebooks(1)
            z_q0 = self.mimi.decode_latent(codes[:, :1, :])
            self.mimi.set_num_codebooks(8)
        return z_q0
    
    def _get_acoustic_gt(self, codes):
        with torch.no_grad():
            n_ac = codes.shape[1] - 1
            self.mimi.set_num_codebooks(n_ac)
            z_ac = self.mimi.decode_latent(codes[:, 1:, :])
            self.mimi.set_num_codebooks(8)
        return z_ac
    
    def forward(self, z_post, codes):
        z_q0 = self._get_q0(codes)
        C = self.content_extractor(z_q0)
        S = self.speaker_encoder(z_post)
        A_pred = self.acoustic_predictor(C, S)
        z_vc = C + A_pred
        
        # Also get ground truth acoustic for loss
        z_ac_gt = self._get_acoustic_gt(codes)
        return z_vc, C, S, A_pred, z_ac_gt


# ── Training ─────────────────────────────────────────────────────────────
TRAIN_SPKS = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266'][:40]
VAL_SPKS = ['p267','p268','p269','p270','p271','p272','p273','p274','p275','p276']

splitter = MimiSplitterGen(mimi).to(device)
print("Params:", sum(p.numel() for p in splitter.parameters() if p.requires_grad))

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def encode_speakers(spk_list, n_utt=5):
    samples = []
    for spk_idx, spk in enumerate(spk_list):
        files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:n_utt]
        for f in files:
            d, sr = sf.read(f)
            if d.ndim > 1: d = d.mean(axis=1)
            if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
            if len(d) < SAFE_LEN: d = np.pad(d, (0, SAFE_LEN - len(d)))
            d = d[:SAFE_LEN]
            x = torch.from_numpy(d).float().view(1,1,-1).to(device)
            with torch.no_grad():
                z, codes = mimi_encode(x, mimi)
            samples.append({'z':z.squeeze(0).cpu(),'codes':codes.squeeze(0).cpu(),'spk':spk_idx})
    return samples

print("Pre-encoding..."); t0=time.time()
train_data = encode_speakers(TRAIN_SPKS, 5)
val_data = encode_speakers(VAL_SPKS, 5)
print(f"Train:{len(train_data)} Val:{len(val_data)} ({time.time()-t0:.1f}s)")

mse = nn.MSELoss()
opt = AdamW(splitter.parameters(), lr=1e-3, weight_decay=1e-5)
BATCH = 8; EPOCHS = 200

print()
print("Stage 1: Same-speaker acoustic prediction...", flush=True)

for epoch in range(EPOCHS):
    idxs = torch.randperm(len(train_data))
    total_recon, total_ac = 0.0, 0.0
    nb = 0
    
    for i in range(0, len(train_data), BATCH):
        batch = [train_data[j] for j in idxs[i:i+BATCH]]
        zb = torch.stack([s['z'] for s in batch]).to(device)
        cb = torch.stack([s['codes'] for s in batch]).to(device)
        
        zv, C, S, A_pred, z_ac_gt = splitter(zb, cb)
        
        L_recon = mse(zv, zb)           # full latent reconstruction
        L_acoustic = mse(A_pred, z_ac_gt)  # acoustic residual prediction
        
        loss = L_recon + 2.0 * L_acoustic  # weight acoustic prediction higher
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
        opt.step()
        
        total_recon += L_recon.item()
        total_ac += L_acoustic.item()
        nb += 1
    
    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        n = max(nb, 1)
        print(f"  E{epoch:3d} | Recon={total_recon/n:.4f} Acoustic={total_ac/n:.4f}", flush=True)

# Save
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": splitter.state_dict()}, "checkpoints/mimi_acoustic_gen.pt")

# ── Stage 2: VC test ───────────────────────────────────────────────────
print()
print("=== VC: p255 -> origin (Acoustic Generator) ===")
splitter.eval()

d_src, sr = sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim > 1: d_src = d_src.mean(axis=1)
if sr != SR: d_src = signal.resample(d_src, int(len(d_src)*SR/sr))
safe_src = (len(d_src)//STRIDE)*STRIDE; d_src = d_src[:safe_src]
x_src = torch.from_numpy(d_src).float().view(1,1,-1).to(device)
with torch.no_grad():
    z_src, codes_src = mimi_encode(x_src, mimi)
    z_q0_src = splitter._get_q0(codes_src)
    C_src = splitter.content_extractor(z_q0_src)

d_tgt, sr = sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim > 1: d_tgt = d_tgt.mean(axis=1)
if sr != SR: d_tgt = signal.resample(d_tgt, int(len(d_tgt)*SR/sr))
safe_tgt = (len(d_tgt)//STRIDE)*STRIDE; d_tgt = d_tgt[:safe_tgt]
x_tgt = torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)
with torch.no_grad():
    z_tgt, codes_tgt = mimi_encode(x_tgt, mimi)
    S_tgt = splitter.speaker_encoder(z_tgt)

# Generate acoustic from content + target speaker
with torch.no_grad():
    A_pred = splitter.acoustic_predictor(C_src, S_tgt)
    z_vc = C_src + A_pred
    x_vc = mimi_decode_latent(mimi, z_vc)

x_rt = mimi_decode_latent(mimi, z_src)
vc_np = x_vc[0,0].cpu().numpy()[:len(d_src)]
rt_np = x_rt[0,0].cpu().numpy()[:len(d_src)]

# Spectral
from scipy.signal import stft
def ana(a, label):
    f,_,Z = stft(a, fs=SR, nperseg=512, noverlap=256)
    mag = np.abs(Z); total = mag.sum() + 1e-8
    c = np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2], axis=0)
    c /= (mag[:len(f)//2].sum(axis=0) + 1e-8)
    vhigh = mag[(f>=4000)&(f<8000)].sum()/total*100
    low = mag[(f>=0)&(f<300)].sum()/total*100
    mid = mag[(f>=300)&(f<2000)].sum()/total*100
    print(f"  {label:15s} RMS={np.sqrt(np.mean(a**2)):.3f} Cent={np.mean(c):.0f}Hz "
          f"Low={low:.1f}% Mid={mid:.1f}% VHigh={vhigh:.1f}%")

print()
ana(d_src, "Source (p255)")
ana(d_tgt, "Target (origin)")
ana(vc_np, "VC acoustic gen")
ana(rt_np, "Mimi RT")

sf.write("checkpoints/vc_acgen_p255_origin.wav", vc_np, SR)
sf.write("/Users/asill/Desktop/vc_acgen_p255_origin.wav", vc_np, SR)
sf.write("/Users/asill/Desktop/vc_acgen_src.wav", d_src, SR)
sf.write("/Users/asill/Desktop/vc_acgen_tgt.wav", d_tgt, SR)
print()
print("Saved: Desktop/vc_acgen_*.wav")
print("Done!")
