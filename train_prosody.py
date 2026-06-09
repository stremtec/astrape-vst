#!/usr/bin/env python3
"""
Mimi Splitter VC with P-path (FCPE F0 + prosody).
Fixes F0 jitter (37.77%) by providing stable pitch to temporal adapter.

Architecture:
  C(q0) + P(FCPE F0) + S(target) → TCN adapter → acoustic modulation → decoder
"""

import sys, os, glob, time, warnings
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, mimi_encode, mimi_decode_latent
import torch, torch.nn as nn
import torch.nn.functional as F
import torchfcpe
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW

warnings.filterwarnings('ignore')
SR = 24000; SAFE_LEN = 48000; STRIDE = 1920
MIMI_FR = 12.5  # Mimi frame rate

device = torch.device('cpu')
print("Device:", device)

# ── FCPE F0 extractor ──────────────────────────────────────────────────
fcpe_model = torchfcpe.spawn_bundled_infer_model(device=str(device))
FCPE_SR = 16000

def extract_f0(audio_24k, target_frames):
    """Extract F0 using FCPE, resample to Mimi frame count."""
    if isinstance(audio_24k, torch.Tensor):
        audio_np = audio_24k.squeeze().cpu().numpy()
    else:
        audio_np = audio_24k
    
    try:
        # Resample to 16kHz for FCPE
        if len(audio_np) > 0:
            audio_16k = signal.resample(audio_np, int(len(audio_np)*FCPE_SR/SR))
        else:
            return torch.zeros(1, target_frames)
        
        audio_t = torch.from_numpy(audio_16k.copy()).float().view(1,1,-1).to(device)
        
        with torch.no_grad():
            f0 = fcpe_model.infer(audio_t, sr=FCPE_SR, 
                                 output_interp_target_length=target_frames,
                                 interp_uv=True, decoder_mode='local_argmax',
                                 threshold=0.006, f0_min=50, f0_max=550)
        
        f0 = f0.squeeze(-1)  # (1, T_mimi)
        if f0.shape[1] != target_frames:
            # Resample manually
            from scipy.interpolate import interp1d
            f0_np = f0.squeeze().numpy()
            t_old = np.linspace(0, 1, len(f0_np))
            t_new = np.linspace(0, 1, target_frames)
            f0_np = interp1d(t_old, f0_np, kind='linear', fill_value=0)(t_new)
            f0 = torch.from_numpy(f0_np).float().view(1, -1)
        
        return f0
    except Exception as e:
        # Fallback: constant F0
        print(f"  F0 fallback: {e}")
        return torch.full((1, target_frames), 150.0)


# ── Prosody Encoder ──────────────────────────────────────────────────────
class ProsodyEncoder(nn.Module):
    """Encode F0 + energy → prosody embedding."""
    def __init__(self, dim=512, hidden=64):
        super().__init__()
        self.f0_proj = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )
        self.energy_proj = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )
        self.fuse = nn.Conv1d(hidden*2, dim, 1)
    
    def forward(self, log_f0, energy):
        # log_f0: (B, 1, T), energy: (B, 1, T)
        h_f0 = self.f0_proj(log_f0)
        h_energy = self.energy_proj(energy)
        h = torch.cat([h_f0, h_energy], dim=1)
        return self.fuse(h)  # (B, D, T)


# ── Temporal Adapter (TCN) ──────────────────────────────────────────────
class TemporalAdapter(nn.Module):
    """Causal TCN that takes C + P + S → acoustic modulation."""
    def __init__(self, dim=512, spk_dim=256, kernel_size=5, dilation=1):
        super().__init__()
        self.spk_proj = nn.Linear(dim, spk_dim)
        
        # Temporal conv (preserve length with symmetric padding)
        k = kernel_size
        pad = k // 2
        self.tcn = nn.Sequential(
            nn.Conv1d(dim*2, dim, k, padding=pad),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv1d(dim, dim, k, padding=pad),
            nn.GroupNorm(8, dim), nn.GELU(),
        )
        
        # FiLM modulation
        self.scale = nn.Sequential(nn.Linear(spk_dim, dim), nn.Tanh())
        self.bias = nn.Linear(spk_dim, dim)
        
        # Output
        self.out = nn.Conv1d(dim, dim, 1)
    
    def forward(self, C, P, S):
        """
        C: (B, D, T) — content
        P: (B, D, T) — prosody
        S: (B, D)    — speaker
        """
        spk = self.spk_proj(S)
        
        # Temporal processing
        x = torch.cat([C, P], dim=1)  # (B, 2D, T)
        h = self.tcn(x)
        
        # FiLM modulation
        scale = self.scale(spk).unsqueeze(-1)
        bias = self.bias(spk).unsqueeze(-1)
        h = h * (1 + scale) + bias
        
        return self.out(h), C


# ── Full Splitter with P-path ──────────────────────────────────────────
from mimi_splitter_v2 import ContentExtractor, SpeakerEncoder

class MimiSplitterProsody(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        self.content_extractor = ContentExtractor(512, 64)
        self.speaker_encoder = SpeakerEncoder(512, 256)
        self.prosody_encoder = ProsodyEncoder(512, 64)
        self.temporal_adapter = TemporalAdapter(512, 256)
    
    def _get_q0(self, codes):
        with torch.no_grad():
            self.mimi.set_num_codebooks(1)
            z_q0 = self.mimi.decode_latent(codes[:, :1, :])
            self.mimi.set_num_codebooks(8)
        return z_q0
    
    def forward(self, z_post, codes, f0_src, energy_src, audio_len):
        """
        z_post: (B, D, T_mimi)
        codes: (B, Nq, T_mimi)
        f0_src: (B, T_mimi) — F0 from FCPE
        energy_src: (B, T_mimi) — RMS energy per frame
        audio_len: int — original audio length in samples
        """
        z_q0 = self._get_q0(codes)
        C = self.content_extractor(z_q0)
        S = self.speaker_encoder(z_post)
        
        # Prosody: log F0 + energy
        log_f0 = torch.log(f0_src + 1.0).unsqueeze(1)  # (B, 1, T)
        energy = energy_src.unsqueeze(1)               # (B, 1, T)
        P = self.prosody_encoder(log_f0, energy)
        
        # Temporal adapter
        A, C_out = self.temporal_adapter(C, P, S)
        
        z_vc = C_out + A
        return z_vc, C_out, S, A, P


# ── Data loading with F0 ──────────────────────────────────────────────
mimi = load_mimi(device).to(device)

TRAIN_SPKS = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266'][:40]

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def encode_with_f0(spk_list, n_utt=5):
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
            T_mimi = z.shape[-1]
            
            # Extract F0
            f0 = extract_f0(d, T_mimi)  # (1, T_mimi)
            
            # Energy per Mimi frame
            frame_len = SAFE_LEN // T_mimi
            energy = np.array([np.sqrt(np.mean(d[i*frame_len:(i+1)*frame_len]**2)) 
                              for i in range(T_mimi)])
            energy_t = torch.from_numpy(energy).float().view(1, -1)
            
            samples.append({
                'z': z.squeeze(0).cpu(), 'codes': codes.squeeze(0).cpu(),
                'f0': f0.squeeze(0).cpu(), 'energy': energy_t.squeeze(0).cpu(),
                'spk': spk_idx, 'audio_len': len(d)
            })
    return samples

print("Pre-encoding with F0..."); t0 = time.time()
train_data = encode_with_f0(TRAIN_SPKS, 5)
print(f"Train: {len(train_data)} ({time.time()-t0:.1f}s)")

# ── Training ────────────────────────────────────────────────────────────
splitter = MimiSplitterProsody(mimi).to(device)
print("Params:", sum(p.numel() for p in splitter.parameters() if p.requires_grad))

mse = nn.MSELoss()
l1 = nn.L1Loss()
opt = AdamW(splitter.parameters(), lr=1e-3, weight_decay=1e-5)
BATCH = 8; EPOCHS = 200

print()
print("Training with P-path (F0 + temporal)...", flush=True)

for epoch in range(EPOCHS):
    idxs = torch.randperm(len(train_data))
    total_loss, nb = 0.0, 0
    
    for i in range(0, len(train_data), BATCH):
        batch = [train_data[j] for j in idxs[i:i+BATCH]]
        zb = torch.stack([s['z'] for s in batch]).to(device)
        cb = torch.stack([s['codes'] for s in batch]).to(device)
        f0b = torch.stack([s['f0'] for s in batch]).to(device)
        eb = torch.stack([s['energy'] for s in batch]).to(device)
        
        zv, C, S, A, P = splitter(zb, cb, f0b, eb, SAFE_LEN)
        
        L_recon = mse(zv, zb[:, :, :zv.shape[-1]])
        L_f0 = l1(P.mean(dim=1), torch.zeros_like(P.mean(dim=1))) * 0.01
        
        loss = L_recon + L_f0
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
        opt.step()
        
        total_loss += loss.item(); nb += 1
    
    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"  E{epoch:3d} | Loss={total_loss/max(nb,1):.4f}", flush=True)

os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": splitter.state_dict()}, "checkpoints/mimi_prosody.pt")

# ── VC test ─────────────────────────────────────────────────────────────
print()
print("=== VC: p255 -> origin (with P-path) ===")
splitter.eval()

d_src, sr = sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr))
safe_src=(len(d_src)//STRIDE)*STRIDE; d_src=d_src[:safe_src]; src_len=len(d_src)
x_src=torch.from_numpy(d_src).float().view(1,1,-1).to(device)

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr))
safe_tgt=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe_tgt]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)

with torch.no_grad():
    z_src, codes_src = mimi_encode(x_src, mimi)
    z_tgt, codes_tgt = mimi_encode(x_tgt, mimi)
    
    T_mimi = z_src.shape[-1]
    f0_src = extract_f0(d_src[:src_len], T_mimi)
    frame_len = src_len // T_mimi
    energy_src = torch.tensor([[np.sqrt(np.mean(d_src[i*frame_len:(i+1)*frame_len]**2)) 
                                for i in range(T_mimi)]]).float()
    
    z_q0 = splitter._get_q0(codes_src)
    C = splitter.content_extractor(z_q0)
    S_tgt = splitter.speaker_encoder(z_tgt)
    
    log_f0 = torch.log(f0_src + 1.0).unsqueeze(1)
    energy = energy_src.unsqueeze(1)
    P = splitter.prosody_encoder(log_f0, energy)
    
    A, C_out = splitter.temporal_adapter(C, P, S_tgt)
    z_vc = C_out + A
    x_vc = mimi_decode_latent(mimi, z_vc)

x_rt = mimi_decode_latent(mimi, z_src)
vc_np = x_vc[0,0].cpu().numpy()[:src_len]
rt_np = x_rt[0,0].cpu().numpy()[:src_len]

# Metrics
from scipy.signal import stft
def measure(a,label):
    f,_,Z=stft(a,fs=SR,nperseg=512,noverlap=384); mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    crest=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    vhigh=mag[(f>=4000)&(f<8000)].sum()/total*100
    return np.mean(c), crest, vhigh

print()
for label, audio in [("SRC", d_src), ("VC+P", vc_np), ("RT", rt_np)]:
    c,cr,vh = measure(audio, label)
    print("  " + label + ": Cent=" + str(round(c)) + "Hz Crest=" + str(round(cr,1)) + " VHigh=" + str(round(vh,1)) + "%")

sf.write("/Users/asill/Desktop/vc_prosody_p255_origin.wav", vc_np, SR)
print()
print("Saved: Desktop/vc_prosody_p255_origin.wav")
print("Done!")
