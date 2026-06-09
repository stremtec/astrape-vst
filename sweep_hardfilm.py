#!/usr/bin/env python3
"""
Hard-FiLM Sweep: calibrate β injection to reduce decoder OOD.
Sweeps (g, λ): source acoustic gate × target beta scale.
Goal: maintain centroid >1300Hz while reducing OOD → less jitter.
"""
import sys, os
sys.path.insert(0, '/Users/asill/btrv5')

import torch, torch.nn.functional as F
import numpy as np, soundfile as sf
from scipy import signal
from collections import defaultdict

SR=24000; STRIDE=1920; SAFE_LEN=48000
device=torch.device('cpu')

from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
mimi=load_mimi(device).to(device); mimi.eval()
splitter=MimiSplitterV2(mimi,n_content=1).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt",map_location='cpu')['model_state_dict'])
splitter.eval()

ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

# Load audio
d_src,sr=sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr))
safe=(len(d_src)//STRIDE)*STRIDE; d_src=d_src[:safe]; src_len=len(d_src)
x_src=torch.from_numpy(d_src).float().view(1,1,-1).to(device)

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr))
safe=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)

# Train S norms for calibration
print("Computing train S norm statistics...")
TRAIN_SPKS=['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256']
train_S_norms=[]
for spk in TRAIN_SPKS[:20]:
    import glob
    files=sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))
    for f in files[:2]:
        d,sr=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
        safe=(len(d)//STRIDE)*STRIDE; d=d[:safe]
        x=torch.from_numpy(d).float().view(1,1,-1).to(device)
        with torch.no_grad():
            z,_=mimi_encode(x,mimi)
            S=splitter.speaker_encoder(z)
        train_S_norms.append(S.norm().item())
train_median_S_norm=np.median(train_S_norms)
print(f"  Train S norm median: {train_median_S_norm:.1f}")

# Extract all components once
with torch.no_grad():
    z_src,codes_src=mimi_encode(x_src,mimi)
    z_tgt,codes_tgt=mimi_encode(x_tgt,mimi)
    
    mimi.set_num_codebooks(1)
    z_q0=mimi.decode_latent(codes_src[:,:1,:]); mimi.set_num_codebooks(8)
    C=splitter.content_extractor(z_q0)
    
    S_tgt=splitter.speaker_encoder(z_tgt)
    S_tgt_norm=S_tgt.norm().item()
    
    n_ac=codes_src.shape[1]-1
    mimi.set_num_codebooks(n_ac)
    z_ac=mimi.decode_latent(codes_src[:,1:,:]); mimi.set_num_codebooks(8)
    
    # Source acoustic residual (time-centered to reduce speaker leak)
    z_ac_res = z_ac - z_ac.mean(dim=-1, keepdim=True)
    
    # Beta from target speaker
    beta_raw = splitter.acoustic_adapter.bias(S_tgt)
    
    # Mimi RT latent for reference
    z_rt = z_src
    
    # Reference metrics
    x_rt=mimi_decode_latent(mimi,z_rt)
    x_base,_=mimi_encode(x_src,mimi) # dummy to get original VC
    A_base=splitter.acoustic_adapter(z_ac,S_tgt,C)
    z_base=C+A_base; x_base=mimi_decode_latent(mimi,z_base)

print(f"  S_tgt norm: {S_tgt_norm:.1f} (train median: {train_median_S_norm:.1f}, ratio: {S_tgt_norm/train_median_S_norm:.2f})")
print(f"  beta_raw norm: {beta_raw.norm():.1f}")

# ── Metrics ──────────────────────────────────────────────────────────
def measure_audio(a):
    a=a-np.mean(a)
    f,_,Z=signal.stft(a,fs=SR,nperseg=512,noverlap=384)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    # Jitter
    fl,hp=int(SR*0.04),int(SR*0.01); fs=[]
    for i in range(0,len(a)-fl,hp):
        fr=a[i:i+fl]
        if np.sqrt(np.mean(fr**2))<0.001: fs.append(0); continue
        corr=np.correlate(fr,fr,mode='full'); corr=corr[len(corr)//2:]; corr=corr/(corr[0]+1e-8)
        pks=signal.find_peaks(corr,distance=10)[0]
        if len(pks)==0: fs.append(0); continue
        f0=SR/pks[0]; fs.append(f0 if 50<f0<400 else 0)
    fs=np.array(fs); v=fs>0
    j=np.mean(np.abs(np.diff(fs[v])))/np.mean(fs[v])*100 if v.sum()>3 else 0
    return np.mean(c),vh,cr,j

# Baseline
src_c,src_vh,src_cr,src_j=measure_audio(d_src[:src_len])
x_base_np=x_base[0,0].cpu().numpy()[:src_len]
base_c,base_vh,base_cr,base_j=measure_audio(x_base_np)

print()
print(f"  SOURCE: Cent={src_c:.0f}Hz Jitter={src_j:.1f}% VHigh={src_vh:.1f}% Crest={src_cr:.1f}")
print(f"  BASE FiLM: Cent={base_c:.0f}Hz Jitter={base_j:.1f}% VHigh={base_vh:.1f}% Crest={base_cr:.1f}")
print(f"  z_base vs z_rt L2: {torch.norm(z_base-z_rt):.1f}")

# ── Sweep ─────────────────────────────────────────────────────────────
print()
print("=" * 100)
print("  HARD-FiLM SWEEP: g (source gate) × λ (target scale)")
print("=" * 100)
print(f"  {'g':>5s} {'λ':>5s} {'β_norm':>7s} {'Calib':>6s} {'Cent':>6s} {'Jitter':>7s} {'VHigh':>6s} {'Crest':>6s} {'L2(z_vc,z_rt)':>13s} {'cos(z_vc,z_rt)':>13s} {'OOD↓':>6s} {'Notes':>20s}")
print("  " + "-" * 120)

results = []
best = {'centroid':0, 'jitter':999, 'l2':999, 'combo':-999}

# Sweep modes
modes = [
    # (name, g_range, λ_range, norm_calib)
    ("raw_scale", [0.0], [1.0], False),
    ("norm_calib", [0.0], [1.0, 0.8, 0.6, 0.4], True),
    ("g_sweep", [0.0, 0.03, 0.05, 0.10], [1.0, 0.8, 0.6], True),
    ("g_only", [0.0, 0.03, 0.05, 0.10, 0.15, 0.20], [1.0], True),
]

for mode_name, g_vals, lam_vals, norm_calib in modes:
    for g in g_vals:
        for lam in lam_vals:
            with torch.no_grad():
                # Beta calibration
                if norm_calib:
                    S_cal = S_tgt / (S_tgt_norm + 1e-8) * train_median_S_norm
                    beta = splitter.acoustic_adapter.bias(S_cal)
                else:
                    beta = beta_raw
                
                beta = beta * lam
                
                # Build z_vc
                z_mod = g * z_ac_res + beta.unsqueeze(-1).expand(-1,-1,z_ac.shape[-1])
                z_vc = C + z_mod
                x_vc = mimi_decode_latent(mimi, z_vc)
            
            vc_np = x_vc[0,0].cpu().numpy()[:src_len]
            c,vh,cr,j = measure_audio(vc_np)
            l2_dist = torch.norm(z_vc - z_rt).item()
            cos_dist = F.cosine_similarity(z_vc.reshape(-1), z_rt.reshape(-1), dim=0).item()
            
            b_norm = beta.norm().item()
            calib_str = "ON" if norm_calib else "OFF"
            
            # Score
            ood_score = l2_dist  # lower = better
            
            notes = ""
            if c > 1300: notes += "SHIFT★ "
            if j < 25: notes += "J_OK "
            if cos_dist > 0.85: notes += "IN_DIST "
            if ood_score < 200: notes += "OOD_OK "
            
            # Combo score: centroid high, jitter low, L2 low
            combo = (c/1500) * 2 - (j/40) - (l2_dist/300)
            
            print(f"  {g:5.2f} {lam:5.1f} {b_norm:6.1f} {calib_str:>6s} {c:5.0f}Hz {j:6.1f}% {vh:5.1f}% {cr:5.1f} {l2_dist:12.1f} {cos_dist:12.4f} {ood_score:5.0f} {notes:<20s}")
            
            if combo > best['combo']:
                best = {'centroid':c, 'jitter':j, 'l2':l2_dist, 'combo':combo,
                        'g':g, 'lam':lam, 'norm_calib':norm_calib, 'vh':vh, 'cr':cr}

# ── Best result ──────────────────────────────────────────────────────
print()
print("=" * 100)
print(f"  BEST: g={best['g']:.2f} λ={best['lam']:.1f} norm_calib={best['norm_calib']}")
print(f"  Centroid={best['centroid']:.0f}Hz Jitter={best['jitter']:.1f}% VHigh={best['vh']:.1f}% Crest={best['cr']:.1f}")
print(f"  L2(z_vc,z_rt)={best['l2']:.1f}")

# Compare to base FiLM
print()
print(f"  BASE → BEST comparison:")
print(f"    Centroid: {base_c:.0f} → {best['centroid']:.0f}Hz")
print(f"    Jitter:   {base_j:.1f} → {best['jitter']:.1f}%")
print(f"    L2 OOD:   {torch.norm(z_base-z_rt):.1f} → {best['l2']:.1f}")

# Save best
with torch.no_grad():
    if best['norm_calib']:
        S_cal = S_tgt / (S_tgt_norm + 1e-8) * train_median_S_norm
        beta_best = splitter.acoustic_adapter.bias(S_cal) * best['lam']
    else:
        beta_best = beta_raw * best['lam']
    z_mod_best = best['g'] * z_ac_res + beta_best.unsqueeze(-1).expand(-1,-1,z_ac.shape[-1])
    z_vc_best = C + z_mod_best
    x_best = mimi_decode_latent(mimi, z_vc_best)

sf.write('/Users/asill/Desktop/vc_hardfilm_best.wav', x_best[0,0].cpu().numpy()[:src_len], SR)
print(f"  Saved: Desktop/vc_hardfilm_best.wav")
print()
print("Done!")
