#!/usr/bin/env python3
"""Hard-FiLM v2: sweep β scale + refine scale to find jitter-centroid tradeoff."""
import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
import torch, torch.nn.functional as F
import numpy as np, soundfile as sf
from scipy import signal

SR=24000; STRIDE=1920; SAFE_LEN=48000
device=torch.device('cpu')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
mimi=load_mimi(device).to(device); mimi.eval()
splitter=MimiSplitterV2(mimi,n_content=1).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt",map_location='cpu')['model_state_dict'])
splitter.eval()

ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

# Load
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

# Extract components
with torch.no_grad():
    z_src,codes_src=mimi_encode(x_src,mimi)
    z_tgt,codes_tgt=mimi_encode(x_tgt,mimi)
    mimi.set_num_codebooks(1); z_q0=mimi.decode_latent(codes_src[:,:1,:]); mimi.set_num_codebooks(8)
    C=splitter.content_extractor(z_q0)
    S_tgt=splitter.speaker_encoder(z_tgt)
    n_ac=codes_src.shape[1]-1
    mimi.set_num_codebooks(n_ac); z_ac=mimi.decode_latent(codes_src[:,1:,:]); mimi.set_num_codebooks(8)
    
    # FiLM components
    scale=splitter.acoustic_adapter.scale(S_tgt)
    bias=splitter.acoustic_adapter.bias(S_tgt)
    z_mod = z_ac * (1+torch.tanh(scale.unsqueeze(-1))) + bias.unsqueeze(-1)
    
    # Content refine
    z_in = C + z_mod
    h = z_in.transpose(1,2)
    h = splitter.acoustic_adapter.norm(h).transpose(1,2)
    z_ref = splitter.acoustic_adapter.refine(h)
    
    # Baseline
    A_full = z_mod + z_ref
    z_full = C + A_full
    x_full = mimi_decode_latent(mimi, z_full)
    
    # β-only baseline (no refine)
    A_beta = z_mod  # ≈ β because γ≈-30
    z_beta = C + A_beta
    x_beta = mimi_decode_latent(mimi, z_beta)
    
    z_rt = z_src

def measure(a):
    a=a-np.mean(a)
    f,_,Z=signal.stft(a,fs=SR,nperseg=512,noverlap=384)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
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

# Source baseline
src_c,_,_,src_j=measure(d_src[:src_len])

# Full FiLM
full_c,full_vh,full_cr,full_j=measure(x_full[0,0].cpu().numpy()[:src_len])
# β-only
beta_c,beta_vh,beta_cr,beta_j=measure(x_beta[0,0].cpu().numpy()[:src_len])

print(f"  SOURCE:  Cent={src_c:.0f}Hz Jitter={src_j:.1f}%")
print(f"  β-only:  Cent={beta_c:.0f}Hz Jitter={beta_j:.1f}% Crest={beta_cr:.1f} VHigh={beta_vh:.1f}%")
print(f"  FULL:    Cent={full_c:.0f}Hz Jitter={full_j:.1f}% Crest={full_cr:.1f} VHigh={full_vh:.1f}%")
print()
print("=" * 90)
print("  SWEEP: refine scale α (0=β-only, 1=full FiLM)")
print("=" * 90)
print(f"  {'α':>5s} {'Cent':>6s} {'Jitter':>7s} {'Crest':>6s} {'VHigh':>6s} {'L2(vc,rt)':>11s} {'Combo':>6s} {'Notes':>20s}")
print("  " + "-" * 80)

best_combo=-999; best_alpha=0
results=[]

for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
    with torch.no_grad():
        A = z_mod + alpha * z_ref
        z_vc = C + A
        x_vc = mimi_decode_latent(mimi, z_vc)
    
    vc_np=x_vc[0,0].cpu().numpy()[:src_len]
    c,vh,cr,j=measure(vc_np)
    l2=torch.norm(z_vc - z_rt).item()
    
    # Combo: prefer high centroid + low jitter
    combo = (c/1500)*3 - (j/30)*2 - (l2/300)
    
    notes=""
    if c>1400: notes+="FULL_SHIFT "
    elif c>1300: notes+="GOOD_SHIFT "
    elif c>1180: notes+="PARTIAL "
    else: notes+="WEAK "
    if j<10: notes+="SILKY "
    elif j<20: notes+="SMOOTH "
    elif j<30: notes+="OK "
    else: notes+="JITTERY "
    
    print(f"  {alpha:4.2f} {c:5.0f}Hz {j:6.1f}% {cr:5.1f} {vh:5.1f}% {l2:10.1f} {combo:5.2f} {notes:<20s}")
    
    if combo>best_combo:
        best_combo=combo; best_alpha=alpha
        best_vc=vc_np; best_c=c; best_j=j; best_l2=l2

# Save best
sf.write('/Users/asill/Desktop/vc_refine_sweep_best.wav', best_vc, SR)
print()
print(f"  BEST α={best_alpha:.2f}: Cent={best_c:.0f}Hz Jitter={best_j:.1f}% L2={best_l2:.1f}")
print(f"  vs FULL: Cent delta={best_c-full_c:.0f}Hz Jitter delta={best_j-full_j:.1f}%")
print(f"  Saved: Desktop/vc_refine_sweep_best.wav")

# Also save a few key points
for alpha, label in [(0.0, "beta_only"), (best_alpha, "best"), (1.0, "full")]:
    with torch.no_grad():
        A = z_mod + alpha * z_ref
        z_vc = C + A
        x_vc = mimi_decode_latent(mimi, z_vc)
    sf.write(f'/Users/asill/Desktop/vc_refine_a{alpha:.2f}.wav', x_vc[0,0].cpu().numpy()[:src_len], SR)
    
print("  Also saved: Desktop/vc_refine_a0.00.wav (β-only), aX.XX.wav (best), a1.00.wav (full)")
print("Done!")
