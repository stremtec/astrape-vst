#!/usr/bin/env python3
"""
Minimal fix: temporal smoothing on FiLM adapter output.
Addresses jitter without breaking speaker transfer.
"""
import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, soundfile as sf, numpy as np
from scipy import signal, ndimage

device = torch.device('cpu')
SR = 24000; STRIDE = 1920

mimi = load_mimi(device).to(device)

# Load the successful FiLM model (60-spk, centroid=1429Hz)
splitter = MimiSplitterV2(mimi).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location='cpu')['model_state_dict'])
splitter.eval()
print("Loaded: 60-spk FiLM model")

# Load audio
ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

d_src, sr = sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr))
safe=(len(d_src)//STRIDE)*STRIDE; d_src=d_src[:safe]; src_len=len(d_src)
x_src=torch.from_numpy(d_src).float().view(1,1,-1).to(device)

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr))
safe=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)

with torch.no_grad():
    z_src, codes_src = mimi_encode(x_src, mimi)
    z_tgt, codes_tgt = mimi_encode(x_tgt, mimi)
    
    # Get q0 content
    mimi.set_num_codebooks(1)
    z_q0 = mimi.decode_latent(codes_src[:,:1,:])
    mimi.set_num_codebooks(8)
    C = splitter.content_extractor(z_q0)
    
    # Get speaker from target
    S_tgt = splitter.speaker_encoder(z_tgt)
    
    # Get acoustic from source
    n_ac = codes_src.shape[1] - 1
    mimi.set_num_codebooks(n_ac)
    z_ac = mimi.decode_latent(codes_src[:,1:,:])
    mimi.set_num_codebooks(8)
    
    # FiLM adapter
    A = splitter.acoustic_adapter(z_ac, S_tgt, C)
    
    # ── Temporal smoothing ──────────────────────────────────────────
    # Apply per-channel temporal smoothing to reduce jitter
    # Kernel size: 3 frames at 12.5Hz = 240ms → smooth fast variations
    A_np = A.squeeze(0).cpu().numpy()  # (D, T)
    
    # Test different smoothing strengths
    results = {}
    for sigma in [0.0, 0.5, 1.0, 1.5, 2.0]:
        if sigma == 0:
            A_smooth = A_np
        else:
            A_smooth = ndimage.gaussian_filter1d(A_np, sigma=sigma, axis=1)
        
        z_vc = C.squeeze(0).cpu().numpy() + A_smooth
        z_vc_t = torch.from_numpy(z_vc).float().view(1, 512, -1).to(device)
        x_vc = mimi_decode_latent(mimi, z_vc_t)
        results[sigma] = x_vc[0,0].cpu().numpy()[:src_len]
    
    # No smoothing baseline
    z_vc_base = C + A
    x_vc_base = mimi_decode_latent(mimi, z_vc_base)

# ── Analyze ───────────────────────────────────────────────────────────
from scipy.signal import stft
def measure(a, label):
    f,_,Z=stft(a,fs=SR,nperseg=512,noverlap=384); mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    crest=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    vhigh=mag[(f>=4000)&(f<8000)].sum()/total*100
    return np.mean(c), crest, vhigh

# Also compute jitter
def jitter_metric(a, label):
    a = a - np.mean(a)
    flen = int(SR*0.04); hop = int(SR*0.01)
    f0s = []
    for i in range(0, len(a)-flen, hop):
        frame = a[i:i+flen]
        if np.sqrt(np.mean(frame**2)) < 0.001: f0s.append(0); continue
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]; corr=corr/(corr[0]+1e-8)
        peaks = signal.find_peaks(corr, distance=10)[0]
        if len(peaks)==0: f0s.append(0); continue
        f0 = SR/peaks[0]
        f0s.append(f0 if 50<f0<400 else 0)
    f0s=np.array(f0s); v=f0s>0
    if v.sum()<3: return 0
    jitter = np.mean(np.abs(np.diff(f0s[v])))/np.mean(f0s[v])*100
    return jitter

print()
print("=" * 70)
print("TEMPORAL SMOOTHING RESULTS")
print("=" * 70)
print("  sigma   Centroid   Crest   VHigh   Jitter")
print("  " + "-" * 50)

x_base_np = x_vc_base[0,0].cpu().numpy()[:src_len]
c,cr,vh = measure(x_base_np, "base")
j = jitter_metric(x_base_np, "base")
print("  base     " + str(round(c)) + "Hz     " + str(round(cr,1)) + "     " + str(round(vh,1)) + "%    " + str(round(j,1)) + "%")

best_sigma = 0
best_jitter = j
for sigma in [0.5, 1.0, 1.5, 2.0]:
    a = results[sigma]
    c,cr,vh = measure(a, "s"+str(sigma))
    j_new = jitter_metric(a, "s"+str(sigma))
    marker = ""
    if j_new < best_jitter:
        best_jitter = j_new
        best_sigma = sigma
        marker = " ★"
    print("  s=" + str(round(sigma,1)) + "    " + str(round(c)) + "Hz     " + str(round(cr,1)) + "     " + str(round(vh,1)) + "%    " + str(round(j_new,1)) + "%" + marker)

# Save best
sf.write('/Users/asill/Desktop/vc_smooth_best.wav', results[best_sigma][:src_len], SR)
sf.write('/Users/asill/Desktop/vc_smooth_base.wav', x_base_np[:src_len], SR)
print()
print("Best sigma=" + str(best_sigma) + " (jitter=" + str(round(best_jitter,1)) + "%)")
print("Saved: Desktop/vc_smooth_best.wav, vc_smooth_base.wav")
