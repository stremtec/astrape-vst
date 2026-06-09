#!/usr/bin/env python3
"""
Mimi Splitter v2: Fix fundamental splitter flaws.
- A_clean: speaker-adversarial acoustic (not source-leaky)
- S_robust: domain-augmented speaker (not EQ/loudness shortcut)
- Decoder manifold constraint: keep z_vc in-distribution
"""
import sys, os, glob, time, random
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, torch.nn as nn
import torch.nn.functional as F
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW

SR=24000; SAFE_LEN=48000; STRIDE=1920
BATCH=8; EPOCHS=200

device=torch.device('cpu')
print("Device:", device)

mimi=load_mimi(device).to(device)
splitter=MimiSplitterV2(mimi,n_content=1).to(device)
print("Params:", sum(p.numel() for p in splitter.parameters() if p.requires_grad))

TRAIN_SPKS=[f'p{i}' for i in range(225,265) if i!=235 and i!=242][:40]
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

# ── Domain augmentation ───────────────────────────────────────────────
from scipy.signal import butter, sosfilt

def augment_audio(audio):
    """Random loudnorm, lowpass, or EQ augmentation."""
    r = random.random()
    if r < 0.25:  # loudnorm
        rms = np.sqrt(np.mean(audio**2))
        scale = np.random.uniform(0.5, 2.0)
        return audio * (scale / (rms + 1e-8)) * rms
    elif r < 0.5:  # lowpass
        cutoff = np.random.choice([4000, 6000, 8000])
        sos = butter(4, cutoff, btype='low', fs=SR, output='sos')
        return sosfilt(sos, audio)
    elif r < 0.75:  # high-shelf boost/cut
        sos_hs = butter(2, 5000, btype='high', fs=SR, output='sos')
        hf = sosfilt(sos_hs, audio)
        gain = 10**(np.random.uniform(-6, 6)/20)
        return audio + hf * (gain - 1)
    return audio  # no aug

# ── Classifiers ──────────────────────────────────────────────────────
class SpkClf(nn.Module):
    def __init__(self,n=40): super().__init__(); self.net=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,n))
    def forward(self,x):
        if x.dim()==3: x=x.mean(dim=-1)
        return self.net(x)

from torch.autograd import Function
class GR(Function):
    @staticmethod
    def forward(ctx,x,a): ctx.alpha=a; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.alpha,None

# ── Data ──────────────────────────────────────────────────────────────
def encode_speakers(spk_list, n_utt=5):
    samples=[]
    for spk_idx, spk in enumerate(spk_list):
        files=sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:n_utt]
        for f in files:
            d,sr=sf.read(f)
            if d.ndim>1: d=d.mean(axis=1)
            if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
            if len(d)<SAFE_LEN: d=np.pad(d,(0,SAFE_LEN-len(d)))
            d=d[:SAFE_LEN]
            # Store original + augmented version
            d_aug = augment_audio(d.copy())
            x=torch.from_numpy(d).float().view(1,1,-1).to(device)
            x_aug=torch.from_numpy(d_aug).float().view(1,1,-1).to(device)
            with torch.no_grad():
                z,codes=mimi_encode(x,mimi)
                z_aug,_=mimi_encode(x_aug,mimi)
            samples.append({
                'z':z.squeeze(0).cpu(), 'codes':codes.squeeze(0).cpu(),
                'z_aug':z_aug.squeeze(0).cpu(),
                'spk':spk_idx, 'audio': d, 'audio_aug': d_aug
            })
    return samples

print("Pre-encoding with augmentation..."); t0=time.time()
train_data=encode_speakers(TRAIN_SPKS, 5)
print("Train:", len(train_data), "({:.0f}s)".format(time.time()-t0))

# ── Training ──────────────────────────────────────────────────────────
mse=nn.MSELoss(); l1=nn.L1Loss(); ce=nn.CrossEntropyLoss()
opt=AdamW(splitter.parameters(),lr=1e-3,weight_decay=1e-5)
# Adversarial classifiers
clf_C=SpkClf(len(TRAIN_SPKS)).to(device)
clf_A=SpkClf(len(TRAIN_SPKS)).to(device)  # NEW: adversarial on A
clf_S=SpkClf(len(TRAIN_SPKS)).to(device)
opt_clf=AdamW(list(clf_C.parameters())+list(clf_A.parameters())+list(clf_S.parameters()),lr=1e-3)

# Track decoder distribution stats for norm clamping
z_rt_norms=[]

print()
print("Training splitter v2 (A_clean + S_robust)...", flush=True)

for epoch in range(EPOCHS):
    idxs=torch.randperm(len(train_data)); tr,ta,tc,ts,tn,nb=0,0,0,0,0,0
    for i in range(0,len(train_data),BATCH):
        batch=[train_data[j] for j in idxs[i:i+BATCH]]
        zb=torch.stack([s['z'] for s in batch]).to(device)
        cb=torch.stack([s['codes'] for s in batch]).to(device)
        spk=torch.tensor([s['spk'] for s in batch]).to(device)
        
        zv,C,S,A=splitter(zb,cb)
        
        # 1. Reconstruction
        L_recon=mse(zv,zb)
        
        # 2. Adversarial on C (speaker-clean content)
        L_adv_C=ce(clf_C(GR.apply(C.mean(dim=-1),1.0)),spk)
        
        # 3. Adversarial on A (speaker-free acoustic) ★ NEW
        L_adv_A=ce(clf_A(GR.apply(A.mean(dim=-1),1.0)),spk)
        
        # 4. Speaker on S (strong identity)
        L_spk_S=ce(clf_S(S),spk)
        
        # 5. Decoder distribution constraint: penalize large z_vc norm
        zv_norm_penalty = torch.relu(zv.norm(dim=1).mean() - 80) * 0.01
        
        loss = L_recon + 0.5*L_adv_C + 0.5*L_adv_A + 0.5*L_spk_S + zv_norm_penalty
        opt.zero_grad(); opt_clf.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(),1.0)
        opt.step(); opt_clf.step()
        tr+=L_recon.item(); ta+=L_adv_A.item(); tc+=L_adv_C.item(); ts+=L_spk_S.item()
        tn+=zv_norm_penalty.item(); nb+=1
    
    if epoch%20==0 or epoch==EPOCHS-1:
        n=max(nb,1)
        print("  E{:3d} Recon={:.4f} Adv_C={:.2f} Adv_A={:.2f} Spk_S={:.2f} NormPen={:.4f}".format(
            epoch,tr/n,tc/n,ta/n,ts/n,tn/n), flush=True)

os.makedirs("checkpoints",exist_ok=True)
torch.save({"model_state_dict":splitter.state_dict()},"checkpoints/mimi_splitter_v2_clean.pt")

# ── VC test ──────────────────────────────────────────────────────────
print()
print("=== VC: p255 -> origin (Splitter v2) ===")
splitter.eval()

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

with torch.no_grad():
    z_src,codes_src=mimi_encode(x_src,mimi)
    z_tgt,codes_tgt=mimi_encode(x_tgt,mimi)
    
    mimi.set_num_codebooks(1); z_q0=mimi.decode_latent(codes_src[:,:1,:]); mimi.set_num_codebooks(8)
    C=splitter.content_extractor(z_q0)
    S_tgt=splitter.speaker_encoder(z_tgt)
    n_ac=codes_src.shape[1]-1
    mimi.set_num_codebooks(n_ac); z_ac=mimi.decode_latent(codes_src[:,1:,:]); mimi.set_num_codebooks(8)
    A=splitter.acoustic_adapter(z_ac,S_tgt,C)
    z_vc=C+A; x_vc=mimi_decode_latent(mimi,z_vc)

x_rt=mimi_decode_latent(mimi,z_src)
vc_np=x_vc[0,0].cpu().numpy()[:src_len]

from scipy.signal import stft
def measure(a):
    f,_,Z=stft(a,fs=SR,nperseg=512,noverlap=384); mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
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

c,vh,cr,j=measure(vc_np)
l2=torch.norm(z_vc - z_src).item()
print("  Centroid: {}Hz  Jitter: {:.1f}%  VHigh: {:.1f}%  Crest: {:.1f}  L2: {:.0f}".format(round(c),j,vh,cr,l2))

# Probe A speaker leak
A_leak = splitter.acoustic_adapter(z_ac, splitter.speaker_encoder(z_src), C)
cos_A = F.cosine_similarity(A_leak.mean(dim=-1), A.mean(dim=-1)).item()
print("  A cosine(src_A, tgt_A): {:.3f}".format(cos_A))

sf.write('/Users/asill/Desktop/vc_splitter_v2.wav', vc_np, SR)
print("Saved: Desktop/vc_splitter_v2.wav")
print("Done!")
