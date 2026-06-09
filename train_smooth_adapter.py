#!/usr/bin/env python3
"""
Retrain FiLM adapter with temporal smoothness loss.
Same architecture (n_content=1 + FiLM), just add L_smooth penalty.
Fix jitter at training time, not post-hoc.
"""
import sys, os, glob, time
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, torch.nn as nn
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW

SR = 24000; SAFE_LEN = 48000; STRIDE = 1920
BATCH = 8; EPOCHS = 300
SMOOTH_WEIGHT = 0.3  # weight for temporal smoothness (reduced)

device = torch.device('cpu')
print("Device:", device, "| Smoothness weight:", SMOOTH_WEIGHT)

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi, n_content=1).to(device)

TRAIN_SPKS = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266'][:40]

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

# Classifiers for adversarial (keep C clean, S strong)
class SpkClf(nn.Module):
    def __init__(self, n=40): super().__init__(); self.net=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,n))
    def forward(self,x):
        if x.dim()==3: x=x.mean(dim=-1)
        return self.net(x)

from torch.autograd import Function
class GR(Function):
    @staticmethod
    def forward(ctx,x,a): ctx.alpha=a; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.alpha,None

def encode_speakers(spk_list, n_utt=5):
    samples = []
    for spk_idx, spk in enumerate(spk_list):
        files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:n_utt]
        for f in files:
            d,sr=sf.read(f)
            if d.ndim>1: d=d.mean(axis=1)
            if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
            if len(d)<SAFE_LEN: d=np.pad(d,(0,SAFE_LEN-len(d)))
            d=d[:SAFE_LEN]
            x=torch.from_numpy(d).float().view(1,1,-1).to(device)
            with torch.no_grad(): z,codes=mimi_encode(x,mimi)
            samples.append({'z':z.squeeze(0).cpu(),'codes':codes.squeeze(0).cpu(),'spk':spk_idx})
    return samples

print("Pre-encoding..."); t0=time.time()
train_data = encode_speakers(TRAIN_SPKS, 5)
print("Train:", len(train_data), "({:.0f}s)".format(time.time()-t0))

mse=nn.MSELoss(); l1=nn.L1Loss(); ce=nn.CrossEntropyLoss()
opt=AdamW(splitter.parameters(),lr=1e-3,weight_decay=1e-5)
clf_C=SpkClf(len(TRAIN_SPKS)).to(device); clf_S=SpkClf(len(TRAIN_SPKS)).to(device)
clf_A=SpkClf(len(TRAIN_SPKS)).to(device)
opt_clf=AdamW(list(clf_C.parameters())+list(clf_S.parameters())+list(clf_A.parameters()),lr=1e-3)

print()
print("Training with temporal smoothness loss...", flush=True)

for epoch in range(EPOCHS):
    idxs=torch.randperm(len(train_data)); tr,tsm,tc,ta,ts,nb=0,0,0,0,0,0
    for i in range(0,len(train_data),BATCH):
        batch=[train_data[j] for j in idxs[i:i+BATCH]]
        zb=torch.stack([s['z'] for s in batch]).to(device)
        cb=torch.stack([s['codes'] for s in batch]).to(device)
        spk=torch.tensor([s['spk'] for s in batch]).to(device)
        
        zv,C,S,A=splitter(zb,cb)
        
        L_recon=mse(zv,zb)
        # Temporal smoothness: penalize frame-to-frame A difference
        L_smooth=mse(A[:,:,1:], A[:,:,:-1]) if A.shape[-1]>1 else 0
        # Adversarial: C→chance, A→chance, S→spk
        L_adv_C=ce(clf_C(GR.apply(C.mean(dim=-1),1.0)),spk)
        L_adv_A=ce(clf_A(GR.apply(A.mean(dim=-1),0.3)),spk)
        L_spk_S=ce(clf_S(S),spk)
        
        loss=L_recon+SMOOTH_WEIGHT*L_smooth+0.5*L_adv_C+0.3*L_adv_A+0.5*L_spk_S
        opt.zero_grad(); opt_clf.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(),1.0)
        opt.step(); opt_clf.step()
        tr+=L_recon.item(); tsm+=L_smooth; tc+=L_adv_C.item(); ta+=L_adv_A.item(); ts+=L_spk_S.item(); nb+=1
    
    if epoch%20==0 or epoch==EPOCHS-1:
        n=max(nb,1)
        print("  E{:3d} Recon={:.4f} Smooth={:.4f} C_adv={:.2f} A_adv={:.2f} S_spk={:.2f}".format(
            epoch,tr/n,tsm/n,tc/n,ta/n,ts/n), flush=True)

os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict":splitter.state_dict()},"checkpoints/mimi_smooth_adapter.pt")

# VC test
print()
print("=== VC: p255 -> origin (smooth adapter) ===")
splitter.eval()

d_src,sr=sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr))
safe=(len(d_src)//STRIDE)*STRIDE; d_src=d_src[:safe]
x_src=torch.from_numpy(d_src).float().view(1,1,-1).to(device)

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr))
safe=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)

with torch.no_grad():
    z_src,codes_src=mimi_encode(x_src,mimi)
    z_tgt,codes_tgt=mimi_encode(x_tgt,mimi)
    
    mimi.set_num_codebooks(1)
    z_q0=mimi.decode_latent(codes_src[:,:1,:]); mimi.set_num_codebooks(8)
    C=splitter.content_extractor(z_q0)
    S_tgt=splitter.speaker_encoder(z_tgt)
    n_ac=codes_src.shape[1]-1
    mimi.set_num_codebooks(n_ac)
    z_ac=mimi.decode_latent(codes_src[:,1:,:]); mimi.set_num_codebooks(8)
    A=splitter.acoustic_adapter(z_ac,S_tgt,C)
    z_vc=C+A; x_vc=mimi_decode_latent(mimi,z_vc)

x_rt=mimi_decode_latent(mimi,z_src)
vc_np=x_vc[0,0].cpu().numpy()[:len(d_src)]
rt_np=x_rt[0,0].cpu().numpy()[:len(d_src)]

# Jitter measurement
def jitter_metric(a):
    a=a-np.mean(a); fl=int(SR*0.04); hp=int(SR*0.01); fs=[]
    for i in range(0,len(a)-fl,hp):
        f=a[i:i+fl]
        if np.sqrt(np.mean(f**2))<0.001: fs.append(0); continue
        c=np.correlate(f,f,mode='full'); c=c[len(c)//2:]; c=c/(c[0]+1e-8)
        pks=signal.find_peaks(c,distance=10)[0]
        if len(pks)==0: fs.append(0); continue
        f0=SR/pks[0]; fs.append(f0 if 50<f0<400 else 0)
    fs=np.array(fs); v=fs>0
    if v.sum()<3: return 0
    return np.mean(np.abs(np.diff(fs[v])))/np.mean(fs[v])*100

from scipy.signal import stft
def measure(a):
    f,_,Z=stft(a,fs=SR,nperseg=512,noverlap=384); mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
    return np.mean(c),cr,vh

c,cr,vh=measure(vc_np); j=jitter_metric(vc_np)
print()
print("  Centroid: {}Hz  Crest: {:.1f}  VHigh: {:.1f}%  Jitter: {:.1f}%".format(round(c),cr,vh,j))

sf.write('/Users/asill/Desktop/vc_smooth_adapter.wav',vc_np,SR)
print("Saved: Desktop/vc_smooth_adapter.wav")
print("Done!")
