#!/usr/bin/env python3
"""Mimi Splitter with q0+q1 content (n_content=2) — better audio quality."""
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
SR = 24000; SAFE_LEN = 48000; BATCH_SIZE = 8; EPOCHS = 300

TRAIN_SPKS = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266'][:40]
VAL_SPKS = ['p267','p268','p269','p270','p271','p272','p273','p274','p275','p276']
TEST_SPKS = ['p277','p278','p279','p280','p281','p282','p283','p284','p285']

device = torch.device('cpu')
print("Device:", device, "n_content=2", flush=True)

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi, n_content=2).to(device)
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
            samples.append({'z':z.squeeze(0).cpu(),'codes':codes.squeeze(0).cpu(),'spk':spk_idx,'name':spk})
    return samples

print("Pre-encoding..."); t0=time.time()
train_data = encode_speakers(TRAIN_SPKS, 5)
val_data = encode_speakers(VAL_SPKS, 5)
test_data = encode_speakers(TEST_SPKS, 5)
print(f"Train:{len(train_data)} Val:{len(val_data)} Test:{len(test_data)} ({time.time()-t0:.1f}s)")

class SpkClf(nn.Module):
    def __init__(self, n_spk=40): super().__init__(); self.net=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,n_spk))
    def forward(self,x):
        if x.dim()==3: x=x.mean(dim=-1)
        return self.net(x)

from torch.autograd import Function
class GR(Function):
    @staticmethod
    def forward(ctx,x,a): ctx.alpha=a; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.alpha,None

mse=nn.MSELoss(); ce=nn.CrossEntropyLoss()
opt=AdamW(splitter.parameters(),lr=1e-3,weight_decay=1e-5)
clf_C=SpkClf(len(TRAIN_SPKS)).to(device); clf_S=SpkClf(len(TRAIN_SPKS)).to(device)
opt_clf=AdamW(list(clf_C.parameters())+list(clf_S.parameters()),lr=1e-3)

def probe(splitter, data, n_spk):
    splitter.eval(); C,S,y=[],[],[]
    for s in data:
        z=s['z'].unsqueeze(0).to(device); codes=s['codes'].unsqueeze(0).to(device)
        with torch.no_grad(): _,C_i,S_i,_=splitter(z,codes)
        C.append(C_i.squeeze(0).mean(dim=1).cpu().numpy()); S.append(S_i.squeeze(0).cpu().numpy()); y.append(s['spk'])
    splitter.train()
    X_C=np.stack(C); X_S=np.stack(S); y=np.array(y)
    scaler=StandardScaler(); clf=LogisticRegression(max_iter=2000,random_state=42,C=0.1)
    cv=min(3,len(set(y))) if len(set(y))>=2 else 2
    try: sC=cross_val_score(clf,scaler.fit_transform(X_C),y,cv=cv); sS=cross_val_score(clf,scaler.fit_transform(X_S),y,cv=cv)
    except: return 0,0
    return sC.mean()*100,sS.mean()*100

print()
print("Training...",flush=True)
best_val=0
for epoch in range(EPOCHS):
    idxs=torch.randperm(len(train_data)); tr,tadv,tspk=0,0,0; nb=0
    for i in range(0,len(train_data),BATCH_SIZE):
        batch=[train_data[j] for j in idxs[i:i+BATCH_SIZE]]
        zb=torch.stack([s['z'] for s in batch]).to(device)
        cb=torch.stack([s['codes'] for s in batch]).to(device)
        spk=torch.tensor([s['spk'] for s in batch]).to(device)
        zv,C,S,A=splitter(zb,cb)
        Lr=mse(zv,zb)
        Ladv=ce(clf_C(GR.apply(C.mean(dim=-1),1.0)),spk)
        Lspk=ce(clf_S(S),spk)
        loss=Lr+0.5*Ladv+0.5*Lspk
        opt.zero_grad(); opt_clf.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(),1.0)
        opt.step(); opt_clf.step()
        tr+=Lr.item(); tadv+=Ladv.item(); tspk+=Lspk.item(); nb+=1
    if epoch%20==0 or epoch==EPOCHS-1:
        n=max(nb,1)
        tc,ts=probe(splitter,train_data[:50],len(TRAIN_SPKS))
        vc,vs=probe(splitter,val_data,len(VAL_SPKS))
        xc,xs=probe(splitter,test_data,len(TEST_SPKS))
        print(f"  E{epoch:3d} R={tr/n:.3f} Adv={tadv/n:.2f} Spk={tspk/n:.2f} | "
              f"T C={tc:.0f}% S={ts:.0f}% | V C={vc:.0f}% S={vs:.0f}% | X C={xc:.0f}% S={xs:.0f}%",flush=True)
        if vs>best_val: best_val=vs; os.makedirs("checkpoints",exist_ok=True); torch.save({"model_state_dict":splitter.state_dict()},"checkpoints/mimi_nc2.pt")

print()
print("=== VC: p255 -> origin (n_content=2) ===")
splitter.eval()

d_src,sr=sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr))
safe_src=(len(d_src)//1920)*1920; d_src=d_src[:safe_src]; src_len=len(d_src)
x_src=torch.from_numpy(d_src).float().view(1,1,-1).to(device)
with torch.no_grad(): z_src,codes_src=mimi_encode(x_src,mimi)

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr))
safe_tgt=(len(d_tgt)//1920)*1920; d_tgt=d_tgt[:safe_tgt]; src_tgt=len(d_tgt)
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)
with torch.no_grad(): z_tgt,codes_tgt=mimi_encode(x_tgt,mimi)

with torch.no_grad():
    mimi.set_num_codebooks(2)
    z_content=mimi.decode_latent(codes_src[:,:2,:])
    mimi.set_num_codebooks(8)
    C=splitter.content_extractor(z_content)
    S=splitter.speaker_encoder(z_tgt)
    mimi.set_num_codebooks(6)
    z_ac=mimi.decode_latent(codes_src[:,2:,:])
    mimi.set_num_codebooks(8)
    A=splitter.acoustic_adapter(z_ac,S,C)
    z_vc=C+A; x_vc=mimi_decode_latent(mimi,z_vc)

x_rt=mimi_decode_latent(mimi,z_src)
vc_np=x_vc[0,0].cpu().numpy()[:src_len]
rt_np=x_rt[0,0].cpu().numpy()[:src_len]

sf.write("checkpoints/vc_nc2_p255_origin.wav",vc_np,SR)
sf.write("checkpoints/vc_nc2_src_rt.wav",rt_np,SR)
sf.write("checkpoints/vc_nc2_src.wav",d_src,SR)
sf.write("checkpoints/vc_nc2_tgt.wav",d_tgt,SR)

# Spectral
from scipy.signal import stft
def ana(a,label):
    f,_,Z=stft(a,fs=SR,nperseg=512,noverlap=256); mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    print(f"  {label:15s} RMS={np.sqrt(np.mean(a**2)):.3f} Cent={np.mean(c):.0f}Hz VHigh={mag[(f>=4000)&(f<8000)].sum()/total*100:.1f}%")

print()
ana(d_src,"Source (p255)")
ana(d_tgt,"Target (origin)")
ana(vc_np,"VC n_content=2")
ana(rt_np,"Mimi RT")

sf.write("/Users/asill/Desktop/vc_nc2_p255_origin.wav",vc_np,SR)
sf.write("/Users/asill/Desktop/vc_nc2_src.wav",d_src,SR)
sf.write("/Users/asill/Desktop/vc_nc2_tgt.wav",d_tgt,SR)
print()
print("Saved to Desktop: vc_nc2_*.wav")
print()
print("Done!")
