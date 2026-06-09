#!/usr/bin/env python3
"""CausalContentStudent v1 — trains on CACHED mel features (fast)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, time, soundfile as sf
from scipy import signal as scipy_signal
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SR=44100; HOP_44k=int(SR/25)  # 1764
BATCH=32; EPOCHS=80
device=torch.device('cpu')

# ── Encoder (no mel frontend — uses cached mel) ─────────────────────
class CausalTCNEncoder(nn.Module):
    def __init__(self, in_dim=80, hidden=256, out_dim=5, num_layers=4, kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(
                nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),
                nn.GroupNorm(8,hidden),nn.GELU(),
                nn.Conv1d(hidden,hidden,1),
            ))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        self.proj_out=nn.Conv1d(hidden,out_dim,1)
        self.embed_head=nn.Conv1d(out_dim,768,1)
    
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers:
            r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h)
        fsq=self.proj_out(h)
        embed=self.embed_head(fsq)
        return fsq, embed

# ── Data ──────────────────────────────────────────────────────────────
MEL_DIR="/Users/asill/btrv5/data/mio_mel"

def load_mel(idx):
    d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    return d['logmel'],d['fsq_5d'],d['fsq_tokens'],d['ce_768']

meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz")
n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl=idxs[int(n*0.8):]
print("Train: {} Val: {}".format(len(tr),len(vl)))

model=CausalTCNEncoder().to(device)
opt=AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training on cached mel (fast)...")
for epoch in range(EPOCHS):
    model.train(); tl=0; nb=0
    perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        bd=[load_mel(tr[j]) for j in bi]
        max_T=max(d[1].shape[0] for d in bd)
        xs=[]; ys=[]; yl=[]
        for logmel,fsq,_,_ in bd:
            m=torch.from_numpy(logmel).float()
            y=torch.from_numpy(fsq).float()
            xs.append(F.pad(m,(0,max_T-m.shape[1])))
            ys.append(F.pad(y.T,(0,max_T-y.shape[0])).T)
            yl.append(y.shape[0])
        xb=torch.stack(xs).to(device); yb=torch.stack(ys).transpose(1,2).to(device)
        fsq_pred,_=model(xb)
        # Trim to min length
        Tp=min(fsq_pred.shape[2],yb.shape[2])
        mask=torch.zeros(yb.shape[0],Tp,device=device)
        for j,l in enumerate(yl): mask[j,:min(l,Tp)]=1
        loss=((fsq_pred[:,:,:Tp]-yb[:,:,:Tp])*mask.unsqueeze(1)).pow(2).sum()/(mask.sum()*5+1e-8)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl+=loss.item(); nb+=1
    sched.step()
    # Val
    model.eval(); vl_=0; vn=0
    with torch.no_grad():
        for j in vl[:20]:
            logmel,fsq,_,_=load_mel(j)
            m=torch.from_numpy(logmel).float().unsqueeze(0).to(device)
            y=torch.from_numpy(fsq).float().T.unsqueeze(0).to(device)
            pred,_=model(m); T=min(pred.shape[2],y.shape[2])
            vl_+=F.mse_loss(pred[:,:,:T],y[:,:,:T]).item(); vn+=1
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} tr={:.4f} val={:.4f}".format(epoch,tl/max(nb,1),vl_/max(vn,1)))

os.makedirs("checkpoints",exist_ok=True)
torch.save(model.state_dict(),"checkpoints/causal_student_v1.pt")

# ── Teacher plug-in ──────────────────────────────────────────────────
print()
print("=== Teacher Decoder Plug-in ===")
from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
teacher.eval()

d,sr=sf.read("/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac")
if d.ndim>1: d=d.mean(axis=1)
if sr!=SR: d=scipy_signal.resample(d,int(len(d)*SR/sr))
d=d[:SR*3]; alen=len(d)

x_t=torch.from_numpy(d).float().unsqueeze(0)
with torch.inference_mode():
    ft=teacher.encode(x_t,return_content=True,return_global=True)
    ge=ft.global_embedding
    wav_t=teacher.decode(global_embedding=ge,content_token_indices=ft.content_token_indices,
                        target_audio_length=alen)

# Student prediction via mel
import torchaudio
mel_spec=torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,n_fft=512,hop_length=int(16000*20/1000),
    n_mels=80,f_min=80,f_max=7600,center=False,power=2)

audio_16k=scipy_signal.resample(d[:alen],int(alen*16000/SR))
mel=mel_spec(torch.from_numpy(audio_16k).float().view(1,1,-1)).squeeze(1)
logmel=torch.log(mel.clamp(min=1e-5))

with torch.inference_mode():
    fsq_pred,embed_pred=model(logmel)
    fsq_t=fsq_pred.squeeze(0).T
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_t.unsqueeze(0))
    z_q=teacher.local_quantizer.proj_out(z_q)
    wav_s=teacher.decode(global_embedding=ge,content_embedding=z_q.squeeze(0),
                        target_audio_length=alen)

from scipy.signal import stft
def m(a):
    a=a-np.mean(a)
    f,_,Z=stft(a,fs=SR,nperseg=1024,noverlap=768)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)
    c/=(mag[:len(f)//2].sum(axis=0)+1e-8)
    return np.mean(c)

wt=wav_t.cpu().numpy()[:alen]; ws=wav_s.cpu().numpy()[:alen]
ct=m(wt); cs=m(ws)
print("  Teacher centroid: {:.0f}Hz".format(ct))
print("  Student centroid: {:.0f}Hz".format(cs))
print("  Delta: {:.0f}Hz".format(cs-ct))

tt=ft.content_token_indices.numpy()
# Get student tokens from raw FSQ 5-dim
st=teacher.local_quantizer.fsq.codes_to_indices(fsq_t.unsqueeze(0)).squeeze(0).numpy()
match=(tt[:len(st)]==st[:len(tt)]).mean()*100
print("  Token match: {:.1f}%".format(match))

sf.write('/Users/asill/Desktop/mio_student_v1.wav',ws,SR)
print("Saved: Desktop/mio_student_v1.wav")
print("Done!")
