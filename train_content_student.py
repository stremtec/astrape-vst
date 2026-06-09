#!/usr/bin/env python3
"""Causal Content Student: streaming conv encoder → predict Mio FSQ 5-dim scalars."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, glob, time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SR=44100  # MioCodec sample rate
CONTENT_RATE=25  # Hz
HOP_SAMPLES=SR//CONTENT_RATE  # 1764 samples per content frame
BATCH=16; EPOCHS=100

device=torch.device('cpu')
print("Device:", device)

# ── Causal Conv Encoder ──────────────────────────────────────────────
class CausalContentEncoder(nn.Module):
    """Streaming encoder: raw audio → 5-dim FSQ scalars at 25Hz."""
    def __init__(self, in_ch=1, hidden=256, out_dim=5, kernel_size=15, num_layers=6):
        super().__init__()
        
        # Strided causal convolutions: each layer downsamples by 2-3x
        # 44100Hz → 25Hz requires ~1764x downsampling
        # Strategy: conv stride=7,3,3,3,3,~3 = 7*3*3*3*3*3 = 1701 ≈ 1764
        strides=[7,3,3,3,3,3]  # total ~1701x
        chs=[hidden//4, hidden//2, hidden, hidden, hidden, hidden]
        
        self.layers=nn.ModuleList()
        in_c=in_ch
        for i,(stride,out_c) in enumerate(zip(strides,chs)):
            # Causal: pad left only
            pad=(kernel_size-1)//2
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_c,out_c,kernel_size,stride=stride,
                         padding=pad, padding_mode='replicate'),
                nn.GroupNorm(min(8,out_c),out_c),
                nn.GELU(),
            ))
            in_c=out_c
        
        # Final projection
        self.proj=nn.Conv1d(hidden,out_dim,1)
    
    def forward(self,x):
        # x: (B, C, T_audio)
        h=x
        for layer in self.layers:
            h=layer(h)
        return self.proj(h)  # (B, 5, T_content)

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"

def load_sample(idx):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,idx))
    return d['fsq_5d'],d['audio']  # (T,5), (T_audio,)

# Split
meta=np.load("{}/meta.npz".format(DATA_DIR))
n_samples=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n_samples)
train_idx=idxs[:int(n_samples*0.8)]
val_idx=idxs[int(n_samples*0.8):]

print("Train: {} Val: {} Total: {}".format(len(train_idx),len(val_idx),n_samples))

# ── Training ──────────────────────────────────────────────────────────
model=CausalContentEncoder(in_ch=1,hidden=256,out_dim=5).to(device)
opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)
mse=nn.MSELoss()

print("Training causal content student (predict FSQ 5-dim)...")
best_loss=float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss=0; nb=0
    
    for idx in train_idx[np.random.permutation(len(train_idx))]:
        fsq_5d,audio=load_sample(idx)
        # fsq_5d: (T,5), audio: (T_audio,)
        T_content=fsq_5d.shape[0]
        
        # Trim audio to match content frames
        audio_len=T_content*HOP_SAMPLES
        audio=audio[:audio_len] if len(audio)>=audio_len else np.pad(audio,(0,audio_len-len(audio)))
        
        x=torch.from_numpy(audio).float().view(1,1,-1).to(device)
        y=torch.from_numpy(fsq_5d).float().T.unsqueeze(0).to(device)  # (1,5,T)
        
        pred=model(x)  # (1,5,T_pred)
        # Trim to match
        Tp=min(pred.shape[2],y.shape[2])
        loss=mse(pred[:,:,:Tp],y[:,:,:Tp])
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        train_loss+=loss.item(); nb+=1
    
    sched.step()
    
    # Validation
    model.eval()
    val_loss=0; vn=0
    with torch.no_grad():
        for idx in val_idx[:20]:
            fsq_5d,audio=load_sample(idx)
            T_content=fsq_5d.shape[0]
            audio_len=T_content*HOP_SAMPLES
            audio=audio[:audio_len] if len(audio)>=audio_len else np.pad(audio,(0,audio_len-len(audio)))
            x=torch.from_numpy(audio).float().view(1,1,-1).to(device)
            y=torch.from_numpy(fsq_5d).float().T.unsqueeze(0).to(device)
            with torch.no_grad():
                pred=model(x)
            Tp=min(pred.shape[2],y.shape[2])
            val_loss+=mse(pred[:,:,:Tp],y[:,:,:Tp]).item(); vn+=1
    
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} train={:.4f} val={:.4f}".format(epoch,train_loss/max(nb,1),val_loss/max(vn,1)))
    
    if val_loss<best_loss:
        best_loss=val_loss
        os.makedirs("checkpoints",exist_ok=True)
        torch.save(model.state_dict(),"checkpoints/causal_content_student.pt")

# ── Test: predict + decode through teacher ────────────────────────────
print()
print("Testing student → teacher decoder plug-in...")
model.eval()

# Load teacher
from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
teacher.eval()

# Test on p255
import soundfile as sf
d,sr=sf.read("/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac")
if d.ndim>1: d=d.mean(axis=1)
from scipy import signal
if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
d=d[:SR*3]
audio_len=len(d)

# Teacher content (ground truth)
x_t=torch.from_numpy(d).float().unsqueeze(0)
with torch.inference_mode():
    feat_t=teacher.encode(x_t,return_content=True,return_global=True)
    ge_t=feat_t.global_embedding
    # Teacher waveform
    wav_t=teacher.decode(global_embedding=ge_t,
                        content_token_indices=feat_t.content_token_indices,
                        target_audio_length=audio_len)

# Student content
T_content=feat_t.content_token_indices.shape[0]
audio_pad=np.pad(d,(0,max(0,T_content*HOP_SAMPLES-len(d))))
x_s=torch.from_numpy(audio_pad[:T_content*HOP_SAMPLES]).float().view(1,1,-1).to(device)
with torch.inference_mode():
    fsq_pred=model(x_s)  # (1,5,T)
    # Quantize through FSQ
    fsq_pred_t=fsq_pred.squeeze(0).T  # (T,5)
    z_q,stu_tokens=teacher.local_quantizer.fsq.encode(fsq_pred_t.unsqueeze(0))
    z_q=teacher.local_quantizer.proj_out(z_q)  # (1,T,768)
    stu_tokens=stu_tokens.squeeze(0)  # (T,)
    
    # Decode through teacher with student tokens
    wav_stu=teacher.decode(global_embedding=ge_t,
                          content_embedding=z_q.squeeze(0),
                          target_audio_length=audio_len)

# Compare
from scipy.signal import stft
def measure(a,sr=SR):
    a=a-np.mean(a)
    f,_,Z=stft(a,fs=sr,nperseg=1024,noverlap=768)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    return np.mean(c)

wt=wav_t.cpu().numpy()[:audio_len]
ws=wav_stu.cpu().numpy()[:audio_len]
ct=measure(wt,'teacher')
cs=measure(ws,'student')
print("  Teacher self-recon centroid: {:.0f}Hz".format(ct))
print("  Student→Teacher centroid:    {:.0f}Hz".format(cs))
print("  Delta: {:.0f}Hz".format(cs-ct))

# Token match rate
t_tokens=feat_t.content_token_indices.cpu().numpy()
s_tokens=stu_tokens.cpu().numpy()
match=(t_tokens==s_tokens).mean()*100
print("  Token match rate: {:.1f}%".format(match))

sf.write('/Users/asill/Desktop/mio_student_test.wav',ws,SR)
sf.write('/Users/asill/Desktop/mio_teacher_self.wav',wt,SR)
print("  Saved: Desktop/mio_student_test.wav, mio_teacher_self.wav")
print()
print("Done!")
