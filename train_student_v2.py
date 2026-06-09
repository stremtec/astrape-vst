#!/usr/bin/env python3
"""
Stage 1 v2: Causal Content Student with FSQ-anchored 768d residual.
FSQ path = stable base, residual embed = corrects toward teacher.
Loss: cosine + L1 on content embedding, auxiliary FSQ.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BATCH=32; EPOCHS=50
device=torch.device('cpu')

# ── Model ──────────────────────────────────────────────────────────────
class CausalContentStudentV2(nn.Module):
    """FSQ anchor + residual 768d embed."""
    def __init__(self,in_dim=80,hidden=256,out_dim=5,num_layers=4,kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(
                nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),
                nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        # FSQ head (stable base)
        self.fsq_head=nn.Conv1d(hidden,out_dim,1)  # 5d
        self.fsq_to_emb=nn.Conv1d(out_dim,768,1)    # 5→768 (like teacher proj_out)
        # Residual head (corrects toward teacher)
        self.residual_head=nn.Conv1d(hidden,768,1)  # direct 768d correction
    
    def forward(self,x,residual_scale=0.3):
        h=self.proj_in(x)
        for layer in self.layers:
            r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h)
        fsq=self.fsq_head(h)  # (B,5,T)
        # FSQ path: 5d → 768d via learned projection
        emb_fsq=self.fsq_to_emb(fsq)  # (B,768,T)
        # Residual: direct correction
        emb_res=self.residual_head(h)  # (B,768,T)
        # Combined: FSQ base + scaled residual
        emb_final=emb_fsq+residual_scale*emb_res
        return fsq, emb_fsq, emb_final

# ── Data ──────────────────────────────────────────────────────────────
MEL_DIR="/Users/asill/btrv5/data/mio_mel"
meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz")
n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl=idxs[int(n*0.8):]

def load_data(idx):
    d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    return torch.from_numpy(d['logmel']).float(),torch.from_numpy(d['fsq_5d']).float(),torch.from_numpy(d['ce_768']).float()

# Compute teacher embed stats for normalization
all_ce=[]
for i in tr[:50]:
    _,_,ce=load_data(i); all_ce.append(ce)
all_ce_cat=torch.cat(all_ce,dim=0)
ce_mean=all_ce_cat.mean(dim=0); ce_std=all_ce_cat.std(dim=0)
print("Teacher CE: mean={:.3f} std={:.3f}".format(ce_mean.mean(),ce_std.mean()))

# ── Training ──────────────────────────────────────────────────────────
model=CausalContentStudentV2().to(device); model.train()
# Start from v1 checkpoint
try: model.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'),strict=False)
except: print("Training from scratch")

opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training Stage 1 v2 (FSQ-anchored 768d residual)...")

for epoch in range(EPOCHS):
    model.train(); tr_loss=0; tr_fsq=0; tr_emb=0; nb=0
    perm=np.random.permutation(len(tr))
    
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; fsqs=[]; ces=[]
        for j in bi:
            mel,fsq,ce=load_data(tr[j])
            mels.append(mel); fsqs.append(fsq); ces.append(ce)
        
        max_T=max(f.shape[0] for f in fsqs)
        # Pad
        mel_b=torch.stack([F.pad(m,(0,max_T-m.shape[1])) for m in mels]).to(device)
        # fsq_b: pad T dim, keep as (B,T,5), then → (B,5,T)
        fsq_padded=[F.pad(f,(0,0,0,max_T-f.shape[0])) for f in fsqs]
        fsq_b=torch.stack(fsq_padded).transpose(1,2).to(device)  # (B,5,T)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T-ce.shape[0])) for ce in ces]).to(device)  # (B,T,768)
        
        # Dynamic residual scale: start small, increase over training
        r_scale=min(0.5,0.1+0.4*epoch/EPOCHS)
        fsq_pred,emb_fsq,emb_final=model(mel_b,residual_scale=r_scale)
        
        # Trim to min
        Tp=min(fsq_pred.shape[2],fsq_b.shape[2])
        
        # FSQ loss (auxiliary)
        L_fsq=F.mse_loss(fsq_pred[:,:,:Tp],fsq_b[:,:,:Tp])
        
        # Content embedding losses (main)
        # Cosine similarity loss — use batch-wise per-sample
        emb_p=emb_final; emb_t=ce_b.transpose(1,2)  # (B,768,T)
        Tp=min(emb_p.shape[2],emb_t.shape[2])
        cos_sim=F.cosine_similarity(emb_p[:,:,:Tp].reshape(emb_p.shape[0],-1),
                                     emb_t[:,:,:Tp].reshape(emb_t.shape[0],-1),dim=1).mean()
        L_cos=1-cos_sim
        
        # L1 on normalized embedding
        L_l1=F.l1_loss(emb_p[:,:,:Tp],emb_t[:,:,:Tp])
        
        loss=L_cos+0.3*L_l1+0.1*L_fsq
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_loss+=loss.item(); tr_fsq+=L_fsq.item(); tr_emb+=L_cos.item(); nb+=1
    
    sched.step()
    
    # Val
    model.eval(); v_emb=0; vn=0
    with torch.no_grad():
        for j in vl[:20]:
            mel,fsq,ce=load_data(j)
            mel=mel.unsqueeze(0).to(device); ce=ce.unsqueeze(0).to(device)
            _,_,emb=model(mel,residual_scale=0.3)
            emb_t=ce.transpose(1,2)
            Tp=min(emb.shape[2],emb_t.shape[2])
            cos=F.cosine_similarity(emb[:,:,:Tp].reshape(1,-1),emb_t[:,:,:Tp].reshape(1,-1),dim=1).mean()
            v_emb+=cos.item(); vn+=1
    model.train()
    
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f} fsq={:.4f} emb_cos={:.4f} val_cos={:.4f} r={:.2f}".format(
            epoch,tr_loss/max(nb,1),tr_fsq/max(nb,1),tr_emb/max(nb,1),v_emb/max(vn,1),r_scale))

os.makedirs("checkpoints",exist_ok=True)
torch.save(model.state_dict(),"checkpoints/causal_student_v2.pt")

# ── Test ──────────────────────────────────────────────────────────────
print()
print("=== Content Embedding Quality Test ===")
model.eval()

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

import soundfile as sf; from scipy import signal as scipy_signal; import torchaudio
SR_=44100

d,sr=sf.read("/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac")
if d.ndim>1: d=d.mean(axis=1)
if sr!=SR_: d=scipy_signal.resample(d,int(len(d)*SR_/sr))
d=d[:SR_*3]; alen=len(d)

mel_spec=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)
audio_16k=scipy_signal.resample(d[:alen],int(alen*16000/SR_))
mel=mel_spec(torch.from_numpy(audio_16k).float().view(1,1,-1)).squeeze(1)
logmel=torch.log(mel.clamp(min=1e-5))

# Teacher content
x_t=torch.from_numpy(d).float().unsqueeze(0)
with torch.inference_mode():
    ft=teacher.encode(x_t,return_content=True,return_global=True)
    ce_t=ft.content_embedding  # (T,768)
    
    # Student FSQ path
    fsq_s,emb_fsq_s,emb_final_s=model(logmel.unsqueeze(0),residual_scale=0.3)
    fsq_s=fsq_s.squeeze(0).T
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_s.unsqueeze(0))
    ce_stu_fsq=teacher.local_quantizer.proj_out(z_q).squeeze(0)  # student FSQ path
    
    # Student residual embed
    ce_stu_res=emb_final_s.squeeze(0).T  # (T,768)
    
    # Cosine comparison
    T=min(ce_t.shape[0],ce_stu_fsq.shape[0],ce_stu_res.shape[0])
    cos_fsq=F.cosine_similarity(ce_stu_fsq[:T].flatten(),ce_t[:T].flatten(),dim=0).item()
    cos_res=F.cosine_similarity(ce_stu_res[:T].flatten(),ce_t[:T].flatten(),dim=0).item()

print("  FSQ path cos:     {:.4f}".format(cos_fsq))
print("  Residual path cos: {:.4f}".format(cos_res))
print("  Improvement:       {:+.4f}".format(cos_res-cos_fsq))

# Stage 3b test with residual embed
print()
print("--- Stage 3b with residual embed ---")
# Load mel decoder
from train_stage3aG import CausalMelDecoder
dec=CausalMelDecoder(); dec.load_state_dict(torch.load("checkpoints/causal_mel_decoder.pt",map_location='cpu')); dec.eval()
for p in dec.parameters(): p.requires_grad=False

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR_: d_tgt=scipy_signal.resample(d_tgt,int(len(d_tgt)*SR_/sr))
d_tgt=d_tgt[:SR_*3]
x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)
with torch.inference_mode():
    ft_tgt=teacher.encode(x_tgt,return_content=False,return_global=True)
    ge_tgt=ft_tgt.global_embedding.unsqueeze(0)
    ge_zero=torch.zeros(1,128)
    wav_vc=teacher.decode(global_embedding=ge_tgt.squeeze(0),content_token_indices=ft.content_token_indices,target_audio_length=alen)

# Reference mel
we=torchaudio.transforms.MelSpectrogram(sample_rate=SR_,n_fft=1024,hop_length=1764,n_mels=80,f_min=80,f_max=14000,center=False,power=1)
mel_ref=we(wav_vc.unsqueeze(0).unsqueeze(1)); mel_ref=torch.log(mel_ref.squeeze(1).clamp(min=1e-5))

# Test FSQ path
mel_fsq=dec(ce_stu_fsq.unsqueeze(0),ge_tgt).squeeze(0)
# Test residual path
mel_res=dec(ce_stu_res.unsqueeze(0),ge_tgt).squeeze(0)
# Test residual + target vs zero
mel_res_zero=dec(ce_stu_res.unsqueeze(0),ge_zero).squeeze(0)

def mcos(pred,tgt):
    T=min(pred.shape[1],tgt.shape[2])
    return F.cosine_similarity(pred[:,:T].flatten(),tgt[:,:,:T].flatten(),dim=0).item()

print("  FSQ path mel cos:   {:.4f}".format(mcos(mel_fsq,mel_ref)))
print("  Residual mel cos:   {:.4f}".format(mcos(mel_res,mel_ref)))
print("  Residual zeroG cos: {:.4f}".format(mcos(mel_res_zero,mel_ref)))
delta=mcos(mel_res,mel_ref)-mcos(mel_res_zero,mel_ref)
print("  Global effect:      {:.4f}".format(delta))

print()
print("Done!")
