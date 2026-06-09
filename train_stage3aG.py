#!/usr/bin/env python3
"""
Stage 3a-G: Retrain causal mel decoder with VC pair distillation.
Input: teacher content_src + global_tgt → teacher VC mel
Forces decoder to use global conditioning for speaker transfer.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, numpy as np, os, time, random, warnings
from scipy import signal as scipy_signal
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings('ignore')

SR=44100; N_MELS=80; MEL_HOP=int(SR/25); BATCH=8; EPOCHS=40
device=torch.device('cpu')

# ── Decoder ────────────────────────────────────────────────────────────
class AdaLNZero(nn.Module):
    def __init__(self,dim,cond_dim,eps=1e-5):
        super().__init__()
        self.norm=nn.LayerNorm(dim,eps=eps,elementwise_affine=False)
        self.proj=nn.Sequential(nn.SiLU(),nn.Linear(cond_dim,3*dim))
        nn.init.zeros_(self.proj[1].weight); nn.init.zeros_(self.proj[1].bias)
    def forward(self,x,cond):
        xn=self.norm(x); shift,scale,gate=self.proj(cond).chunk(3,dim=-1)
        return xn*(1+scale)+shift, gate

class CausalDecoderBlock(nn.Module):
    def __init__(self,dim=512,cond_dim=128,n_heads=8,ff_mult=4,dropout=0.1):
        super().__init__()
        self.adaln=AdaLNZero(dim,cond_dim); self.adaln2=AdaLNZero(dim,cond_dim)
        self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),
                              nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x,cond):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn,gate=self.adaln(x,cond); attn_out=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]
        x=x+gate*attn_out; xn2,gate2=self.adaln2(x,cond); ff_out=self.ff(xn2); x=x+gate2*ff_out
        return x

class CausalMelDecoder(nn.Module):
    def __init__(self,cd=768,cond_dim=128,hidden=512,n_layers=4,n_heads=8,n_mels=80):
        super().__init__()
        self.proj_in=nn.Linear(cd,hidden)
        self.blocks=nn.ModuleList([CausalDecoderBlock(hidden,cond_dim,n_heads) for _ in range(n_layers)])
        self.norm_out=nn.LayerNorm(hidden); self.proj_out=nn.Linear(hidden,n_mels)
    def forward(self,ce,ge):
        x=self.proj_in(ce); cond=ge.unsqueeze(1)
        for b in self.blocks: x=b(x,cond)
        x=self.norm_out(x); return self.proj_out(x).transpose(1,2)

# ── Teacher ────────────────────────────────────────────────────────────
from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

mel_ext=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=1024,hop_length=MEL_HOP,n_mels=N_MELS,f_min=80,f_max=14000,center=False,power=1)

def extract_mel(waveform):
    if isinstance(waveform,torch.Tensor): w=waveform.detach().cpu().numpy()
    else: w=waveform
    w=w-np.mean(w)
    mel=mel_ext(torch.from_numpy(w).float().view(1,1,-1))
    return torch.log(mel.squeeze(1).clamp(min=1e-5))

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
meta=np.load("{}/meta.npz".format(DATA_DIR)); spk_names=meta['spk_names']; n=len(meta['spk_names'])
unique_spks=sorted(set(spk_names))

# Split speakers
np.random.RandomState(42).shuffle(unique_spks)
tr_spks=unique_spks[:20]; vl_spks=unique_spks[20:25]

# Build pairs: for each src utterance, pair with random target speakers
print("Building VC pair dataset...")
pairs=[]
for i in range(n):
    if spk_names[i] not in tr_spks: continue
    # Self-recon pair (30%)
    pairs.append({'src':i,'tgt':i,'type':'self'})
    # VC pairs: 3 random targets from different speakers (70%)
    other_spks=[s for s in tr_spks if s!=spk_names[i]]
    for tgt_spk in random.sample(other_spks,min(3,len(other_spks))):
        tgt_utts=[j for j in range(n) if spk_names[j]==tgt_spk]
        if tgt_utts:
            tgt_idx=random.choice(tgt_utts)
            pairs.append({'src':i,'tgt':tgt_idx,'type':'vc'})

print("Train pairs: {} (self={}, vc={})".format(len(pairs),
    sum(1 for p in pairs if p['type']=='self'),sum(1 for p in pairs if p['type']=='vc')))

# Pre-compute content embeddings + mels
print("Pre-computing teacher VC mels...")
src_ce={}; tgt_ge={}; vc_mels={}

for i in range(n):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,i))
    src_ce[i]=torch.from_numpy(d['ce_768']).float()
    tgt_ge[i]=torch.from_numpy(d['ge_128']).float()

# Generate VC mels for pairs (lazy: compute on first use)
vc_mel_cache={}
def get_vc_mel(src_idx,tgt_idx):
    key=(src_idx,tgt_idx)
    if key in vc_mel_cache: return vc_mel_cache[key]
    d_src=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,src_idx))
    audio_src=d_src['audio']; alen=len(audio_src)
    x_src=torch.from_numpy(audio_src[:SR*3]).float().unsqueeze(0)
    ge_tgt=tgt_ge[tgt_idx].unsqueeze(0)
    with torch.inference_mode():
        ft_src=teacher.encode(x_src,return_content=True,return_global=False)
        ce=ft_src.content_embedding
        wav=teacher.decode(global_embedding=ge_tgt.squeeze(0),
                          content_token_indices=ft_src.content_token_indices,
                          target_audio_length=alen)
    mel=extract_mel(wav)
    vc_mel_cache[key]=mel
    if len(vc_mel_cache)%50==0: print("  {} VC mels computed".format(len(vc_mel_cache)))
    return mel

# ── Training ───────────────────────────────────────────────────────────
decoder=CausalMelDecoder().to(device); decoder.train()
# Start from Stage 3a checkpoint
try: decoder.load_state_dict(torch.load("checkpoints/causal_mel_decoder.pt",map_location='cpu'))
except: print("Starting from scratch (no checkpoint)")

opt=AdamW(decoder.parameters(),lr=5e-4,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training with VC pair distillation...")
for epoch in range(EPOCHS):
    random.shuffle(pairs); tr_loss=0; nb=0
    for i in range(0,len(pairs),BATCH):
        batch=pairs[i:i+BATCH]
        ce_list=[]; ge_list=[]; mel_list=[]
        for p in batch:
            ce=src_ce[p['src']]; ge=tgt_ge[p['tgt']]
            mel=get_vc_mel(p['src'],p['tgt'])
            ce_list.append(ce); ge_list.append(ge); mel_list.append(mel.squeeze(0))
        
        # Pad
        max_Tc=max(ce.shape[0] for ce in ce_list)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_Tc-ce.shape[0])) for ce in ce_list]).to(device)
        ge_b=torch.stack(ge_list).to(device)
        
        max_Tm=max(mel.shape[1] for mel in mel_list)
        mel_b=torch.stack([F.pad(mel,(0,max_Tm-mel.shape[1])) for mel in mel_list]).to(device)
        
        mel_pred=decoder(ce_b,ge_b)
        Tp=min(mel_pred.shape[2],mel_b.shape[2])
        loss=F.l1_loss(mel_pred[:,:,:Tp],mel_b[:,:,:Tp])
        
        # Delta-mel loss
        if Tp>1:
            dp=mel_pred[:,:,1:Tp]-mel_pred[:,:,:Tp-1]
            dt=mel_b[:,:,1:Tp]-mel_b[:,:,:Tp-1]
            loss=loss+0.3*F.l1_loss(dp,dt)
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.0); opt.step()
        tr_loss+=loss.item(); nb+=1
    sched.step()
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f}".format(epoch,tr_loss/max(nb,1)))

os.makedirs("checkpoints",exist_ok=True)
torch.save(decoder.state_dict(),"checkpoints/causal_mel_decoder_vc.pt")

# ── Test: global sensitivity ──────────────────────────────────────────
print()
print("=== Global Sensitivity Test (p255→origin) ===")
decoder.eval()
for p in decoder.parameters(): p.requires_grad=False

import soundfile as sf, glob
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
files=sorted(glob.glob("{}/p255/p255_*_mic1.flac".format(ROOT)))
d_src,sr_s=sf.read(files[0])
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr_s!=SR: d_src=scipy_signal.resample(d_src,int(len(d_src)*SR/sr_s))
d_src=d_src[:SR*3]; alen=len(d_src)

d_tgt,sr_t=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr_t!=SR: d_tgt=scipy_signal.resample(d_tgt,int(len(d_tgt)*SR/sr_t))
d_tgt=d_tgt[:SR*3]

x_src=torch.from_numpy(d_src).float().unsqueeze(0)
x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)

with torch.inference_mode():
    ft_src=teacher.encode(x_src,return_content=True,return_global=True)
    ft_tgt=teacher.encode(x_tgt,return_content=False,return_global=True)
    ce_t=ft_src.content_embedding.unsqueeze(0)
    ge_src=ft_src.global_embedding.unsqueeze(0)
    ge_tgt=ft_tgt.global_embedding.unsqueeze(0)
    ge_wrong=ge_src  # use source as wrong
    ge_zero=torch.zeros(1,128)

    # Also get a random VCTK global
    d_other,_=sf.read(glob.glob("{}/p226/p226_*_mic1.flac".format(ROOT))[0])
    if d_other.ndim>1: d_other=d_other.mean(axis=1)
    d_other=scipy_signal.resample(d_other,int(len(d_other)*SR/sr_s))[:SR*3]
    x_other=torch.from_numpy(d_other).float().unsqueeze(0)
    ft_other=teacher.encode(x_other,return_content=False,return_global=True)
    ge_other=ft_other.global_embedding.unsqueeze(0)

    # Teacher VC mel as target
    wav_vc=teacher.decode(global_embedding=ge_tgt.squeeze(0),
                         content_token_indices=ft_src.content_token_indices,
                         target_audio_length=alen)
    mel_vc=extract_mel(wav_vc)

def run(ge,label):
    mel=decoder(ce_t,ge).squeeze(0)
    T=min(mel.shape[1],mel_vc.shape[2])
    cos=F.cosine_similarity(mel[:,:T].flatten(),mel_vc[:,:,:T].flatten(),dim=0).item()
    l1=F.l1_loss(mel[:,:T],mel_vc[:,:,:T]).item()
    print("  {}: Cos={:.4f} L1={:.4f}".format(label,cos,l1))
    return cos

c_tgt=run(ge_tgt,"tgt global")
c_src=run(ge_src,"src global")
c_other=run(ge_other,"other global")
c_zero=run(ge_zero,"zero global")

print()
print("Global ranking:")
ranking=sorted([("tgt",c_tgt),("src",c_src),("other",c_other),("zero",c_zero)],key=lambda x:-x[1])
for i,(name,cos) in enumerate(ranking):
    flag=" ★ TARGET WINS" if i==0 and name=="tgt" else " ⚠ wrong wins" if i==0 else ""
    print("  {}. {} Cos={:.4f}{}".format(i+1,name,cos,flag))

if ranking[0][0]=="tgt":
    print()
    print("GLOBAL CONDITIONING FIXED — target global produces best mel")
else:
    print()
    print("Global conditioning still weak — need more VC pairs")

print()
print("Done!")
