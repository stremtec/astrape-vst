#!/usr/bin/env python3
"""
Stage 3b: Diagnostic — student content + global → mel decoder.
Compares A/B/C/D paths to isolate failure mode.
Key question: does global conditioning actually change the mel?
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, numpy as np, os, warnings, soundfile as sf
from scipy import signal as scipy_signal
warnings.filterwarnings('ignore')

SR=44100; N_MELS=80; MEL_HOP=int(SR/25)
device=torch.device('cpu')

# ── Content Student ────────────────────────────────────────────────────
class CausalTCNEncoder(nn.Module):
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
        self.proj_out=nn.Conv1d(hidden,out_dim,1)
        self.embed_head=nn.Conv1d(out_dim,768,1)
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers:
            r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h); fsq=self.proj_out(h); embed=self.embed_head(fsq)
        return fsq, embed

# ── Mel Decoder ────────────────────────────────────────────────────────
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

# ── Load models ────────────────────────────────────────────────────────
content_stu=CausalTCNEncoder()
content_stu.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'))
content_stu.eval()

mel_dec=CausalMelDecoder()
mel_dec.load_state_dict(torch.load("checkpoints/causal_mel_decoder.pt",map_location='cpu'))
mel_dec.eval()
for p in mel_dec.parameters(): p.requires_grad=False
content_stu.eval()
for p in content_stu.parameters(): p.requires_grad=False

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
teacher.eval()

mel_ext=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=1024,hop_length=MEL_HOP,n_mels=N_MELS,f_min=80,f_max=14000,center=False,power=1)

# ── Test pairs ─────────────────────────────────────────────────────────
import glob
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

src_spk='p255'; tgt_spk='origin'
files=sorted(glob.glob("{}/{}/{}_*_mic1.flac".format(ROOT,src_spk,src_spk)))
d_src,sr_s=sf.read(files[0])
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr_s!=SR: d_src=scipy_signal.resample(d_src,int(len(d_src)*SR/sr_s))
d_src=d_src[:SR*3]; alen=len(d_src)

d_tgt,sr_t=sf.read("/Users/asill/Downloads/{}.mp3".format(tgt_spk))
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr_t!=SR: d_tgt=scipy_signal.resample(d_tgt,int(len(d_tgt)*SR/sr_t))
d_tgt=d_tgt[:SR*3]

x_src=torch.from_numpy(d_src).float().unsqueeze(0)
x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)

# Extract teacher features
with torch.inference_mode():
    ft_src=teacher.encode(x_src,return_content=True,return_global=True)
    ft_tgt=teacher.encode(x_tgt,return_content=False,return_global=True)
    ce_t=ft_src.content_embedding.unsqueeze(0)  # (1,T,768)
    ge_src=ft_src.global_embedding.unsqueeze(0)
    ge_tgt=ft_tgt.global_embedding.unsqueeze(0)

# Extract student content via FSQ path
audio_16k=scipy_signal.resample(d_src[:alen],int(alen*16000/SR))
mel_in=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)
mel=mel_in(torch.from_numpy(audio_16k).float().view(1,1,-1)).squeeze(1)
logmel=torch.log(mel.clamp(min=1e-5))

with torch.inference_mode():
    fsq_pred,_=content_stu(logmel)  # (1,5,T)
    fsq_stu=fsq_pred.squeeze(0).T  # (T,5)
    # Teacher FSQ quantize → proj_out
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_stu.unsqueeze(0))
    ce_s_fsq=teacher.local_quantizer.proj_out(z_q)  # (1,T,768)

# Target mel (teacher self-recon)
wav_t=teacher.decode(global_embedding=ge_tgt.squeeze(0),content_token_indices=ft_src.content_token_indices,target_audio_length=alen)
mel_tgt=mel_ext(wav_t.unsqueeze(0).unsqueeze(0)); mel_tgt=torch.log(mel_tgt.squeeze(1).clamp(min=1e-5)).squeeze(0)

# ── 4-way comparison ──────────────────────────────────────────────────
print("="*80)
print("  STAGE 3b: STUDENT CONTENT + GLOBAL → MEL DECODER")
print("="*80)
print("  {}→{}".format(src_spk,tgt_spk))
print()

def run_mel(ce,ge,use_teacher_dec=False):
    ce_t=ce.transpose(1,2) if ce.shape[1]==768 and ce.shape[2]!=768 else ce
    if use_teacher_dec:
        # Need tokens: quantize through teacher FSQ
        # For teacher CE: already 768d continuous, can't get tokens easily
        # Use teacher decode path for waveform, then mel
        return None  # skip for now
    else:
        return mel_dec(ce_t,ge).squeeze(0)

def metrics(pred,target):
    T=min(pred.shape[1],target.shape[1]); p=pred[:,:T]; t=target[:,:T]
    l1=F.l1_loss(p,t).item()
    cos=F.cosine_similarity(p.flatten(),t.flatten(),dim=0).item()
    return l1,cos

# A: teacher content + target global → causal decoder
mel_A=run_mel(ce_t,ge_tgt)
# B: student content (FSQ path) + target global → causal decoder
mel_B=run_mel(ce_s_fsq,ge_tgt)
# C: student content (FSQ path) + source global (same-speaker reference)
mel_C=run_mel(ce_s_fsq,ge_src)
# D: student content (FSQ path) + zero global (no speaker)
mel_D=run_mel(ce_s_fsq,torch.zeros_like(ge_src))

# Also: teacher content + source global
mel_A2=run_mel(ce_t,ge_src)

results={}
for name,mel in [("A: teaC+tgtG",mel_A),("B: stuC+tgtG",mel_B),
                  ("C: stuC+srcG",mel_C),("D: stuC+zeroG",mel_D),
                  ("A2: teaC+srcG",mel_A2)]:
    l1,cos=metrics(mel,mel_tgt)
    results[name]={'l1':l1,'cos':cos}
    print("  {:<16s} L1={:.4f} Cos={:.4f}".format(name,l1,cos))

# ── Global sensitivity ────────────────────────────────────────────────
print()
print("--- Global Sensitivity (same content, different global) ---")
# Compare mel from different globals with same content
for label,(m1,m2) in [("tgt vs src (teaC)",(mel_A,mel_A2)),
                        ("tgt vs src (stuC)",(mel_B,mel_C)),
                        ("tgt vs zero (stuC)",(mel_B,mel_D))]:
    T=min(m1.shape[1],m2.shape[1])
    cos=F.cosine_similarity(m1[:,:T].flatten(),m2[:,:T].flatten(),dim=0).item()
    l1=F.l1_loss(m1[:,:T],m2[:,:T]).item()
    print("  {}: L1-diff={:.4f} Cos={:.4f}".format(label,l1,cos))
    if cos<0.98: print("    → GLOBAL HAS EFFECT (different mels)")
    else: print("    → GLOBAL NEGLIGIBLE (nearly identical mels)")

# ── Student vs Teacher content quality ────────────────────────────────
print()
print("--- Content Quality (same global, different content) ---")
T=min(mel_A.shape[1],mel_B.shape[1])
cos_AB=F.cosine_similarity(mel_A[:,:T].flatten(),mel_B[:,:T].flatten(),dim=0).item()
l1_AB=F.l1_loss(mel_A[:,:T],mel_B[:,:T]).item()
print("  teaC vs stuC (both tgtG): L1={:.4f} Cos={:.4f}".format(l1_AB,cos_AB))
print("  → student content preserves {:.1f}% of teacher mel quality".format(cos_AB/0.93*100))

print()
print("Done!")
