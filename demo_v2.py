#!/usr/bin/env python3
"""V2 Demo: Transformer content student → decoder → waveform."""
import torch, numpy as np, soundfile as sf, glob, torchaudio, math, os
from scipy import signal as scipy_signal
SR=44100
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_len=2000):
        super().__init__(); pe=torch.zeros(max_len,dim)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): T=x.size(1); return x+self.pe[:,:T,:].contiguous()
class CausalTransformerBlock(nn.Module):
    def __init__(self,dim=384,n_heads=6,ff_mult=4,dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.norm2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn=self.norm1(x); a=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]; x=x+a
        xn=self.norm2(x); x=x+self.ff(xn); return x
class ContentStudentV2(nn.Module):
    def __init__(self,in_dim=80,hidden=384,n_layers=6,n_heads=6,out_dim=5,kernel=5):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(in_dim,hidden,kernel,padding=kernel//2),nn.GELU(),nn.Conv1d(hidden,hidden,kernel,padding=kernel//2),nn.GELU())
        self.pos_enc=PositionalEncoding(hidden)
        self.blocks=nn.ModuleList([CausalTransformerBlock(hidden,n_heads) for _ in range(n_layers)])
        self.norm=nn.LayerNorm(hidden); self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1)
        self.content_head=nn.Conv1d(hidden,768,1)
    def forward(self,x):
        h=self.stem(x); h=h.transpose(1,2); h=self.pos_enc(h)
        for block in self.blocks: h=block(h)
        h=self.norm(h).transpose(1,2); h=self.down(h)
        return self.content_head(h)

# Load
stu=ContentStudentV2(hidden=256,n_layers=4,n_heads=4)
stu.load_state_dict(torch.load('checkpoints/causal_student_v2.pt',map_location='cpu'),strict=False); stu.eval()
for p in stu.parameters(): p.requires_grad=False

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

ROOT='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
OUT='/Users/asill/Desktop/mio_vc_demo_v2'
os.makedirs(OUT,exist_ok=True)

mel_in=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)

# Origin target
d_origin,sr_o=sf.read('/Users/asill/Downloads/origin.mp3')
if d_origin.ndim>1: d_origin=d_origin.mean(axis=1)
if sr_o!=SR: d_origin=scipy_signal.resample(d_origin,int(len(d_origin)*SR/sr_o))
sf.write(f'{OUT}/00_origin_target.wav',d_origin[:SR*5],SR)

male_spks=['p255','p256','p258','p260','p262','p265','p270']
for spk in male_spks:
    files=glob.glob(f'{ROOT}/{spk}/{spk}_*_mic1.flac'); 
    if not files: continue
    d_src,sr=sf.read(files[0])
    if d_src.ndim>1: d_src=d_src.mean(axis=1)
    if sr!=SR: d_src=scipy_signal.resample(d_src,int(len(d_src)*SR/sr))
    d_src=d_src[:SR*3]; alen=len(d_src)
    x_s=torch.from_numpy(d_src).float().unsqueeze(0)
    x_o=torch.from_numpy(d_origin[:SR*3]).float().unsqueeze(0)
    
    with torch.inference_mode():
        fs=teacher.encode(x_s,return_content=True,return_global=True)
        fo=teacher.encode(x_o,return_content=False,return_global=True)
        ge_o=fo.global_embedding
        
        # Teacher VC
        w_t=teacher.decode(global_embedding=ge_o,content_token_indices=fs.content_token_indices,target_audio_length=alen)
        
        # v2 Student VC
        a16=scipy_signal.resample(d_src[:alen],int(alen*16000/SR))
        mel=mel_in(torch.from_numpy(a16).float().view(1,1,-1))
        lm=torch.log(mel.squeeze(1).clamp(min=1e-5))  # (1,80,T) — correct for Conv1d
        ce_v2=stu(lm)  # (1,768,T)
        w_s=teacher.decode(global_embedding=ge_o,content_embedding=ce_v2.squeeze(0).T,target_audio_length=alen)
    
    sf.write(f'{OUT}/{spk}_01_original.wav',d_src,SR)
    sf.write(f'{OUT}/{spk}_02_teacher_vc.wav',w_t.numpy()[:alen],SR)
    sf.write(f'{OUT}/{spk}_03_v2_student.wav',w_s.numpy()[:alen],SR)
    print(f'{spk} done')

# Also generate with v1 hard FSQ for comparison
class StudentV1(nn.Module):
    def __init__(self,in_dim=80,hidden=256,out_dim=5,num_layers=4,kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p),nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1)
        self.proj_out=nn.Conv1d(hidden,out_dim,1)
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers: r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h); return self.proj_out(h)

stu_v1=StudentV1(); stu_v1.load_state_dict(torch.load('checkpoints/causal_student_v1.pt',map_location='cpu'),strict=False); stu_v1.eval()
for p in stu_v1.parameters(): p.requires_grad=False

for spk in ['p255','p256']:
    files=glob.glob(f'{ROOT}/{spk}/{spk}_*_mic1.flac')
    d_src,sr=sf.read(files[0])
    if d_src.ndim>1: d_src=d_src.mean(axis=1)
    if sr!=SR: d_src=scipy_signal.resample(d_src,int(len(d_src)*SR/sr))
    d_src=d_src[:SR*3]; alen=len(d_src)
    x_s=torch.from_numpy(d_src).float().unsqueeze(0)
    with torch.inference_mode():
        fs=teacher.encode(x_s,return_content=True,return_global=True)
        fo=teacher.encode(torch.from_numpy(d_origin[:SR*3]).float().unsqueeze(0),return_content=False,return_global=True)
        a16=scipy_signal.resample(d_src[:alen],int(alen*16000/SR))
        mel2=mel_in(torch.from_numpy(a16).float().view(1,1,-1))
        lm2=torch.log(mel2.squeeze(1).clamp(min=1e-5))  # (1,80,T)
        z5=stu_v1(lm2).squeeze(0).T
        zq,_=teacher.local_quantizer.fsq.encode(z5.unsqueeze(0))
        ce_v1=teacher.local_quantizer.proj_out(zq)
        w_v1=teacher.decode(global_embedding=fo.global_embedding,content_embedding=ce_v1.squeeze(0),target_audio_length=alen)
    sf.write(f'{OUT}/{spk}_04_v1_hard.wav',w_v1.numpy()[:alen],SR)
    print(f'{spk} v1 comparison done')

print()
print('Done! Files in ' + OUT + '/')
print('  _01 = original | _02 = teacher VC | _03 = v2 student | _04 = v1 hard (for p255/256)')
