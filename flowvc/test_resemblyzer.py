"""Resemblyzer + Encoder Kanade Splitter → Q-Space Converter VC."""
import sys,os,time,random; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess
from moshi.models import loaders; from pathlib import Path
from scipy import signal
import numpy as np
from resemblyzer import VoiceEncoder

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
STRIDE=1920; SR=24000

# Resemblyzer for text-independent speaker embedding
ve = VoiceEncoder()

def get_spk_emb(audio_np):
    """Extract Resemblyzer speaker embedding (256-dim)."""
    # Resample to 16kHz for Resemblyzer
    if len(audio_np) > 16000*10:
        audio_np = audio_np[:16000*10]
    audio_16k = signal.resample(audio_np, int(len(audio_np)*16000/SR))
    return torch.from_numpy(ve.embed_utterance(audio_16k.astype(np.float32))).float()

# Content extractor (Encoder Kanade)
class ContentExtractor(nn.Module):
    def __init__(self, mimi):
        super().__init__(); self.mimi=mimi
        self.c_bn=nn.Sequential(nn.Conv1d(512,64,1),nn.GELU(),nn.Conv1d(64,512,1))
    def get_z_q(self, audio):
        with torch.no_grad():
            z=self.mimi.encode_to_latent(audio,quantize=False)
            codes=self.mimi.quantizer.encode(z)
            return self.mimi.quantizer.decode(codes)
    def extract(self, audio):
        with torch.no_grad(): enc=self.mimi.encoder(audio)
        h=enc.transpose(1,2); tt=self.mimi.encoder_transformer.transformer; shallow=[]
        for i,layer in enumerate(tt.layers):
            h=layer(h)
            if i in[0,1,2]: shallow.append(h)
        f=torch.stack(shallow,0).mean(0).transpose(1,2)
        return f+self.c_bn(f)  # (B,512,T_enc)

# Converter (Resemblyzer spk → Q-Space)
class ResemblyzerConverter(nn.Module):
    def __init__(self, spk_dim=256, z_dim=512):
        super().__init__()
        self.down=nn.Conv1d(z_dim,z_dim,4,stride=2,padding=1)
        self.spk_proj=nn.Sequential(nn.Linear(spk_dim,z_dim),nn.GELU(),nn.Linear(z_dim,z_dim))
        self.gamma=nn.Sequential(nn.Linear(z_dim,z_dim),nn.GELU(),nn.Linear(z_dim,z_dim))
        self.beta=nn.Sequential(nn.Linear(z_dim,z_dim),nn.GELU(),nn.Linear(z_dim,z_dim))
        self.ref=nn.Conv1d(z_dim,z_dim,3,padding=1)
    def forward(self, c_src, s_tgt):
        cz=self.down(c_src); sp=self.spk_proj(s_tgt)
        g=self.gamma(sp).unsqueeze(-1); b=self.beta(sp).unsqueeze(-1)
        m=cz.mean(2,keepdim=True); st=cz.std(2,keepdim=True)+1e-5
        return cz+self.ref((cz-m)/st*g+b)

def load(path,dur=2):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float()

def load_any(path,dur=None):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    if dur is not None: L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float()

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Pre-extract speaker embeddings for all training speakers
spks=['p225','p226','p227','p228','p229']; utts=['001','002','003']
print('Extracting Resemblyzer speaker embeddings...')
spk_embs={}
for s in spks:
    # Use first 3 utterances to build speaker profile
    embs=[]
    for u in utts[:3]:
        a=load_any(f'{base}/{s}/{s}_{u}_mic1.flac').numpy()
        embs.append(get_spk_emb(a))
    spk_embs[s]=torch.stack(embs).mean(0)  # average across utterances
    print(f'  {s}: dim={spk_embs[s].shape[0]}')

# Check speaker separation
print(f'Speaker cosines:')
for s1 in spks:
    for s2 in spks:
        if s1<s2:
            cos=F.cosine_similarity(spk_embs[s1].unsqueeze(0),spk_embs[s2].unsqueeze(0),dim=-1).item()
            print(f'  {s1}↔{s2}: {cos:.4f}')

# Train converter
ce=ContentExtractor(mimi); cv=ResemblyzerConverter()
opt=torch.optim.AdamW(list(ce.parameters())+list(cv.parameters()), lr=1e-3)

# Cache z_q for training
zq_cache={}
for s in spks:
    for u in utts:
        a=load(f'{base}/{s}/{s}_{u}_mic1.flac').unsqueeze(0).unsqueeze(0)
        zq_cache[(s,u)]=ce.get_z_q(a)
print(f'Cached {len(zq_cache)} z_q')

# Training pairs: any source-target combination
pairs=[]
for s in spks:
    for t in spks:
        if s==t: continue
        for u in utts:
            if (s,u) in zq_cache and (t,u) in zq_cache:
                pairs.append((s,t,u))
random.shuffle(pairs)
print(f'{len(pairs)} pairs')

print('Training converter (latent only, fast)...')
t0=time.time()
for step in range(100):
    random.shuffle(pairs); lt=0
    for s,t,u in pairs:
        a_s=load(f'{base}/{s}/{s}_{u}_mic1.flac').unsqueeze(0).unsqueeze(0)
        c_s=ce.extract(a_s)
        s_tgt=spk_embs[t].unsqueeze(0).to(c_s.device)
        zq_tgt=zq_cache[(t,u)]
        
        zvc=cv(c_s, s_tgt)
        T=min(zvc.shape[2], zq_tgt.shape[2])
        loss=F.mse_loss(zvc[:,:,:T], zq_tgt[:,:,:T])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ce.parameters())+list(cv.parameters()),1.0)
        opt.step()
        lt+=loss.item()
    if step%20==0: print(f'  step {step:3d}: loss={lt/len(pairs):.4f} [{time.time()-t0:.1f}s]')

print(f'Done [{time.time()-t0:.1f}s]')

# Test
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tw.wav'],capture_output=True)

def convert(src_a, tgt_a_np):
    with torch.no_grad():
        c_src=ce.extract(src_a)
        s_tgt=get_spk_emb(tgt_a_np.numpy()).unsqueeze(0)
        zvc=cv(c_src, s_tgt)
        zu=mimi._to_encoder_framerate(zvc)
        if mimi.decoder_transformer: (zt,)=mimi.decoder_transformer(zu)
        else: zt=zu
        return mimi.decoder(zt)

src_a=load_any(f'{base}/p225/p225_001_mic1.flac').unsqueeze(0).unsqueeze(0)
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac')
tgt_c=load_any('/tmp/tw.wav',dur=None)

out='/Users/asill/research5'
for name,ta in[('parallel',tgt_p),('cross',tgt_c)]:
    va=convert(src_a, ta)
    Tc=min(va.shape[2],src_a.shape[2],ta.unsqueeze(0).unsqueeze(0).shape[2])
    zv=mimi.encode_to_latent(va[:,:,:Tc],quantize=False)
    zs=mimi.encode_to_latent(src_a[:,:,:Tc],quantize=False)
    zt=mimi.encode_to_latent(ta.unsqueeze(0).unsqueeze(0)[:,:,:Tc],quantize=False)
    T2=min(zv.shape[2],zs.shape[2],zt.shape[2])
    cs=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zs[:,:,:T2].reshape(-1),dim=0)
    ct=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zt[:,:,:T2].reshape(-1),dim=0)
    print(f'{name}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'{out}/mimi_res_{name}.wav',va[0,0,:Tc].numpy(),SR)
print(f'✅ {out}/mimi_res_*.wav')
