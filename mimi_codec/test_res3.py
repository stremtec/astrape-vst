"""Resemblyzer VC - fixed + fast."""
import sys,os,random; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal
import numpy as np
from resemblyzer import VoiceEncoder

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
SR=24000; STRIDE=1920; ve=VoiceEncoder()

def get_spk(audio_np):
    a=audio_np[:16000*10]; a16=signal.resample(a,int(len(a)*16000/SR))
    return torch.from_numpy(ve.embed_utterance(a16.astype(np.float32))).float()

class CE(nn.Module):
    def __init__(self,mimi):
        super().__init__(); self.mimi=mimi
        self.c_bn=nn.Sequential(nn.Conv1d(512,64,1),nn.GELU(),nn.Conv1d(64,512,1))
    def extract(self,a):
        with torch.no_grad(): enc=self.mimi.encoder(a)
        h=enc.transpose(1,2); tt=self.mimi.encoder_transformer.transformer; sh=[]
        for i,l in enumerate(tt.layers):
            h=l(h)
            if i in[0,1,2]: sh.append(h)
        f=torch.stack(sh,0).mean(0).transpose(1,2)
        return f+self.c_bn(f)

class CV(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Conv1d(512,512,kernel_size=4,stride=2,padding=1)
        self.sp=nn.Linear(256,512)
        self.ref=nn.Conv1d(512,512,3,padding=1)
    def forward(self,c,s):
        if c.shape[2]%2!=0: c=F.pad(c,(0,1))
        cz=self.down(c)
        sp=self.sp(s).unsqueeze(-1)
        h=cz+sp.expand(-1,-1,cz.shape[2])
        return cz+self.ref(h)

def load_any(path,dur=None):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    if dur is not None: L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float()

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks=['p225','p226','p227']; utts=['001','002']

# Speaker embeddings (Resemblyzer)
spk_e={}
for s in spks:
    es=[]
    for u in utts:
        a=load_any(f'{base}/{s}/{s}_{u}_mic1.flac').numpy()
        es.append(get_spk(a))
    spk_e[s]=torch.stack(es).mean(0)

# Cache z_q
ce=CE(mimi); cv=CV()
zq={}
for s in spks:
    for u in utts:
        a=load_any(f'{base}/{s}/{s}_{u}_mic1.flac').unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z=mimi.encode_to_latent(a,quantize=False)
            c=mimi.quantizer.encode(z)
            zq[(s,u)]=mimi.quantizer.decode(c)

pairs=[(s,t,u) for s in spks for t in spks for u in utts if s!=t]
random.shuffle(pairs)
opt=torch.optim.AdamW(list(ce.parameters())+list(cv.parameters()), lr=1e-3)

print(f'Training {len(pairs)} pairs...')
t0=time.time()
for step in range(5):
    random.shuffle(pairs); lt=0
    for s,t,u in pairs:
        a_s=load_any(f'{base}/{s}/{s}_{u}_mic1.flac').unsqueeze(0).unsqueeze(0)
        c_s=ce.extract(a_s); s_t=spk_e[t].unsqueeze(0); z_t=zq[(t,u)]
        zvc=cv(c_s,s_t); Tq=z_t.shape[2]; T=min(zvc.shape[2],z_t.shape[2])
        loss=F.mse_loss(zvc[:,:,:T],z_t[:,:,:T])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ce.parameters())+list(cv.parameters()),1.0)
        opt.step()
        lt+=loss.item()
    print(f'  step {step}: loss={lt/len(pairs):.4f} [{time.time()-t0:.1f}s]')
print(f'Done [{time.time()-t0:.1f}s]')

# Test
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tz5.wav'],capture_output=True)
src_a=load_any(f'{base}/p225/p225_001_mic1.flac').unsqueeze(0).unsqueeze(0)
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac')
tgt_c=load_any('/tmp/tz5.wav',dur=None)

out='/Users/asill/research5'
for nm,ta in[('parallel',tgt_p),('cross',tgt_c)]:
    with torch.no_grad():
        cs=ce.extract(src_a); st=get_spk(ta.numpy()).unsqueeze(0)
        zv=cv(cs,st); zu=mimi._to_encoder_framerate(zv)
        if mimi.decoder_transformer: (zt,)=mimi.decoder_transformer(zu)
        else: zt=zu
        va=mimi.decoder(zt)
    Tc=min(va.shape[2],src_a.shape[2],ta.shape[0])
    zv2=mimi.encode_to_latent(va[:,:,:Tc],quantize=False)
    zs=mimi.encode_to_latent(src_a[:,:,:Tc],quantize=False)
    zt2=mimi.encode_to_latent(ta.unsqueeze(0).unsqueeze(0)[:,:,:Tc],quantize=False)
    T2=min(zv2.shape[2],zs.shape[2],zt2.shape[2])
    cs2=F.cosine_similarity(zv2[:,:,:T2].reshape(-1),zs[:,:,:T2].reshape(-1),dim=0)
    ct2=F.cosine_similarity(zv2[:,:,:T2].reshape(-1),zt2[:,:,:T2].reshape(-1),dim=0)
    print(f'{nm}: cos_src={cs2:.4f} cos_tgt={ct2:.4f} Δ={ct2-cs2:+.4f}')
    sf.write(f'{out}/mimi_res_{nm}.wav',va[0,0,:Tc].numpy(),SR)
print('✅')
