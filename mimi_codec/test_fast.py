"""Fast test: Improved Splitter + Converter (latent training + audio fine-tune)."""
import sys,os,time,random; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess
from moshi.models import loaders; from pathlib import Path
from scipy import signal
import numpy as np

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
STRIDE=1920

class ImpSplitter(nn.Module):
    def __init__(self,mimi):
        super().__init__(); self.mimi=mimi
        self.c_bn=nn.Sequential(nn.Conv1d(512,64,1),nn.GELU(),nn.Conv1d(64,512,1))
        self.s_net=nn.Sequential(nn.Conv1d(512,256,7,padding=3),nn.GELU(),nn.Conv1d(256,512,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(512,512))
    def get_z_q(self,audio):
        with torch.no_grad():
            z=self.mimi.encode_to_latent(audio,quantize=False); codes=self.mimi.quantizer.encode(z)
            return self.mimi.quantizer.decode(codes)
    def split(self,audio):
        with torch.no_grad(): enc=self.mimi.encoder(audio)
        h=enc.transpose(1,2); tt=self.mimi.encoder_transformer.transformer; shallow=[]; deep=[]
        for i,layer in enumerate(tt.layers):
            h=layer(h)
            if i in[0,1,2]: shallow.append(h)
            if i in[3,4,5,6,7]: deep.append(h)
        return torch.stack(shallow,0).mean(0).transpose(1,2), torch.stack(deep,0).mean(0).transpose(1,2)
    def forward(self,audio):
        fs,fd=self.split(audio); return fs+self.c_bn(fs), self.s_net(fd)

class ImpConverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Conv1d(512,512,4,stride=2,padding=1)
        self.gamma=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,512))
        self.beta=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,512))
        self.ref=nn.Conv1d(512,512,3,padding=1)
    def forward(self,c,s):
        cz=self.down(c); g=self.gamma(s).unsqueeze(-1); b=self.beta(s).unsqueeze(-1)
        m=cz.mean(2,keepdim=True); st=cz.std(2,keepdim=True)+1e-5
        return cz+self.ref((cz-m)/st*g+b)

def load(path,dur=2):
    data,sr=sf.read(path)
    if sr!=24000: data=signal.resample(data,int(len(data)*24000/sr),axis=0)
    L=dur*24000-(dur*24000%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

def load_any(path,dur=None):
    data,sr=sf.read(path)
    if sr!=24000: data=signal.resample(data,int(len(data)*24000/sr),axis=0)
    if dur is not None: L=dur*24000-(dur*24000%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Resemblyzer test
try:
    from resemblyzer import VoiceEncoder
    ve=VoiceEncoder()
    subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','16000','-ac','1','-sample_fmt','s16','/tmp/r16.wav'],capture_output=True)
    s16=load_any(f'{base}/p225/p225_001_mic1.flac').numpy(); s16=signal.resample(s16,int(len(s16)*16000/24000))
    t16=load_any('/tmp/r16.wav',dur=None).numpy()
    p16=load_any(f'{base}/p226/p226_001_mic1.flac').numpy(); p16=signal.resample(p16,int(len(p16)*16000/24000))
    es=ve.embed_utterance(s16); et=ve.embed_utterance(t16); ep=ve.embed_utterance(p16)
    print(f'Resemblyzer: src↔tgt={np.dot(es,et):.4f} src↔p226={np.dot(es,ep):.4f} tgt↔p226={np.dot(et,ep):.4f}')
except Exception as e: print(f'Resemblyzer: {e}')

# Train latent-only
sp=ImpSplitter(mimi); cv=ImpConverter()
opt=torch.optim.AdamW(list(sp.parameters())+list(cv.parameters()), lr=1e-3)
spks=['p225','p226','p227','p228','p229']; utts=['001','002','003']

zq={}
for s in spks:
    for u in utts:
        a=load(f'{base}/{s}/{s}_{u}_mic1.flac'); zq[(s,u)]=(a,sp.get_z_q(a))
print(f'Cached {len(zq)}')

tp=[]; skp=[]
for u in utts:
    sw=[s for s in spks if (s,u) in zq]
    for s in sw:
        for t in sw:
            if s!=t: tp.append((s,t,u))
for s in spks:
    uw=[u for u in utts if (s,u) in zq]
    for u1 in uw:
        for u2 in uw:
            if u1!=u2: skp.append((s,u1,u2))
random.shuffle(tp); random.shuffle(skp)
print(f'{len(tp)} text, {len(skp)} spk pairs')

print('Latent training...')
t0=time.time()
for step in range(150):
    random.shuffle(tp); random.shuffle(skp); lt=0
    for s,t,u in tp:
        a_s,zq_s=zq[(s,u)]; a_t,zq_t=zq[(t,u)]
        c_s,s_s=sp(a_s); c_t,s_t=sp(a_t)
        T=min(c_s.shape[2],c_t.shape[2])
        loss_c=F.mse_loss(c_s[:,:,:T],c_t[:,:,:T])+(1-F.cosine_similarity(c_s[:,:,:T].reshape(-1),c_t[:,:,:T].reshape(-1),dim=0))**2*0.5
        s_sim=F.cosine_similarity(s_s,s_t,dim=-1).mean()
        loss_s=torch.relu(s_sim-0.1)
        loss=F.mse_loss(cv(c_s[:,:,:T],s_t),zq_t[:,:,:T//2])+0.3*loss_c+0.3*loss_s
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(sp.parameters())+list(cv.parameters()),1.0); opt.step()
        lt+=loss.item()
    for s,u1,u2 in skp:
        a1,_=zq[(s,u1)]; a2,_=zq[(s,u2)]
        _,s1=sp(a1); _,s2=sp(a2)
        loss=(1-F.cosine_similarity(s1,s2,dim=-1).mean())**2
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sp.parameters(),1.0); opt.step()
    if step%30==0: print(f'  step {step:3d}: loss={lt/len(tp):.4f} s_sim={s_sim.item():.4f} [{time.time()-t0:.1f}s]')

print(f'Done [{time.time()-t0:.1f}s]')

# Quick audio fine-tune
print('Audio fine-tune...')
for step in range(10):
    random.shuffle(tp); lt=0
    for s,t,u in tp[:10]:
        a_s,zq_s=zq[(s,u)]; a_t,zq_t=zq[(t,u)]
        c_s,s_s=sp(a_s); _,s_t=sp(a_t)
        zvc=cv(c_s,s_t)
        zu=mimi._to_encoder_framerate(zvc)
        if mimi.decoder_transformer: (zt,)=mimi.decoder_transformer(zu)
        else: zt=zu
        av=mimi.decoder(zt)
        Ta=min(av.shape[2],a_s.shape[2],a_t.shape[2])
        loss=F.mse_loss(av[:,:,:Ta],a_t[:,:,:Ta])+0.2*F.mse_loss(zvc[:,:,:min(zvc.shape[2],zq_t.shape[2])],zq_t[:,:,:min(zvc.shape[2],zq_t.shape[2])])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(sp.parameters())+list(cv.parameters()),1.0); opt.step()
        lt+=loss.item()
    print(f'  step {step:3d}: audio_loss={lt/10:.4f}')

# Test
def conv(s,a,t_a):
    with torch.no_grad():
        c_s,_=sp(s); _,s_t=sp(t_a)
        zvc=cv(c_s,s_t); zu=mimi._to_encoder_framerate(zvc)
        if mimi.decoder_transformer: (zt,)=mimi.decoder_transformer(zu)
        else: zt=zu
        return mimi.decoder(zt)

subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tz.wav'],capture_output=True)
src2=load_any(f'{base}/p225/p225_001_mic1.flac').unsqueeze(0).unsqueeze(0)
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac').unsqueeze(0).unsqueeze(0)
tgt_c=load_any('/tmp/tz.wav',dur=None).unsqueeze(0).unsqueeze(0)

for name,t in[('parallel',tgt_p),('cross',tgt_c)]:
    va=conv(src2,t); Tc=min(va.shape[2],src2.shape[2],t.shape[2])
    zv=mimi.encode_to_latent(va[:,:,:Tc],quantize=False)
    zs=mimi.encode_to_latent(src2[:,:,:Tc],quantize=False)
    zt=mimi.encode_to_latent(t[:,:,:Tc],quantize=False)
    T2=min(zv.shape[2],zs.shape[2],zt.shape[2])
    cs=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zs[:,:,:T2].reshape(-1),dim=0)
    ct=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zt[:,:,:T2].reshape(-1),dim=0)
    print(f'{name}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'/Users/asill/research5/mimi_imp_{name}.wav',va[0,0,:Tc].numpy(),24000)
print('✅')
