"""Improved Splitter + Resemblyzer speaker embedding test."""
import sys; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time, random, os
from moshi.models import loaders; from pathlib import Path
from scipy import signal
import numpy as np

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

STRIDE=1920

# ====== IMPROVED SPLITTER ======
class ImprovedSplitter(nn.Module):
    """Content: enc layers 0-2 (shallow). Speaker: enc layers 3-7 (deep)."""
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        # Content bottleneck (64-dim)
        self.c_bn = nn.Sequential(
            nn.Conv1d(512, 64, 1), nn.GELU(),
            nn.Conv1d(64, 512, 1),
        )
        # Speaker: larger network with contrastive head
        self.s_conv = nn.Sequential(
            nn.Conv1d(512, 256, 7, padding=3), nn.GELU(),
            nn.Conv1d(256, 512, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.s_proj = nn.Linear(512, 256)  # projection for contrastive loss
        
    def get_z_q(self, audio):
        with torch.no_grad():
            z=self.mimi.encode_to_latent(audio,quantize=False)
            codes=self.mimi.quantizer.encode(z)
            return self.mimi.quantizer.decode(codes)
    
    def split(self, audio):
        with torch.no_grad():
            enc=self.mimi.encoder(audio)
        h=enc.transpose(1,2)
        tt=self.mimi.encoder_transformer.transformer
        shallow=[]; deep=[]
        for i,layer in enumerate(tt.layers):
            h=layer(h)
            if i in[0,1,2]: shallow.append(h)
            if i in[3,4,5,6,7]: deep.append(h)
        f_s=torch.stack(shallow,0).mean(0).transpose(1,2)
        f_d=torch.stack(deep,0).mean(0).transpose(1,2)
        return f_s,f_d
    
    def forward(self, audio):
        f_s,f_d=self.split(audio)
        c=f_s+self.c_bn(f_s)              # (B, 512, T_enc)
        s_raw=self.s_conv(f_d)            # (B, 512)
        s=self.s_proj(s_raw)              # (B, 256) for contrastive
        return c, s, s_raw

# ====== CONVERTER (same as before) ======
class CleanConverter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.downsample=nn.Conv1d(dim,dim,4,stride=2,padding=1)
        self.gamma=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.beta=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.refine=nn.Conv1d(dim,dim,3,padding=1)
    
    def forward(self, c_src, s_tgt):
        c_zq=self.downsample(c_src)
        gamma=self.gamma(s_tgt).unsqueeze(-1)
        beta=self.beta(s_tgt).unsqueeze(-1)
        mean=c_zq.mean(dim=2,keepdim=True)
        std=c_zq.std(dim=2,keepdim=True)+1e-5
        c_norm=(c_zq-mean)/std
        return c_zq+self.refine(c_norm*gamma+beta)

# ====== RESEMBLYZER SPEAKER EMBEDDING ======
try:
    from resemblyzer import VoiceEncoder
    voice_encoder = VoiceEncoder()
    print('Resemblyzer loaded')
    HAS_RESEMBLYZER = True
except:
    print('Resemblyzer not available')
    HAS_RESEMBLYZER = False

def load(path, dur=2):
    data,sr=sf.read(path)
    if sr!=24000: data=signal.resample(data,int(len(data)*24000/sr),axis=0)
    L=dur*24000-(dur*24000%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float()

def load_any(path, dur=None):
    data,sr=sf.read(path)
    if sr!=24000: data=signal.resample(data,int(len(data)*24000/sr),axis=0)
    if dur is not None: L=dur*24000-(dur*24000%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float()

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Test Resemblyzer on origin.mp3
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','16000','-ac','1','-sample_fmt','s16','/tmp/tr_16k.wav'],capture_output=True)

if HAS_RESEMBLYZER:
    # Test speaker similarity
    src16=load_any(f'{base}/p225/p225_001_mic1.flac'); src16_wav = signal.resample(src16.numpy(), int(len(src16)*16000/24000))
    tgt16=load_any('/tmp/tr_16k.wav',dur=None)
    p226_16=load_any(f'{base}/p226/p226_001_mic1.flac'); p226_wav = signal.resample(p226_16.numpy(), int(len(p226_16)*16000/24000))
    
    e_src=voice_encoder.embed_utterance(src16_wav)
    e_tgt=voice_encoder.embed_utterance(tgt16.numpy())
    e_p226=voice_encoder.embed_utterance(p226_wav)
    
    cos_st=np.dot(e_src,e_tgt)
    cos_s6=np.dot(e_src,e_p226)
    cos_t6=np.dot(e_tgt,e_p226)
    print(f'Resemblyzer: src↔tgt={cos_st:.4f} src↔p226={cos_s6:.4f} tgt↔p226={cos_t6:.4f}')
    print(f'  → different speakers ARE different ✅' if cos_st<0.5 else '  → poor separation')

# ====== TRAINING ======
print('\
Training improved splitter + converter...')
splitter=ImprovedSplitter(mimi)
converter=CleanConverter()
opt=torch.optim.AdamW(list(splitter.parameters())+list(converter.parameters()), lr=8e-4)

# Multi-speaker training
spks=['p225','p226','p227','p228','p229']; utts=['001','002','003']

# Pre-cache
zq_cache={}
with torch.no_grad():
    for s in spks:
        for u in utts:
            f=f'{base}/{s}/{s}_{u}_mic1.flac'
            if not os.path.isfile(f): continue
            zq_cache[(s,u)]=splitter.get_z_q(load_any(f).unsqueeze(0).unsqueeze(0))
            # Also cache audio
            zq_cache[(s,u,'audio')]=load_any(f).unsqueeze(0).unsqueeze(0)

# Pairs: same-text (content match) + same-spk (speaker match)
text_pairs=[]; spk_pairs=[]
for u in utts:
    sw=[s for s in spks if (s,u) in zq_cache]
    for s in sw:
        for t in sw:
            if s!=t: text_pairs.append((s,t,u))

for s in spks:
    uw=[u for u in utts if (s,u) in zq_cache]
    for u1 in uw:
        for u2 in uw:
            if u1!=u2: spk_pairs.append((s,u1,u2))

random.shuffle(text_pairs); random.shuffle(spk_pairs)
print(f'Text pairs: {len(text_pairs)}, Spk pairs: {len(spk_pairs)}')

t0=time.time()
for step in range(150):
    random.shuffle(text_pairs); random.shuffle(spk_pairs)
    lt=0; lc=0; ls=0; n=0
    
    for s,t,u in text_pairs[:30]:  # Limit per step for speed
        src_a=zq_cache[(s,u,'audio')]; tgt_a=zq_cache[(t,u,'audio')]
        c_src,s_src,s_src_raw=splitter(src_a)
        c_tgt,s_tgt,s_tgt_raw=splitter(tgt_a)
        zq_tgt=splitter.get_z_q(tgt_a)
        
        # Content: same text → similar
        T_enc=min(c_src.shape[2],c_tgt.shape[2])
        loss_c=F.mse_loss(c_src[:,:,:T_enc],c_tgt[:,:,:T_enc])
        c_cos=F.cosine_similarity(c_src[:,:,:T_enc].reshape(-1),c_tgt[:,:,:T_enc].reshape(-1),dim=0)
        loss_c+=(1-c_cos)**2*0.5
        
        # Speaker: different speakers → push apart (contrastive)
        s_sim=F.cosine_similarity(s_src,s_tgt,dim=-1).mean()
        loss_s=torch.relu(s_sim-0.1)
        
        # Converter
        zq_vc=converter(c_src[:,:,:T_enc],s_tgt_raw)
        zq_up=mimi._to_encoder_framerate(zq_vc)
        if mimi.decoder_transformer: (z_tr,)=mimi.decoder_transformer(zq_up)
        else: z_tr=zq_up
        audio_vc=mimi.decoder(z_tr)
        T_a=min(audio_vc.shape[2],src_a.shape[2],tgt_a.shape[2])
        loss_r=F.mse_loss(audio_vc[:,:,:T_a],tgt_a[:,:,:T_a])
        
        T_z=min(zq_vc.shape[2],zq_tgt.shape[2])
        loss_z=0.1*F.mse_loss(zq_vc[:,:,:T_z],zq_tgt[:,:,:T_z])
        
        loss=loss_r+0.5*loss_c+0.5*loss_s+loss_z
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(splitter.parameters())+list(converter.parameters()),1.0)
        opt.step()
        lt+=loss.item(); lc+=loss_c.item(); ls+=loss_s.item(); n+=1
    
    # Speaker consistency: same speaker → similar
    for s,u1,u2 in spk_pairs[:15]:
        a1=zq_cache[(s,u1,'audio')]; a2=zq_cache[(s,u2,'audio')]
        _,s1,_=splitter(a1); _,s2,_=splitter(a2)
        loss_spk=(1-F.cosine_similarity(s1,s2,dim=-1).mean())**2
        opt.zero_grad(); loss_spk.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(),1.0)
        opt.step()
    
    if step%30==0:
        print(f'  step {step:3d}: recon={lt/n:.4f} content={lc/n:.4f} spk={ls/n:.4f} s_sim={s_sim.item():.4f} [{time.time()-t0:.1f}s]')

# Test
print('\
Testing...')
def convert(src_a,tgt_a):
    with torch.no_grad():
        c_src,_,s_src_raw=splitter(src_a); _,_,s_tgt_raw=splitter(tgt_a)
        zq_vc=converter(c_src,s_tgt_raw)
        zq_up=mimi._to_encoder_framerate(zq_vc)
        if mimi.decoder_transformer: (z_tr,)=mimi.decoder_transformer(zq_up)
        else: z_tr=zq_up
        return mimi.decoder(z_tr)

subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/ty.wav'],capture_output=True)
src2=load_any(f'{base}/p225/p225_001_mic1.flac').unsqueeze(0).unsqueeze(0)
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac').unsqueeze(0).unsqueeze(0)
tgt_c=load_any('/tmp/ty.wav',dur=None).unsqueeze(0).unsqueeze(0)

for name,t in[('parallel',tgt_p),('cross',tgt_c)]:
    vca=convert(src2,t)
    Tc=min(vca.shape[2],src2.shape[2],t.shape[2])
    zv=mimi.encode_to_latent(vca[:,:,:Tc],quantize=False)
    zs=mimi.encode_to_latent(src2[:,:,:Tc],quantize=False)
    zt=mimi.encode_to_latent(t[:,:,:Tc],quantize=False)
    T2=min(zv.shape[2],zs.shape[2],zt.shape[2])
    cs=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zs[:,:,:T2].reshape(-1),dim=0)
    ct=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zt[:,:,:T2].reshape(-1),dim=0)
    print(f'{name}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'/Users/asill/research5/mimi_imp_{name}.wav',vca[0,0,:Tc].numpy(),24000)
print('✅')
