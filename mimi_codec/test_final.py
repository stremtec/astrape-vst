"""Final VC test: Encoder Kanade Splitter + Q-Space Converter."""
import sys; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

STRIDE=1920
def load(path, dur=2):
    data,sr=sf.read(path)
    if sr!=24000: data=signal.resample(data,int(len(data)*24000/sr),axis=0)
    L=dur*24000-(dur*24000%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

class FinalSplitter(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        self.c_bn = nn.Sequential(nn.Conv1d(512,64,1),nn.GELU(),nn.Conv1d(64,512,1))
        self.s_net = nn.Sequential(nn.Conv1d(512,256,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(256,512))
    
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
            if i in[5,6,7]: deep.append(h)
        f_s=torch.stack(shallow,0).mean(0).transpose(1,2)
        f_d=torch.stack(deep,0).mean(0).transpose(1,2)
        return f_s,f_d
    
    def forward(self, audio):
        f_s,f_d=self.split(audio)
        c=f_s+self.c_bn(f_s)
        s=self.s_net(f_d)
        return c,s

class FinalConverter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.gamma=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.beta=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.conv=nn.Sequential(nn.Conv1d(dim,dim,5,padding=2),nn.GELU(),nn.Conv1d(dim,dim,5,padding=2))
    
    def forward(self, z_q_src, s_tgt):
        gamma=self.gamma(s_tgt).unsqueeze(-1)
        beta=self.beta(s_tgt).unsqueeze(-1)
        mean=z_q_src.mean(dim=2,keepdim=True)
        std=z_q_src.std(dim=2,keepdim=True)+1e-5
        z_norm=(z_q_src-mean)/std
        z_mod=z_norm*gamma+beta
        return z_q_src+self.conv(z_mod)

splitter=FinalSplitter(mimi)
converter=FinalConverter()
opt=torch.optim.AdamW(list(splitter.parameters())+list(converter.parameters()), lr=5e-4)

src=load(f'{base}/p225/p225_001_mic1.flac')
tgt=load(f'{base}/p226/p226_001_mic1.flac')

print('Training...')
t0=time.time()
for step in range(50):
    c_src,s_src=splitter(src)
    c_tgt,s_tgt=splitter(tgt)
    zq_src=splitter.get_z_q(src)
    zq_tgt=splitter.get_z_q(tgt)
    
    T_enc=min(c_src.shape[2],c_tgt.shape[2])
    loss_c=F.mse_loss(c_src[:,:,:T_enc],c_tgt[:,:,:T_enc])
    c_cos=F.cosine_similarity(c_src[:,:,:T_enc].reshape(-1),c_tgt[:,:,:T_enc].reshape(-1),dim=0)
    loss_c+=(1-c_cos)**2*0.5
    
    s_cos=F.cosine_similarity(s_src,s_tgt,dim=-1).mean()
    loss_s=torch.relu(s_cos-0.2)+(1-F.cosine_similarity(s_src,-s_tgt.detach(),dim=-1).mean())*0.1
    
    zq_vc=converter(zq_src,s_tgt)
    zq_up=mimi._to_encoder_framerate(zq_vc)
    if mimi.decoder_transformer: (z_tr,)=mimi.decoder_transformer(zq_up)
    else: z_tr=zq_up
    audio_vc=mimi.decoder(z_tr)
    
    T_a=min(audio_vc.shape[2],src.shape[2],tgt.shape[2])
    loss_r=F.mse_loss(audio_vc[:,:,:T_a],tgt[:,:,:T_a])
    
    T_z=min(zq_vc.shape[2],zq_tgt.shape[2])
    loss_z=0.1*F.mse_loss(zq_vc[:,:,:T_z],zq_tgt[:,:,:T_z])
    
    loss=loss_r+0.5*loss_c+0.3*loss_s+loss_z
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(splitter.parameters())+list(converter.parameters()),1.0)
    opt.step()
    
    if step%10==0:
        print(f'  step {step:3d}: recon={loss_r.item():.4f} c={loss_c.item():.4f} s={loss_s.item():.4f} c_cos={c_cos.item():.4f} s_cos={s_cos.item():.4f}')

print(f'Done [{time.time()-t0:.1f}s]')

# Test
def convert(src_a,tgt_a):
    with torch.no_grad():
        _,s_tgt=splitter(tgt_a)
        zq_src=splitter.get_z_q(src_a)
        zq_vc=converter(zq_src,s_tgt)
        zq_up=mimi._to_encoder_framerate(zq_vc)
        if mimi.decoder_transformer: (z_tr,)=mimi.decoder_transformer(zq_up)
        else: z_tr=zq_up
        return mimi.decoder(z_tr)

subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tf.wav'],capture_output=True)

def load_any(path, dur=None):
    data,sr=sf.read(path)
    if sr!=24000: data=signal.resample(data,int(len(data)*24000/sr),axis=0)
    if dur is not None: L=dur*24000-(dur*24000%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

src2=load_any(f'{base}/p225/p225_001_mic1.flac')
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac')
tgt_c=load_any('/tmp/tf.wav',dur=None)

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
    sf.write(f'/Users/asill/research5/mimi_final_{name}.wav',vca[0,0,:Tc].numpy(),24000)
print('✅')
