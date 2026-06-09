#!/usr/bin/env python3
"""
Stage 3a: Causal Mel Decoder Student.
Input: teacher content_embedding + global_embedding
Output: mel spectrogram (computed from teacher waveform)
AdaLN-Zero speaker conditioning, strictly causal.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, numpy as np, os, time, soundfile as sf
from scipy import signal as scipy_signal
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SR=44100; CONTENT_RATE=25; BATCH=8; EPOCHS=60
N_MELS=80; MEL_HOP=320  # ~138Hz mel rate
device=torch.device('cpu')

# ── AdaLN-Zero (same as MioCodec) ────────────────────────────────────
class AdaLNZero(nn.Module):
    def __init__(self,dim,cond_dim,eps=1e-5):
        super().__init__()
        self.norm=nn.LayerNorm(dim,eps=eps,elementwise_affine=False)
        self.proj=nn.Sequential(nn.SiLU(),nn.Linear(cond_dim,3*dim))
        nn.init.zeros_(self.proj[1].weight); nn.init.zeros_(self.proj[1].bias)
    def forward(self,x,cond):
        # x: (B,T,D), cond: (B,1,D_cond) or (B,T,D_cond)
        x_n=self.norm(x)
        shift,scale,gate=self.proj(cond).chunk(3,dim=-1)
        return x_n*(1+scale)+shift, gate

# ── Causal Decoder Block ─────────────────────────────────────────────
class CausalDecoderBlock(nn.Module):
    def __init__(self,dim=512,cond_dim=128,n_heads=8,ff_mult=4,dropout=0.1):
        super().__init__()
        self.adaln=AdaLNZero(dim,cond_dim)
        self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.adaln2=AdaLNZero(dim,cond_dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),
                              nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    
    def forward(self,x,cond):
        # x: (B,T,D), cond: (B,1,cond_dim)
        # Causal attention mask: lower triangular
        T=x.shape[1]
        mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        
        xn,gate=self.adaln(x,cond)
        attn_out=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]
        x=x+gate*attn_out
        
        xn2,gate2=self.adaln2(x,cond)
        ff_out=self.ff(xn2)
        x=x+gate2*ff_out
        return x

# ── Mel Decoder Student ───────────────────────────────────────────────
class CausalMelDecoder(nn.Module):
    def __init__(self,content_dim=768,cond_dim=128,hidden=512,n_layers=4,n_heads=8,n_mels=80):
        super().__init__()
        self.proj_in=nn.Linear(content_dim,hidden)
        self.blocks=nn.ModuleList([
            CausalDecoderBlock(hidden,cond_dim,n_heads) for _ in range(n_layers)
        ])
        self.norm_out=nn.LayerNorm(hidden)
        self.proj_out=nn.Linear(hidden,n_mels)
    
    def forward(self,content_emb,global_emb):
        # content_emb: (B,T_c,768), global_emb: (B,128)
        x=self.proj_in(content_emb)  # (B,T,hidden)
        cond=global_emb.unsqueeze(1)  # (B,1,128)
        
        for block in self.blocks:
            x=block(x,cond)
        
        x=self.norm_out(x)
        mel=self.proj_out(x)  # (B,T,n_mels)
        return mel.transpose(1,2)  # (B,n_mels,T)

# ── Teacher for target mel ────────────────────────────────────────────
from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
teacher.eval()

mel_extractor=torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,n_fft=1024,hop_length=MEL_HOP,n_mels=N_MELS,
    f_min=80,f_max=14000,center=False,power=1)

# Use content-rate mel hop for teacher
MEL_HOP_CONTENT=int(SR/CONTENT_RATE)  # 1764 samples = 40ms
mel_extractor_ct=torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,n_fft=1024,hop_length=MEL_HOP_CONTENT,n_mels=N_MELS,
    f_min=80,f_max=14000,center=False,power=1)

def extract_target_mel(waveform):
    """Compute mel spectrogram at CONTENT RATE (25Hz)."""
    if isinstance(waveform,torch.Tensor):
        w=waveform.detach().cpu().numpy()
    else: w=waveform
    w=w-np.mean(w)
    mel=mel_extractor_ct(torch.from_numpy(w).float().view(1,1,-1))
    return torch.log(mel.squeeze(1).clamp(min=1e-5))  # (1,n_mels,T_25hz)

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
meta=np.load("{}/meta.npz".format(DATA_DIR))
n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl_idx=idxs[int(n*0.8):]
print("Train: {} Val: {}".format(len(tr),len(vl_idx)))

# ── Pre-compute teacher mels for training ─────────────────────────────
print("Pre-computing teacher mel targets...")
teacher_mels={}
for i in range(n):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,i))
    audio=d['audio']; ce=d['ce_768']; ge=d['ge_128']
    alen=len(audio)
    # Teacher self-recon waveform
    x_t=torch.from_numpy(audio[:SR*3]).float().unsqueeze(0)
    with torch.inference_mode():
        feat=teacher.encode(x_t,return_content=True,return_global=True)
        wav=teacher.decode(global_embedding=torch.from_numpy(ge).float(),
                          content_token_indices=feat.content_token_indices,
                          target_audio_length=alen)
    mel_target=extract_target_mel(wav)
    teacher_mels[i]=mel_target
    if i%50==0: print("  {}/{}".format(i,n))

print("Target mels ready")

# ── Training ──────────────────────────────────────────────────────────
model=CausalMelDecoder().to(device); model.train()
opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training causal mel decoder...")
for epoch in range(EPOCHS):
    tr_loss=0; nb=0
    perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        # Stack content + global embeddings (variable T_c)
        bd=[np.load("{}/sample_{:04d}.npz".format(DATA_DIR,tr[j])) for j in bi]
        max_Tc=max(d['ce_768'].shape[0] for d in bd)
        ces=[]; ges=[]; mel_tgts=[]
        for jj,j in enumerate(bi):
            d=bd[jj]
            ce=torch.from_numpy(d['ce_768']).float()
            ge=torch.from_numpy(d['ge_128']).float()
            ces.append(F.pad(ce,(0,0,0,max_Tc-ce.shape[0])))
            ges.append(ge)
            mel_tgts.append(teacher_mels[tr[j]])
        
        ce_b=torch.stack(ces).to(device)  # (B,T_c,768)
        ge_b=torch.stack(ges).to(device)  # (B,128)
        mel_pred=model(ce_b,ge_b)  # (B,n_mels,T_pred)
        
        # Match mel lengths
        max_Tm=max(m.shape[2] for m in mel_tgts)  # (1,n_mels,T), T is dim 2
        mel_tgt_b=torch.stack([F.pad(m,(0,max_Tm-m.shape[2])) for m in mel_tgts]).to(device)
        mel_tgt_b=mel_tgt_b.squeeze(1)  # (B,1,n_mels,T) → (B,n_mels,T)
        Tp=min(mel_pred.shape[2],mel_tgt_b.shape[2])
        loss=F.l1_loss(mel_pred[:,:,:Tp],mel_tgt_b[:,:,:Tp])
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_loss+=loss.item(); nb+=1
    
    sched.step()
    
    # Val (avoid name collision: vl_idx for indices, v_loss for loss)
    model.eval(); v_loss=0; vn=0
    with torch.no_grad():
        for j in vl_idx[:10]:
            d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,j))
            ce=torch.from_numpy(d['ce_768']).float().unsqueeze(0).to(device)
            ge=torch.from_numpy(d['ge_128']).float().unsqueeze(0).to(device)
            mel_pred=model(ce,ge)
            mel_tgt=teacher_mels[j].to(device)  # already (1,n_mels,T)
            Tp=min(mel_pred.shape[2],mel_tgt.shape[2])
            v_loss+=F.l1_loss(mel_pred[:,:,:Tp],mel_tgt[:,:,:Tp]).item(); vn+=1
    model.train()
    
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} tr={:.4f} val={:.4f}".format(epoch,tr_loss/max(nb,1),v_loss/max(vn,1)))

os.makedirs("checkpoints",exist_ok=True)
torch.save(model.state_dict(),"checkpoints/causal_mel_decoder.pt")

# ── Test ──────────────────────────────────────────────────────────────
print()
print("=== Mel Decoder Test ===")
model.eval()

d,sr=sf.read("/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac")
if d.ndim>1: d=d.mean(axis=1)
if sr!=SR: d=scipy_signal.resample(d,int(len(d)*SR/sr))
d=d[:SR*3]; alen=len(d)

x_t=torch.from_numpy(d).float().unsqueeze(0)
with torch.inference_mode():
    ft=teacher.encode(x_t,return_content=True,return_global=True)
    ce=ft.content_embedding.unsqueeze(0)  # (1,T,768)
    ge=ft.global_embedding.unsqueeze(0)   # (1,128)
    mel_stu=model(ce.to(device),ge.to(device))
    # Teacher waveform → mel
    wav=teacher.decode(global_embedding=ge.squeeze(0),content_token_indices=ft.content_token_indices,target_audio_length=alen)
    mel_tgt=extract_target_mel(wav)

# Compare
mel_s=mel_stu.squeeze(0).cpu(); mel_t=mel_tgt.squeeze(0)
l1=F.l1_loss(mel_s[:,:mel_t.shape[1]],mel_t).item()
cos=F.cosine_similarity(mel_s[:,:mel_t.shape[1]].flatten(),mel_t.flatten(),dim=0).item()
print("  Mel L1: {:.4f}  Cosine: {:.4f}".format(l1,cos))

# Save
np.save('/tmp/mel_student.npy',mel_s.numpy())
np.save('/tmp/mel_teacher.npy',mel_t.numpy())
print("  Saved mels to /tmp/")
print("Done!")
