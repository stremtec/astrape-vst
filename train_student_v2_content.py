#!/usr/bin/env python3
"""
CausalContentStudent v2: Stronger encoder + teacher intermediate distillation.
- Conv stem + causal Transformer encoder
- Heads: content_embed_768 (MAIN), pre_fsq_768, fsq_5d
- Loss: cosine+L1 on content_embed, cosine on pre_fsq, MSE on fsq_5d
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

BATCH=4; EPOCHS=60
device=torch.device('cpu')

# ── Positional Encoding ────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_len=2000):
        super().__init__()
        pe=torch.zeros(max_len,dim)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): 
        T=x.size(1)
        return x+self.pe[:,:T,:].contiguous()

# ── Causal Transformer Block ───────────────────────────────────────────
class CausalTransformerBlock(nn.Module):
    def __init__(self,dim=384,n_heads=6,ff_mult=4,dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.norm2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn=self.norm1(x); a=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]; x=x+a
        xn=self.norm2(x); x=x+self.ff(xn); return x

# ── Content Student v2 ─────────────────────────────────────────────────
class ContentStudentV2(nn.Module):
    def __init__(self,in_dim=80,hidden=384,n_layers=6,n_heads=6,out_dim=5,kernel=5):
        super().__init__()
        # Conv stem: 80→384
        self.stem=nn.Sequential(
            nn.Conv1d(in_dim,hidden,kernel,padding=kernel//2,padding_mode='replicate'),nn.GELU(),
            nn.Conv1d(hidden,hidden,kernel,padding=kernel//2,padding_mode='replicate'),nn.GELU(),
        )
        self.pos_enc=PositionalEncoding(hidden)
        self.blocks=nn.ModuleList([CausalTransformerBlock(hidden,n_heads) for _ in range(n_layers)])
        self.norm=nn.LayerNorm(hidden)
        # Downsample 50Hz→25Hz
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        # Heads
        self.content_head=nn.Conv1d(hidden,768,1)  # MAIN: content embedding
        self.prefsq_head=nn.Conv1d(hidden,768,1)   # AUX: pre-FSQ intermediate
        self.fsq_head=nn.Conv1d(hidden,out_dim,1)  # AUX: FSQ 5d
    
    def forward(self,x):
        # x: (B,80,T50)
        h=self.stem(x)  # (B,H,T50)
        h=h.transpose(1,2)  # (B,T50,H)
        h=self.pos_enc(h)
        for block in self.blocks: h=block(h)
        h=self.norm(h).transpose(1,2)  # (B,H,T50)
        h=self.down(h)  # (B,H,T25)
        ce=self.content_head(h)  # (B,768,T25)
        pf=self.prefsq_head(h)   # (B,768,T25)
        fsq=self.fsq_head(h)     # (B,5,T25)
        return ce, pf, fsq

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR="/Users/asill/btrv5/data/mio_intermediate"
meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz"); n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl=idxs[int(n*0.8):]

# Also load mel data
MEL_DIR="/Users/asill/btrv5/data/mio_mel"

def load(idx):
    mel_d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    inter_d=np.load("{}/inter_{:04d}.npz".format(DATA_DIR,idx))
    return (torch.from_numpy(mel_d['logmel']).float(),         # (80,T50)
            torch.from_numpy(inter_d['ce_768']).float(),       # (T25,768)
            torch.from_numpy(inter_d['pre_fsq_768']).float(),  # (T25,768)
            torch.from_numpy(inter_d['fsq_5d']).float())       # (T25,5)

# ── Training ──────────────────────────────────────────────────────────
model=ContentStudentV2(hidden=256,n_layers=4,n_heads=4).to(device); model.train()
opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training ContentStudent v2 ({} params)".format(sum(p.numel() for p in model.parameters())))
for epoch in range(EPOCHS):
    tr_loss=0; tr_ce=0; tr_pf=0; tr_fsq=0; nb=0
    perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; ces=[]; pfs=[]; fsqs=[]
        for j in bi:
            mel,ce,pf,fsq=load(tr[j]); mels.append(mel); ces.append(ce); pfs.append(pf); fsqs.append(fsq)
        max_T50=max(m.shape[1] for m in mels)  # T50
        max_T25=max(c.shape[0] for c in ces)    # T25
        
        xb=torch.stack([F.pad(m,(0,max_T50-m.shape[1])) for m in mels]).to(device)  # (B,80,T50)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T25-ce.shape[0])) for ce in ces]).to(device)  # (B,T25,768)
        pf_b=torch.stack([F.pad(pf,(0,0,0,max_T25-pf.shape[0])) for pf in pfs]).to(device)
        fsq_b=torch.stack([F.pad(fsq,(0,0,0,max_T25-fsq.shape[0])) for fsq in fsqs]).to(device)
        
        ce_p,pf_p,fsq_p=model(xb)  # (B,768,T25_pred), (B,768,T25_pred), (B,5,T25_pred)
        Tp=min(ce_p.shape[2],ce_b.shape[1])
        
        # Main: content embedding
        ep=ce_p[:,:,:Tp]; et=ce_b[:,:Tp,:].transpose(1,2)  # (B,768,Tp)
        cos_ce=F.cosine_similarity(ep.reshape(ep.shape[0],-1),et.reshape(et.shape[0],-1),dim=1).mean()
        L_ce=(1-cos_ce)+0.3*F.l1_loss(ep,et)
        
        # Aux: pre-FSQ
        pp=pf_p[:,:,:Tp]; pt=pf_b[:,:Tp,:].transpose(1,2)
        cos_pf=F.cosine_similarity(pp.reshape(pp.shape[0],-1),pt.reshape(pt.shape[0],-1),dim=1).mean()
        L_pf=(1-cos_pf)+0.1*F.l1_loss(pp,pt)
        
        # Aux: FSQ 5d
        fp=fsq_p[:,:,:Tp]; ft=fsq_b[:,:Tp,:].transpose(1,2)
        L_fsq=F.mse_loss(fp,ft)
        
        loss=L_ce+0.3*L_pf+0.1*L_fsq
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_loss+=loss.item(); tr_ce+=L_ce.item(); tr_pf+=L_pf.item(); tr_fsq+=L_fsq.item(); nb+=1
    sched.step()
    
    if epoch%20==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f} ce={:.4f} pf={:.4f} fsq={:.4f}".format(
            epoch,tr_loss/max(nb,1),tr_ce/max(nb,1),tr_pf/max(nb,1),tr_fsq/max(nb,1)))

os.makedirs("checkpoints",exist_ok=True)
torch.save(model.state_dict(),"checkpoints/causal_student_v2.pt")

# ── Quick test ────────────────────────────────────────────────────────
print()
print("=== V2 Content Quality Test ===")
model.eval()
mel,ce_t,pf_t,fsq_t=load(120); mel=mel.unsqueeze(0)
with torch.no_grad():
    ce_p,pf_p,fsq_p=model(mel)
    ce_s=ce_p.squeeze(0).T; T=min(ce_s.shape[0],ce_t.shape[0])
    cos=F.cosine_similarity(ce_s[:T].flatten(),ce_t[:T].flatten(),dim=0).item()
    fcos=[F.cosine_similarity(ce_s[i:i+1].flatten(),ce_t[i:i+1].flatten(),dim=0).item() for i in range(T)]
print("  Content cos: {:.4f}  frame median: {:.3f}  anti-corr: {:.1f}%".format(
    cos,np.median(fcos),(np.array(fcos)<0).mean()*100))
print("Done!")
