#!/usr/bin/env python3
"""ContentStudent v2-FINAL: 384dim, 6-layer Transformer, 60 spk, teacher distillation."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, time, math
from torch.optim import AdamW; from torch.optim.lr_scheduler import CosineAnnealingLR

BATCH=4; EPOCHS=100; device='cpu'

class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_len=2000):
        super().__init__(); pe=torch.zeros(max_len,dim)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): T=x.size(1); return x+self.pe[:,:T,:].contiguous()
class CausalTransformerBlock(nn.Module):
    def __init__(self,dim=384,n_heads=8,ff_mult=4,dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.norm2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn=self.norm1(x); a=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]; x=x+a
        xn=self.norm2(x); x=x+self.ff(xn); return x
class ContentStudentV2(nn.Module):
    def __init__(self,in_dim=80,hidden=384,n_layers=6,n_heads=8,out_dim=5,kernel=5):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(in_dim,hidden,kernel,padding=kernel//2),nn.GELU(),nn.Conv1d(hidden,hidden,kernel,padding=kernel//2),nn.GELU())
        self.pos_enc=PositionalEncoding(hidden)
        self.blocks=nn.ModuleList([CausalTransformerBlock(hidden,n_heads) for _ in range(n_layers)])
        self.norm=nn.LayerNorm(hidden); self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1)
        self.content_head=nn.Conv1d(hidden,768,1); self.prefsq_head=nn.Conv1d(hidden,768,1); self.fsq_head=nn.Conv1d(hidden,out_dim,1)
    def forward(self,x):
        h=self.stem(x); h=h.transpose(1,2); h=self.pos_enc(h)
        for block in self.blocks: h=block(h)
        h=self.norm(h).transpose(1,2); h=self.down(h)
        return self.content_head(h),self.prefsq_head(h),self.fsq_head(h)

# ── Data: use ALL samples ──────────────────────────────────────────────
MEL_DIR="/Users/asill/btrv5/data/mio_mel"; INTER_DIR="/Users/asill/btrv5/data/mio_intermediate"
meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz"); n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.85)]; vl=idxs[int(n*0.85):]  # 85% train for more data
print("Train: {} Val: {} ({} unique speakers)".format(len(tr),len(vl),len(set(meta['spk_names'][tr]))))

def load(idx):
    md=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    id=np.load("{}/inter_{:04d}.npz".format(INTER_DIR,idx))
    return (torch.from_numpy(md['logmel']).float(),torch.from_numpy(id['ce_768']).float(),
            torch.from_numpy(id['pre_fsq_768']).float())

# ── Training ──────────────────────────────────────────────────────────
model=ContentStudentV2().to(device); model.train()
# Start from v2 256dim checkpoint
try:
    v2_ckpt=torch.load("checkpoints/causal_student_v2.pt",map_location='cpu')
    model.load_state_dict(v2_ckpt,strict=False)
    print("Loaded v2 256dim checkpoint (partial)")
except: print("Training from scratch")

opt=AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5); sched=CosineAnnealingLR(opt,T_max=EPOCHS)
print("Params:",sum(p.numel() for p in model.parameters()),"| Epochs:",EPOCHS)

print("Training V2-FINAL (384dim, 6-layer, 8-heads)...")
for epoch in range(EPOCHS):
    tr_l=0; nb=0; perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; ces=[]; pfs=[]
        for j in bi: mel,ce,pf=load(tr[j]); mels.append(mel); ces.append(ce); pfs.append(pf)
        max_T50=max(m.shape[1] for m in mels); max_T25=max(c.shape[0] for c in ces)
        xb=torch.stack([F.pad(m,(0,max_T50-m.shape[1])) for m in mels]).to(device)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T25-ce.shape[0])) for ce in ces]).to(device)
        pf_b=torch.stack([F.pad(pf,(0,0,0,max_T25-pf.shape[0])) for pf in pfs]).to(device)
        
        ce_p,pf_p,_=model(xb); Tp=min(ce_p.shape[2],ce_b.shape[1])
        ep=ce_p[:,:,:Tp]; et=ce_b[:,:Tp,:].transpose(1,2)
        cos=F.cosine_similarity(ep.reshape(ep.shape[0],-1),et.reshape(et.shape[0],-1),dim=1).mean()
        L_ce=(1-cos)+0.3*F.l1_loss(ep,et)
        pp=pf_p[:,:,:Tp]; pt=pf_b[:,:Tp,:].transpose(1,2)
        cos_pf=F.cosine_similarity(pp.reshape(pp.shape[0],-1),pt.reshape(pt.shape[0],-1),dim=1).mean()
        L_pf=(1-cos_pf)*0.3
        loss=L_ce+L_pf
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_l+=loss.item(); nb+=1
    sched.step()
    if epoch%20==0 or epoch==EPOCHS-1:
        # Val check
        model.eval(); vcos=0; vn=0
        with torch.no_grad():
            for j in vl[:10]:
                mel,ce,_=load(j); mel=mel.unsqueeze(0).to(device); ce=ce.unsqueeze(0).to(device)
                cp,_,_=model(mel); Tp=min(cp.shape[2],ce.shape[1])
                ep=cp[:,:,:Tp]; et=ce[:,:Tp,:].transpose(1,2)
                vcos+=F.cosine_similarity(ep.reshape(1,-1),et.reshape(1,-1),dim=1).mean().item(); vn+=1
        model.train()
        print("  E{:3d} loss={:.4f} val_cos={:.4f}".format(epoch,tr_l/max(nb,1),vcos/max(vn,1)))

os.makedirs("checkpoints",exist_ok=True); torch.save(model.state_dict(),"checkpoints/causal_student_v2_final.pt")
print("Saved: checkpoints/causal_student_v2_final.pt")
print("Done!")
