#!/usr/bin/env python3
"""ContentStudent v2: Bigger TCN + teacher intermediate distillation (fast on CPU)."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, time
from torch.optim import AdamW; from torch.optim.lr_scheduler import CosineAnnealingLR
BATCH=16; EPOCHS=80; device='cpu'

class ContentStudentV2(nn.Module):
    def __init__(self,in_dim=80,hidden=384,out_dim=5,n_layers=6,kernel=7):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(n_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(
                nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),
                nn.GroupNorm(min(16,hidden),hidden),nn.GELU(),
                nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        self.content_head=nn.Conv1d(hidden,768,1)
        self.prefsq_head=nn.Conv1d(hidden,768,1)
        self.fsq_head=nn.Conv1d(hidden,out_dim,1)
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers: r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h)
        return self.content_head(h),self.prefsq_head(h),self.fsq_head(h)

MEL_DIR="/Users/asill/btrv5/data/mio_mel"; INTER_DIR="/Users/asill/btrv5/data/mio_intermediate"
meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz"); n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl=idxs[int(n*0.8):]

def load(idx):
    md=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    id=np.load("{}/inter_{:04d}.npz".format(INTER_DIR,idx))
    return (torch.from_numpy(md['logmel']).float(),torch.from_numpy(id['ce_768']).float(),
            torch.from_numpy(id['pre_fsq_768']).float(),torch.from_numpy(id['fsq_5d']).float())

model=ContentStudentV2().to(device); model.train()
print("Params:",sum(p.numel() for p in model.parameters()))
opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5); sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training Bigger TCN v2 (384dim, 6-layer)...")
for epoch in range(EPOCHS):
    tr_l=0; nb=0; perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; ces=[]; pfs=[]
        for j in bi:
            mel,ce,pf,fsq=load(tr[j]); mels.append(mel); ces.append(ce); pfs.append(pf)
        max_T50=max(m.shape[1] for m in mels); max_T25=max(c.shape[0] for c in ces)
        xb=torch.stack([F.pad(m,(0,max_T50-m.shape[1])) for m in mels]).to(device)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T25-ce.shape[0])) for ce in ces]).to(device)
        pf_b=torch.stack([F.pad(pf,(0,0,0,max_T25-pf.shape[0])) for pf in pfs]).to(device)
        
        ce_p,pf_p,fsq_p=model(xb); Tp=min(ce_p.shape[2],ce_b.shape[1])
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
    if epoch%20==0 or epoch==EPOCHS-1: print("  E{:3d} loss={:.4f}".format(epoch,tr_l/max(nb,1)))

os.makedirs("checkpoints",exist_ok=True); torch.save(model.state_dict(),"checkpoints/causal_student_v2b.pt")

# Test
print()
model.eval()
mel,ce_t,_,_=load(120); mel=mel.unsqueeze(0).to(device)
with torch.no_grad():
    ce_p,_,_=model(mel); ce_s=ce_p.squeeze(0).T.cpu(); ce_t=ce_t.to('cpu')
    T=min(ce_s.shape[0],ce_t.shape[0])
    cos=F.cosine_similarity(ce_s[:T].flatten(),ce_t[:T].flatten(),dim=0).item()
    fcos=[F.cosine_similarity(ce_s[i:i+1].flatten(),ce_t[i:i+1].flatten(),dim=0).item() for i in range(T)]
print("  Content cos: {:.4f}  fmed: {:.3f}  anti-corr: {:.1f}%".format(cos,np.median(fcos),(np.array(fcos)<0).mean()*100))
print("Done!")
