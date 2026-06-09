"""LongCat combinatorial speaker probe — where is the speaker info?"""
import sys, os; sys.path.insert(0,'/tmp/LongCat-Audio-Codec'); os.chdir('/tmp/LongCat-Audio-Codec')
import torch, torch.nn as nn, soundfile as sf, random, numpy as np
from scipy import signal
from networks.semantic_codec.model_loader import load_encoder

SR=24000; BASE='/tmp/LongCat-Audio-Codec'
encoder=load_encoder(f'{BASE}/configs/LongCatAudioCodec_encoder.yaml', torch.device('cpu'))

def load_any(path,dur=None):
    d,sr=sf.read(path)
    if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr),axis=0)
    if dur is not None: L=dur*SR; d=d[:L]
    else: L=len(d); d=d[:L]
    if d.ndim>1: d=d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks=sorted([d for d in os.listdir(base) if os.path.isdir(f'{base}/{d}') and d.startswith('p')])[:20]
spk_to_idx={s:i for i,s in enumerate(spks)}; n_spk=len(spks)

samples=[]
for s in spks:
    uts=sorted([f for f in os.listdir(f'{base}/{s}') if f.endswith('.flac')])[:3]
    for u in uts:
        a=load_any(f'{base}/{s}/{u}')
        with torch.no_grad():
            sem,aco=encoder(a)
        T=min(aco.shape[2],sem.shape[1])
        samples.append((sem[0,:T], aco[0,0,:T], aco[0,1,:T], aco[0,2,:T], spk_to_idx[s]))

random.shuffle(samples)
n_train=int(len(samples)*0.8)
train_s=samples[:n_train]; val_s=samples[n_train:]
print(f'{len(samples)} samples, {n_spk} speakers')

V=8200; S=8200  # LongCat vocab sizes

class CombiProbe(nn.Module):
    def __init__(self, sources):
        super().__init__()
        self.sources=sources  # list of 'sem','cb0','cb1','cb2'
        total_dim=0
        self.embs=nn.ModuleDict()
        for src in sources:
            v=S if src=='sem' else V
            dim=32
            self.embs[src]=nn.Embedding(v,dim)
            total_dim+=dim
        self.net=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(total_dim,n_spk))
    def forward(self, sem,cb0,cb1,cb2):
        feats=[]
        for src in self.sources:
            t={'sem':sem,'cb0':cb0,'cb1':cb1,'cb2':cb2}[src]
            feats.append(self.embs[src](t.long()))
        h=torch.cat(feats,dim=-1).transpose(1,2)
        return self.net(h)

combos=[
    ['cb0'],['cb1'],['cb2'],
    ['cb0','cb1'],['cb0','cb2'],['cb1','cb2'],
    ['cb0','cb1','cb2'],
    ['sem'],['sem','cb0'],['sem','cb0','cb1','cb2'],
]

print()
header = f"{'Features':>25s} {'Top-1':>8s} {'Top-3':>8s} {'Top-5':>8s}"
print(header)
print('-'*55)

for combo in combos:
    probe=CombiProbe(combo); opt=torch.optim.AdamW(probe.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for _ in range(30):
        random.shuffle(train_s)
        for s in train_s[:30]:
            sem,cb0,cb1,cb2,label=s[0],s[1],s[2],s[3],s[4]
            sem=sem.unsqueeze(0); cb0=cb0.unsqueeze(0); cb1=cb1.unsqueeze(0); cb2=cb2.unsqueeze(0)
            loss=ce(probe(sem,cb0,cb1,cb2),torch.tensor([label]))
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    top1=0; top3=0; top5=0
    for s in val_s:
        sem,cb0,cb1,cb2,label=s[0],s[1],s[2],s[3],s[4]
        sem=sem.unsqueeze(0); cb0=cb0.unsqueeze(0); cb1=cb1.unsqueeze(0); cb2=cb2.unsqueeze(0)
        logits=probe(sem,cb0,cb1,cb2)
        _,idx=logits.topk(5,dim=-1)
        if label==idx[0,0]: top1+=1
        if label in idx[0,:3]: top3+=1
        if label in idx[0,:5]: top5+=1
    N=len(val_s)
    name='+'.join(combo)
    print(f'{name:>25s} {top1/N:7.4f} {top3/N:7.4f} {top5/N:7.4f}')

rand_line = f"{'random':>25s} {1/n_spk:7.4f} {3/n_spk:7.4f} {5/n_spk:7.4f}"
print(rand_line)
