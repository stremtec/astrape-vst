"""LongCat CB speaker probe."""
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
spk_to_idx={s:i for i,s in enumerate(spks)}
n_spk=len(spks)
print(f'{n_spk} speakers')

samples=[]
for s in spks:
    uts=sorted([f for f in os.listdir(f'{base}/{s}') if f.endswith('.flac')])[:3]
    for u in uts:
        a=load_any(f'{base}/{s}/{u}')
        with torch.no_grad():
            sem,aco=encoder(a)
        T=min(aco.shape[2],sem.shape[1])
        samples.append((aco[0,0,:T], aco[0,1,:T], aco[0,2,:T], sem[0,:T], spk_to_idx[s]))

random.shuffle(samples)
n_train=int(len(samples)*0.8)
train_s=samples[:n_train]; val_s=samples[n_train:]

V=8200  # LongCat token range 0~8000+
class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(V,64)
        self.net=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(64,n_spk))
    def forward(self,x):
        return self.net(self.emb(x).transpose(1,2))

print()
print(f'{"CB":>6s} {"SpkAcc":>8s}')
print('-'*20)
for cb in range(3):
    probe=Probe(); opt=torch.optim.AdamW(probe.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss()
    for _ in range(30):
        random.shuffle(train_s)
        for s in train_s[:30]:
            t=s[cb].unsqueeze(0).long()
            loss=ce(probe(t),torch.tensor([s[4]]))
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    acc=sum(probe(s[cb].unsqueeze(0).long()).argmax(-1).item()==s[4] for s in val_s)/len(val_s)
    print(f'  CB{cb}   {acc:.4f}')
print(f'  rnd   {1/n_spk:.4f}')
