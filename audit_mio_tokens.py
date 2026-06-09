#!/usr/bin/env python3
"""MioCodec: deep content token analysis."""
import torch, numpy as np, soundfile as sf
from scipy import signal
from collections import Counter
import sys
sys.path.insert(0,'/Users/asill/btrvrc0/.venv/lib/python3.12/site-packages')
from miocodec.model import MioCodecModel

SR=44100
model=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
model.eval()

SPKS=['p225','p226','p227','p228','p229','p255','p256','p257','p258','p259']
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
import glob

print("="*70)
print("CONTENT TOKEN DEEP ANALYSIS")
print("="*70)

# Collect tokens from all speakers
all_tokens=[]
all_embeddings=[]
spk_token_stats={}

for spk in SPKS:
    files=sorted(glob.glob("{}/{}/{}_*_mic1.flac".format(ROOT,spk,spk)))
    spk_tokens=[]
    for f in files[:5]:
        d,s=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if s!=SR: d=signal.resample(d,int(len(d)*SR/s))
        d=d[:SR*2]
        x=torch.from_numpy(d).float().unsqueeze(0)
        with torch.inference_mode():
            feat=model.encode(x,return_content=True,return_global=False)
        tokens=feat.content_token_indices.cpu().numpy()
        spk_tokens.append(tokens)
        all_tokens.append(tokens)
    spk_token_stats[spk]=np.concatenate(spk_tokens)

# Global stats
all_flat=np.concatenate(all_tokens)
vocab_size=all_flat.max()+1

print()
print("Vocabulary size:", vocab_size)
print("Total tokens:", len(all_flat))
print("Unique tokens used:", len(np.unique(all_flat)))
print("Usage ratio: {:.1f}%".format(len(np.unique(all_flat))/vocab_size*100))

# Entropy
counts=Counter(all_flat)
probs=np.array([c/len(all_flat) for c in counts.values()])
entropy=-np.sum(probs*np.log2(probs+1e-8))
print("Global entropy: {:.3f} bits (max: {:.1f})".format(entropy,np.log2(vocab_size)))

# Top tokens
print()
print("Top 10 token frequencies:")
for tok,count in counts.most_common(10):
    print("  {}: {} ({:.2f}%)".format(tok,count,count/len(all_flat)*100))

# Per-speaker entropy
print()
print("Per-speaker token stats:")
print("  Speaker  Unique   Entropy  Top1%  Top3%")
print("  " + "-"*50)
for spk in SPKS:
    tokens=spk_token_stats[spk]
    c=Counter(tokens)
    p=np.array([v/len(tokens) for v in c.values()])
    e=-np.sum(p*np.log2(p+1e-8))
    top1=c.most_common(1)[0][1]/len(tokens)*100
    top3=sum(v for _,v in c.most_common(3))/len(tokens)*100
    print("  {}  {:5d}    {:6.3f}  {:5.1f}% {:5.1f}%".format(spk,len(c),e,top1,top3))

# Temporal analysis
print()
print("Temporal dynamics (per-frame token change rate):")
for spk in SPKS[:3]:
    tokens=spk_token_stats[spk]
    # Flatten all utterances
    changes=np.sum(tokens[:-1]!=tokens[1:])
    rate=changes/(len(tokens)-1)*100
    print("  {}: {:.1f}% frame-to-frame change".format(spk,rate))

# FSQ dimension analysis
print()
print("FSQ quantizer structure:")
q=model.local_quantizer.fsq
print("  FSQ levels:", q.levels)
print("  FSQ dim:", q.dim)
print("  Codebook size:", model.local_quantizer.all_codebook_size)
print("  Product: ", end="")
p=1
for l in q.levels: p*=l
print(p)

# Check if content embedding clusters by speaker
from sklearn.decomposition import PCA
all_ce=[]
all_spk=[]
for spk in SPKS:
    files=sorted(glob.glob("{}/{}/{}_*_mic1.flac".format(ROOT,spk,spk)))
    for f in files[:3]:
        d,s=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if s!=SR: d=signal.resample(d,int(len(d)*SR/s))
        d=d[:SR*2]
        x=torch.from_numpy(d).float().unsqueeze(0)
        with torch.inference_mode():
            feat=model.encode(x,return_content=True,return_global=False)
        # Mean-pool content embedding
        ce=feat.content_embedding.mean(dim=0).cpu().numpy()
        all_ce.append(ce)
        all_spk.append(SPKS.index(spk))

ce_arr=np.array(all_ce)
pca=PCA(n_components=2)
ce_2d=pca.fit_transform(ce_arr)
# Check speaker separation
from sklearn.metrics import silhouette_score
try:
    sil=silhouette_score(ce_2d,all_spk)
    print()
    print("Content embedding PCA silhouette: {:.3f} (low=speaker-mixed, high=separated)".format(sil))
except: pass

print()
print("Done!")
