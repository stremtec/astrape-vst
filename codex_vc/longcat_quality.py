"""LongCat quality experiments — fast version with precomputed embeddings."""
import sys, os; sys.path.insert(0,'/tmp/LongCat-Audio-Codec'); os.chdir('/tmp/LongCat-Audio-Codec')
import torch, soundfile as sf, numpy as np, time, json
from scipy import signal
from pathlib import Path
from networks.semantic_codec.model_loader import load_encoder, load_decoder

SR=24000
OUT=Path('/Users/asill/research5/longcat_experiments')
for d in ['ema_embedding','fast_slow','baselines']:
    (OUT/d).mkdir(parents=True,exist_ok=True)

print("Loading...")
encoder=load_encoder('configs/LongCatAudioCodec_encoder.yaml', torch.device('cpu'))
decoder=load_decoder('configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml', torch.device('cpu'))

# Precomputed CB0 embeddings
CB0_EMBS=torch.from_numpy(np.load('/tmp/longcat_cb0_embs.npy')).float()  # (8100, 1024)
MAX_TOK=CB0_EMBS.shape[0]-1

def tokens_to_emb(tokens_np):
    """Fast lookup from precomputed table."""
    return CB0_EMBS[tokens_np.astype(np.int64)].T.unsqueeze(0)  # (1, 1024, T)

def emb_to_tokens(emb):
    """Fast nearest-neighbor using precomputed table."""
    B,D,T=emb.shape
    e_flat=emb[0].T  # (T, 1024)
    dists=torch.cdist(e_flat, CB0_EMBS)  # (T, 8100)
    return dists.argmin(dim=1).numpy()

def load_audio(path,dur=2):
    d,sr=sf.read(path)
    if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr),axis=0)
    L=int(dur*SR); d=d[:L]
    if d.ndim>1: d=d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
src=load_audio(f'{base}/p255/p255_001_mic1.flac')
tgt=load_audio(f'{base}/p226/p226_001_mic1.flac')

with torch.no_grad():
    sem_src,aco_src=encoder(src); sem_tgt,aco_tgt=encoder(tgt)
    T=min(sem_src.shape[1],sem_tgt.shape[1],aco_src.shape[2],aco_tgt.shape[2])
    sem_s=sem_src[:,:T]; aco_s=aco_src[:,:,:T]; aco_t=aco_tgt[:,:,:T]
    cb0_src_np=aco_s[0,0,:].numpy(); cb0_tgt_np=aco_t[0,0,:].numpy()

# Test round-trip
emb_src=tokens_to_emb(cb0_src_np)
tokens_rt=emb_to_tokens(emb_src)
match=(tokens_rt==cb0_src_np).mean()
print(f"Round-trip accuracy: {match:.4f}")

def causal_ema_emb(emb, alpha):
    B,D,T=emb.shape
    out=torch.zeros_like(emb)
    running=emb[:,:,0].clone()
    out[:,:,0]=running
    for t in range(1,T):
        running=alpha*running+(1-alpha)*emb[:,:,t]
        out[:,:,t]=running
    return out

results=[]
def evaluate(name, cb0_vc_np, **params):
    aco_vc=aco_s.clone()
    aco_vc[0,0,:]=torch.from_numpy(cb0_vc_np)
    vc=decoder(sem_s,aco_vc)
    sem_vc,_=encoder(vc)
    sem_s2,_=encoder(src); sem_t2,_=encoder(tgt)
    T2=min(sem_vc.shape[1],sem_s2.shape[1],sem_t2.shape[1])
    cs=torch.nn.functional.cosine_similarity(sem_vc[:,:T2].float().reshape(-1),sem_s2[:,:T2].float().reshape(-1),dim=0)
    ct=torch.nn.functional.cosine_similarity(sem_vc[:,:T2].float().reshape(-1),sem_t2[:,:T2].float().reshape(-1),dim=0)
    r={'name':name,'cos_src':cs.item(),'cos_tgt':ct.item(),'delta':(ct-cs).item(),
       'rms':vc.detach().squeeze().std().item()}
    r.update(params)
    results.append(r)
    sf.write(str(OUT/f'{name}.wav'),vc.detach().squeeze().numpy(),SR)
    return r

# ── Token EMA baseline ──
print()
print("=== Token EMA Baselines ===")
for alpha in [0.85,0.88,0.90,0.92,0.94,0.96,0.98]:
    cb0=np.zeros_like(cb0_tgt_np)
    r=int(cb0_tgt_np[0]); cb0[0]=r
    for t in range(1,len(cb0_tgt_np)):
        r=int(alpha*r+(1-alpha)*cb0_tgt_np[t])
        cb0[t]=r
    r=evaluate(f'baselines/token_ema_a{alpha}', cb0, alpha=alpha, method='token_ema')
    print(f"  token_ema α={alpha}: Δ={r['delta']:+.4f}")

# ── Embedding EMA ──
print()
print("=== Embedding EMA ===")
emb_tgt=tokens_to_emb(cb0_tgt_np)
for alpha in [0.85,0.88,0.90,0.92,0.94,0.96,0.98]:
    emb_smooth=causal_ema_emb(emb_tgt, alpha)
    cb0=emb_to_tokens(emb_smooth)
    r=evaluate(f'ema_embedding/a{alpha}', cb0, alpha=alpha, method='emb_ema')
    print(f"  emb_ema α={alpha}: Δ={r['delta']:+.4f}")

# ── Source-fast + Target-slow ──
print()
print("=== Source-fast + Target-slow ===")
emb_src=tokens_to_emb(cb0_src_np)
for alpha in [0.88,0.90,0.92,0.94,0.96]:
    slow_src=causal_ema_emb(emb_src, alpha)
    fast_src=emb_src-slow_src
    slow_tgt=causal_ema_emb(emb_tgt, alpha)
    for beta in [0.0,0.25,0.5,0.75,1.0]:
        emb_mix=slow_tgt+beta*fast_src
        cb0=emb_to_tokens(emb_mix)
        r=evaluate(f'fast_slow/a{alpha}_b{beta}', cb0, alpha=alpha, beta=beta, method='fast_slow')

# ── Summary ──
best_delta=max(results,key=lambda r:r['delta'])
best_content=max(results,key=lambda r:r['cos_src'])
print()
print(f"Best Δ: {best_delta['name']} Δ={best_delta['delta']:+.4f}")
print(f"Best cos_src: {best_content['name']} cos_src={best_content['cos_src']:.4f}")
print(f"Total: {len(results)} results")

with open(OUT/'metrics.json','w') as f:
    json.dump(results,f,indent=2)
print(f"Saved to {OUT}/metrics.json")
