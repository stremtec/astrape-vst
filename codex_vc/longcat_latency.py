"""LongCat low-latency CB0 VC experiment."""
import sys, os; sys.path.insert(0,'/tmp/LongCat-Audio-Codec'); os.chdir('/tmp/LongCat-Audio-Codec')
import torch, soundfile as sf, numpy as np, time, json
from scipy import signal, stats
from pathlib import Path
from networks.semantic_codec.model_loader import load_encoder, load_decoder

SR=24000; NL=chr(10); FRAME_MS=60  # ~16.6Hz
OUT=Path('/Users/asill/research5/longcat_latency')

print("Loading...")
encoder=load_encoder('configs/LongCatAudioCodec_encoder.yaml', torch.device('cpu'))
decoder=load_decoder('configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml', torch.device('cpu'))

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
    sem_s=sem_src[:,:T]; aco_s=aco_src[:,:,:T]
    cb0_src=aco_s[0,0,:].numpy(); cb0_tgt=aco_tgt[0,0,:T].numpy()

# Target-side precomputation (offline)
print("Precomputing target style cache...")
tgt_hist=dict(zip(*np.unique(cb0_tgt,return_counts=True)))
tgt_mode=int(stats.mode(cb0_tgt,keepdims=True)[0][0])
tgt_topk=sorted(tgt_hist.items(),key=lambda x:-x[1])
tgt_topk_tokens=np.array([t for t,_ in tgt_topk])

# CB0 embedding cache for nearest-neighbor
CACHE=Path('/Users/asill/research5/longcat_vc_quality_min/cache')
CB0_EMBS=torch.load(CACHE/'cb0_emb_table.pt', weights_only=True)
MAX_TOK=CB0_EMBS.shape[0]-1

def emb_to_tokens(emb):
    e_flat=emb[0].T; dists=torch.cdist(e_flat, CB0_EMBS)
    return dists.argmin(dim=1).numpy()

# EMA teacher
cb0_teacher=np.zeros_like(cb0_tgt)
r=int(cb0_tgt[0]); cb0_teacher[0]=r
for t in range(1,len(cb0_tgt)):
    r=int(0.94*r+0.06*cb0_tgt[t])
    cb0_teacher[t]=r

# ── Methods ──
def method_global_mode():
    return np.full_like(cb0_src, tgt_mode)

def method_topk_nearest(k):
    target_set=set(tgt_topk_tokens[:k])
    out=np.zeros_like(cb0_src)
    for t in range(len(cb0_src)):
        if cb0_src[t] in target_set:
            out[t]=cb0_src[t]
        else:
            # Find nearest target token
            src_emb=CB0_EMBS[cb0_src[t]]
            best=None; best_d=float('inf')
            for tok in target_set:
                d=((CB0_EMBS[tok]-src_emb)**2).sum().item()
                if d<best_d: best_d=d; best=tok
            out[t]=best
    return out

def method_histogram_rank_remap():
    # Map source CB0 rank to target CB0 rank
    src_hist=dict(zip(*np.unique(cb0_src,return_counts=True)))
    src_ranked=sorted(src_hist.items(),key=lambda x:-x[1])
    tgt_ranked=tgt_topk
    src_to_tgt={}
    for i,(stok,_) in enumerate(src_ranked):
        ti=min(i,len(tgt_ranked)-1)
        src_to_tgt[stok]=tgt_ranked[ti][0]
    out=np.zeros_like(cb0_src)
    for t in range(len(cb0_src)):
        out[t]=src_to_tgt.get(cb0_src[t], tgt_mode)
    return out

def method_ema(alpha):
    out=np.zeros_like(cb0_tgt)
    r=int(cb0_tgt[0]); out[0]=r
    for t in range(1,len(cb0_tgt)):
        r=int(alpha*r+(1-alpha)*cb0_tgt[t])
        out[t]=r
    return out

def method_causal_mode(window):
    out=np.zeros_like(cb0_tgt)
    for t in range(len(cb0_tgt)):
        start=max(0,t-window+1)
        out[t]=stats.mode(cb0_tgt[start:t+1],keepdims=True)[0][0]
    return out

def method_hysteresis(persist=2):
    out=np.zeros_like(cb0_tgt)
    out[0]=cb0_src[0]  # start with source
    candidate=None; candidate_count=0
    for t in range(1,len(cb0_tgt)):
        new_candidate=cb0_tgt[t]
        if new_candidate==candidate:
            candidate_count+=1
            if candidate_count>=persist:
                out[t]=candidate
        else:
            candidate=new_candidate; candidate_count=1
            out[t]=out[t-1]  # keep previous
    return out

# ── Evaluate ──
results=[]
def evaluate(name, cb0_vc_np, latency_frames=0, mapper_ms=0):
    if cb0_vc_np.max()>MAX_TOK or cb0_vc_np.min()<0:
        print(f"  SKIP {name}: token OOB"); return
    t0=time.time()
    aco_vc=aco_s.clone(); aco_vc[0,0,:]=torch.from_numpy(cb0_vc_np)
    vc=decoder(sem_s,aco_vc)
    decode_ms=(time.time()-t0)*1000
    vc_np=vc.detach().squeeze().numpy()
    total_latency=latency_frames*FRAME_MS+mapper_ms+decode_ms
    
    sem_vc,_=encoder(vc)
    sem_s2,_=encoder(src); sem_t2,_=encoder(tgt)
    T2=min(sem_vc.shape[1],sem_s2.shape[1],sem_t2.shape[1])
    cs=torch.nn.functional.cosine_similarity(sem_vc[:,:T2].float().reshape(-1),sem_s2[:,:T2].float().reshape(-1),dim=0)
    ct=torch.nn.functional.cosine_similarity(sem_vc[:,:T2].float().reshape(-1),sem_t2[:,:T2].float().reshape(-1),dim=0)
    
    row={'name':name,'delta':(ct-cs).item(),'cos_src':cs.item(),'cos_tgt':ct.item(),
         'rms':float(np.sqrt(np.mean(vc_np**2))),
         'added_ms':latency_frames*FRAME_MS+mapper_ms,
         'total_ms':total_latency,'decode_ms':decode_ms,'mapper_ms':mapper_ms,
         'latency_frames':latency_frames}
    results.append(row)
    (OUT/'wavs').mkdir(exist_ok=True)
    sf.write(str(OUT/'wavs'/f'{name}.wav'), vc_np, SR)
    return row

# ── Run ──
print()
print("=== Low-Latency CB0 Methods ===")
evaluate('all_source', cb0_src, latency_frames=0)
evaluate('ema_0.94_teacher', cb0_teacher, latency_frames=0, mapper_ms=0)  # teacher ref

# 0-latency
evaluate('global_mode', method_global_mode(), latency_frames=0, mapper_ms=1)
for k in [4,8,16,32,64]:
    evaluate(f'topk_nearest_{k}', method_topk_nearest(k), latency_frames=0, mapper_ms=k*0.1)
evaluate('histogram_remap', method_histogram_rank_remap(), latency_frames=0, mapper_ms=1)

# Short EMA
for alpha in [0.50,0.60,0.70,0.80]:
    evaluate(f'ema_{alpha:.2f}', method_ema(alpha), latency_frames=1, mapper_ms=1)

# Causal mode 2fr
evaluate('mode_2fr', method_causal_mode(2), latency_frames=2, mapper_ms=1)

# Hysteresis
evaluate('hysteresis_2', method_hysteresis(2), latency_frames=1, mapper_ms=1)

# ── Report ──
print()
print("=== Latency Report ===")
hdr=f"{'method':<22s} {'Δ':>8s} {'cos_src':>8s} {'cos_tgt':>8s} {'add_ms':>7s} {'total_ms':>8s} {'RMS':>8s}"
print(hdr); print('-'*80)
for r in sorted(results, key=lambda r: r['delta'], reverse=True):
    print(f"{r['name']:<22s} {r['delta']:+8.4f} {r['cos_src']:8.4f} {r['cos_tgt']:8.4f} "
          f"{r['added_ms']:7.0f} {r['total_ms']:8.0f} {r['rms']:8.4f}")

with open(OUT/'latency_report.md','w') as f:
    f.write(hdr+NL); f.write('-'*80+NL)
    for r in sorted(results, key=lambda r: r['delta'], reverse=True):
        f.write(f"{r['name']:<22s} {r['delta']:+8.4f} {r['cos_src']:8.4f} {r['cos_tgt']:8.4f} "
                f"{r['added_ms']:7.0f} {r['total_ms']:8.0f} {r['rms']:8.4f}{NL}")
print(f"Saved to {OUT}/latency_report.md")
