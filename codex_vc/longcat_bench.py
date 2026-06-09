"""LongCat minimal quality benchmark — resumable, cached."""
import sys, os; sys.path.insert(0,'/tmp/LongCat-Audio-Codec'); os.chdir('/tmp/LongCat-Audio-Codec')
import torch, soundfile as sf, numpy as np, time, json
from scipy import signal
from pathlib import Path
from networks.semantic_codec.model_loader import load_encoder, load_decoder

SR=24000; NL=chr(10)
OUT=Path('/Users/asill/research5/longcat_vc_quality_min')
CACHE=OUT/'cache'; WAVS=OUT/'wavs'
for d in [CACHE,WAVS]: d.mkdir(parents=True,exist_ok=True)
METRICS_FILE=OUT/'metrics.jsonl'
SUMMARY_FILE=OUT/'summary.md'

print("Loading models...")
encoder=load_encoder('configs/LongCatAudioCodec_encoder.yaml', torch.device('cpu'))
decoder=load_decoder('configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml', torch.device('cpu'))

def load_audio(path,dur=2):
    d,sr=sf.read(path)
    if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr),axis=0)
    L=int(dur*SR); d=d[:L]
    if d.ndim>1: d=d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

if (CACHE/'src_tokens.pt').exists():
    print("Loading cached tokens...")
    cache=torch.load(CACHE/'src_tokens.pt', weights_only=True)
    sem_src=cache['sem_src']; aco_src=cache['aco_src']
    sem_tgt=cache['sem_tgt']; aco_tgt=cache['aco_tgt']
else:
    print("Encoding source and target...")
    src=load_audio(f'{base}/p255/p255_001_mic1.flac')
    tgt=load_audio(f'{base}/p226/p226_001_mic1.flac')
    with torch.no_grad():
        sem_src,aco_src=encoder(src); sem_tgt,aco_tgt=encoder(tgt)
    T=min(sem_src.shape[1],sem_tgt.shape[1],aco_src.shape[2],aco_tgt.shape[2])
    sem_src=sem_src[:,:T]; aco_src=aco_src[:,:,:T]
    sem_tgt=sem_tgt[:,:T]; aco_tgt=aco_tgt[:,:,:T]
    torch.save({'sem_src':sem_src,'aco_src':aco_src,'sem_tgt':sem_tgt,'aco_tgt':aco_tgt}, CACHE/'src_tokens.pt')
    print(f"  Cached: T={T}")

cb0_src_np=aco_src[0,0,:].numpy()
cb0_tgt_np=aco_tgt[0,0,:].numpy()
sem_s=sem_src; aco_s=aco_src

if (CACHE/'cb0_emb_table.pt').exists():
    print("Loading CB0 embedding table...")
    CB0_EMBS=torch.load(CACHE/'cb0_emb_table.pt', weights_only=True)
else:
    print("Building CB0 embedding table...")
    max_valid=0
    for tok in range(10000):
        try:
            t=torch.tensor([[[tok]]]).long()
            e,_,_=decoder.acoustic_quantizer.from_codes(t)
            max_valid=tok
        except: break
    print(f"  Max valid token: {max_valid}")
    embs=torch.zeros(max_valid+1,1024)
    for tok in range(0,max_valid+1,5):
        t=torch.tensor([[[tok]]]).long()
        e,_,_=decoder.acoustic_quantizer.from_codes(t)
        embs[tok]=e.detach()[0,:,0]
    for tok in range(max_valid+1):
        if embs[tok].sum()==0:
            lo=tok-(tok%5); hi=min(lo+5,max_valid)
            a=(tok-lo)/max(1,hi-lo)
            embs[tok]=embs[lo]*(1-a)+embs[hi]*a
    CB0_EMBS=embs
    torch.save(CB0_EMBS, CACHE/'cb0_emb_table.pt')

MAX_TOK=CB0_EMBS.shape[0]-1

def tokens_to_emb(tokens_np):
    return CB0_EMBS[tokens_np.astype(np.int64)].T.unsqueeze(0)

def emb_to_tokens(emb):
    e_flat=emb[0].T
    dists=torch.cdist(e_flat, CB0_EMBS)
    return dists.argmin(dim=1).numpy()

emb_test=tokens_to_emb(cb0_src_np)
rt=emb_to_tokens(emb_test)
rt_acc=(rt==cb0_src_np).mean()
print(f"Round-trip accuracy: {rt_acc:.4f}")

existing=set()
if METRICS_FILE.exists():
    with open(METRICS_FILE) as f:
        for line in f:
            try: existing.add(json.loads(line)['name'])
            except: pass
print(f"Existing results: {len(existing)}")

def causal_ema_emb(emb, alpha):
    B,D,T=emb.shape
    out=torch.zeros_like(emb)
    running=emb[:,:,0].clone()
    out[:,:,0]=running
    for t in range(1,T):
        running=alpha*running+(1-alpha)*emb[:,:,t]
        out[:,:,t]=running
    return out

def make_cb0_token_ema(alpha):
    cb0=np.zeros_like(cb0_tgt_np)
    r=int(cb0_tgt_np[0]); cb0[0]=r
    for t in range(1,len(cb0_tgt_np)):
        r=int(alpha*r+(1-alpha)*cb0_tgt_np[t])
        cb0[t]=r
    return cb0

def make_cb0_emb_ema(alpha):
    emb_tgt=tokens_to_emb(cb0_tgt_np)
    emb_smooth=causal_ema_emb(emb_tgt, alpha)
    return emb_to_tokens(emb_smooth)

def make_fastslow(alpha, beta):
    emb_src=tokens_to_emb(cb0_src_np)
    emb_tgt=tokens_to_emb(cb0_tgt_np)
    slow_tgt=causal_ema_emb(emb_tgt, alpha)
    slow_src=causal_ema_emb(emb_src, alpha)
    fast_src=emb_src-slow_src
    emb_mix=slow_tgt+beta*fast_src
    return emb_to_tokens(emb_mix)

src=load_audio(f'{base}/p255/p255_001_mic1.flac')
tgt=load_audio(f'{base}/p226/p226_001_mic1.flac')

def evaluate(name, cb0_vc_np, expensive=False):
    if name in existing:
        print(f"  SKIP {name} (already done)")
        return
    print(f"  {name}...", end='', flush=True)
    t0=time.time()
    if cb0_vc_np.max()>MAX_TOK or cb0_vc_np.min()<0:
        print(f" ERROR: token out of range")
        return
    aco_vc=aco_s.clone(); aco_vc[0,0,:]=torch.from_numpy(cb0_vc_np)
    vc=decoder(sem_s,aco_vc)
    decode_ms=(time.time()-t0)*1000
    vc_np=vc.detach().squeeze().numpy()
    row={'name':name,'rms':float(np.sqrt(np.mean(vc_np**2))),
         'peak':float(np.max(np.abs(vc_np))),'decode_ms':decode_ms,
         'edit_src':float(np.mean(cb0_vc_np!=cb0_src_np)),
         'edit_tgt':float(np.mean(cb0_vc_np!=cb0_tgt_np)),
         'n_unique_cb0':int(len(np.unique(cb0_vc_np)))}
    if expensive:
        sem_vc,_=encoder(vc)
        sem_s2,_=encoder(src); sem_t2,_=encoder(tgt)
        T2=min(sem_vc.shape[1],sem_s2.shape[1],sem_t2.shape[1])
        cs=torch.nn.functional.cosine_similarity(sem_vc[:,:T2].float().reshape(-1),sem_s2[:,:T2].float().reshape(-1),dim=0)
        ct=torch.nn.functional.cosine_similarity(sem_vc[:,:T2].float().reshape(-1),sem_t2[:,:T2].float().reshape(-1),dim=0)
        row['cos_src']=cs.item(); row['cos_tgt']=ct.item(); row['delta']=(ct-cs).item()
    sf.write(str(WAVS/f'{name}.wav'), vc_np, SR)
    with open(METRICS_FILE,'a') as f:
        f.write(json.dumps(row)+NL); f.flush()
    dt=time.time()-t0
    print(f" done ({dt:.1f}s)" if not expensive else f" done ({dt:.1f}s) [expensive]")
    existing.add(name)
    return row

print()
print("=== Running candidates ===")
candidates=[]
candidates.append(('baseline_all_source', lambda: cb0_src_np, False))
candidates.append(('token_ema_0.90', lambda: make_cb0_token_ema(0.90), False))
candidates.append(('token_ema_0.94', lambda: make_cb0_token_ema(0.94), True))
candidates.append(('token_ema_0.96', lambda: make_cb0_token_ema(0.96), False))
candidates.append(('emb_ema_0.90', lambda: make_cb0_emb_ema(0.90), False))
candidates.append(('emb_ema_0.94', lambda: make_cb0_emb_ema(0.94), True))
candidates.append(('fastslow_a0.94_b0.25', lambda: make_fastslow(0.94,0.25), False))
candidates.append(('fastslow_a0.94_b0.50', lambda: make_fastslow(0.94,0.50), True))
candidates.append(('fastslow_a0.94_b0.75', lambda: make_fastslow(0.94,0.75), False))
for name, fn, expensive in candidates:
    evaluate(name, fn(), expensive=expensive)

print()
print("=== Summary ===")
results=[]
with open(METRICS_FILE) as f:
    for line in f: results.append(json.loads(line))
hdr=f"{'method':<30s} {'delta':>8s} {'cos_src':>8s} {'cos_tgt':>8s} {'RMS':>8s} {'dec_ms':>8s}"
print(hdr); print('-'*70)
for r in results:
    d=r.get('delta','-'); cs=r.get('cos_src','-'); ct=r.get('cos_tgt','-')
    d_str=f"{d:+.4f}" if isinstance(d,float) else str(d)
    cs_str=f"{cs:.4f}" if isinstance(cs,float) else str(cs)
    ct_str=f"{ct:.4f}" if isinstance(ct,float) else str(ct)
    print(f"{r['name']:<30s} {d_str:>8s} {cs_str:>8s} {ct_str:>8s} {r['rms']:8.4f} {r['decode_ms']:8.0f}")
with open(SUMMARY_FILE,'w') as f:
    f.write(hdr+NL); f.write('-'*70+NL)
    for r in results:
        d=r.get('delta','-'); cs=r.get('cos_src','-'); ct=r.get('cos_tgt','-')
        d_str=f"{d:+.4f}" if isinstance(d,float) else str(d)
        cs_str=f"{cs:.4f}" if isinstance(cs,float) else str(cs)
        ct_str=f"{ct:.4f}" if isinstance(ct,float) else str(ct)
        f.write(f"{r['name']:<30s} {d_str:>8s} {cs_str:>8s} {ct_str:>8s} {r['rms']:8.4f} {r['decode_ms']:8.0f}{NL}")
print(f"Saved to {SUMMARY_FILE}")
