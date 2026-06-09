"""LongCat live latency benchmark — true pipeline measurement."""
import sys, os; sys.path.insert(0,'/tmp/LongCat-Audio-Codec'); os.chdir('/tmp/LongCat-Audio-Codec')
import torch, soundfile as sf, numpy as np, time, json
from scipy import signal, stats
from pathlib import Path
from networks.semantic_codec.model_loader import load_encoder, load_decoder

SR=24000; NL=chr(10); FRAME_MS=60  # ~16.6Hz
OUT=Path('/Users/asill/research5/longcat_latency_v2')
CACHE=OUT/'cache'; WAVS=OUT/'wavs'
for d in [CACHE,WAVS]: d.mkdir(parents=True,exist_ok=True)
LIVE_LOG=OUT/'live_latency.jsonl'
QUAL_LOG=OUT/'quality_metrics.jsonl'

# ── Device ──
if torch.backends.mps.is_available():
    device=torch.device('mps')
    print(f"Device: MPS")
else:
    device=torch.device('cpu')
    print(f"Device: CPU")

# ── Load models (outside live path) ──
t0=time.time()
encoder=load_encoder('configs/LongCatAudioCodec_encoder.yaml', device)
decoder=load_decoder('configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml', device)
encoder.eval(); decoder.eval()
print(f"Model loading: {(time.time()-t0)*1000:.0f}ms")

def load_audio(path,dur=2):
    d,sr=sf.read(path)
    if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr),axis=0)
    L=int(dur*SR); d=d[:L]
    if d.ndim>1: d=d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# ── Target precomputation (OFFLINE) ──
print("Target precompute (offline)...")
tgt=load_audio(f'{base}/p226/p226_001_mic1.flac')
with torch.inference_mode():
    sem_tgt,aco_tgt_raw=encoder(tgt)
cb0_tgt=aco_tgt_raw[0,0,:].cpu().numpy()
tgt_hist=dict(zip(*np.unique(cb0_tgt,return_counts=True)))
tgt_topk=sorted(tgt_hist.items(),key=lambda x:-x[1])
tgt_mode=int(stats.mode(cb0_tgt,keepdims=True)[0][0])
target_cache={'mode':tgt_mode,'topk_tokens':{},'histogram':tgt_hist}
for k in [4,8,12,16,24,32]:
    target_cache['topk_tokens'][k]=np.array([t for t,_ in tgt_topk[:k]])
CB0_EMBS=torch.load('/Users/asill/research5/longcat_vc_quality_min/cache/cb0_emb_table.pt',weights_only=True).to(device)
target_cache['topk_embs']={}
for k,tokens in target_cache['topk_tokens'].items():
    target_cache['topk_embs'][k]=CB0_EMBS[tokens.astype(np.int64)]

# ── Source encoding (LIVE) ──
src=load_audio(f'{base}/p255/p255_001_mic1.flac')
with torch.inference_mode():
    t_enc0=time.time()
    sem_src,aco_src=encoder(src)
    source_encode_ms=(time.time()-t_enc0)*1000

T=min(sem_src.shape[1],aco_src.shape[2])
sem_s=sem_src[:,:T]; aco_s=aco_src[:,:,:T]
cb0_src=aco_s[0,0,:].cpu().numpy()

# ── Fast top-k nearest mapping ──
def fast_topk_nearest(k):
    tokens=target_cache['topk_tokens'][k]
    embs=target_cache['topk_embs'][k]  # (K, 1024) on device
    
    t0=time.time()
    src_embs=CB0_EMBS[torch.from_numpy(cb0_src.astype(np.int64)).to(device)]  # (T, 1024)
    dists=torch.cdist(src_embs, embs)  # (T, K)
    nearest_idx=dists.argmin(dim=1).cpu().numpy()  # (T,)
    cb0_out=tokens[nearest_idx]
    mapping_ms=(time.time()-t0)*1000
    return cb0_out, mapping_ms

def fast_global_mode():
    t0=time.time()
    cb0_out=np.full_like(cb0_src, target_cache['mode'])
    mapping_ms=(time.time()-t0)*1000
    return cb0_out, mapping_ms

# ── Decode (LIVE) ──
def decode_cb0(cb0_vc_np):
    aco_vc=aco_s.clone()
    aco_vc[0,0,:]=torch.from_numpy(cb0_vc_np).to(device)
    with torch.inference_mode():
        t0=time.time()
        vc=decoder(sem_s,aco_vc)
        decode_ms=(time.time()-t0)*1000
    return vc, decode_ms

# ── Quality evaluation (OFFLINE, not in live path) ──
def eval_quality(name, vc):
    t0=time.time()
    vc_np=vc.detach().cpu().squeeze().numpy()
    
    # Re-encode for cosine (evaluation only!)
    vc_t=torch.from_numpy(vc_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        sem_vc,_=encoder(vc_t)
    sem_s2=sem_s
    with torch.inference_mode():
        sem_t2,_=encoder(tgt)
    
    T2=min(sem_vc.shape[1],sem_s2.shape[1],sem_t2.shape[1])
    sv=sem_vc[:,:T2].float().reshape(-1); ss=sem_s2[:,:T2].float().reshape(-1)
    st2=sem_t2[:,:T2].float().reshape(-1)
    cs=torch.nn.functional.cosine_similarity(sv,ss,dim=0).item()
    ct=torch.nn.functional.cosine_similarity(sv,st2,dim=0).item()
    
    row={'name':name,'delta':ct-cs,'cos_src':cs,'cos_tgt':ct,
         'rms':float(np.sqrt(np.mean(vc_np**2)))}
    eval_ms=(time.time()-t0)*1000
    
    sf.write(str(WAVS/f'{name}.wav'), vc_np, SR)
    with open(QUAL_LOG,'a') as f: f.write(json.dumps(row)+NL); f.flush()
    return row, eval_ms

# ── Run benchmarks ──
print()
print("=== Live Latency Benchmark ===")
hdr=f"{'method':<22s} {'Δ':>8s} {'cos_src':>8s} {'enc_ms':>7s} {'map_ms':>7s} {'dec_ms':>7s} {'live_ms':>8s} {'eval_ms':>7s}"
print(hdr); print('-'*95)

with open(LIVE_LOG,'a') as lf:
    lf.write('# '+hdr+NL)
    
    for method_name, cb0_fn in [
        ('all_source', lambda: (cb0_src.copy(), 0)),
        ('global_mode', fast_global_mode),
        ('topk_4', lambda: fast_topk_nearest(4)),
        ('topk_8', lambda: fast_topk_nearest(8)),
        ('topk_12', lambda: fast_topk_nearest(12)),
        ('topk_16', lambda: fast_topk_nearest(16)),
        ('topk_24', lambda: fast_topk_nearest(24)),
        ('topk_32', lambda: fast_topk_nearest(32)),
    ]:
        cb0_vc, map_ms=cb0_fn()
        vc, dec_ms=decode_cb0(cb0_vc)
        total_live_ms=source_encode_ms+map_ms+dec_ms
        
        quality, eval_ms=eval_quality(method_name, vc)
        
        line=(f"{method_name:<22s} {quality['delta']:+8.4f} {quality['cos_src']:8.4f} "
              f"{source_encode_ms:7.0f} {map_ms:7.1f} {dec_ms:7.0f} "
              f"{total_live_ms:8.0f} {eval_ms:7.0f}")
        print(line)
        lf.write(line+NL); lf.flush()

# ── Chunk-size test ──
print()
print("=== Chunk-size Latency Test ===")
for dur_s in [0.25, 0.50, 1.0, 2.0]:
    src_chunk=load_audio(f'{base}/p255/p255_001_mic1.flac', dur=dur_s)
    with torch.inference_mode():
        t0=time.time(); sem_c,aco_c=encoder(src_chunk)
        enc_ms=(time.time()-t0)*1000
        T_c=sem_c.shape[1]
        t0=time.time(); vc_c=decoder(sem_c,aco_c)
        dec_ms=(time.time()-t0)*1000
    rtf=(enc_ms+dec_ms)/(dur_s*1000)
    print(f"  {dur_s:.2f}s ({T_c}frames): enc={enc_ms:.0f}ms dec={dec_ms:.0f}ms total={enc_ms+dec_ms:.0f}ms RTF={rtf:.2f}")

# ── Summary ──
print()
print("=== Summary ===")
with open(LIVE_LOG) as f:
    lines=[l for l in f if not l.startswith('#')]
print(hdr); print('-'*95)
for line in lines: print(line.rstrip())

with open(OUT/'summary.md','w') as f:
    f.write(hdr+NL); f.write('-'*95+NL)
    for line in lines: f.write(line)
print(f"Saved to {OUT}/")
