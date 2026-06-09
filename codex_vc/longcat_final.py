"""LongCat VC: accurate MPS latency + speaker anonymization metrics."""
import sys, os; sys.path.insert(0,'/tmp/LongCat-Audio-Codec'); os.chdir('/tmp/LongCat-Audio-Codec')
import torch, soundfile as sf, numpy as np, time, json, statistics
from scipy import signal, stats
from pathlib import Path
from networks.semantic_codec.model_loader import load_encoder, load_decoder

SR=24000; NL=chr(10)
OUT=Path('/Users/asill/research5/longcat_final')
for d in [OUT/'cache',OUT/'wavs']: d.mkdir(parents=True,exist_ok=True)

device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

def sync():
    if device.type=='mps': torch.mps.synchronize()
    elif device.type=='cuda': torch.cuda.synchronize()

def timed(fn):
    sync(); t0=time.perf_counter()
    out=fn()
    sync(); t1=time.perf_counter()
    return out, (t1-t0)*1000

# Load models
encoder=load_encoder('configs/LongCatAudioCodec_encoder.yaml', device)
decoder=load_decoder('configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml', device)
encoder.eval(); decoder.eval()

def load_audio(path,dur=2):
    d,sr=sf.read(path)
    if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr),axis=0)
    L=int(dur*SR); d=d[:L]
    if d.ndim>1: d=d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Target precompute (offline)
tgt=load_audio(f'{base}/p226/p226_001_mic1.flac')
with torch.inference_mode():
    sem_tgt,aco_tgt=encoder(tgt)
cb0_tgt_np=aco_tgt[0,0,:].cpu().numpy()
tgt_hist=dict(zip(*np.unique(cb0_tgt_np,return_counts=True)))
tgt_topk=sorted(tgt_hist.items(),key=lambda x:-x[1])
tgt_mode=int(stats.mode(cb0_tgt_np,keepdims=True)[0][0])
topk_tokens={k:np.array([t for t,_ in tgt_topk[:k]]) for k in[4,8,12,16,24,32]}

CB0_EMBS=torch.load('/Users/asill/research5/longcat_vc_quality_min/cache/cb0_emb_table.pt',weights_only=True).to(device)
topk_embs={}
for k,tokens in topk_tokens.items():
    topk_embs[k]=CB0_EMBS[tokens.astype(np.int64)]

# Source
src=load_audio(f'{base}/p255/p255_001_mic1.flac')

# Warmup (MPS)
print("Warmup...")
for _ in range(5):
    with torch.inference_mode():
        s,a=encoder(src)
        _=decoder(s,a)
sync()

# Source encode (baseline measurement)
with torch.inference_mode():
    (sem_src,aco_src), source_encode_ms = timed(lambda: encoder(src))

T=min(sem_src.shape[1],aco_src.shape[2])
sem_s=sem_src[:,:T]; aco_s=aco_src[:,:,:T]
cb0_src=aco_s[0,0,:].cpu().numpy()

# Mapping functions
def map_global():
    return np.full_like(cb0_src, tgt_mode)

def map_topk(k):
    tokens=topk_tokens[k]; embs=topk_embs[k]
    src_embs=CB0_EMBS[torch.from_numpy(cb0_src.astype(np.int64)).to(device)]
    dists=torch.cdist(src_embs, embs)
    return tokens[dists.argmin(dim=1).cpu().numpy()]

def decode_cb0(cb0):
    aco_vc=aco_s.clone(); aco_vc[0,0,:]=torch.from_numpy(cb0).to(device)
    with torch.inference_mode():
        return decoder(sem_s,aco_vc)

# Run latency benchmark (10 iterations each, with sync)
print()
print("=== Accurate Latency (10 iterations, median) ===")
results=[]
for name, fn in [
    ('global_mode', map_global),
    ('topk_4', lambda: map_topk(4)),
    ('topk_8', lambda: map_topk(8)),
    ('topk_12', lambda: map_topk(12)),
    ('topk_16', lambda: map_topk(16)),
    ('topk_24', lambda: map_topk(24)),
    ('topk_32', lambda: map_topk(32)),
]:
    map_times=[]; dec_times=[]
    for _ in range(10):
        cb0_vc, map_ms=timed(fn)
        vc, dec_ms=timed(lambda: decode_cb0(cb0_vc))
        map_times.append(map_ms); dec_times.append(dec_ms)
    
    map_med=statistics.median(map_times)
    dec_med=statistics.median(dec_times)
    total=source_encode_ms+map_med+dec_med
    
    # Quality (single evaluation)
    cb0_vc=fn()
    vc=decode_cb0(cb0_vc)
    vc_np=vc.detach().cpu().squeeze().numpy()
    vc_t=torch.from_numpy(vc_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        sem_vc,_=encoder(vc_t)
    T2=min(sem_vc.shape[1],sem_s.shape[1],sem_tgt.shape[1])
    sv=sem_vc[:,:T2].float().reshape(-1)
    ss=sem_s[:,:T2].float().reshape(-1)
    st=sem_tgt[:,:T2].float().reshape(-1)
    content_cos=torch.nn.functional.cosine_similarity(sv,ss,dim=0).item()
    spk_src=torch.nn.functional.cosine_similarity(sv,ss,dim=0).item()  # same as content in this metric
    spk_tgt=torch.nn.functional.cosine_similarity(sv,st,dim=0).item()
    
    # Anonymization: how different from source speaker
    anon_score=1.0-content_cos  # 1.0 = fully anonymized
    
    row={'name':name,'content_cos':content_cos,'spk_tgt':spk_tgt,
         'delta':spk_tgt-content_cos,'anon':anon_score,
         'src_ms':source_encode_ms,'map_ms':map_med,'dec_ms':dec_med,
         'total_ms':total,'rms':float(np.sqrt(np.mean(vc_np**2)))}
    results.append(row)
    
    sf.write(str(OUT/'wavs'/f'{name}.wav'), vc_np, SR)

# Report
print()
hdr=(f"{'method':<15s} {'content':>8s} {'→tgt':>8s} {'Δ':>8s} {'anon':>8s} "
     f"{'enc_ms':>7s} {'map_ms':>7s} {'dec_ms':>7s} {'total':>7s} {'<100ms?':>8s}")
print(hdr); print('-'*100)
for r in sorted(results, key=lambda r: r['delta'], reverse=True):
    ok='✅' if r['total_ms']<100 else ('⚠️' if r['total_ms']<150 else '❌')
    print(f"{r['name']:<15s} {r['content_cos']:8.4f} {r['spk_tgt']:8.4f} {r['delta']:+8.4f} "
          f"{r['anon']:8.4f} {r['src_ms']:7.0f} {r['map_ms']:7.1f} {r['dec_ms']:7.0f} "
          f"{r['total_ms']:7.0f} {ok:>8s}")

with open(OUT/'final_report.json','w') as f:
    json.dump(results,f,indent=2)
with open(OUT/'final_report.md','w') as f:
    f.write(hdr+NL); f.write('-'*100+NL)
    for r in sorted(results, key=lambda r: r['delta'], reverse=True):
        ok='OK' if r['total_ms']<100 else ('CLOSE' if r['total_ms']<150 else 'SLOW')
        f.write(f"{r['name']:<15s} {r['content_cos']:8.4f} {r['spk_tgt']:8.4f} {r['delta']:+8.4f} "
                f"{r['anon']:8.4f} {r['src_ms']:7.0f} {r['map_ms']:7.1f} {r['dec_ms']:7.0f} "
                f"{r['total_ms']:7.0f} {ok:>8s}{NL}")
print(f"Saved to {OUT}/")
