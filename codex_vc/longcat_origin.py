"""LongCat VC quality: p255(EN male) → origin.mp3(KR female)."""
import sys, os; sys.path.insert(0,'/tmp/LongCat-Audio-Codec'); os.chdir('/tmp/LongCat-Audio-Codec')
import torch, soundfile as sf, numpy as np, time, json, subprocess
from scipy import signal, stats
from pathlib import Path
from networks.semantic_codec.model_loader import load_encoder, load_decoder

SR=24000; NL=chr(10); FRAME_MS=60
OUT=Path('/Users/asill/research5/longcat_origin_eval')
WAVS=OUT/'wavs'; WAVS.mkdir(parents=True,exist_ok=True)

device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

def sync():
    if device.type=='mps': torch.mps.synchronize()

# Load models
encoder=load_encoder('configs/LongCatAudioCodec_encoder.yaml', device)
decoder=load_decoder('configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml', device)
encoder.eval(); decoder.eval()

def load_audio(path,dur=2):
    d,sr=sf.read(path)
    if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr),axis=0)
    if dur is not None: L=int(dur*SR); d=d[:L]
    else: L=len(d); d=d[:L]
    if d.ndim>1: d=d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Convert origin.mp3 to 24kHz
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1',
                '-sample_fmt','s16','-t','3','/tmp/origin_24k.wav'],capture_output=True)

# Load source (p255) and target (origin)
src=load_audio(f'{base}/p255/p255_001_mic1.flac')
tgt=load_audio('/tmp/origin_24k.wav', dur=None)

# Encode
with torch.inference_mode():
    sem_src,aco_src=encoder(src)
    sem_tgt,aco_tgt=encoder(tgt)
T=min(sem_src.shape[1],sem_tgt.shape[1],aco_src.shape[2],aco_tgt.shape[2])
sem_s=sem_src[:,:T]; aco_s=aco_src[:,:,:T]; aco_t=aco_tgt[:,:,:T]
cb0_src=aco_s[0,0,:].cpu().numpy()
cb0_tgt=aco_t[0,0,:T].cpu().numpy()

# F0 analysis
def compute_f0(wav,sr=SR):
    from scipy.signal import correlate
    wav=wav-np.mean(wav); fl=int(0.025*sr); sl=int(0.010*sr)
    nf=(len(wav)-fl)//sl+1
    if nf<=0: return np.array([])
    f0s=[]
    for i in range(nf):
        frame=wav[i*sl:i*sl+fl]*np.hanning(fl)
        corr=correlate(frame,frame,mode='full'); corr=corr[len(corr)//2:]
        corr[:10]=0
        if corr.max()>0:
            pk=corr.argmax()
            if pk>0: f0=sr/pk
            else: f0=0
        else: f0=0
        f0s.append(f0 if 50<f0<500 else 0)
    f0s=np.array(f0s)
    return f0s[f0s>0]

# Target precomputation
tgt_hist=dict(zip(*np.unique(cb0_tgt,return_counts=True)))
tgt_topk=sorted(tgt_hist.items(),key=lambda x:-x[1])
tgt_mode=int(stats.mode(cb0_tgt,keepdims=True)[0][0])
topk_tokens={k:np.array([t for t,_ in tgt_topk[:k]]) for k in[4,8,12,16,24,32]}

CB0_EMBS=torch.load('/Users/asill/research5/longcat_vc_quality_min/cache/cb0_emb_table.pt',weights_only=True).to(device)
topk_embs={k:CB0_EMBS[tokens.astype(np.int64)] for k,tokens in topk_tokens.items()}

def map_global():
    return np.full_like(cb0_src, tgt_mode)

def map_topk(k):
    tokens=topk_tokens[k]; embs=topk_embs[k]
    src_embs=CB0_EMBS[torch.from_numpy(cb0_src.astype(np.int64)).to(device)]
    dists=torch.cdist(src_embs, embs)
    return tokens[dists.argmin(dim=1).cpu().numpy()]

def map_ema(alpha):
    out=np.zeros_like(cb0_tgt)
    r=int(cb0_tgt[0]); out[0]=r
    for t in range(1,len(cb0_tgt)):
        r=int(alpha*r+(1-alpha)*cb0_tgt[t])
        out[t]=r
    return out

def decode_cb0(cb0):
    aco_vc=aco_s.clone(); aco_vc[0,0,:]=torch.from_numpy(cb0).to(device)
    with torch.inference_mode():
        return decoder(sem_s,aco_vc)

# Compute source/target reference metrics
src_np=src.cpu().squeeze().numpy()
tgt_np=tgt.cpu().squeeze().numpy()
f0_src=compute_f0(src_np[:len(src_np)])
f0_tgt=compute_f0(tgt_np[:len(tgt_np)])
f0_src_mean=np.median(f0_src) if len(f0_src)>0 else 0
f0_tgt_mean=np.median(f0_tgt) if len(f0_tgt)>0 else 0
print(f"Source F0 median: {f0_src_mean:.0f}Hz, Target F0 median: {f0_tgt_mean:.0f}Hz")

# Speaker reference embeddings (via LongCat re-encode of source/target audio)
with torch.inference_mode():
    sem_s_ref,_=encoder(src)
    sem_t_ref,_=encoder(tgt)

def compute_metrics(name, vc):
    vc_np=vc.detach().cpu().squeeze().numpy()
    vc_t=torch.from_numpy(vc_np).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Re-encode for semantic cosine
    with torch.inference_mode():
        sem_vc,_=encoder(vc_t)
    T2=min(sem_vc.shape[1],sem_s_ref.shape[1],sem_t_ref.shape[1])
    sv=sem_vc[:,:T2].float().reshape(-1)
    ss=sem_s_ref[:,:T2].float().reshape(-1)
    st=sem_t_ref[:,:T2].float().reshape(-1)
    
    content_cos=torch.nn.functional.cosine_similarity(sv,ss,dim=0).item()
    spk_src=content_cos
    spk_tgt=torch.nn.functional.cosine_similarity(sv,st,dim=0).item()
    delta=spk_tgt-spk_src
    
    # Anonymization
    # Reference: source self-similarity (how similar is source to itself)
    src_self=torch.nn.functional.cosine_similarity(ss,ss,dim=0).item()  # ~1.0
    # Random speaker similarity (source vs random different speaker = target)
    src_random=torch.nn.functional.cosine_similarity(ss,st,dim=0).item()
    anon=1.0-max(0,min(1,(spk_src-src_random)/(src_self-src_random+1e-8)))
    
    # F0
    f0_vc=compute_f0(vc_np[:len(vc_np)])
    f0_vc_mean=np.median(f0_vc) if len(f0_vc)>0 else 0
    f0_dist_src=abs(f0_vc_mean-f0_src_mean) if f0_src_mean>0 and f0_vc_mean>0 else 0
    f0_dist_tgt=abs(f0_vc_mean-f0_tgt_mean) if f0_tgt_mean>0 and f0_vc_mean>0 else 0
    
    # Audio stats
    rms=float(np.sqrt(np.mean(vc_np**2)))
    peak=float(np.max(np.abs(vc_np)))
    silent=float(np.mean(np.abs(vc_np)<1e-4))
    
    row={'name':name,'content_cos':content_cos,'spk_src':spk_src,'spk_tgt':spk_tgt,
         'delta':delta,'anon':anon,'f0_vc':f0_vc_mean,
         'f0_dist_src':f0_dist_src,'f0_dist_tgt':f0_dist_tgt,
         'rms':rms,'peak':peak,'silent_ratio':silent}
    
    sf.write(str(WAVS/f'{name}.wav'), vc_np, SR)
    return row

# Run methods
print()
print("=== p255 → origin.mp3 Quality Evaluation ===")
results=[]

for name, fn in [
    ('all_source', lambda: cb0_src.copy()),
    ('global_mode', map_global),
    ('topk_4', lambda: map_topk(4)),
    ('topk_8', lambda: map_topk(8)),
    ('topk_12', lambda: map_topk(12)),
    ('topk_16', lambda: map_topk(16)),
    ('topk_24', lambda: map_topk(24)),
    ('topk_32', lambda: map_topk(32)),
    ('ema_0.70', lambda: map_ema(0.70)),
    ('ema_0.94', lambda: map_ema(0.94)),
]:
    print(f"  {name}...", end='', flush=True)
    cb0=fn(); vc=decode_cb0(cb0)
    row=compute_metrics(name, vc)
    results.append(row)
    print(f" Δ={row['delta']:+.4f} content={row['content_cos']:.4f} anon={row['anon']:.4f} F0={row['f0_vc']:.0f}Hz")

# Report
print()
hdr=(f"{'method':<16s} {'content':>8s} {'→src':>8s} {'→tgt':>8s} {'Δ':>8s} "
     f"{'anon':>8s} {'F0vc':>7s} {'F0→src':>8s} {'F0→tgt':>8s} {'RMS':>8s}")
print(hdr); print('-'*105)
for r in sorted(results, key=lambda r: r['delta'], reverse=True):
    print(f"{r['name']:<16s} {r['content_cos']:8.4f} {r['spk_src']:8.4f} {r['spk_tgt']:8.4f} "
          f"{r['delta']:+8.4f} {r['anon']:8.4f} {r['f0_vc']:7.0f} "
          f"{r['f0_dist_src']:8.0f} {r['f0_dist_tgt']:8.0f} {r['rms']:8.4f}")

# Summary
with open(OUT/'metrics.jsonl','w') as f:
    for r in results: f.write(json.dumps(r)+NL)
with open(OUT/'summary.md','w') as f:
    f.write('# LongCat VC: p255(EN male) → origin.mp3(KR female)'+NL+NL)
    f.write(f'Src F0: {f0_src_mean:.0f}Hz, Tgt F0: {f0_tgt_mean:.0f}Hz'+NL+NL)
    f.write(hdr+NL); f.write('-'*105+NL)
    for r in sorted(results, key=lambda r: r['delta'], reverse=True):
        f.write(f"{r['name']:<16s} {r['content_cos']:8.4f} {r['spk_src']:8.4f} {r['spk_tgt']:8.4f} "
                f"{r['delta']:+8.4f} {r['anon']:8.4f} {r['f0_vc']:7.0f} "
                f"{r['f0_dist_src']:8.0f} {r['f0_dist_tgt']:8.0f} {r['rms']:8.4f}{NL}")
    
    # Best picks
    best_anon=max(results,key=lambda r:r['anon'])
    best_tgt=max(results,key=lambda r:r['spk_tgt'])
    best_content=max(results,key=lambda r:r['content_cos'])
    best_delta=max(results,key=lambda r:r['delta'])
    f.write(NL+'## Best Picks'+NL)
    f.write(f"- Best anonymization: {best_anon['name']} (anon={best_anon['anon']:.4f})"+NL)
    f.write(f"- Best target following: {best_tgt['name']} (→tgt={best_tgt['spk_tgt']:.4f})"+NL)
    f.write(f"- Best content: {best_content['name']} (content={best_content['content_cos']:.4f})"+NL)
    f.write(f"- Best delta: {best_delta['name']} (Δ={best_delta['delta']:+.4f})"+NL)

print(f"Saved to {OUT}/")
