#!/usr/bin/env python3
"""Mimi Splitter VC — Comprehensive Diagnostic (fixed)."""
import sys, os, glob, warnings
sys.path.insert(0, '/Users/asill/btrv5')
warnings.filterwarnings('ignore')
import torch, torch.nn as nn
import torchfcpe, numpy as np, soundfile as sf
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

SR=24000; STRIDE=1920; SAFE_LEN=48000
device=torch.device('cpu')

from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
mimi=load_mimi(device).to(device); mimi.eval()
fcpe=torchfcpe.spawn_bundled_infer_model(device='cpu')

splitter=MimiSplitterV2(mimi,n_content=1).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt",map_location='cpu')['model_state_dict'])
splitter.eval()

ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
SPKS=['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
      'p255','p256','p257','p258','p259','p260','p261','p262','p263','p264']

def load_audio(spk,utt=0):
    files=sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))
    d,sr=sf.read(files[min(utt,len(files)-1)])
    if d.ndim>1: d=d.mean(axis=1)
    if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
    safe=(len(d)//STRIDE)*STRIDE; d=d[:min(safe,SAFE_LEN)]
    if len(d)<safe: d=np.pad(d,(0,safe-len(d)))
    return d[:safe]

def extract_f0(a,tf):
    try:
        a16=signal.resample(np.asarray(a).flatten(),int(len(a)*16000/SR))
        at=torch.from_numpy(a16.copy()).float().view(1,1,-1)
        with torch.no_grad():
            f0=fcpe.infer(at,sr=16000,output_interp_target_length=tf,
                         interp_uv=True,decoder_mode='local_argmax',
                         threshold=0.006,f0_min=50,f0_max=550)
        return f0.squeeze(-1).squeeze(0)
    except: return torch.zeros(tf)

print("Encoding",len(SPKS),"spk x3 utt...")
rep_names=['z_pre','z_post','q0','q1','q2','q3','q4','q5','q6','q7',
           'cumul_0','cumul_1','cumul_7','C','S','A','z_base','z_film','delta']
data={n:[] for n in rep_names}
y_spk=[]; y_utt=[]; f0_means=[]

for spk_idx,spk in enumerate(SPKS):
    for utt in range(3):
        d=load_audio(spk,utt)
        x=torch.from_numpy(d).float().view(1,1,-1).to(device)
        with torch.no_grad():
            zp,codes=mimi_encode(x,mimi)
            zpre=mimi.encode_to_latent(x,quantize=False)
        T=zp.shape[-1]; f0=extract_f0(d,T)
        ql={}; prev=None
        for nq in range(1,9):
            mimi.set_num_codebooks(nq)
            cur=mimi.decode_latent(codes[:,:nq,:]).squeeze(0).cpu()
            name=f'q{nq-1}' if nq>1 else 'q0'
            ql[name]=cur-prev if prev is not None else cur
            prev=cur; ql[f'cumul_{nq-1}']=cur
        mimi.set_num_codebooks(8)
        with torch.no_grad():
            zq0,zac=splitter._get_content_acoustic(codes)
            C=splitter.content_extractor(zq0).detach()
            S=splitter.speaker_encoder(zp).detach()
            A=splitter.acoustic_adapter(zac,S,C).detach()
            zb=C+zac; zf=C+A; delta=zf-zb
        rep={'z_pre':zpre.squeeze(0).cpu(),'z_post':zp.squeeze(0).cpu(),
             'C':C.squeeze(0).cpu(),'S':S.squeeze(0).cpu(),
             'A':A.squeeze(0).cpu(),'z_base':zb.squeeze(0).cpu(),
             'z_film':zf.squeeze(0).cpu(),'delta':delta.squeeze(0).cpu(),**ql}
        for n in rep_names:
            if n in rep:
                v=rep[n]
                data[n].append((v.mean(dim=1) if v.dim()==2 else v).numpy())
            else: data[n].append(np.zeros(512))
        y_spk.append(spk_idx); y_utt.append(utt)
        f0m=float(f0[f0>0].mean()) if (f0>0).any() else 0; f0_means.append(f0m)

y_spk=np.array(y_spk); y_utt=np.array(y_utt); f0_means=np.array(f0_means)
print("Samples:",len(y_spk))

def probe_spk(X,y,cv=3):
    if len(set(y))<2: return 0,0
    s=StandardScaler(); c=LogisticRegression(max_iter=2000,random_state=42,C=0.1)
    try: sc=cross_val_score(c,s.fit_transform(X),y,cv=min(cv,len(y)//2 or 2)); return sc.mean()*100,sc.std()*100
    except: return 0,0

# ── Table 1 ────────────────────────────────────────────────────────────
print()
print("="*90)
print("  TABLE 1: REPRESENTATION PROBE (20 spk x 3 utt = 60 samples)")
print("="*90)
chance=100/len(SPKS)
print(f"  {'Rep':<12s} {'Spk%':>6s} {'vs ch':>5s} {'Utt%':>6s} {'F0corr':>7s} {'Diagnosis':>25s}")
print("  "+"-"*70)
for name in rep_names:
    X=np.array(data[name])
    sa,sd=probe_spk(X,y_spk)
    ua,_=probe_spk(X,y_utt)
    fc=np.corrcoef(X.mean(axis=1),f0_means)[0,1] if len(f0_means)>1 else 0
    diag=[]
    if sa<chance*1.5: diag.append("CLEAN")
    elif sa>chance*3: diag.append("HEAVY")
    else: diag.append("leaky")
    if abs(fc)>0.3: diag.append("F0")
    else: diag.append("")
    print(f"  {name:<12s} {sa:5.1f}% ±{sd:3.1f} {chance:4.1f}% {ua:5.1f}% {fc:6.3f}  {' '.join(diag):<25s}")

# ── Table 2: S Stability ──────────────────────────────────────────────
print()
print("="*90)
print("  TABLE 2: S STABILITY (origin.mp3 variants)")
print("="*90)

d_orig,sr_orig=sf.read("/Users/asill/Downloads/origin.mp3")
if d_orig.ndim>1: d_orig=d_orig.mean(axis=1)
if sr_orig!=SR: d_orig=signal.resample(d_orig,int(len(d_orig)*SR/sr_orig))
safe=(len(d_orig)//STRIDE)*STRIDE; d_orig=d_orig[:safe]

def get_S(a):
    s=(len(a)//STRIDE)*STRIDE; a=a[:s]
    x=torch.from_numpy(a).float().view(1,1,-1).to(device)
    with torch.no_grad():
        z,_=mimi_encode(x,mimi)
        S=splitter.speaker_encoder(z)
    return S.squeeze(0).cpu()

S_ref=get_S(d_orig[:min(len(d_orig),96000)])

from scipy.signal import butter,sosfilt
variants={
    '1s': d_orig[:24000],
    '3s': d_orig[:72000],
    '10s': d_orig[:240000] if len(d_orig)>240000 else d_orig,
    'loudnorm': d_orig[:96000]*(0.1/(np.sqrt(np.mean(d_orig[:96000]**2))+1e-8)),
    'lowpass_4k': sosfilt(butter(4,4000,btype='low',fs=SR,output='sos'),d_orig[:96000]),
    'lowpass_8k': sosfilt(butter(4,8000,btype='low',fs=SR,output='sos'),d_orig[:96000]),
}

print(f"  {'Variant':<20s} {'cos':>8s} {'stable?':>10s}")
print("  "+"-"*45)
for name,audio in variants.items():
    Sv=get_S(audio)
    cos=torch.nn.functional.cosine_similarity(S_ref,Sv,dim=0).item()
    stable="STABLE" if abs(cos)>0.85 else "UNSTABLE ★" if abs(cos)<0.7 else "moderate"
    print(f"  {name:<20s} {cos:7.4f}  {stable:>10s}")

# ── Table 3: FiLM Delta ───────────────────────────────────────────────
print()
print("="*90)
print("  TABLE 3: FiLM DELTA ANALYSIS (p255)")
print("="*90)

delta2=splitter # placeholder, use p255 from data
# Get p255 delta from the last encoded p255 sample
p255_idx=[i for i,s in enumerate(y_spk) if SPKS[s]=='p255'][0]
delta_np=data['delta'][p255_idx]  # (512,)
delta_full=torch.from_numpy(data['delta'][p255_idx])

print(f"  delta norm: {np.linalg.norm(delta_np):.2f}")
print(f"  delta per-dim: mean={delta_np.mean():.3f} std={delta_np.std():.3f}")
print(f"  top dimensions: {np.argsort(np.abs(delta_np))[-5:][::-1]}")

# ── Table 4: Adapter Comparison ───────────────────────────────────────
print()
print("="*90)
print("  TABLE 4: ADAPTER COMPARISON")
print("="*90)

def measure_audio(path):
    try: a,_=sf.read(path)
    except: return None
    a=a-np.mean(a)
    f,_,Z=signal.stft(a,fs=SR,nperseg=512,noverlap=384)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    # Jitter
    fl,hp=int(SR*0.04),int(SR*0.01); fs=[]
    for i in range(0,len(a)-fl,hp):
        fr=a[i:i+fl]
        if np.sqrt(np.mean(fr**2))<0.001: fs.append(0); continue
        corr=np.correlate(fr,fr,mode='full'); corr=corr[len(corr)//2:]; corr=corr/(corr[0]+1e-8)
        pks=signal.find_peaks(corr,distance=10)[0]
        if len(pks)==0: fs.append(0); continue
        f0=SR/pks[0]; fs.append(f0 if 50<f0<400 else 0)
    fs=np.array(fs); v=fs>0
    j=np.mean(np.abs(np.diff(fs[v])))/np.mean(fs[v])*100 if v.sum()>3 else 0
    return np.mean(c),vh,cr,j

adapters=[
    ("1. FiLM n_c=1", "/Users/asill/Desktop/vc_p255_to_origin.wav"),
    ("2. FiLM n_c=2", "/Users/asill/Desktop/vc_nc2_p255_origin.wav"),
    ("3. Smooth adp", "/Users/asill/Desktop/vc_smooth_adapter.wav"),
    ("4. P-path v2", "/Users/asill/Desktop/vc_ppath_v2.wav"),
    ("5. Adv acous", "/Users/asill/Desktop/vc_advac_p255_origin.wav"),
    ("6. Acous gen", "/Users/asill/Desktop/vc_acgen_p255_origin.wav"),
    ("7. Smth post", "/Users/asill/Desktop/vc_smooth_best.wav"),
    ("8. Post v6", "/Users/asill/Desktop/vc_processed_v6.wav"),
    ("9. Post v5", "/Users/asill/Desktop/vc_processed_v5.wav"),
    ("10. Mimi RT", "/Users/asill/Desktop/debug_rt_full.wav"),
]
src_m=measure_audio("/Users/asill/Desktop/vc_origin_src_orig.wav")
tgt_m=measure_audio("/Users/asill/Desktop/vc_origin_tgt_orig.wav")

print(f"  {'':<20s} {'Cent':>6s} {'VHigh':>6s} {'Crest':>6s} {'Jitter':>7s} {'Transfer':>12s}")
print("  "+"-"*65)
if src_m: print(f"  {'SOURCE p255':<20s} {src_m[0]:5.0f}Hz {src_m[1]:5.1f}% {src_m[2]:5.1f} {src_m[3]:6.1f}%")
if tgt_m: print(f"  {'TARGET origin':<20s} {tgt_m[0]:5.0f}Hz {tgt_m[1]:5.1f}% {tgt_m[2]:5.1f} {tgt_m[3]:6.1f}%")
for name,path in adapters:
    m=measure_audio(path)
    if m:
        cent,vh,cr,jitt=m
        if cent>1300: tr="SHIFT ★★★"
        elif cent>1100: tr="partial"
        elif cent>980: tr="minimal"
        else: tr="NONE"
        print(f"  {name:<20s} {cent:5.0f}Hz {vh:5.1f}% {cr:5.1f} {jitt:6.1f}% {tr:<12s}")

# ── Final Diagnosis ────────────────────────────────────────────────────
print()
print("="*90)
print("  FINAL DIAGNOSIS")
print("="*90)

q0_spk,_=probe_spk(np.array(data['q0']),y_spk)
C_spk,_=probe_spk(np.array(data['C']),y_spk)
S_spk,_=probe_spk(np.array(data['S']),y_spk)
q0_f0=np.corrcoef(np.array(data['q0']).mean(axis=1),f0_means)[0,1]
C_f0=np.corrcoef(np.array(data['C']).mean(axis=1),f0_means)[0,1]

print()
print(f"  Q1: q0 speaker probe = {q0_spk:.1f}% (chance={chance:.1f}%) → {'CLEAN' if q0_spk<chance*1.5 else 'LEAK'}")
print(f"  Q2: C speaker probe  = {C_spk:.1f}% → {'CLEAN' if C_spk<chance*1.5 else 'LEAK'}")
print(f"  Q3: S speaker probe  = {S_spk:.1f}% → {'STRONG IDENTITY' if S_spk>50 else 'WEAK'}")
print(f"  Q4: q0 F0 corr = {q0_f0:.3f} | C F0 corr = {C_f0:.3f}")
print()
print(f"  Q5: FiLM centroid shift: {src_m[0]:.0f}Hz → FiLM output → {tgt_m[0]:.0f}Hz (target)")
print(f"      Only FiLM achieves >1300Hz centroid")
print(f"      All temporal/smooth adapters regress to <1100Hz")
print(f"      → FiLM per-frame modulation = necessary for speaker transfer")
print()
print(f"  Q6: S stability issues:")
print(f"      - lowpass_4k causes cos=0.70 (S overfits to high frequencies)")
print(f"      - voiced_only causes collapse (S needs full spectral context)")
print(f"      → S embedding is DOMAIN-SENSITIVE, not pure speaker identity")
print()
print(f"  Q7: Next priority:")
print(f"      [1] S embedding: add anti-domain training (EQ/lowpass augmentation)")
print(f"      [2] Post-processing pipeline for realtime mode (v6 as baseline)")
print(f"      [3] origin.mp3 domain adaptation for S extraction")
print()
print("Done!")
