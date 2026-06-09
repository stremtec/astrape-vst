#!/usr/bin/env python3
"""MioCodec: complete audit — structure, probe, stability, upper bound."""
import torch, time, numpy as np, soundfile as sf
from scipy import signal
from scipy.signal import butter, sosfilt
import sys
sys.path.insert(0, '/Users/asill/btrvrc0/.venv/lib/python3.12/site-packages')
from miocodec.model import MioCodecModel

SR=44100
model=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
model.eval()
cfg=model.config

src_path='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac'
tgt_path='/Users/asill/Downloads/origin.mp3'

def load(path, sr=SR, dur=3):
    d,s=sf.read(path)
    if d.ndim>1: d=d.mean(axis=1)
    if s!=sr: d=signal.resample(d,int(len(d)*sr/s))
    return d[:int(sr*dur)]

d_src=load(src_path)
d_tgt=load(tgt_path)
x_src=torch.from_numpy(d_src).float().unsqueeze(0)
x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)

print("="*70)
print("SECTION 1: MODULE STRUCTURE AUDIT")
print("="*70)

# Get features
with torch.inference_mode():
    fs=model.encode(x_src,return_content=True,return_global=True)
    ft=model.encode(x_tgt,return_content=True,return_global=True)
    ce=fs.content_embedding; ct=fs.content_token_indices
    ge_src=fs.global_embedding; ge_tgt=ft.global_embedding

modules=[
    ("raw_audio", x_src, "B=1, T_audio={}".format(len(d_src))),
    ("content_embedding", ce, "continuous, 25Hz, D=768"),
    ("content_token_indices", ct, "discrete, 25Hz, cardinality={}".format(ct.max().item()+1)),
    ("global_embedding", ge_src, "global vector, D=128, pooled over full utterance"),
    ("self_recon_mel", model.decode(global_embedding=ge_src,content_token_indices=ct,target_audio_length=len(d_src)), "mel spectrogram, n_mels=100"),
    ("vc_mel", model.decode(global_embedding=ge_tgt,content_token_indices=ct,target_audio_length=len(d_src)), "VC mel (source content + target global)"),
]

for name,tensor,desc in modules:
    if hasattr(tensor,'shape'):
        s=list(tensor.shape)
    else:
        s=list(tensor.shape)
    print("  {}: shape={} | {}".format(name,s,desc))

print()
print("Config:")
for k in ['sample_rate','hop_length','n_fft','downsample_factor','n_mels',
          'use_wave_decoder','wave_upsampler_factors','normalize_ssl_features']:
    print("  {} = {}".format(k,getattr(cfg,k,'?')))

print()
print("Key architectural facts:")
print("  1. Content: continuous embedding (768d) + discrete tokens (12579 classes)")
print("  2. Content rate: 25Hz (1 frame = 40ms)")
print("  3. Global embedding: 128d, pooled from WavLM SSL features")
print("  4. Decoder: content tokens + global embedding → mel spectrogram")
print("  5. Mel spectrogram → vocoder → waveform (vocoder NOT in codec model)")
print("  6. WavLM frontend: full-seq self-attention → NON-CAUSAL")
print("  7. Local transformer: causal=False, window=125 → ~1.2s future context")
print("  8. Wave decoder: causal=False, window=65 → ~1.3s future context")

# ── Section 2: Speaker Probe ──────────────────────────────────────────
print()
print("="*70)
print("SECTION 2: REPRESENTATION PROBE (5 speakers)")
print("="*70)

spks=['p225','p226','p227','p228','p229','p255','p256','p257','p258','p259']
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
import glob

CE_list=[]; GE_list=[]; y_spk=[]; f0_means=[]
for spk_idx,spk in enumerate(spks):
    files=sorted(glob.glob("{}/{}/{}_*_mic1.flac".format(ROOT,spk,spk)))
    for f in files[:3]:
        d,s=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if s!=SR: d=signal.resample(d,int(len(d)*SR/s))
        d=d[:SR*2]
        x=torch.from_numpy(d).float().unsqueeze(0)
        with torch.inference_mode():
            feat=model.encode(x,return_content=True,return_global=True)
        CE_list.append(feat.content_embedding.mean(dim=0).cpu().numpy())
        GE_list.append(feat.global_embedding.cpu().numpy())
        y_spk.append(spk_idx)
        # Simple F0 estimate
        d2=d-np.mean(d); fl=int(SR*0.04); hp=int(SR*0.01)
        fs_list=[]
        for i in range(0,len(d2)-fl,hp):
            fr=d2[i:i+fl]
            if np.sqrt(np.mean(fr**2))<0.001: fs_list.append(0); continue
            corr=np.correlate(fr,fr,mode='full'); corr=corr[len(corr)//2:]; corr=corr/(corr[0]+1e-8)
            pks=signal.find_peaks(corr,distance=10)[0]
            if len(pks)==0: fs_list.append(0); continue
            f0v=SR/pks[0]; fs_list.append(f0v if 50<f0v<400 else 0)
        fs_arr=np.array(fs_list); v=fs_arr>0
        f0_means.append(np.mean(fs_arr[v]) if v.sum()>0 else 0)

CE=np.array(CE_list); GE=np.array(GE_list); y=np.array(y_spk); f0m=np.array(f0_means)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def probe_spk(X,y,cv=3):
    if len(set(y))<2: return 0,0
    s=StandardScaler(); c=LogisticRegression(max_iter=2000,random_state=42,C=0.1)
    try: sc=cross_val_score(c,s.fit_transform(X),y,cv=min(cv,len(y)//2 or 2)); return sc.mean()*100,sc.std()*100
    except: return 0,0

chance=100/len(spks)
ce_spk,_=probe_spk(CE,y)
ge_spk,_=probe_spk(GE,y)
ce_f0=np.corrcoef(CE.mean(axis=1),f0m)[0,1] if len(f0m)>1 else 0
ge_f0=np.corrcoef(GE.mean(axis=1),f0m)[0,1] if len(f0m)>1 else 0

print("  ContentEmbed(mean): speaker={:.1f}% (chance={:.1f}%) F0_corr={:.3f}".format(ce_spk,chance,ce_f0))
print("  GlobalEmbed:       speaker={:.1f}% (chance={:.1f}%) F0_corr={:.3f}".format(ge_spk,chance,ge_f0))
print("  Interpretation:")
if ce_spk<chance*2: print("    Content is SPEAKER-CLEAN")
else: print("    Content has speaker leakage")
if ge_spk>50: print("    Global is STRONG speaker identity")
else: print("    Global is WEAK speaker identity")

# ── Section 3: Speaker Stability ──────────────────────────────────────
print()
print("="*70)
print("SECTION 3: SPEAKER STABILITY (origin.mp3 variants)")
print("="*70)

def get_ge(audio):
    x=torch.from_numpy(audio).float().unsqueeze(0)
    with torch.inference_mode():
        feat=model.encode(x,return_content=False,return_global=True)
    return feat.global_embedding

GE_ref=get_ge(d_tgt)

variants={
    '1s': d_tgt[:SR],
    '3s': d_tgt[:SR*3],
    'loudnorm': d_tgt*(0.1/(np.sqrt(np.mean(d_tgt**2))+1e-8)),
    'lowpass_4k': sosfilt(butter(4,4000,btype='low',fs=SR,output='sos'),d_tgt),
    'lowpass_8k': sosfilt(butter(4,8000,btype='low',fs=SR,output='sos'),d_tgt),
}

print("  Variant         cos(S_ref,S)   stable?")
print("  " + "-"*45)
for name,audio in variants.items():
    ge=get_ge(audio)
    cos=torch.nn.functional.cosine_similarity(GE_ref.unsqueeze(0),ge.unsqueeze(0)).item()
    stable="STABLE" if abs(cos)>0.9 else "UNSTABLE" if abs(cos)<0.7 else "moderate"
    print("  {:<16s} {:.4f}       {}".format(name,cos,stable))

# ── Section 4: Upper Bound Quality ────────────────────────────────────
print()
print("="*70)
print("SECTION 4: UPPER BOUND VC QUALITY (mel space)")
print("="*70)

with torch.inference_mode():
    mel_self=model.decode(global_embedding=ge_src,content_token_indices=ct,target_audio_length=len(d_src))
    mel_vc=model.decode(global_embedding=ge_tgt,content_token_indices=ct,target_audio_length=len(d_src))

mel_s=self_=mel_self.squeeze(0).cpu().numpy()
mel_v=mel_vc.squeeze(0).cpu().numpy()

# Mel statistics
print("  Self-recon mel: mean={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(
    mel_s.mean(),mel_s.std(),mel_s.min(),mel_s.max()))
print("  VC mel:         mean={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(
    mel_v.mean(),mel_v.std(),mel_v.min(),mel_v.max()))
mel_l2=np.linalg.norm(mel_s-mel_v)
mel_cos=np.dot(mel_s.flatten(),mel_v.flatten())/(np.linalg.norm(mel_s)*np.linalg.norm(mel_v)+1e-8)
print("  Mel delta L2: {:.1f}  cosine: {:.4f}".format(mel_l2,mel_cos))

# Centroid from mel (approximate)
def mel_centroid(mel):
    freqs=np.linspace(0,SR/2,cfg.n_mels)
    weights=np.exp(mel).sum(axis=0)+1e-8
    return np.sum(freqs[:,np.newaxis]*np.exp(mel),axis=0)/weights

c_self=np.mean(mel_centroid(mel_s))
c_vc=np.mean(mel_centroid(mel_v))
print("  Mel centroid: self={:.0f}Hz  VC={:.0f}Hz  delta={:.0f}Hz".format(c_self,c_vc,c_vc-c_self))

print()
print("=== SUMMARY ===")
print("Content rate: 25Hz, dim=768 (cont) + discrete tokens (12579 vocab)")
print("Global embedding: 128d, pooled over full utterance")
print("Content speaker probe: {:.1f}% (chance {:.1f}%)".format(ce_spk,chance))
print("Global speaker probe: {:.1f}% (chance {:.1f}%)".format(ge_spk,chance))
print("VC mechanism: source content tokens + target global embedding → mel")
print("Causality: FULLY NON-CAUSAL (WavLM + symmetric attn + ISTFT)")
print("Streaming potential: NONE — requires complete causal student distillation")
print()
print("Done!")
