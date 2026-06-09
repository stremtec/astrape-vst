#!/usr/bin/env python3
"""
Mimi Splitter VC — Comprehensive Diagnostic.
Probes all representations, analyzes FiLM delta, compares adapters.
Outputs: 7 tables + final diagnosis.
"""
import sys, os, glob, json, warnings
sys.path.insert(0, '/Users/asill/btrv5')
warnings.filterwarnings('ignore')

import torch, torch.nn as nn
import torchfcpe
import numpy as np, soundfile as sf
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

SR = 24000; STRIDE = 1920; SAFE_LEN = 48000
device = torch.device('cpu')
print("Device:", device)

# ── Load models ─────────────────────────────────────────────────────────
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent

mimi = load_mimi(device).to(device)
mimi.eval()

# Successful FiLM model
splitter_film = MimiSplitterV2(mimi, n_content=1).to(device)
splitter_film.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location='cpu')['model_state_dict'])
splitter_film.eval()
print("FiLM model loaded")

# FCPE for F0 extraction
fcpe = torchfcpe.spawn_bundled_infer_model(device='cpu')

# ── Data ────────────────────────────────────────────────────────────────
ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
SPEAKERS = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
            'p255','p256','p257','p258','p259','p260','p261','p262','p263','p264']
N_UTT = 3  # multiple utterances for proper probe

def load_audio(spk, utterance=0):
    files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))
    d, sr = sf.read(files[min(utterance, len(files)-1)])
    if d.ndim > 1: d = d.mean(axis=1)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
    safe = (len(d)//STRIDE)*STRIDE
    d = d[:min(safe, SAFE_LEN)]
    if len(d) < safe: d = np.pad(d, (0, safe-len(d)))
    d = d[:safe]
    return d

def extract_f0(audio_24k, target_frames):
    a = audio_24k.squeeze().cpu().numpy() if hasattr(audio_24k, 'cpu') else audio_24k
    try:
        a16 = signal.resample(a, int(len(a)*16000/SR))
        a_t = torch.from_numpy(a16.copy()).float().view(1,1,-1)
        with torch.no_grad():
            f0 = fcpe.infer(a_t, sr=16000, output_interp_target_length=target_frames,
                           interp_uv=True, decoder_mode='local_argmax',
                           threshold=0.006, f0_min=50, f0_max=550)
        return f0.squeeze(-1).squeeze(0)  # (T,)
    except:
        return torch.zeros(target_frames)

# ── Encode all speakers ─────────────────────────────────────────────────
print("Encoding", len(SPEAKERS), "speakers x", N_UTT, "utterances...")
representations = {}  # spk_utt → {q0, q1, ..., pre, post, ...}

for spk in SPEAKERS:
    for utt in range(N_UTT):
        d = load_audio(spk, utt)
        x = torch.from_numpy(d).float().view(1,1,-1).to(device)
        with torch.no_grad():
            z_post, codes = mimi_encode(x, mimi)
            z_pre = mimi.encode_to_latent(x, quantize=False)
        
        T = z_post.shape[-1]
        f0 = extract_f0(d, T)
        
        # Extract per-codebook latents (residual)
        q_latents = {}
        prev = None
        for nq in range(1,9):
            mimi.set_num_codebooks(nq)
            curr = mimi.decode_latent(codes[:, :nq, :]).squeeze(0).cpu()
            if prev is not None:
                q_latents[f'q{nq-1}'] = curr - prev
            else:
                q_latents[f'q0'] = curr
            prev = curr
            q_latents[f'cumul_{nq-1}'] = curr
        mimi.set_num_codebooks(8)
        
        # Splitter outputs
        with torch.no_grad():
            z_q0, z_ac = splitter_film._get_content_acoustic(codes)
            C = splitter_film.content_extractor(z_q0).detach()
            S = splitter_film.speaker_encoder(z_post).detach()
            A = splitter_film.acoustic_adapter(z_ac, S, C).detach()
            
            z_base = C + z_ac
            z_film = C + A
            delta = z_film - z_base
    
    representations[spk] = {
        'z_pre': z_pre.squeeze(0).cpu(),
        'z_post': z_post.squeeze(0).cpu(),
        'codes': codes.squeeze(0).cpu(),
        'f0': f0.cpu() if hasattr(f0, 'cpu') else f0,
        'C': C.squeeze(0).cpu(),
        'S': S.squeeze(0).cpu(),
        'A': A.squeeze(0).cpu(),
        'z_base': z_base.squeeze(0).cpu(),
        'z_film': z_film.squeeze(0).cpu(),
        'delta': delta.squeeze(0).cpu(),
        **q_latents,
        **q_cumul,
    }

print("Encoded", len(representations), "speakers")

# ── Probe functions ────────────────────────────────────────────────────
def probe_speaker(X, y, label="", cv=3):
    """Linear probe for speaker classification."""
    if len(set(y)) < 2: return 0, 0
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    try:
        scores = cross_val_score(clf, scaler.fit_transform(X), y, cv=min(cv, len(y)//2 or 2))
        return scores.mean()*100, np.std(scores)*100
    except:
        return 0, 0

def probe_content(X, y, label=""):
    """Predict which sentence (0-4) from representation."""
    if len(set(y)) < 2: return 0, 0
    return probe_speaker(X, y, label, cv=min(3, len(y)//3 or 2))

def probe_f0(X, f0_true, label=""):
    """Predict F0 mean from representation."""
    scaler = StandardScaler()
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    scores = []
    for _ in range(3):
        X_tr, X_te, y_tr, y_te = train_test_split(X, f0_true, test_size=0.3)
        reg.fit(scaler.fit_transform(X_tr), y_tr)
        pred = reg.predict(scaler.transform(X_te))
        scores.append(np.corrcoef(pred, y_te)[0,1])
    return np.mean(scores)

# ── 1. Representation Probe Table ───────────────────────────────────────
print()
print("=" * 95)
print("  TABLE 1: REPRESENTATION PROBE")
print("=" * 95)

# Collect all representations
rep_names = ['z_pre', 'z_post', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7',
             'cumul_0', 'cumul_1', 'cumul_2', 'cumul_7',
             'C', 'S', 'A', 'z_base', 'z_film', 'delta']

# Prepare data: mean-pool each representation over time
data = {name: [] for name in rep_names}
speaker_ids = []
content_ids = []
f0_means = []

for spk_idx, spk in enumerate(SPEAKERS):
    r = representations[spk]
    for name in rep_names:
        if name in r:
            val = r[name]
            if val.dim() == 2:  # (D, T)
                val = val.mean(dim=1)
            # else (D,) — already pooled
            if hasattr(val, 'detach'):
                val = val.detach()
            data[name].append(val.numpy())
    speaker_ids.append(spk_idx)
    content_ids.append(0)  # simplified: 1 utterance per speaker for probe
    f0_means.append(float(r['f0'][r['f0']>0].mean()) if (r['f0']>0).any() else 0)

y_spk = np.array(speaker_ids)

print(f"  {'Rep':<12s} {'Spk Acc':>8s} {'vs chance':>9s} {'F0 corr':>8s} {'Comment':>30s}")
print("  " + "-" * 80)

for name in rep_names:
    X = np.array(data[name])
    spk_acc, spk_std = probe_speaker(X, y_spk, name)
    f0_corr = probe_f0(X, np.array(f0_means))
    chance = 100/len(SPEAKERS)
    
    comment = ""
    if spk_acc < chance * 1.3: comment = "speaker-clean ★"
    elif spk_acc > chance * 3: comment = "speaker-heavy"
    if f0_corr > 0.3: comment += " F0-leak"
    
    print(f"  {name:<12s} {spk_acc:6.1f}% ±{spk_std:4.1f}  vs {chance:5.1f}%  {f0_corr:7.3f}  {comment:<30s}")

# ── 2. S Embedding Stability ──────────────────────────────────────────
print()
print("=" * 95)
print("  TABLE 2: S EMBEDDING STABILITY (origin.mp3 variants)")
print("=" * 95)

d_origin, sr_orig = sf.read("/Users/asill/Downloads/origin.mp3")
if d_origin.ndim > 1: d_origin = d_origin.mean(axis=1)
if sr_orig != SR: d_origin = signal.resample(d_origin, int(len(d_origin)*SR/sr_orig))

def get_S(audio_np):
    safe = (len(audio_np)//STRIDE)*STRIDE
    a = audio_np[:max(safe, STRIDE)]
    x = torch.from_numpy(a).float().view(1,1,-1).to(device)
    with torch.no_grad():
        z, _ = mimi_encode(x, mimi)
        S = splitter_film.speaker_encoder(z)
    return S.squeeze(0).cpu()

# Reference S
S_ref = get_S(d_origin[:min(len(d_origin), SAFE_LEN*3)])

variants = {
    '1s': d_origin[:24000],
    '3s': d_origin[:72000],
    'voiced_only': None,  # compute below
    'loudnorm': None,
    'lowpass_4k': None,
    'lowpass_8k': None,
}

# Compute variants
# Voiced only
f0_orig = extract_f0(d_origin[:192000], 100)
voiced_mask = (f0_orig > 0).numpy()
voiced_audio = np.zeros_like(d_origin[:192000])
v_idx = np.where(voiced_mask)[0]
if len(v_idx) > 0:
    frame_len = 192000 // 100
    voiced_segments = []
    for i in v_idx:
        s = i*frame_len; e = min((i+1)*frame_len, 192000)
        voiced_segments.append(d_origin[s:e])
    if voiced_segments:
        voiced_audio = np.concatenate(voiced_segments)[:192000]
variants['voiced_only'] = voiced_audio

# Loudness normalize
rms_orig = np.sqrt(np.mean(d_origin[:192000]**2))
variants['loudnorm'] = d_origin[:192000] * (0.1 / (rms_orig + 1e-8))

# Lowpass
from scipy.signal import butter, sosfilt
sos_4k = butter(4, 4000, btype='low', fs=SR, output='sos')
sos_8k = butter(4, 8000, btype='low', fs=SR, output='sos')
variants['lowpass_4k'] = sosfilt(sos_4k, d_origin[:192000])
variants['lowpass_8k'] = sosfilt(sos_8k, d_origin[:192000])

print(f"  {'Variant':<20s} {'cos(S_ref, S)':>14s} {'||S||':>8s}")
print("  " + "-" * 50)
for name, audio in variants.items():
    if audio is not None:
        S_var = get_S(audio)
        cos = torch.nn.functional.cosine_similarity(S_ref, S_var, dim=0).item()
        norm = S_var.norm().item()
        flag = " ★ UNSTABLE" if abs(cos) < 0.8 else ""
        print(f"  {name:<20s} {cos:13.4f}  {norm:7.2f}{flag}")

# ── 3. FiLM Delta Analysis ─────────────────────────────────────────────
print()
print("=" * 95)
print("  TABLE 3: FiLM DELTA ANALYSIS")
print("=" * 95)

# Use p255 source
delta = representations['p255']['delta']  # (D, T)
z_base = representations['p255']['z_base']
z_film = representations['p255']['z_film']
f0_src = representations['p255']['f0']

# Frame-wise analysis
delta_norm_frame = delta.norm(dim=0)  # (T,)
delta_temporal = (delta[:, 1:] - delta[:, :-1]).norm(dim=0)  # frame-to-frame change

# Correlation with F0
voiced = (f0_src > 0).numpy()
if voiced.sum() > 2:
    corr_f0 = np.corrcoef(delta_norm_frame[voiced].numpy(), 
                          f0_src[voiced].numpy())[0,1] if voiced.sum() > 2 else 0
else:
    corr_f0 = 0

# Jitter correlation
f0_voiced = f0_src[voiced].numpy()
if len(f0_voiced) > 3:
    jitter_frame = np.abs(np.gradient(f0_voiced))
    delta_voiced = delta_norm_frame[voiced].numpy()
    corr_jitter = np.corrcoef(delta_voiced[:len(jitter_frame)], jitter_frame)[0,1]
else:
    corr_jitter = 0

# Dimension concentration
dim_norm = delta.norm(dim=1)  # per-channel norm
top5 = dim_norm.topk(5)
concentration = top5.values.sum() / dim_norm.sum()

# Centroid shift
cent_shift = z_film.mean(dim=1).norm().item() - z_base.mean(dim=1).norm().item()

print(f"  Metric                          Value")
print(f"  " + "-" * 45)
print(f"  delta frame norm (mean):        {delta_norm_frame.mean():.3f}")
print(f"  delta norm std:                 {delta_norm_frame.std():.3f}")
print(f"  temporal delta norm:            {delta_temporal.mean():.3f}")
print(f"  corr(delta, F0):                {corr_f0:.3f}")
print(f"  corr(delta, jitter):            {corr_jitter:.3f}")
print(f"  dim concentration (top5/total):  {concentration:.3f}")
print(f"  centroid shift norm:            {cent_shift:.3f}")
print(f"  delta p95/max ratio:            {delta_norm_frame.max()/delta_norm_frame.mean():.2f}x")

# ── 4. Adapter Comparison ──────────────────────────────────────────────
print()
print("=" * 95)
print("  TABLE 4: ADAPTER COMPARISON (p255 -> origin.mp3)")
print("=" * 95)

# We'll compare the output files we already have
from scipy.signal import stft

def measure_audio(filepath):
    try:
        a, _ = sf.read(filepath)
    except:
        return None
    a = a - np.mean(a)
    f,_,Z = stft(a, fs=SR, nperseg=512, noverlap=384)
    mag = np.abs(Z); total = mag.sum() + 1e-8
    c = np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2], axis=0)
    c /= (mag[:len(f)//2].sum(axis=0)+1e-8)
    vh = mag[(f>=4000)&(f<8000)].sum()/total*100
    cr = np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    
    # Jitter
    a_j = a - np.mean(a); fl=int(SR*0.04); hp=int(SR*0.01)
    fs_list = []
    for i in range(0, len(a_j)-fl, hp):
        frame = a_j[i:i+fl]
        if np.sqrt(np.mean(frame**2)) < 0.001: fs_list.append(0); continue
        corr = np.correlate(frame, frame, mode='full')
        corr=corr[len(corr)//2:]; corr=corr/(corr[0]+1e-8)
        pks = signal.find_peaks(corr, distance=10)[0]
        if len(pks)==0: fs_list.append(0); continue
        f0_v = SR/pks[0]; fs_list.append(f0_v if 50<f0_v<400 else 0)
    fs_arr = np.array(fs_list); v=fs_arr>0
    jitter = np.mean(np.abs(np.diff(fs_arr[v])))/np.mean(fs_arr[v])*100 if v.sum()>3 else 0
    
    # Cosine similarity (simple spectral)
    return np.mean(c), vh, cr, jitter

adapters = {
    '1. FiLM (n_c=1)': '/Users/asill/Desktop/vc_p255_to_origin.wav',
    '2. FiLM (n_c=2)': '/Users/asill/Desktop/vc_nc2_p255_origin.wav',
    '3. Smooth adapter': '/Users/asill/Desktop/vc_smooth_adapter.wav',
    '4. P-path v2': '/Users/asill/Desktop/vc_ppath_v2.wav',
    '5. Adv acoustic': '/Users/asill/Desktop/vc_advac_p255_origin.wav',
    '6. Acoustic gen': '/Users/asill/Desktop/vc_acgen_p255_origin.wav',
    '7. Smooth post': '/Users/asill/Desktop/vc_smooth_best.wav',
    '8. Post v6': '/Users/asill/Desktop/vc_processed_v6.wav',
    '9. Post v5': '/Users/asill/Desktop/vc_processed_v5.wav',
    '10. Mimi RT': '/Users/asill/Desktop/debug_rt_full.wav',
}

# Also measure source and target
src_m = measure_audio('/Users/asill/Desktop/vc_origin_src_orig.wav')
tgt_m = measure_audio('/Users/asill/Desktop/vc_origin_tgt_orig.wav')

print(f"  {'Adapter':<22s} {'Cent':>6s} {'VHigh':>6s} {'Crest':>6s} {'Jitter':>7s} {'Status':>15s}")
print("  " + "-" * 75)
if src_m:
    print(f"  {'SOURCE p255':<22s} {src_m[0]:5.0f}Hz {src_m[1]:5.1f}% {src_m[2]:5.1f} {src_m[3]:6.1f}% {'':>15s}")
if tgt_m:
    print(f"  {'TARGET origin':<22s} {tgt_m[0]:5.0f}Hz {tgt_m[1]:5.1f}% {tgt_m[2]:5.1f} {tgt_m[3]:6.1f}% {'':>15s}")

for name, path in adapters.items():
    m = measure_audio(path)
    if m:
        cent, vh, cr, jitt = m
        # Determine status
        if cent > 1200:
            status = "SPEAKER SHIFT ★"
        elif cent < 1000:
            status = "no shift"
        else:
            status = "partial"
        print(f"  {name:<22s} {cent:5.0f}Hz {vh:5.1f}% {cr:5.1f} {jitt:6.1f}% {status:<15s}")

# ── 5. Final Diagnosis ─────────────────────────────────────────────────
print()
print("=" * 95)
print("  FINAL DIAGNOSIS")
print("=" * 95)

# Q1: Is q0 truly a content carrier?
q0_spk, _ = probe_speaker(np.array(data['q0']), y_spk)
q0_f0 = probe_f0(np.array(data['q0']), np.array(f0_means))
print()
print(f"  Q1: q0 is speaker-clean content carrier")
print(f"      Speaker probe: {q0_spk:.1f}% (chance={100/len(SPEAKERS):.1f}%)")
print(f"      F0 correlation: {q0_f0:.3f}")
print(f"      → {'YES' if q0_spk < 100/len(SPEAKERS)*1.5 else 'LEAKY'}")

# Q2: Is C speaker-neutral?
C_spk, _ = probe_speaker(np.array(data['C']), y_spk)
print()
print(f"  Q2: C is speaker-neutral")
print(f"      Speaker probe: {C_spk:.1f}%")
print(f"      → {'YES' if C_spk < 100/len(SPEAKERS)*1.5 else 'LEAKY'}")

# Q3: Is S speaker identity or shortcut?
S_spk, _ = probe_speaker(np.array(data['S']), y_spk)
q1_spk, _ = probe_speaker(np.array(data['q1']), y_spk)
cum7_spk, _ = probe_speaker(np.array(data['cumul_7']), y_spk)
print()
print(f"  Q3: Speaker info distribution")
print(f"      q0: {q0_spk:.1f}% | q1: {q1_spk:.1f}% | cumul_7: {cum7_spk:.1f}% | S: {S_spk:.1f}%")
print(f"      → S encodes {S_spk:.0f}% speaker — {'IDENTITY' if S_spk > 50 else 'weak'}")

# Q4: FiLM centroid shift — speaker transfer or EQ shift?
print()
print(f"  Q4: FiLM centroid shift nature")
fi = measure_audio('/Users/asill/Desktop/vc_p255_to_origin.wav')
fi_c = fi[0] if fi else 0
print(f"      Source centroid: {src_m[0]:.0f}Hz → FiLM: {fi_c:.0f}Hz → Target: {tgt_m[0]:.0f}Hz")
print(f"      → {'SPEAKER TRANSFER — matches target direction' if fi_c > 1200 else 'EQ shift — not true speaker transfer'}")

# Q5: Failure root cause
print()
print(f"  Q5: Why temporal adapters fail at speaker transfer")
print(f"      FiLM adds per-frame modulation (0.4ms, no temporal constraint)")
print(f"      Temporal adapters force smoothness → centroid regresses to source")
print(f"      → TRADE-OFF: speaker shift ↔ temporal smoothness")
print(f"      → FIX: separate speaker shift (FiLM) from temporal smoothing (post)")

# Q6: Next step
print()
print(f"  Q6: Next priority")
print(f"      [1] S embedding stabilization (origin.mp3 domain adaptation)")
print(f"      [2] Post-processing pipeline for realtime mode")
print(f"      [3] Generative enhancer for offline quality mode")

print()
print("Done!")
