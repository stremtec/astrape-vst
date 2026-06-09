#!/usr/bin/env python3
"""Artifact diagnosis: #1 VC output (n_content=1 FiLM, centroid 1429Hz)."""
import numpy as np, soundfile as sf
from scipy import signal

SR = 24000

src, _ = sf.read('/Users/asill/Desktop/vc_origin_src_orig.wav')
tgt, _ = sf.read('/Users/asill/Desktop/vc_origin_tgt_orig.wav')
vc1, _ = sf.read('/Users/asill/Desktop/vc_p255_to_origin.wav')
rt, _ = sf.read('/Users/asill/Desktop/debug_rt_full.wav')

T = min(len(src), len(vc1), len(rt))
src, vc1, rt = src[:T], vc1[:T], rt[:T]
tgt = tgt[:min(len(tgt), 96000)]

print("=" * 70)
print("ARTIFACT DIAGNOSIS: n_content=1 + FiLM (centroid=1429Hz)")
print("=" * 70)

def analyze(a, label):
    f, t, Z = signal.stft(a, fs=SR, nperseg=512, noverlap=384)
    mag = np.abs(Z)
    phase = np.angle(Z)
    total = mag.sum() + 1e-8
    
    c = np.sum(f[:len(f)//2, np.newaxis] * mag[:len(f)//2], axis=0)
    c /= (mag[:len(f)//2].sum(axis=0) + 1e-8)
    
    low = mag[(f>=0)&(f<300)].sum()/total*100
    mid = mag[(f>=300)&(f<2000)].sum()/total*100
    high = mag[(f>=2000)&(f<4000)].sum()/total*100
    vhigh = mag[(f>=4000)&(f<8000)].sum()/total*100
    ultra = mag[(f>=8000)].sum()/total*100
    
    spec = np.mean(mag, axis=1)
    peaks_idx = signal.find_peaks(spec[:len(f)//2], distance=5, prominence=spec.max()*0.01)[0]
    h_energy = spec[peaks_idx].sum() if len(peaks_idx) > 0 else 0
    n_energy = spec.sum() - h_energy
    hnr = 10 * np.log10(h_energy / (n_energy + 1e-8)) if n_energy > 0 else 99
    
    geo_mean = np.exp(np.mean(np.log(mag + 1e-8)))
    arith_mean = np.mean(mag)
    flatness = geo_mean / (arith_mean + 1e-8)
    
    cumsum = np.cumsum(spec)
    roll95 = f[np.searchsorted(cumsum, cumsum[-1]*0.95)] if cumsum[-1] > 0 else 0
    roll85 = f[np.searchsorted(cumsum, cumsum[-1]*0.85)] if cumsum[-1] > 0 else 0
    
    rms = np.sqrt(np.mean(a**2))
    crest = np.max(np.abs(a)) / (rms + 1e-8)
    zcr = np.sum(np.abs(np.diff(np.sign(a)))) / (2 * len(a))
    phase_std = np.std(np.diff(phase, axis=1))
    
    print()
    print("  " + label + ":")
    print("    Centroid:        " + str(round(np.mean(c))) + " Hz")
    print("    HNR:             " + str(round(hnr, 1)) + " dB")
    print("    Flatness:        " + str(round(flatness, 3)) + " (0=tonal, 1=noise)")
    print("    Crest factor:    " + str(round(crest, 1)) + " (3-5=normal)")
    print("    ZCR:             " + str(round(zcr, 3)))
    print("    Rolloff 85%:     " + str(round(roll85)) + " Hz")
    print("    Rolloff 95%:     " + str(round(roll95)) + " Hz")
    print("    Phase std:       " + str(round(phase_std, 3)))
    print("    Bands: L=" + str(round(low,1)) + "% M=" + str(round(mid,1)) +
          "% H=" + str(round(high,1)) + "% VH=" + str(round(vhigh,1)) + "% UH=" + str(round(ultra,1)) + "%")
    
    return {'centroid': np.mean(c), 'hnr': hnr, 'flatness': flatness,
            'crest': crest, 'zcr': zcr, 'roll95': roll95,
            'phase_std': phase_std, 'vhigh': vhigh, 'low': low, 'mid': mid, 'high': high}

r_src = analyze(src, "SOURCE (p255 original)")
r_rt = analyze(rt, "MIMI RT (encode→decode)")
r_vc1 = analyze(vc1, "VC #1 (n_content=1)")
r_tgt = analyze(tgt, "TARGET (origin.mp3)")

print()
print("=" * 70)
print("DEGRADATION CHAIN")
print("=" * 70)

keys = ['centroid','hnr','flatness','crest','vhigh','roll95','phase_std']

print()
print("--- Codec degradation (SRC -> RT) ---")
for k in keys:
    d = r_rt[k] - r_src[k]
    arrow = "v" if d < 0 else "^"
    print("  " + k + ": " + str(round(r_src[k],1)) + " -> " + str(round(r_rt[k],1)) + " (" + arrow + str(round(abs(d),1)) + ")")

print()
print("--- Splitter degradation (RT -> VC) ---")
for k in keys:
    d = r_vc1[k] - r_rt[k]
    arrow = "v" if d < 0 else "^"
    print("  " + k + ": " + str(round(r_rt[k],1)) + " -> " + str(round(r_vc1[k],1)) + " (" + arrow + str(round(abs(d),1)) + ")")

print()
print("--- Distance to Target (VC vs TGT) ---")
for k in ['centroid','hnr','flatness','crest','vhigh','roll95']:
    gap_vc = abs(r_vc1[k] - r_tgt[k])
    gap_src = abs(r_src[k] - r_tgt[k])
    better = "BETTER" if gap_vc < gap_src else "worse"
    print("  " + k + ": SRC gap=" + str(round(gap_src,1)) + " VC gap=" + str(round(gap_vc,1)) + " " + better)

print()
print("=" * 70)
print("ROOT CAUSE")
print("=" * 70)

vhigh_loss = (r_src['vhigh'] - r_vc1['vhigh']) / (r_src['vhigh'] + 1e-8) * 100
hnr_drop = r_src['hnr'] - r_vc1['hnr']
flatness_rise = r_vc1['flatness'] - r_src['flatness']

print()
print("1. High-frequency loss: " + str(round(vhigh_loss)) + "% of VHigh energy lost")
print("   Verdict: " + ("CRITICAL - main quality bottleneck" if vhigh_loss > 40 else "moderate"))

print()
print("2. HNR degradation: " + str(round(hnr_drop,1)) + " dB drop")
print("   Verdict: " + ("Noise/hiss dominant" if hnr_drop > 6 else "acceptable"))

print()
print("3. Spectral flatness: +" + str(round(flatness_rise,3)))
print("   Verdict: " + ("Buzzy/metallic artifact" if flatness_rise > 0.05 else "acceptable"))

print()
print("4. Crest factor: SRC=" + str(round(r_src['crest'],1)) + " VC=" + str(round(r_vc1['crest'],1)))
print("   Verdict: " + ("Clipping/compression" if r_vc1['crest'] < 3 else "acceptable"))

print()
done = "Done!"
print(done)
