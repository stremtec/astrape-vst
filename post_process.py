#!/usr/bin/env python3
"""
Post-processing chain for Mimi Splitter VC output.
Fixes: crest factor spike (11.9) + VHigh loss (44%).
Chain: de-ess 2-5kHz → high-shelf 6kHz+ → soft limiter → normalize.
"""
import numpy as np, soundfile as sf
from scipy import signal

SR = 24000

# Load VC output (#1 successful)
vc, _ = sf.read('/Users/asill/Desktop/vc_p255_to_origin.wav')
src, _ = sf.read('/Users/asill/Desktop/vc_origin_src_orig.wav')
tgt, _ = sf.read('/Users/asill/Desktop/vc_origin_tgt_orig.wav')

def measure(a, label):
    f, t, Z = signal.stft(a, fs=SR, nperseg=512, noverlap=384)
    mag = np.abs(Z)
    total = mag.sum() + 1e-8
    c = np.sum(f[:len(f)//2, np.newaxis] * mag[:len(f)//2], axis=0)
    c /= (mag[:len(f)//2].sum(axis=0) + 1e-8)
    vhigh = mag[(f>=4000)&(f<8000)].sum()/total*100
    low = mag[(f>=0)&(f<300)].sum()/total*100
    mid = mag[(f>=300)&(f<2000)].sum()/total*100
    high = mag[(f>=2000)&(f<4000)].sum()/total*100
    rms = np.sqrt(np.mean(a**2))
    crest = np.max(np.abs(a)) / (rms + 1e-8)
    
    # HNR
    spec = np.mean(mag, axis=1)
    peaks = signal.find_peaks(spec[:len(f)//2], distance=5, prominence=spec.max()*0.01)[0]
    h_energy = spec[peaks].sum() if len(peaks) > 0 else 0
    n_energy = spec.sum() - h_energy
    hnr = 10*np.log10(h_energy/(n_energy+1e-8))
    
    # Flatness
    geo = np.exp(np.mean(np.log(mag + 1e-8)))
    arith = np.mean(mag)
    flat = geo/(arith+1e-8)
    
    return {'centroid': np.mean(c), 'vhigh': vhigh, 'low': low, 'mid': mid, 'high': high,
            'rms': rms, 'crest': crest, 'hnr': hnr, 'flatness': flat}

# ── Step 1: De-esser (2-5kHz dynamic taming) ────────────────────────
print("Step 1: 2-5kHz de-esser...")

# Design bandpass filter for 2-5kHz detection
sos_bp = signal.butter(4, [2000, 5000], btype='band', fs=SR, output='sos')
band_energy = signal.sosfilt(sos_bp, vc)

# Compute envelope of band energy
envelope = np.abs(signal.hilbert(band_energy))
# Smooth envelope
from scipy.ndimage import uniform_filter1d
envelope_smooth = uniform_filter1d(envelope, size=int(SR*0.01))  # 10ms window

# Threshold: anything above 2x median gets attenuated
threshold = np.median(envelope_smooth) * 2.0
gain_reduction = np.ones_like(vc)
mask = envelope_smooth > threshold
gain_reduction[mask] = threshold / (envelope_smooth[mask] + 1e-8)
gain_reduction = np.clip(gain_reduction, 0.3, 1.0)  # max -10dB reduction

# Smooth gain changes
gain_reduction = uniform_filter1d(gain_reduction, size=int(SR*0.005))  # 5ms

# Apply broadband reduction (or we could do multiband)
# For simplicity: apply to the de-essed band only
band_attenuated = band_energy * gain_reduction
# Reconstruct: original - band + attenuated_band
vc_deessed = vc - band_energy + band_attenuated

r1 = measure(vc_deessed, "After de-esser")
print("  Crest:", round(r1['crest'], 1), "(was 11.9)")

# ── Step 2: High-shelf boost (6kHz+) ─────────────────────────────────
print("Step 2: 6kHz+ high-shelf...")

# Design high-shelf filter
sos_hs = signal.iirfilter(4, 6000, btype='high', fs=SR, output='sos')
# Get high-freq component
hf_component = signal.sosfilt(sos_hs, vc_deessed)

# Boost by 4dB
boost_db = 4.0
boost_linear = 10 ** (boost_db / 20)
vc_boosted = vc_deessed + hf_component * (boost_linear - 1)

r2 = measure(vc_boosted, "After high-shelf")
print("  VHigh:", round(r2['vhigh'], 1), "% (was", round(r1['vhigh'], 1), "%)")

# ── Step 3: Soft limiter (target crest 4-6) ──────────────────────────
print("Step 3: Soft limiter...")

# Soft knee limiter
threshold_linear = 0.7  # -3dB threshold
knee = 0.3

abs_vc = np.abs(vc_boosted)
gain = np.ones_like(vc_boosted)

# Soft knee compression
over = abs_vc - threshold_linear + knee
mask_over = over > 0
# soft knee compression ratio
ratio = 4.0  # 4:1
gain[mask_over] = (threshold_linear + over[mask_over] / ratio) / (abs_vc[mask_over] + 1e-8)
gain = np.clip(gain, 0.1, 1.0)
gain = uniform_filter1d(gain, size=int(SR*0.002))  # 2ms smoothing

vc_limited = vc_boosted * gain

r3 = measure(vc_limited, "After limiter")
print("  Crest:", round(r3['crest'], 1), "(target 4-6)")

# ── Step 4: Loudness normalize ───────────────────────────────────────
print("Step 4: Loudness normalize...")

# Match RMS to target speaker
target_rms = np.sqrt(np.mean(tgt[:len(vc_limited)]**2)) if len(tgt) >= len(vc_limited) else np.sqrt(np.mean(tgt**2))
# But don't over-amplify — cap at source RMS * 1.5
rms_cap = np.sqrt(np.mean(src**2)) * 1.5
target_rms = min(target_rms, rms_cap)

current_rms = np.sqrt(np.mean(vc_limited**2))
gain_norm = target_rms / (current_rms + 1e-8)
vc_final = vc_limited * gain_norm

r4 = measure(vc_final, "Final (normalized)")
print("  RMS:", round(r4['rms'], 3), "| SRC:", round(np.sqrt(np.mean(src**2)), 3), "| TGT:", round(np.sqrt(np.mean(tgt**2)), 3))

# ── Final comparison ─────────────────────────────────────────────────
print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

for label, audio in [("SOURCE", src), ("TARGET", tgt), ("VC raw", vc), ("VC processed", vc_final)]:
    r = measure(audio[:min(len(audio), len(vc_final))], label)
    print()
    print("  " + label + ":")
    print("    Centroid=" + str(round(r['centroid'])) + "Hz  VHigh=" + str(round(r['vhigh'],1)) +
          "%  Crest=" + str(round(r['crest'],1)) + "  HNR=" + str(round(r['hnr'],1)) + "dB")
    print("    Bands: L=" + str(round(r['low'],1)) + "% M=" + str(round(r['mid'],1)) +
          "% H=" + str(round(r['high'],1)) + "% VH=" + str(round(r['vhigh'],1)) + "%")

# Save
sf.write('/Users/asill/Desktop/vc_processed.wav', vc_final, SR)
print()
print("Saved: Desktop/vc_processed.wav")
print("Done!")
