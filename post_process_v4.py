#!/usr/bin/env python3
"""Post-process v4: frame-aware high-band control + restore presence."""
import numpy as np, soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

SR = 24000

vc, _ = sf.read('/Users/asill/Desktop/vc_p255_to_origin.wav')
src, _ = sf.read('/Users/asill/Desktop/vc_origin_src_orig.wav')

def measure(a):
    f, t, Z = signal.stft(a, fs=SR, nperseg=512, noverlap=384)
    mag = np.abs(Z); total = mag.sum() + 1e-8
    c = np.sum(f[:len(f)//2, np.newaxis] * mag[:len(f)//2], axis=0)
    c /= (mag[:len(f)//2].sum(axis=0) + 1e-8)
    vhigh = mag[(f>=4000)&(f<8000)].sum()/total*100
    crest = np.max(np.abs(a)) / (np.sqrt(np.mean(a**2)) + 1e-8)
    return np.mean(c), vhigh, crest

def filter_band(audio, lo, hi):
    if lo == 0:
        sos = signal.butter(4, hi, btype='low', fs=SR, output='sos')
    elif hi >= SR/2:
        sos = signal.butter(4, lo, btype='high', fs=SR, output='sos')
    else:
        sos = signal.butter(4, [lo, hi], btype='band', fs=SR, output='sos')
    return signal.sosfilt(sos, audio)

# ── Step 1: Multiband compression (2-5k tamed) ──────────────────────
print("Step 1: MB compression...")
bands = [(0,2000,-3,2), (2000,5000,-6,5), (5000,8000,-3,2)]
result = np.zeros_like(vc)
for lo, hi, th_dB, rat in bands:
    band = filter_band(vc, lo, hi)
    env = np.abs(signal.hilbert(band))
    env = uniform_filter1d(env, size=int(SR*0.003))
    th = 10**(th_dB/20)
    gain = np.ones_like(band)
    over = env > th
    gain[over] = (th + (env[over]-th)/rat) / (env[over]+1e-8)
    gain = uniform_filter1d(gain, size=int(SR*0.015))
    result += band * gain

# ── Step 2: Frame-aware high-band enhancement ─────────────────────────
print("Step 2: Frame-aware high-band...")

# Extract bands for analysis
band_high = filter_band(result, 5000, SR/2)  # 5kHz+
band_vhigh = filter_band(result, 8000, SR/2)  # 8kHz+
band_mid = filter_band(result, 300, 5000)      # 300-5k (reference)

# Frame-wise energy analysis
frame_len = int(SR * 0.02)  # 20ms
n_frames = len(result) // frame_len

for i in range(n_frames):
    start = i * frame_len
    end = start + frame_len
    
    high_e = np.sum(band_high[start:end]**2)
    mid_e = np.sum(band_mid[start:end]**2)
    total_e = high_e + mid_e + 1e-8
    high_ratio = high_e / total_e
    
    # Clamp: if >5kHz energy > 60%, pull it down
    if high_ratio > 0.60:
        scale = np.sqrt(0.55 / high_ratio)  # reduce to 55%
        band_high[start:end] *= scale
        band_vhigh[start:end] *= scale * 0.7  # 8k+ attenuated more

# Reconstruct
band_low = filter_band(result, 0, 5000)
# Remove old high, add controlled high
result_clean = band_low - filter_band(result, 5000, SR/2) + band_high

# ── Step 3: Moderate high-shelf (+4dB at 5k+) ─────────────────────
print("Step 3: High-shelf (+4dB)...")
sos_hs = signal.iirfilter(2, 5500, btype='high', fs=SR, output='sos')
hf = signal.sosfilt(sos_hs, result_clean)
boost = 10**(4/20)
result_clean = result_clean + hf * (boost - 1)

# ── Step 4: Restore presence (3-5kHz +2dB) ────────────────────────
print("Step 4: Restore presence...")
band_35k = filter_band(result_clean, 3000, 5000)
result_clean = result_clean + band_35k * 0.26  # ~+2dB

# ── Step 5: Soft limiter ──────────────────────────────────────────────
print("Step 5: Limiter...")
lookahead = int(SR * 0.002)
ceiling = 0.55
env = np.zeros_like(result_clean)
for i in range(len(result_clean)):
    end = min(i + lookahead, len(result_clean))
    env[i] = np.max(np.abs(result_clean[i:end]))
gain = np.ones_like(result_clean)
over = env > ceiling
gain[over] = ceiling / (env[over] + 1e-8)
gain = uniform_filter1d(gain, size=int(SR*0.001))
result_clean = result_clean * gain

# ── Step 6: Normalize ─────────────────────────────────────────────────
target_rms = np.sqrt(np.mean(src**2)) * 1.1
result_final = result_clean * (target_rms / (np.sqrt(np.mean(result_clean**2)) + 1e-8))

# ── Results ───────────────────────────────────────────────────────────
print()
c, vh, cr = measure(result_final)
print("v4 FINAL: Cent=" + str(round(c)) + "Hz VHigh=" + str(round(vh,1)) + "% Crest=" + str(round(cr,1)))

# Frame analysis
f, t, Z = signal.stft(result_final, fs=SR, nperseg=512, noverlap=384)
mag = np.abs(Z)
for frame_idx, sec in [(0, 0.0), (75, 1.5), (87, 1.75)]:
    if frame_idx < mag.shape[1]:
        frame = mag[:, frame_idx]
        total = frame.sum() + 1e-8
        h5 = frame[(f>=5000)].sum()/total*100
        print("  Frame " + str(round(sec,2)) + "s: >5kHz=" + str(round(h5,1)) + "%")

sf.write('/Users/asill/Desktop/vc_processed_v4.wav', result_final[:len(vc)], SR)
print()
print("Saved: Desktop/vc_processed_v4.wav")
