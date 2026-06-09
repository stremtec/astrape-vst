#!/usr/bin/env python3
"""Aggressive post-processing: multiband dynamics + EQ + limiter."""
import numpy as np, soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

SR = 24000

vc, _ = sf.read('/Users/asill/Desktop/vc_p255_to_origin.wav')
src, _ = sf.read('/Users/asill/Desktop/vc_origin_src_orig.wav')
tgt, _ = sf.read('/Users/asill/Desktop/vc_origin_tgt_orig.wav')

def measure(a, label):
    f, t, Z = signal.stft(a, fs=SR, nperseg=512, noverlap=384)
    mag = np.abs(Z); total = mag.sum() + 1e-8
    c = np.sum(f[:len(f)//2, np.newaxis] * mag[:len(f)//2], axis=0)
    c /= (mag[:len(f)//2].sum(axis=0) + 1e-8)
    vhigh = mag[(f>=4000)&(f<8000)].sum()/total*100
    low = mag[(f>=0)&(f<300)].sum()/total*100
    mid = mag[(f>=300)&(f<2000)].sum()/total*100
    crest = np.max(np.abs(a)) / (np.sqrt(np.mean(a**2)) + 1e-8)
    return np.mean(c), vhigh, crest, low, mid

def multiband_compressor(audio, bands, thresholds_db, ratios, attack_ms=5, release_ms=50):
    """Apply compression per frequency band and sum back."""
    result = np.zeros_like(audio)
    for (lo, hi), thresh_db, ratio in zip(bands, thresholds_db, ratios):
        if lo == 0:
            sos = signal.butter(4, hi, btype='low', fs=SR, output='sos')
        elif hi >= SR/2:
            sos = signal.butter(4, lo, btype='high', fs=SR, output='sos')
        else:
            sos = signal.butter(4, [lo, hi], btype='band', fs=SR, output='sos')
        band = signal.sosfilt(sos, audio)
        # Compress this band
        env = np.abs(signal.hilbert(band))
        env = uniform_filter1d(env, size=int(SR*attack_ms/1000))
        thresh_linear = 10 ** (thresh_db / 20)
        gain = np.ones_like(band)
        over = env > thresh_linear
        gain[over] = (thresh_linear + (env[over] - thresh_linear) / ratio) / (env[over] + 1e-8)
        gain = uniform_filter1d(gain, size=int(SR*release_ms/1000))
        result += band * gain
    return result

def soft_clip(audio, threshold=0.8):
    """Tanh-based soft clipping."""
    x = audio / threshold
    return threshold * np.tanh(x)

# ── Process ──────────────────────────────────────────────────────────
print("Multiband compression...")

# Bands: [0-2k, 2k-5k, 5k-8k, 8k+]
bands = [(0, 2000), (2000, 5000), (5000, 8000), (8000, 12000)]
# Compress 2-5k more aggressively (where the transient spike is)
thresholds = [-3, -6, -3, -3]  # dB
ratios = [2, 6, 2, 2]

vc_comp = multiband_compressor(vc, bands, thresholds, ratios, attack_ms=3, release_ms=30)

print("High-shelf boost...")
# High-shelf: boost 6kHz+ by 6dB
sos_hs = signal.iirfilter(2, 6000, btype='high', fs=SR, output='sos')
hf = signal.sosfilt(sos_hs, vc_comp)
boost_db = 6.0
vc_eq = vc_comp + hf * (10**(boost_db/20) - 1)

print("Soft clipping...")
vc_clipped = soft_clip(vc_eq, threshold=0.6)

print("Normalize...")
rms_target = np.sqrt(np.mean(src**2)) * 1.3  # slightly louder than source
rms_current = np.sqrt(np.mean(vc_clipped**2))
vc_final = vc_clipped * (rms_target / (rms_current + 1e-8))

# ── Results ───────────────────────────────────────────────────────────
print()
print("=" * 70)
for label, audio in [("SOURCE", src), ("TARGET", tgt), ("VC raw", vc), ("VC processed", vc_final)]:
    c, vh, cr, lo, mi = measure(audio[:min(len(audio), len(vc))], label)
    print(label + ": Cent=" + str(round(c)) + "Hz VHigh=" + str(round(vh,1)) +
          "% Crest=" + str(round(cr,1)) + " Low=" + str(round(lo,1)) + "% Mid=" + str(round(mi,1)) + "%")

sf.write('/Users/asill/Desktop/vc_processed_v2.wav', vc_final, SR)
print()
print("Saved: Desktop/vc_processed_v2.wav")
