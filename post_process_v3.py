#!/usr/bin/env python3
"""Post-process v3: multiband + hard lookahead limiter."""
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

# ── Multiband compression ─────────────────────────────────────────────
def comp_band(audio, lo, hi, thresh_db, ratio):
    if lo == 0:
        sos = signal.butter(4, hi, btype='low', fs=SR, output='sos')
    elif hi >= SR/2:
        sos = signal.butter(4, lo, btype='high', fs=SR, output='sos')
    else:
        sos = signal.butter(4, [lo, hi], btype='band', fs=SR, output='sos')
    band = signal.sosfilt(sos, audio)
    env = np.abs(signal.hilbert(band))
    env = uniform_filter1d(env, size=int(SR*0.002))  # 2ms attack
    th = 10**(thresh_db/20)
    gain = np.ones_like(band)
    over = env > th
    gain[over] = (th + (env[over]-th)/ratio) / (env[over]+1e-8)
    gain = uniform_filter1d(gain, size=int(SR*0.01))
    return band * gain

# Apply compression: tame 2-5k heavily, others lightly
bands = [(0,2000,-3,3), (2000,5000,-8,8), (5000,8000,-3,3), (8000,12000,-2,2)]
result = np.zeros_like(vc)
for lo, hi, th_dB, rat in bands:
    result += comp_band(vc, lo, hi, th_dB, rat)

print("After MB comp:", end=" ")
c, vh, cr = measure(result)
print("Cent=" + str(round(c)) + "Hz VHigh=" + str(round(vh,1)) + "% Crest=" + str(round(cr,1)))

# ── High-shelf ─────────────────────────────────────────────────────────
sos_hs = signal.iirfilter(2, 6000, btype='high', fs=SR, output='sos')
hf = signal.sosfilt(sos_hs, result)
boost = 10**(8/20)  # +8dB
result = result + hf * (boost - 1)

print("After EQ:", end=" ")
c, vh, cr = measure(result)
print("Cent=" + str(round(c)) + "Hz VHigh=" + str(round(vh,1)) + "% Crest=" + str(round(cr,1)))

# ── Lookahead hard limiter ────────────────────────────────────────────
lookahead_samples = int(SR * 0.003)  # 3ms
ceiling = 0.5  # -6dB

envelope = np.zeros_like(result)
for i in range(len(result)):
    end = min(i + lookahead_samples, len(result))
    envelope[i] = np.max(np.abs(result[i:end]))

gain = np.ones_like(result)
over = envelope > ceiling
gain[over] = ceiling / (envelope[over] + 1e-8)
gain = uniform_filter1d(gain, size=int(SR*0.001))

result = result * gain

print("After limiter:", end=" ")
c, vh, cr = measure(result)
print("Cent=" + str(round(c)) + "Hz VHigh=" + str(round(vh,1)) + "% Crest=" + str(round(cr,1)))

# ── Normalize ────────────────────────────────────────────────────────
target_rms = np.sqrt(np.mean(src**2)) * 1.2
result = result * (target_rms / (np.sqrt(np.mean(result**2)) + 1e-8))

print()
print("FINAL: Cent=" + str(round(c)) + "Hz VHigh=" + str(round(vh,1)) + "% Crest=" + str(round(cr,1)))
print("Target: Crest 4-6, VHigh >10%")

sf.write('/Users/asill/Desktop/vc_processed_v3.wav', result[:len(vc)], SR)
print()
print("Saved: Desktop/vc_processed_v3.wav")
