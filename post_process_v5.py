#!/usr/bin/env python3
"""Post-process v5: simple, clean chain. No frame-aware clamping."""
import numpy as np, soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

SR = 24000

vc, _ = sf.read('/Users/asill/Desktop/vc_p255_to_origin.wav')
src, _ = sf.read('/Users/asill/Desktop/vc_origin_src_orig.wav')
tgt, _ = sf.read('/Users/asill/Desktop/vc_origin_tgt_orig.wav')

def measure(a):
    f, t, Z = signal.stft(a, fs=SR, nperseg=512, noverlap=384)
    mag = np.abs(Z); total = mag.sum() + 1e-8
    c = np.sum(f[:len(f)//2, np.newaxis] * mag[:len(f)//2], axis=0)
    c /= (mag[:len(f)//2].sum(axis=0) + 1e-8)
    vhigh = mag[(f>=4000)&(f<8000)].sum()/total*100
    vh2 = mag[(f>=5000)].sum()/total*100
    crest = np.max(np.abs(a)) / (np.sqrt(np.mean(a**2)) + 1e-8)
    hnr_spec = np.mean(mag, axis=1)
    peaks = signal.find_peaks(hnr_spec[:len(f)//2], distance=5, prominence=hnr_spec.max()*0.01)[0]
    h_e = hnr_spec[peaks].sum() if len(peaks)>0 else 0
    n_e = hnr_spec.sum() - h_e
    hnr = 10*np.log10(h_e/(n_e+1e-8))
    return np.mean(c), vhigh, vh2, crest, hnr

def filter_band(audio, lo, hi):
    if lo == 0:
        sos = signal.butter(4, hi, btype='low', fs=SR, output='sos')
    elif hi >= SR/2:
        sos = signal.butter(4, lo, btype='high', fs=SR, output='sos')
    else:
        sos = signal.butter(4, [lo, hi], btype='band', fs=SR, output='sos')
    return signal.sosfilt(sos, audio)

# ── Initial metrics ──────────────────────────────────────────────────
c_raw, vh_raw, vh2_raw, cr_raw, hn_raw = measure(vc)
print("RAW:  Cent=" + str(round(c_raw)) + "Hz VHigh=" + str(round(vh_raw,1)) +
      "% >5k=" + str(round(vh2_raw,1)) + "% Crest=" + str(round(cr_raw,1)) +
      " HNR=" + str(round(hn_raw,1)) + "dB")

# ── Step 1: Light MB compression on 2-5k only ───────────────────────
print("Step 1: 2-5kHz compression...")
band_25k = filter_band(vc, 2000, 5000)
voice_rest = vc - band_25k

env = np.abs(signal.hilbert(band_25k))
env = uniform_filter1d(env, size=int(SR*0.003))
th = 10**(-6/20)  # -6dB threshold
ratio = 5.0
gain = np.ones_like(band_25k)
over = env > th
gain[over] = (th + (env[over]-th)/ratio) / (env[over]+1e-8)
gain = uniform_filter1d(gain, size=int(SR*0.02))
vc1 = voice_rest + band_25k * gain

c1, vh1, vh21, cr1, hn1 = measure(vc1)
print("  Cent=" + str(round(c1)) + "Hz VHigh=" + str(round(vh1,1)) +
      "% >5k=" + str(round(vh21,1)) + "% Crest=" + str(round(cr1,1)))

# ── Step 2: Gentle EQ ────────────────────────────────────────────────
print("Step 2: EQ...")
# High-shelf +6dB at 6kHz
sos_hs = signal.iirfilter(2, 6000, btype='high', fs=SR, output='sos')
hf = signal.sosfilt(sos_hs, vc1)
boost = 10**(6/20)
vc2 = vc1 + hf * (boost - 1)

# Presence +2dB at 3-4kHz
band_34k = filter_band(vc2, 3000, 4000)
vc2 = vc2 + band_34k * 0.26

c2, vh2, vh22, cr2, hn2 = measure(vc2)
print("  Cent=" + str(round(c2)) + "Hz VHigh=" + str(round(vh2,1)) +
      "% >5k=" + str(round(vh22,1)) + "% Crest=" + str(round(cr2,1)))

# ── Step 3: Soft lookahead limiter ───────────────────────────────────
print("Step 3: Limiter...")
lookahead = int(SR * 0.002)
ceiling = 0.55
env_la = np.zeros_like(vc2)
for i in range(len(vc2)):
    end = min(i + lookahead, len(vc2))
    env_la[i] = np.max(np.abs(vc2[i:end]))
gain_la = np.ones_like(vc2)
over = env_la > ceiling
gain_la[over] = ceiling / (env_la[over] + 1e-8)
gain_la = uniform_filter1d(gain_la, size=int(SR*0.001))
vc3 = vc2 * gain_la

c3, vh3, vh23, cr3, hn3 = measure(vc3)
print("  Cent=" + str(round(c3)) + "Hz VHigh=" + str(round(vh3,1)) +
      "% >5k=" + str(round(vh23,1)) + "% Crest=" + str(round(cr3,1)))

# ── Step 4: Normalize ───────────────────────────────────────────────
target_rms = np.sqrt(np.mean(src**2)) * 1.15
vc_final = vc3 * (target_rms / (np.sqrt(np.mean(vc3**2)) + 1e-8))

cF, vhF, vh2F, crF, hnF = measure(vc_final)
print()
print("FINAL: Cent=" + str(round(cF)) + "Hz VHigh=" + str(round(vhF,1)) +
      "% >5k=" + str(round(vh2F,1)) + "% Crest=" + str(round(crF,1)) +
      " HNR=" + str(round(hnF,1)) + "dB")

# Check frame distribution
f, t, Z = signal.stft(vc_final, fs=SR, nperseg=512, noverlap=384)
mag = np.abs(Z)
frame_h5 = []
for j in range(mag.shape[1]):
    total = mag[:,j].sum()+1e-8
    frame_h5.append(mag[(f>=5000),j].sum()/total*100)
frame_h5 = np.array(frame_h5)
print("  >5kHz: median=" + str(round(np.median(frame_h5),1)) +
      "% p90=" + str(round(np.percentile(frame_h5,90),1)) +
      "% max=" + str(round(np.max(frame_h5),1)) + "%")

sf.write('/Users/asill/Desktop/vc_processed_v5.wav', vc_final[:len(vc)], SR)
print()
print("Saved: Desktop/vc_processed_v5.wav")
