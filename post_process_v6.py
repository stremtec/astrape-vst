#!/usr/bin/env python3
"""Post-process v6: dynamic high-band gate for hiss tail removal."""
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

# ── Step 1: 2-5kHz compression ───────────────────────────────────────
print("Step 1: 2-5kHz compression...")
band_25k = filter_band(vc, 2000, 5000)
rest = vc - band_25k
env = np.abs(signal.hilbert(band_25k))
env = uniform_filter1d(env, size=int(SR*0.003))
th = 10**(-6/20); ratio = 5.0
gain = np.ones_like(band_25k)
over = env > th
gain[over] = (th + (env[over]-th)/ratio) / (env[over]+1e-8)
gain = uniform_filter1d(gain, size=int(SR*0.02))
vc1 = rest + band_25k * gain

# ── Step 2: EQ ───────────────────────────────────────────────────────
print("Step 2: EQ...")
# High-shelf +4dB at 6kHz (reduced from v5's +6dB)
sos_hs = signal.iirfilter(2, 6000, btype='high', fs=SR, output='sos')
hf = signal.sosfilt(sos_hs, vc1)
vc2 = vc1 + hf * (10**(4/20) - 1)

# Reduce 8-12kHz by -2dB
sos_vh = signal.iirfilter(2, 8000, btype='high', fs=SR, output='sos')
vh = signal.sosfilt(sos_vh, vc2)
vc2 = vc2 - vh * 0.2

# Presence 3-4kHz +2.5dB (restore articulation)
band_34 = filter_band(vc2, 3000, 4000)
vc2 = vc2 + band_34 * 0.33

# ── Step 3: Dynamic high-band gate ───────────────────────────────────
print("Step 3: Dynamic high-band gate...")

# Extract bands
band_high = filter_band(vc2, 5000, SR/2)
band_low = vc2 - band_high
band_mid = filter_band(vc2, 300, 5000)

# Frame-wise analysis
frame_len = int(SR * 0.02)  # 20ms
n_frames = len(vc2) // frame_len

# Compute median speech RMS (from mid-band energy)
mid_rms_per_frame = []
for i in range(n_frames):
    s, e = i*frame_len, (i+1)*frame_len
    mid_rms_per_frame.append(np.sqrt(np.mean(band_mid[s:e]**2)))
speech_rms_median = np.median(mid_rms_per_frame)

attenuated_frames = 0
total_frames = 0

for i in range(n_frames):
    s, e = i*frame_len, min((i+1)*frame_len, len(vc2))
    total_frames += 1
    
    high_e = np.sum(band_high[s:e]**2)
    low_e = np.sum(band_low[s:e]**2)
    total_e = high_e + low_e + 1e-8
    high_ratio = high_e / total_e
    
    mid_rms = np.sqrt(np.mean(band_mid[s:e]**2))
    frame_rms = np.sqrt(np.mean(vc2[s:e]**2))
    
    # Gate: attenuate high-band in low-energy frames with excessive >5kHz
    is_hiss_tail = (high_ratio > 0.55 and 
                    mid_rms < speech_rms_median * 0.4)
    
    if is_hiss_tail:
        # Attenuation: reduce high band by 6-10dB depending on severity
        atten_db = min(10, max(4, (high_ratio - 0.5) * 15))
        atten_linear = 10 ** (-atten_db / 20)
        band_high[s:e] *= atten_linear
        attenuated_frames += 1

vc3 = band_low + band_high
print("  Attenuated " + str(attenuated_frames) + "/" + str(total_frames) +
      " frames (" + str(round(attenuated_frames/total_frames*100,1)) + "%)")

# ── Step 4: Limiter ──────────────────────────────────────────────────
print("Step 4: Limiter...")
lookahead = int(SR * 0.002); ceiling = 0.55
env_la = np.zeros_like(vc3)
for i in range(len(vc3)):
    end = min(i + lookahead, len(vc3))
    env_la[i] = np.max(np.abs(vc3[i:end]))
gain_la = np.ones_like(vc3)
over = env_la > ceiling
gain_la[over] = ceiling / (env_la[over] + 1e-8)
gain_la = uniform_filter1d(gain_la, size=int(SR*0.001))
vc4 = vc3 * gain_la

# ── Step 5: Normalize ────────────────────────────────────────────────
target_rms = np.sqrt(np.mean(src**2)) * 1.15
vc_final = vc4 * (target_rms / (np.sqrt(np.mean(vc4**2)) + 1e-8))

# ── Results ───────────────────────────────────────────────────────────
cF, vhF, crF = measure(vc_final)
print()
print("FINAL: Cent=" + str(round(cF)) + "Hz VHigh=" + str(round(vhF,1)) + "% Crest=" + str(round(crF,1)))

# Frame check at problematic timestamps
f, t, Z = signal.stft(vc_final, fs=SR, nperseg=512, noverlap=384)
mag = np.abs(Z)
frame_ms = 512/SR*1000  # ~21ms per frame
for sec in [1.20, 1.25, 1.27, 1.58, 1.71, 1.73]:
    fi = int(sec * SR / 256)  # hop=256
    if fi < mag.shape[1]:
        total = mag[:,fi].sum()+1e-8
        h5 = mag[(f>=5000),fi].sum()/total*100
        print("  t=" + str(round(sec,2)) + "s: >5kHz=" + str(round(h5,1)) + "%")

sf.write('/Users/asill/Desktop/vc_processed_v6.wav', vc_final[:len(vc)], SR)
print()
print("Saved: Desktop/vc_processed_v6.wav")
