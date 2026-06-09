#!/usr/bin/env python3
"""p255 -> origin.mp3 VC test with Mimi Splitter (FiLM adapter)."""
import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, soundfile as sf, numpy as np
from scipy import signal

device = torch.device('cpu')
SR = 24000
SAFE_LEN = 96000  # 4s for longer audio

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location='cpu')['model_state_dict'])
splitter.eval()
print("Model loaded (60-spk FiLM)")

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def load_audio(path, label=""):
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != SR:
        d = signal.resample(d, int(len(d)*SR/sr))
    orig_len = len(d)
    # Pad or truncate to safe_len (multiple of 1920)
    stride = 1920
    safe = (len(d) // stride) * stride
    d = d[:safe]
    if len(d) < safe:
        d = np.pad(d, (0, safe - len(d)))
    x = torch.from_numpy(d).float().view(1, 1, -1).to(device)
    with torch.no_grad():
        z, codes = mimi_encode(x, mimi)
    print("  " + label + ": " + str(round(len(d)/SR,1)) + "s, z=" + str(z.shape) + " codes=" + str(codes.shape))
    return x, z, codes, d, orig_len

print()
print("=== Loading audio ===")

# Source: p255 (English male)
x_src, z_src, codes_src, d_src, src_len = load_audio(
    f"{ROOT}/p255/p255_001_mic1.flac", "p255 source")

# Target: origin.mp3 (Korean female)
x_tgt, z_tgt, codes_tgt, d_tgt, tgt_len = load_audio(
    "/Users/asill/Downloads/origin.mp3", "origin target")

print()
print("=== VC: p255 -> origin ===")

with torch.no_grad():
    # Content from source
    mimi.set_num_codebooks(1)
    z_q0_src = mimi.decode_latent(codes_src[:, :1, :])
    mimi.set_num_codebooks(8)
    C_src = splitter.content_extractor(z_q0_src)

    # Speaker from target
    S_tgt = splitter.speaker_encoder(z_tgt)

    # Acoustic adapter
    n_ac = codes_src.shape[1] - 1
    mimi.set_num_codebooks(n_ac)
    z_ac = mimi.decode_latent(codes_src[:, 1:, :])
    mimi.set_num_codebooks(8)
    A_vc = splitter.acoustic_adapter(z_ac, S_tgt, C_src)

    z_vc = C_src + A_vc
    x_vc = mimi_decode_latent(mimi, z_vc)

# Also make Mimi round-trip outputs
x_src_rt = mimi_decode_latent(mimi, z_src)
x_tgt_rt = mimi_decode_latent(mimi, z_tgt)

# Save
vc_np = x_vc[0,0].cpu().numpy()[:src_len]
sf.write("checkpoints/vc_p255_to_origin.wav", vc_np, SR)
sf.write("checkpoints/vc_origin_src_rt.wav", x_src_rt[0,0].cpu().numpy()[:src_len], SR)
sf.write("checkpoints/vc_origin_tgt_rt.wav", x_tgt_rt[0,0].cpu().numpy()[:tgt_len], SR)
sf.write("checkpoints/vc_origin_src_orig.wav", d_src[:src_len], SR)
sf.write("checkpoints/vc_origin_tgt_orig.wav", d_tgt[:tgt_len], SR)

print("Saved: checkpoints/vc_p255_to_origin.wav")
print("Saved: checkpoints/vc_origin_*.wav")

# Spectral analysis
print()
print("=== Spectral comparison ===")
from scipy.signal import stft

def analyze(a, label, sr=SR):
    a = a - np.mean(a)
    f,_,Z = stft(a, fs=sr, nperseg=512, noverlap=256)
    mag = np.abs(Z)
    centroid = np.sum(f[:f.shape[0]//2,np.newaxis]*mag[:f.shape[0]//2,:], axis=0)
    centroid /= (np.sum(mag[:f.shape[0]//2,:], axis=0)+1e-8)
    rms = np.sqrt(np.mean(a**2))
    # Band energies
    total = np.sum(mag) + 1e-8
    low = np.sum(mag[(f>=80)&(f<300),:])/total*100
    mid = np.sum(mag[(f>=300)&(f<1000),:])/total*100
    high = np.sum(mag[(f>=1000)&(f<4000),:])/total*100
    print("  " + label + ": RMS=" + str(round(rms,3)) +
          " Centroid=" + str(round(np.mean(centroid))) + "Hz" +
          " Low=" + str(round(low,1)) + "% Mid=" + str(round(mid,1)) +
          "% High=" + str(round(high,1)) + "%")

analyze(d_src[:src_len], "Source (p255)")
analyze(d_tgt[:tgt_len], "Target (origin)")
analyze(vc_np, "VC output")
analyze(x_src_rt[0,0].cpu().numpy()[:src_len], "SRC RT")
analyze(x_tgt_rt[0,0].cpu().numpy()[:tgt_len], "TGT RT")

print()
print("Done! Listen: checkpoints/vc_p255_to_origin.wav")
