#!/usr/bin/env python3
"""Stage 4: Voice conversion via MimiSplitter."""

import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch
import soundfile as sf, numpy as np
from scipy import signal

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device:", device, flush=True)

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi).to(device)

ckpt_path = "checkpoints/mimi_splitter_v2_adv.pt"
if os.path.exists(ckpt_path):
    splitter.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    print("Loaded checkpoint:", ckpt_path)
else:
    print("WARNING: No checkpoint, using untrained splitter")

splitter.eval()
SR = 24000
safe_len = 48000

def load_and_encode(path, label=""):
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != SR:
        d = signal.resample(d, int(len(d)*SR/sr))
    if len(d) < safe_len:
        d = np.pad(d, (0, safe_len - len(d)))
    d = d[:safe_len]
    x = torch.from_numpy(d).float().view(1,1,-1).to(device)
    with torch.no_grad():
        z, codes = mimi_encode(x, mimi)
    print("  ", label, "z:", z.shape, "codes:", codes.shape)
    return x, z, codes, d

root = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

# Source: p255 (male)
x_src, z_src, codes_src, d_src = load_and_encode(
    f"{root}/p255/p255_001_mic1.flac", "p255 source")

# Target: p226 (female)
x_tgt, z_tgt, codes_tgt, d_tgt = load_and_encode(
    f"{root}/p226/p226_001_mic1.flac", "p226 target")

print()
print("--- Baselines ---")

# Mimi round-trip
with torch.no_grad():
    x_src_rt = mimi_decode_latent(mimi, z_src)
    x_tgt_rt = mimi_decode_latent(mimi, z_tgt)

sf.write("checkpoints/vc_src_rt.wav", x_src_rt[0,0].cpu().numpy(), SR)
sf.write("checkpoints/vc_tgt_rt.wav", x_tgt_rt[0,0].cpu().numpy(), SR)
sf.write("checkpoints/vc_src_orig.wav", d_src, SR)
sf.write("checkpoints/vc_tgt_orig.wav", d_tgt, SR)

# VC
print()
print("--- VC: p255 -> p226 ---")

with torch.no_grad():
    # Content from source (q0)
    mimi.set_num_codebooks(1)
    z_q0_src = mimi.decode_latent(codes_src[:, :1, :])
    mimi.set_num_codebooks(8)
    C_src = splitter.content_extractor(z_q0_src)

    # Speaker from target
    S_tgt = splitter.speaker_encoder(z_tgt)

    # Acoustic adapter: source acoustic + target speaker conditioning
    n_ac = codes_src.shape[1] - 1
    mimi.set_num_codebooks(n_ac)
    z_ac = mimi.decode_latent(codes_src[:, 1:, :])
    mimi.set_num_codebooks(8)
    A_vc = splitter.acoustic_adapter(z_ac, S_tgt, C_src)

    z_vc = C_src + A_vc
    x_vc = mimi_decode_latent(mimi, z_vc)

sf.write("checkpoints/vc_p255_to_p226.wav", x_vc[0,0].cpu().numpy(), SR)
print("  Saved: checkpoints/vc_p255_to_p226.wav")

# Metrics
print()
print("--- Quality Metrics ---")
from scipy.signal import stft

def compute_metrics(a, b, sr=24000, label=""):
    T = min(len(a), len(b))
    a, b = a[:T], b[:T]
    f1,_,Z1 = stft(a,fs=sr,nperseg=512,noverlap=256)
    f2,_,Z2 = stft(b,fs=sr,nperseg=512,noverlap=256)
    Ts = min(Z1.shape[1], Z2.shape[1])
    lsd = np.mean(np.sqrt(np.mean(
        (np.log10(np.abs(Z1[:,:Ts])+1e-8)-np.log10(np.abs(Z2[:,:Ts])+1e-8))**2, axis=0)))*20
    mse = np.mean((a-b)**2)
    rms_a = np.sqrt(np.mean(a**2))
    rms_b = np.sqrt(np.mean(b**2))
    centroid_a = np.sum(f1[:f1.shape[0]//2,np.newaxis]*np.abs(Z1[:f1.shape[0]//2,:Ts]),axis=0)
    centroid_a /= (np.sum(np.abs(Z1[:f1.shape[0]//2,:Ts]),axis=0)+1e-8)
    centroid_b = np.sum(f2[:f2.shape[0]//2,np.newaxis]*np.abs(Z2[:f2.shape[0]//2,:Ts]),axis=0)
    centroid_b /= (np.sum(np.abs(Z2[:f2.shape[0]//2,:Ts]),axis=0)+1e-8)
    print("  ", label, "LSD=", round(lsd,1), "dB MSE=", round(mse,6),
          "RMS=", round(rms_a,3), "/", round(rms_b,3),
          "Centroid=", round(np.mean(centroid_a)), "/", round(np.mean(centroid_b)), "Hz")
    return lsd, mse

lsd_src, _ = compute_metrics(d_src, x_src_rt[0,0].cpu().numpy(), SR, "SRC -> RT")
lsd_tgt, _ = compute_metrics(d_tgt, x_tgt_rt[0,0].cpu().numpy(), SR, "TGT -> RT")
lsd_vc_src, _ = compute_metrics(d_src, x_vc[0,0].cpu().numpy(), SR, "VC  -> SRC")
lsd_vc_tgt, _ = compute_metrics(d_tgt, x_vc[0,0].cpu().numpy(), SR, "VC  -> TGT")

# Speaker/content analysis
print()
print("--- Latent Analysis ---")
with torch.no_grad():
    _, C_vc, S_vc, _ = splitter(z_vc, codes_src)
    cos_sim = torch.nn.functional.cosine_similarity(S_tgt, S_vc)
    mse_c = torch.nn.functional.mse_loss(C_src, C_vc)
    print("  Speaker cos(VC, target):", round(cos_sim.item(), 4))
    print("  Content MSE(VC, source):", round(mse_c.item(), 4))
    print()
    print("  If VC works: cos_sim > 0.5, content preserved")
    print("  LSD(VC,TGT)=", round(lsd_vc_tgt,1), "dB (lower=more target-like)")

print()
print("Done! Listen: checkpoints/vc_p255_to_p226.wav")
