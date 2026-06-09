#!/usr/bin/env python3
"""Mimi + FiLM latency on MPS — patched for >65536 channel conv limit."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys, time
sys.path.insert(0, '/Users/asill/btrv5')

import torch
import torch.nn.functional as F

# ── Monkey-patch: redirect large-channel MPS conv1d to CPU ────────────
_orig_conv1d = F.conv1d

def _patched_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if (input.device.type == 'mps' and 
        weight.shape[0] > 60000):  # >60k output channels → CPU fallback
        input_cpu = input.to('cpu')
        weight_cpu = weight.to('cpu')
        bias_cpu = bias.to('cpu') if bias is not None else None
        result_cpu = _orig_conv1d(input_cpu, weight_cpu, bias_cpu, 
                                   stride, padding, dilation, groups)
        return result_cpu.to('mps')
    return _orig_conv1d(input, weight, bias, stride, padding, dilation, groups)

F.conv1d = _patched_conv1d
torch.nn.functional.conv1d = _patched_conv1d
# Also patch the MPS conv1d implementation
import torch._C._nn as _nn
if hasattr(_nn, 'conv1d'):
    _orig_mps_conv = _nn.conv1d

print("MPS large-channel conv1d → CPU fallback patched")

# ── Now load everything ──────────────────────────────────────────────
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import soundfile as sf, numpy as np
from scipy import signal

device = torch.device('mps')
print("Device:", device)

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi, n_content=1).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location=device)['model_state_dict'])
splitter.eval()

SR = 24000
CHUNK_MS = 80
CHUNK_SAMPLES = int(SR * CHUNK_MS / 1000)
STRIDE = 1920

# Load audio
ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
d, sr = sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d.ndim > 1: d = d.mean(axis=1)
if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
n_chunks = (len(d) // CHUNK_SAMPLES)
chunks = [d[i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES] for i in range(n_chunks)]
print("Chunks:", n_chunks, "x", CHUNK_MS, "ms =", round(n_chunks*CHUNK_MS/1000,1), "s")

# Warmup
print("Warmup...")
with torch.no_grad():
    x = torch.from_numpy(chunks[0]).float().view(1,1,-1).to(device)
    _ = mimi_encode(x, mimi)
torch.mps.synchronize()
# Also pre-encode target speaker
d_tgt, sr_tgt = sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim > 1: d_tgt = d_tgt.mean(axis=1)
if sr_tgt != SR: d_tgt = signal.resample(d_tgt, int(len(d_tgt)*SR/sr_tgt))
safe_tgt = (len(d_tgt)//STRIDE)*STRIDE; d_tgt = d_tgt[:safe_tgt]
x_tgt = torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)
with torch.no_grad(): z_tgt, _ = mimi_encode(x_tgt, mimi); S_tgt = splitter.speaker_encoder(z_tgt)
torch.mps.synchronize()
time.sleep(0.3)

# Measurement
times_encode, times_split, times_decode = [], [], []
all_output = []
print("Measuring", n_chunks, "chunks...")

for i in range(n_chunks):
    x = torch.from_numpy(chunks[i]).float().view(1,1,-1).to(device)
    
    torch.mps.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(): z, codes = mimi_encode(x, mimi)
    torch.mps.synchronize(); times_encode.append(time.perf_counter()-t0)
    
    torch.mps.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        mimi.set_num_codebooks(1); z_q0=mimi.decode_latent(codes[:,:1,:]); mimi.set_num_codebooks(8)
        C=splitter.content_extractor(z_q0)
        n_ac=codes.shape[1]-1; mimi.set_num_codebooks(n_ac)
        z_ac=mimi.decode_latent(codes[:,1:,:]); mimi.set_num_codebooks(8)
        A=splitter.acoustic_adapter(z_ac,S_tgt,C); z_vc=C+A
    torch.mps.synchronize(); times_split.append(time.perf_counter()-t0)
    
    torch.mps.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(): x_out=mimi_decode_latent(mimi,z_vc)
    torch.mps.synchronize(); times_decode.append(time.perf_counter()-t0)
    all_output.append(x_out[0,0].cpu().numpy())

# Results
te=np.array(times_encode[1:])*1000; ts=np.array(times_split[1:])*1000
td=np.array(times_decode[1:])*1000; tt=te+ts+td

print()
print("=" * 65)
print("  MPS LATENCY (per 80ms chunk)")
print("=" * 65)
print("  {:20s} {:>8s} {:>8s} {:>8s} {:>8s}".format("","mean","p50","p95","max"))
print("  "+"-"*50)
for name,vals in [("Mimi encode",te),("Splitter+FiLM",ts),("Mimi decode",td),("TOTAL",tt)]:
    print("  {:20s} {:7.1f}ms {:7.1f}ms {:7.1f}ms {:7.1f}ms".format(
        name,np.mean(vals),np.median(vals),np.percentile(vals,95),np.max(vals)))

rtf=np.sum(tt)/1000/(n_chunks*CHUNK_MS/1000)
print()
print("  RTF: {:.3f} (vs CPU: 0.386)".format(rtf))
print("  Effective latency: ~{}ms".format(int(80+np.median(tt))))

# Compare CPU vs MPS
print()
print("  ─── CPU vs MPS ───")
print("  {:15s} {:>8s} {:>8s} {:>7s}".format("","CPU","MPS","Δ"))
cpu_enc, cpu_dec, cpu_tot = 18.7, 9.7, 28.6
print("  {:15s} {:7.1f}ms {:7.1f}ms {:6.1f}x".format("Encode",cpu_enc,np.median(te),cpu_enc/np.median(te)))
print("  {:15s} {:7.1f}ms {:7.1f}ms {:6.1f}x".format("Decode",cpu_dec,np.median(td),cpu_dec/np.median(td)))
print("  {:15s} {:7.1f}ms {:7.1f}ms {:6.1f}x".format("Total",cpu_tot,np.median(tt),cpu_tot/np.median(tt)))

full_out=np.concatenate(all_output)[:len(d)]
sf.write('/Users/asill/Desktop/vc_mps_test.wav',full_out,SR)
print("  Saved: Desktop/vc_mps_test.wav")
