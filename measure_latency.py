#!/usr/bin/env python3
"""Mimi + FiLM latency measurement with streaming simulation."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys, time
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch
import soundfile as sf, numpy as np
from scipy import signal

device = torch.device('mps')
print("Device:", device, "(with CPU fallback for large-channel ops)")

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi, n_content=1).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location=device)['model_state_dict'])
splitter.eval()

SR = 24000
CHUNK_MS = 80  # Mimi frame = 80ms
CHUNK_SAMPLES = int(SR * CHUNK_MS / 1000)  # 1920 samples
STRIDE = 1920

# Load audio
ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
d, sr = sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d.ndim > 1: d = d.mean(axis=1)
if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
n_chunks = (len(d) // CHUNK_SAMPLES)
chunks = [d[i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES] for i in range(n_chunks)]

print("Chunks:", n_chunks, "x", CHUNK_MS, "ms =", round(n_chunks*CHUNK_MS/1000,1), "s")
print()

# ── Warmup ───────────────────────────────────────────────────────────
print("Warmup...")
with torch.no_grad():
    x = torch.from_numpy(chunks[0]).float().view(1,1,-1).to(device)
    _ = mimi_encode(x, mimi)
if device.type == 'mps':
    torch.mps.synchronize()
time.sleep(0.5)

# ── Measurement ───────────────────────────────────────────────────────
times_encode = []
times_split = []
times_decode = []

# Pre-encode target speaker (origin.mp3)
d_tgt, sr_tgt = sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim > 1: d_tgt = d_tgt.mean(axis=1)
if sr_tgt != SR: d_tgt = signal.resample(d_tgt, int(len(d_tgt)*SR/sr_tgt))
safe_tgt = (len(d_tgt)//STRIDE)*STRIDE
d_tgt = d_tgt[:safe_tgt]
x_tgt = torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)
with torch.no_grad():
    z_tgt, _ = mimi_encode(x_tgt, mimi)
    S_tgt = splitter.speaker_encoder(z_tgt)
if device.type == 'mps':
    torch.mps.synchronize()

print("Measuring", n_chunks, "chunks...")

# Streaming simulation
all_output = []

for i in range(n_chunks):
    chunk = chunks[i]
    x = torch.from_numpy(chunk).float().view(1,1,-1).to(device)
    
    # Step 1: Encode
    if device.type == 'mps': torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        z, codes = mimi_encode(x, mimi)
    if device.type == 'mps': torch.mps.synchronize()
    times_encode.append(time.perf_counter() - t0)
    
    # Step 2: Splitter (content + speaker + acoustic adapter)
    if device.type == 'mps': torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        mimi.set_num_codebooks(1)
        z_q0 = mimi.decode_latent(codes[:, :1, :])
        mimi.set_num_codebooks(8)
        C = splitter.content_extractor(z_q0)
        n_ac = codes.shape[1] - 1
        mimi.set_num_codebooks(n_ac)
        z_ac = mimi.decode_latent(codes[:, 1:, :])
        mimi.set_num_codebooks(8)
        A = splitter.acoustic_adapter(z_ac, S_tgt, C)
        z_vc = C + A
    if device.type == 'mps': torch.mps.synchronize()
    times_split.append(time.perf_counter() - t0)
    
    # Step 3: Decode
    if device.type == 'mps': torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        x_out = mimi_decode_latent(mimi, z_vc)
    if device.type == 'mps': torch.mps.synchronize()
    times_decode.append(time.perf_counter() - t0)
    
    all_output.append(x_out[0,0].cpu().numpy())

# ── Results ───────────────────────────────────────────────────────────
te = np.array(times_encode[1:]) * 1000  # skip first (warmup)
ts = np.array(times_split[1:]) * 1000
td = np.array(times_decode[1:]) * 1000
tt = te + ts + td

print()
print("=" * 65)
print("  LATENCY BREAKDOWN (per 80ms chunk)")
print("=" * 65)
print("  {:20s} {:>8s} {:>8s} {:>8s} {:>8s}".format("", "mean", "p50", "p95", "max"))
print("  " + "-" * 50)
for name, vals in [("Mimi encode", te), ("Splitter+FiLM", ts), ("Mimi decode", td), ("TOTAL", tt)]:
    print("  {:20s} {:7.1f}ms {:7.1f}ms {:7.1f}ms {:7.1f}ms".format(
        name, np.mean(vals), np.median(vals), np.percentile(vals,95), np.max(vals)))

# RTF
total_audio_s = n_chunks * CHUNK_MS / 1000
total_proc_s = np.sum(tt) / 1000
rtf = total_proc_s / total_audio_s
print()
print("  Total audio: {:.2f}s | Total processing: {:.2f}s | RTF: {:.3f}".format(
    total_audio_s, total_proc_s, rtf))
print("  RTF < 1.0 → real-time capable")
print("  RTF < 0.5 → streaming headroom")

# First-output latency (if simulating streaming)
# In streaming: first chunk needs full encode+split+decode
first_latency_ms = te[0] + ts[0] + td[0]
print()
print("  First-output latency: {:.0f}ms".format(first_latency_ms))
print("  Target: <200ms for conversational VC")

# Also measure model loading overhead
print()
print("  Algorithmic latency: ~80ms (1 Mimi frame)")
print("  + per-frame FiLM: ~0ms (no lookahead)")
print("  → Theoretical minimum: 80ms")

# Save concatenated output for validation
full_out = np.concatenate(all_output)[:len(d)]
sf.write('/Users/asill/Desktop/vc_stream_test.wav', full_out, SR)
print("  Saved: Desktop/vc_stream_test.wav")
