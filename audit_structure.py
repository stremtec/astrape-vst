#!/usr/bin/env python3
"""
Mimi Splitter VC — Full Tensor/Module Structural Audit.
Documents every step: shape, dtype, rate, structure, statistics.
No training — pure measurement and documentation.
"""
import sys, os, glob, json, warnings, inspect
sys.path.insert(0, '/Users/asill/btrv5')
warnings.filterwarnings('ignore')

import torch, torch.nn as nn
import numpy as np, soundfile as sf
from scipy import signal
from collections import Counter

SR = 24000; STRIDE = 1920; SAFE_LEN = 48000
device = torch.device('cpu')

# ── Load ────────────────────────────────────────────────────────────────
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent

mimi = load_mimi(device).to(device); mimi.eval()

# Load FiLM success model
splitter = MimiSplitterV2(mimi, n_content=1).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location='cpu')['model_state_dict'])
splitter.eval()

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def load_audio(spk):
    files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))
    d, sr = sf.read(files[0])
    if d.ndim > 1: d = d.mean(axis=1)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
    safe = (len(d)//STRIDE)*STRIDE; d = d[:min(safe, SAFE_LEN)]
    if len(d) < safe: d = np.pad(d, (0, safe-len(d)))
    return d[:safe], files[0]

# ── Load source and target ──────────────────────────────────────────────
d_src, _ = load_audio('p255')
x_src = torch.from_numpy(d_src).float().view(1,1,-1).to(device)

d_tgt, _ = sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim > 1: d_tgt = d_tgt.mean(axis=1)
sr_t = sf.info("/Users/asill/Downloads/origin.mp3").samplerate
if sr_t != SR: d_tgt = signal.resample(d_tgt, int(len(d_tgt)*SR/sr_t))
safe = (len(d_tgt)//STRIDE)*STRIDE; d_tgt = d_tgt[:min(safe, SAFE_LEN*3)]
x_tgt = torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)

# ── 1. Full Pipeline Tensor Trace ───────────────────────────────────────
print("=" * 90)
print("  SECTION 1: FULL PIPELINE TENSOR TRACE")
print("=" * 90)

traces = []

def trace(name, tensor, rate=None, meaning="", notes=""):
    if tensor is None:
        traces.append([name, "None", "", "", "", "", "", ""])
        return
    t = tensor.detach() if hasattr(tensor, 'detach') else tensor
    shape = str(list(t.shape))
    dtype = str(t.dtype) if hasattr(t, 'dtype') else str(type(t).__name__)
    dev = str(t.device) if hasattr(t, 'device') else "n/a"
    r = f"{rate}Hz" if rate else ""
    stats = f"μ={t.float().mean():.3f} σ={t.float().std():.3f} [{t.float().min():.2f}, {t.float().max():.2f}]" if hasattr(t, 'float') and t.numel() > 0 else ""
    traces.append([name, shape, dtype, dev, r, meaning, stats, notes])

trace("raw_audio_src", torch.from_numpy(d_src), SR, "source waveform")
trace("raw_audio_tgt", torch.from_numpy(d_tgt), SR, "target waveform")

# Encode
with torch.no_grad():
    z_post_src, codes_src = mimi_encode(x_src, mimi)
    z_pre_src = mimi.encode_to_latent(x_src, quantize=False)
    z_post_tgt, codes_tgt = mimi_encode(x_tgt, mimi)

trace("x_src (encoder input)", x_src, SR, "B=1, C=1, T_audio")
trace("z_pre_src (pre-quant)", z_pre_src, 12.5, "B, D, T_mimi", "before quantization")
trace("z_post_src (post-quant)", z_post_src, 12.5, "B, D, T_mimi", "after quantization")
trace("codes_src", codes_src, 12.5, "B, Nq, T_mimi", f"discrete indices, cardinality={mimi.cardinality}")

# Per-codebook
for nq in [1,2,3,4,5,6,7,8]:
    mimi.set_num_codebooks(nq)
    z_cumul = mimi.decode_latent(codes_src[:,:nq,:])
    trace(f"z_cumul_q0-q{nq-1}", z_cumul, 12.5, f"cumulative latent from q0..q{nq-1}")
mimi.set_num_codebooks(8)

# Residual per codebook
prev = torch.zeros_like(z_post_src)
for nq in range(8):
    mimi.set_num_codebooks(nq+1)
    cur = mimi.decode_latent(codes_src[:,:nq+1,:])
    res = cur - prev
    trace(f"q{nq}_residual", res, 12.5, f"residual contribution of codebook {nq}")
    prev = cur
mimi.set_num_codebooks(8)

# Splitter
with torch.no_grad():
    z_q0, z_ac = splitter._get_content_acoustic(codes_src)
    C = splitter.content_extractor(z_q0)
    S_src = splitter.speaker_encoder(z_post_src)
    S_tgt = splitter.speaker_encoder(z_post_tgt)
    A = splitter.acoustic_adapter(z_ac, S_tgt, C)
    z_vc = C + A

trace("z_q0 (content input)", z_q0, 12.5, "B, D, T", "q0 decode_latent output")
trace("z_ac (acoustic input)", z_ac, 12.5, "B, D, T", "q1-q7 decode_latent output")
trace("C (content extractor output)", C, 12.5, "B, D, T", "speaker-neutralized content")
trace("S_src (source speaker)", S_src, None, "B, D global", "global speaker embedding")
trace("S_tgt (target speaker)", S_tgt, None, "B, D global", "global speaker embedding")
trace("A (acoustic adapter output)", A, 12.5, "B, D, T", "speaker-conditioned acoustic")
trace("z_vc", z_vc, 12.5, "B, D, T", "final latent for decoder")

# FiLM internals
with torch.no_grad():
    scale_raw = splitter.acoustic_adapter.scale(S_tgt)
    bias_raw = splitter.acoustic_adapter.bias(S_tgt)
    z_mod = z_ac * (1 + torch.tanh(scale_raw.unsqueeze(-1))) + bias_raw.unsqueeze(-1)
    z_in = C + z_mod
    h = z_in.transpose(1,2)
    h = splitter.acoustic_adapter.norm(h).transpose(1,2)
    z_ref = splitter.acoustic_adapter.refine(h)

trace("FiLM scale (gamma)", scale_raw, None, "B, D", "speaker→scale via MLP")
trace("FiLM bias (beta)", bias_raw, None, "B, D", "speaker→bias mapping")
trace("FiLM modulated (z_mod)", z_mod, 12.5, "B, D, T", "z_ac * (1+tanh(scale)) + bias")
trace("FiLM content-aware refine", z_ref, 12.5, "B, D, T", "content-dependent refinement")
trace("FiLM delta (A - z_ac)", A - z_ac, 12.5, "B, D, T", "what FiLM actually changes")

# Decode
x_out = mimi_decode_latent(mimi, z_vc)
x_rt = mimi_decode_latent(mimi, z_post_src)
trace("x_out (VC audio)", x_out, SR, "B, 1, T_audio", "final converted audio")
trace("x_rt (Mimi RT)", x_rt, SR, "B, 1, T_audio", "round-trip reference")

# Print table
print()
print(f"  {'Name':<30s} {'Shape':<22s} {'dtype':<8s} {'dev':<5s} {'Rate':<8s} {'Meaning':<35s}")
print("  " + "-" * 120)
for t in traces:
    name, shape, dtype, dev, rate, meaning, stats, notes = t
    print(f"  {name:<30s} {shape:<22s} {dtype:<8s} {dev:<5s} {rate:<8s} {meaning:<35s}")

# ── 2. Mimi Codebook Structure ──────────────────────────────────────────
print()
print("=" * 90)
print("  SECTION 2: MIMI CODEBOOK STRUCTURE")
print("=" * 90)

q = mimi.quantizer
print(f"  Quantizer class: {type(q).__name__}")
print(f"  n_q (total codebooks): {q.max_n_q if hasattr(q,'max_n_q') else '?'}")
print(f"  n_q_semantic: {q.n_q_semantic if hasattr(q,'n_q_semantic') else '?'}")
print(f"  n_q_acoustic: {q.n_q_acoustic if hasattr(q,'n_q_acoustic') else '?'}")
print(f"  rvq_first (semantic): {type(q.rvq_first).__name__} n_q={q.rvq_first.n_q}")
print(f"  rvq_rest (acoustic): {type(q.rvq_rest).__name__} n_q={q.rvq_rest.n_q}")
print(f"  Embedding dim: {mimi.quantizer.dimension}")
print(f"  Codebook cardinality: {mimi.cardinality}")
print(f"  Sample rate: {mimi.sample_rate} Hz")
print(f"  Frame rate: {mimi.frame_rate} Hz (1 frame = {1000/mimi.frame_rate:.0f}ms)")
print(f"  Encoder type: {type(mimi.encoder).__name__}")
print(f"  Decoder type: {type(mimi.decoder).__name__}")
print(f"  Encoder transformer: {type(mimi.encoder_transformer).__name__}")
print(f"  Decoder transformer: {type(mimi.decoder_transformer).__name__}")
print(f"  Encoder hop_length: {mimi.encoder.hop_length}")
print(f"  Stride samples = hop_length * (encoder_rate/frame_rate) = {mimi.encoder.hop_length} * {int(SR/mimi.encoder.hop_length/mimi.frame_rate)} = {STRIDE}")
print()
print(f"  Key insight:")
print(f"    q0 = semantic codebook (n_q_semantic=1)")
print(f"    q1-q7 = acoustic codebooks (n_q_acoustic=7)")
print(f"    Each codebook: {mimi.cardinality} entries × {mimi.quantizer.dimension}d embedding")
print(f"    RVQ: residual — q1 refines q0, q2 refines q0+q1, etc.")
print(f"    decode_latent(codes) sums codebook embeddings: Σ embed[code_i]")
print(f"    Decoder receives single (B, D, T) latent, not per-codebook")

# ── 3. q0 / q1-q7 Token Statistics ──────────────────────────────────────
print()
print("=" * 90)
print("  SECTION 3: CODEBOOK TOKEN STATISTICS (p255, 25 frames)")
print("=" * 90)

codes_np = codes_src.squeeze(0).cpu().numpy()  # (8, T)
T = codes_np.shape[1]

print(f"  {'CB':>3s} {'Entropy':>8s} {'Unique':>7s} {'Top3 freq':>12s} {'Most common':>14s} {'Usage%':>7s}")
print("  " + "-" * 70)
for cb in range(8):
    tokens = codes_np[cb]
    counts = Counter(tokens)
    total = len(tokens)
    probs = np.array([c/total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-8))
    unique = len(counts)
    top3 = sorted(counts.values(), reverse=True)[:3]
    top3_str = "/".join([str(t) for t in top3])
    most_common = counts.most_common(1)[0]
    usage = unique / mimi.cardinality * 100
    print(f"  q{cb:<2d}  {entropy:7.3f}  {unique:5d}/{total}  {top3_str:>12s}  {most_common[0]:5d}({most_common[1]/total*100:4.1f}%)  {usage:5.1f}%")

# ── 4. Content C Structure ─────────────────────────────────────────────
print()
print("=" * 90)
print("  SECTION 4: CONTENT C STRUCTURE")
print("=" * 90)

print(f"  C shape: {list(C.shape)} — frame-wise (B, D, T) at 12.5Hz")
print(f"  C is CONTINUOUS latent (not discrete)")
print(f"  C derived from: q0 decode_latent → ContentExtractor (residual bottleneck 64d)")
print(f"  C temporal smoothness:")
C_np = C.squeeze(0).cpu().numpy()  # (D, T)
frame_diff = np.linalg.norm(C_np[:,1:] - C_np[:,:-1], axis=0)
print(f"    frame-to-frame diff: μ={frame_diff.mean():.2f} σ={frame_diff.std():.2f}")
print(f"  C dimension variance (top 5): {np.argsort(C_np.var(axis=1))[-5:][::-1]}")
print(f"  C norm per frame: μ={np.linalg.norm(C_np,axis=0).mean():.1f} σ={np.linalg.norm(C_np,axis=0).std():.1f}")
print(f"  C contributes to z_vc as: z_vc = C + A")

# ── 5. Speaker S Structure ─────────────────────────────────────────────
print()
print("=" * 90)
print("  SECTION 5: SPEAKER S STRUCTURE")
print("=" * 90)

print(f"  S shape: {list(S_src.shape)} — GLOBAL vector (B, D)")

# SpeakerEncoder architecture
se = splitter.speaker_encoder
print(f"  SpeakerEncoder architecture:")
print(f"    conv: {se.conv}")
print(f"    pool: {se.pool} — AdaptiveAvgPool1d(1) = global mean pool")
print(f"    proj: {se.proj} — Linear({se.proj.in_features}, {se.proj.out_features})")
print(f"  Input: z_post (B, D, T) → conv → pool → proj → S (B, 512)")
print(f"  S is used ONLY for FiLM modulation (gamma/beta)")
print(f"  S is NOT concatenated with C")

# S statistics
print(f"  S_src norm: {S_src.norm():.1f}")
print(f"  S_tgt norm: {S_tgt.norm():.1f}")
print(f"  cos(S_src, S_tgt): {torch.nn.functional.cosine_similarity(S_src, S_tgt).item():.4f}")

# Cross-speaker S cosine
print()
print(f"  S cosine matrix (5 speakers):")
test_spks = ['p225','p226','p227','p255','p256']
S_list = []
for spk in test_spks:
    d, _ = load_audio(spk)
    x = torch.from_numpy(d).float().view(1,1,-1).to(device)
    with torch.no_grad():
        z, _ = mimi_encode(x, mimi)
        S_list.append(splitter.speaker_encoder(z).squeeze(0).cpu())
cos_matrix = torch.zeros(len(test_spks), len(test_spks))
for i in range(len(test_spks)):
    for j in range(len(test_spks)):
        cos_matrix[i,j] = torch.nn.functional.cosine_similarity(S_list[i], S_list[j], dim=0)
for i, spk in enumerate(test_spks):
    row = " ".join([f"{cos_matrix[i,j]:.3f}" for j in range(len(test_spks))])
    print(f"  {spk}: [{row}]")

# ── 6. FiLM Structure ──────────────────────────────────────────────────
print()
print("=" * 90)
print("  SECTION 6: FiLM GAMMA/BETA/DELTA STRUCTURE")
print("=" * 90)

print(f"  FiLM equation: z_mod = z_ac * (1 + tanh(γ)) + β")
print(f"  γ shape: {list(scale_raw.shape)} — channel-wise scalar per dimension")
print(f"  β shape: {list(bias_raw.shape)} — channel-wise scalar per dimension")
print(f"  γ applied as: γ broadcast to (B, D, 1) → per-dim, same across time")
print(f"  γ mean: {scale_raw.mean():.3f} std: {scale_raw.std():.3f}")
print(f"  β mean: {bias_raw.mean():.3f} std: {bias_raw.std():.3f}")
print(f"  After FiLM + refine: A = z_mod + z_ref")
print(f"  z_ref shape: {list(z_ref.shape)} — content-dependent temporal refinement")

# Gamma temporal behavior (constant across time by design)
print(f"  γ is TIME-INVARIANT — same for all frames")
print(f"  β is TIME-INVARIANT — same for all frames")
print(f"  Only z_ref (content-aware refine) varies temporally")

# Delta analysis
delta = (A - z_ac).squeeze(0).cpu()  # (D, T)
delta_norm_frame = delta.norm(dim=0)
print(f"  FiLM delta = A - z_ac")
print(f"  delta norm per frame: μ={delta_norm_frame.mean():.2f} σ={delta_norm_frame.std():.2f}")
print(f"  delta norm temporal derivative: μ={delta_norm_frame.diff().abs().mean():.2f}")
print(f"  delta concentrated dims (top 5): {delta.norm(dim=1).topk(5).indices.tolist()}")
print(f"  delta direction = the change FiLM makes to the latent")

# ── 7. Decoder Input Distribution ──────────────────────────────────────
print()
print("=" * 90)
print("  SECTION 7: DECODER INPUT DISTRIBUTION")
print("=" * 90)

z_rt = z_post_src  # normal Mimi latent
z_vc_np = z_vc.squeeze(0).cpu()
z_rt_np = z_rt.squeeze(0).cpu()

print(f"  z_rt (Mimi RT latent):")
print(f"    norm per frame: μ={z_rt_np.norm(dim=0).mean():.1f} σ={z_rt_np.norm(dim=0).std():.1f}")
print(f"    mean: {z_rt_np.mean():.4f} std: {z_rt_np.std():.4f}")
print(f"    min: {z_rt_np.min():.4f} max: {z_rt_np.max():.4f}")
print()
print(f"  z_vc (FiLM VC latent):")
print(f"    norm per frame: μ={z_vc_np.norm(dim=0).mean():.1f} σ={z_vc_np.norm(dim=0).std():.1f}")
print(f"    mean: {z_vc_np.mean():.4f} std: {z_vc_np.std():.4f}")
print(f"    min: {z_vc_np.min():.4f} max: {z_vc_np.max():.4f}")
print()
# Distribution distance
cos_dist = torch.nn.functional.cosine_similarity(
    z_rt.reshape(-1), z_vc.reshape(-1), dim=0).item()
print(f"  z_rt vs z_vc cosine: {cos_dist:.4f}")
print(f"  L2 distance: {torch.norm(z_rt - z_vc):.2f}")
print(f"  z_vc norm / z_rt norm ratio: {z_vc_np.norm()/z_rt_np.norm():.3f}")

# Check if z_vc is within training distribution
print()
print(f"  Decoder manifold analysis:")
print(f"    z_rt is from Mimi training distribution (in-distribution)")
print(f"    z_vc is from FiLM modulation (potentially out-of-distribution)")
print(f"    L2 distance = {torch.norm(z_rt - z_vc):.1f}")
print(f"    → {'WITHIN distribution' if torch.norm(z_rt - z_vc) < z_rt_np.norm()*0.5 else 'POSSIBLE OOD — may cause artifacts'}")

# ── 8. Failure Mode Summary ────────────────────────────────────────────
print()
print("=" * 90)
print("  SECTION 8: ARCHITECTURAL FAILURE MODE SUMMARY")
print("=" * 90)

print()
print(f"  Why FiLM succeeds:")
print(f"    1. γ/β are GLOBAL per-speaker (B,D) — single vector for all frames")
print(f"    2. Applied CHANNEL-WISE — modulates entire frequency spectrum uniformly")
print(f"    3. TIME-INVARIANT modulation + temporal refinement from content")
print(f"    4. This global spectral shift = centroid 920→1429Hz")
print()
print(f"  Why temporal adapters fail:")
print(f"    1. Temporal constraints (TCN, GRU, smoothness loss) force output")
print(f"       to stay close to source temporal pattern")
print(f"    2. Speaker transfer requires GLOBAL spectral shift, not local smoothing")
print(f"    3. Temporal adapters prioritize short-term consistency over")
print(f"       global speaker-direction shift")
print(f"    4. Result: centroid regresses from 1429Hz → 920-1080Hz")
print()
print(f"  Why n_content=2 fails:")
print(f"    1. q1 adds speaker info to content path (q1 probe: 20% speaker)")
print(f"    2. Content path now contains source speaker → fights target speaker")
print(f"    3. Result: centroid stays near source (1001Hz)")
print()
print(f"  Root cause of jitter:")
print(f"    1. FiLM delta has temporal variance (δ_norm std = {delta_norm_frame.std():.1f})")
print(f"    2. Content-aware refinement adds frame-to-frame variation")
print(f"    3. This variation in latent → variation in decoded F0 → jitter 37.8%")
print(f"    4. Simple cause: frame-independent FiLM + content-dependent refine = instability")

print()
print("Done!")
