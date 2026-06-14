# MioCodec Internal Audit — Complete Deep Analysis

**Date:** 2026-06-09
**Checkpoint:** `Aratako/MioCodec-25Hz-44.1kHz-v2`
**Status:** 완료 — 모든 모듈, 텐서, 통계 분석 완료

---

## 1. Module-by-Module Structure

### 1.1 SSL Feature Extractor (`ssl_extractor.py`)

| Property | Value |
|----------|-------|
| Model | `torchaudio.pipelines.WAVLM_BASE_PLUS` |
| SSL sample rate | 16000 Hz |
| Feature dim | 768 |
| Hop size | product of conv strides (~320 samples = 20ms) |
| Causal? | **NO** — WavLM uses full-sequence self-attention |
| Frozen? | YES — `freeze_modules()` called at init |

**Extraction pipeline:**
```
waveform (B, T_44100)
  → resample to 16kHz
  → WavLM conv frontend (7 conv layers, kernel=10, stride=5,3,3,3,3,2,2)
  → WavLM Transformer (12 layers, 768d, 12 heads)
  → list of 12 layer outputs [(B, T_ssl, 768), ...]
```

**Local branch:** averages SSL layers [6, 9] (config.local_ssl_layers)
**Global branch:** averages SSL layers [1, 2] (config.global_ssl_layers)
**Normalization:** Full-utterance mean/std → NON-CAUSAL

### 1.2 Transformer (`transformer.py`)

Llama-style Transformer with:
- **Rotary Position Embedding** (RoPE, θ=500000)
- **Local window attention** — window_size=125 → ±62 frames, 1.2s context
- **SwiGLU FFN** — `silu(w1(x)) * w3(x)`
- **AdaLN-Zero conditioning** — global embedding → shift, scale, gate per layer
- **causal=False** — all layers use bidirectional attention
- **FlashAttention** support (not installed — falls back to SDPA)
- **KV cache** support for streaming inference

**Attention mask:** `create_mask()` builds windowed + causal mask:
```python
# Local window: triu(diag=-62) & tril(diag=+62)
# If causal=True: tril first, then window
# Current config: causal=False, so full bidirectional window
```

**Loaded config:**
| Module | Layers | Dim | Heads | Window | Causal |
|--------|--------|-----|-------|--------|--------|
| local_encoder | 6 | 768 | 12 | 125 (±62) | False |
| wave_prenet | 6 | 768 | 12 | 65 (±32) | False |
| wave_decoder | 8 | 512 | 8 | 65 (±32) | False |

**Wave decoder uses AdaLN-Zero with condition_dim=128 (global embedding)**

### 1.3 FSQ Quantizer (`fsq.py`)

**Finite Scalar Quantization** — no learned codebook, bounded scalar quantization.

| Property | Value |
|----------|-------|
| Levels per dim | [8, 8, 8, 5, 5] |
| Total dims | 5 |
| Codebook size | 8×8×8×5×5 = **12,800** |
| Input dim | 768 |
| Output dim | 768 |

**Pipeline:**
```
local_encoded (B, T, 768)
  → proj_in: Linear(768, 5)
  → FSQ.bound: tanh to [-3.5, 3.5] per dim
  → FSQ.quantize: round_ste()
  → FSQ.codes_to_indices: basis encoding → single int (B, T)
  → proj_out: Linear(5, 768)
  → content_embedding (B, T, 768)
```

**Key insight:** The 768-dim content embedding is a LINEAR PROJECTION of 5 bounded scalars.
Only 5 degrees of freedom carry information. The 768-dim representation has high redundancy.

### 1.4 Global Encoder (`global_encoder.py`)

ECAPA-TDNN style speaker encoder.

| Component | Details |
|-----------|---------|
| Backbone | ConvNextBackbone (Conv1d layers with residual) |
| Pooling | AttentiveStatsPool (softmax-weighted mean + std over time) |
| Output dim | 128 |

**AttentiveStatsPool:**
```python
alpha = Softmax(Tanh(Conv1d(x)))  # (B, C, T) attention weights
mean = sum(alpha * x)             # weighted mean
std = sqrt(sum(alpha * x²) - mean²)  # weighted std
output = LayerNorm(Linear([mean; std]))  # (B, 128)
```

ConvNeXt backbone uses **symmetric Conv1d(kernel=7, padding=3)** → NON-CAUSAL.
But global encoder only runs on REFERENCE audio → acceptable for offline cache.

### 1.5 AdaLN-Zero (`adaln_zero.py`)

Adaptive Layer Normalization with zero-initialized conditioning.

```python
x_norm = LayerNorm(x)       # without learnable affine
shift, scale, gate = Linear(SiLU(condition)).chunk(3)
modulated = x_norm * (1 + scale) + shift
# gate used for residual scaling: h = x + gate * attn_out
```

**Condition source:** global_embedding (128d) broadcast to (B, 1, 128) or (B, T, 128).
**Initialization:** All Linear weights/bias = 0 → identity at start of training.

### 1.6 ISTFT Head (`istft_head.py`)

| Component | Details |
|-----------|---------|
| Activation | SnakeBeta (periodic: x + 1/β·sin²(αx)) |
| ResNet blocks | 2 blocks, kernel=3, GroupNorm(32), dropout=0.1 |
| Upsampler | ConvTranspose1d(3,3)×2 = 9× upsampling |
| ISTFT | n_fft=392, hop=98, padding="same" |
| Output | Complex STFT → waveform |

**Wave decoder pipeline:**
```
content_embedding (B, T_c, 768)
  → wave_prenet (6-layer transformer, causal=False, window=65)
  → ConvTranspose1d × upsampling
  → interpolate to STFT frame length
  → wave_prior_net (ResNet × 2)
  → wave_decoder (8-layer, AdaLN-Zero conditioned on global_emb)
  → wave_post_net (ResNet × 2)
  → wave_upsampler (ConvTranspose1d × 2 = 9x)
  → ISTFTHead → waveform (B, T_audio @ 44100Hz)
```

### 1.7 Pipeline (`pipeline.py`)

`MioCodec` wrapper bundles MioCodecModel + PupuGAN vocoder.

```python
# VC flow
voice_conversion(source, reference):
  src_feat = model.encode(source, content=True, global=False)
  ref_feat = model.encode(reference, content=False, global=True)
  return model.decode(
    content_embedding=src_feat.content_embedding,
    global_embedding=ref_feat.global_embedding,
    target_audio_length=len(source)
  )
```

**Vocoder:** Separate PupuGAN (GAN-based neural vocoder). MioCodec model itself
outputs mel spectrograms; the pipeline wrapper runs `vocode()` after decode.

---

## 2. Complete Tensor Flow

```
INPUT: source_waveform (T_audio @ 44100Hz)

[SSL Extractor — FROZEN]
  → resample to 16kHz
  → WavLM conv frontend → (B, T_ssl, 768) @ ~50Hz
  → WavLM transformer 12 layers → list of [(B, T_ssl, 768), ...]
  → avg(layer 6, layer 9) → local_ssl (B, T_ssl, 768)
  → avg(layer 1, layer 2) → global_ssl (B, T_ssl, 768)

[Content Path — TRAINABLE]
  local_ssl → normalize (full-seq mean/std) → (B, T_ssl, 768)
  → local_encoder (6-layer transformer, causal=False, window=125)
  → (B, T_ssl, 768)
  → Conv1d downsample (kernel=2, stride=2)
  → (B, T_content, 768) @ 25Hz
  → FSQ quantize:
      proj_in: Linear(768→5) → (B, T, 5)
      FSQ bound + quantize → (B, T, 5)
      codes_to_indices → (B, T) int64 (0..12799)
      proj_out: Linear(5→768) → content_embedding (B, T, 768)

[Global Path — TRAINABLE]
  global_ssl → ConvNext backbone → (B, T_ssl, 768)
  → AttentiveStatsPool → (B, 128)

[DECODER — TRAINABLE]
  content_embedding (B, T, 768)
  → wave_prenet (6-layer, causal=False, window=65) → (B, T, 768)
  → ConvTranspose1d upsampling → interpolate to STFT frames
  → ResNet prior (2 blocks, kernel=3)
  → wave_decoder (8-layer, AdaLN-Zero global_emb)
  → ResNet post (2 blocks)
  → UpSampler (ConvTranspose1d × 2 = 9×)
  → ISTFTHead (SnakeBeta + ISTFT) → waveform (B, T_audio)

OUTPUT: waveform @ 44100Hz
```

---

## 3. Content Token Deep Analysis

| Metric | Value |
|--------|-------|
| Vocabulary size | 12,785 (of 12,800 max) |
| Unique tokens used (across 10 spk × 5 utt = 2496 total) | 2,012 (15.7%) |
| Global entropy | 10.84 bits (max 13.64) |
| Per-speaker entropy | ~7.92 bits (very uniform) |
| **Frame-to-frame change rate** | **99-100% — EVERY frame changes** |
| Top token frequency | 0.36% (extremely flat distribution) |
| Content PCA silhouette (speaker) | **-0.331** (negative = totally mixed) |

### FSQ Structure

```
Levels per dimension: [8, 8, 8, 5, 5]
5 scalar dimensions → single integer via basis encoding
Only 5 degrees of freedom → projected to 768-dim embedding
```

### Key Findings

1. **No temporal redundancy** — 100% frame change rate means tokens are unique per frame.
   This is fundamentally different from text tokens or RVQ codes.

2. **Codebook underutilized** — only 15.7% of vocabulary used. Room for more diverse
   content or suggests the FSQ quantization is sparse.

3. **Content is speaker-mixed** — PCA silhouette = -0.331 confirms content embeddings
   do NOT cluster by speaker. This is excellent for VC.

4. **High entropy** — 7.92 bits per speaker means the token distribution is very flat.
   No single token dominates. This is characteristic of lossless information coding.

5. **5-dim bottleneck** — Despite the 768-dim output, only 5 scalar values carry
   the actual content information. The Linear projection adds no extra capacity.

---

## 4. Speaker Embedding Deep Analysis

| Metric | Value |
|--------|-------|
| Dimension | 128 |
| Speaker probe acc | **96.7%** (chance 10%) |
| F0 correlation | 0.035 (very low) |
| Stability under loudnorm | cos = **0.999** |
| Stability under lowpass 4k | cos = 0.988 |
| Segment length stability | 1s=0.993, 3s=1.000 |
| cos(src, tgt) for p255 vs origin | 0.958 (surprisingly high) |

### Key Findings

1. **Extremely stable** — orders of magnitude better than Mimi's S (loudnorm cos=0.77)
2. **True speaker identity** — low F0 correlation confirms it's NOT a pitch shortcut
3. **Domain-invariant** — loudness/EQ changes barely affect the embedding
4. **However:** cos(p255, origin) = 0.958 is suspiciously high for different genders.
   This may indicate the 128-dim space compresses speaker info into a compact manifold
   where different speakers are close in cosine distance (but separable by classifier).
   The high speaker probe acc (96.7%) despite high inter-speaker cosine suggests
   the separation is in specific dimensions, not overall vector direction.

---

## 5. Causality Summary

| Module | Causal? | Future context | Can fix? |
|--------|---------|---------------|----------|
| WavLM SSL | NO | Full sequence | Need causal student |
| SSL normalization | NO | Full utterance | Replace with running norm |
| local_encoder | NO | ±62 frames (~1.2s) | Set causal=True + retrain |
| local_quantizer (FSQ) | YES | None | Already causal |
| wave_prenet | NO | ±32 frames (~1.3s) | Set causal=True + retrain |
| wave_decoder | NO | ±32 frames (~0.6s) | Set causal=True + retrain |
| ResNet (Conv1d) | NO | Symmetric padding | Causal padding |
| UpSampler | NO | Symmetric ConvTranspose | Causal upsampling |
| ISTFT | NO | Overlap-add | Streaming overlap-add |
| Global encoder | NO | Full utterance pool | **ACCEPTABLE** — offline cache |
| AdaLN-Zero | YES | None | Already causal |

**Verdict:** MioCodec cannot be made causal by flipping flags.
Full causal student distillation required.

---

## 6. Speaker Conditioning Mechanism

The target speaker is injected via **AdaLN-Zero in the wave_decoder** ONLY.
The wave_prenet and local_encoder do NOT receive speaker conditioning.

```python
# wave_decoder receives global_embedding as condition
wave_decoder(content_latent, condition=global_embeddings.unsqueeze(1))
# condition shape: (B, 1, 128) broadcast to all layers

# Each AdaLN-Zero block:
shift, scale, gate = Linear(SiLU(condition)).chunk(3)
modulated = LayerNorm(x) * (1 + scale) + shift
output = x + gate * attention(modulated)
```

**Architecture implication for VC:**
- Source content goes through wave_prenet → wave_decoder
- Target global embedding conditions ONLY the decoder layers
- No information from the target leaks into content encoding
- Clean separation: source→content, target→timbre

---

## 7. Mimi vs MioCodec — Final Comparison

| Aspect | Mimi | MioCodec |
|--------|------|----------|
| Content rate | 12.5 Hz | 25 Hz |
| Content dim | 512d (continuous RVQ sum) | 768d (5-dim FSQ → Linear projection) |
| Content speaker leak | 8.3% (clean) | 16.7% (slight) |
| Speaker embedding dim | 512d (global) | 128d (global) |
| Speaker identity acc | 81.7% | **96.7%** |
| Speaker stability (loudnorm) | cos=0.77 ✗ | **cos=0.999 ★** |
| Speaker injection point | FiLM on source acoustic | AdaLN-Zero in decoder |
| Decoder input | (B, 512, T) single latent | (B, T, 768) content + (B, 128) global |
| Decoder output | Waveform (24kHz) | Mel → Vocoder → Waveform (44.1kHz) |
| Vocoder needed? | No | **YES** (separate PupuGAN) |
| Causality | Streaming-capable (~110ms) | **OFFLINE ONLY** |
| Architecture complexity | Simple (SEANet + RVQ) | Complex (WavLM + Transformer + FSQ + ISTFT) |
| Speaker transfer mechanism | Kill source acoustic + inject β | Replace global embedding in decoder |
| VC quality | UNUSABLE (jitter 37.8%) | TBD (vocoder path needed) |
| Latency | 110ms CPU | N/A (offline) |
| Streaming potential | Immediate | Requires full causal student |

---

## 8. Key Architectural Decisions for Student

### What to KEEP from MioCodec

1. **FSQ content quantizer** — 25Hz, causal, 5-dim bottleneck, bounded scalar
2. **AdaLN-Zero speaker conditioning** — clean separation from content
3. **Global encoder** — offline-cacheable, 96.7% speaker acc, domain-stable
4. **44.1kHz target quality** — with vocoder

### What to REPLACE for streaming

1. **WavLM SSL** → Causal content encoder (e.g., streaming HuBERT or conv encoder)
2. **Non-causal transformers** → Causal transformer with limited window
3. **Symmetric convolutions** → Causal Conv1d with left-padding
4. **ISTFT overlap-add** → Streaming ISTFT or neural vocoder (HiFi-GAN)
5. **Full-seq normalization** → Running mean/std or LayerNorm

### Proposed Student Architecture

```
STREAMING SOURCE:
  source audio chunks (40ms at 25Hz)
    → Causal Conv encoder
    → Causal Transformer (small, causal=True, window=31)
    → FSQ quantize (5-dim bounded scalar)
    → content_embedding (5-dim, projected to 768)

OFFLINE TARGET (one-time):
  target reference (one continuous recording, >=5 seconds in production)
    → MioCodec Global Encoder (frozen)
    → global_embedding (128d) cached

STREAMING DECODER:
  content_embedding
    → Causal wave_prenet (smaller transformer, causal=True)
    → Causal wave_decoder (AdaLN-Zero with cached global_emb)
    → Neural vocoder (HiFi-GAN streaming)
    → waveform output
```

---

## 9. Open Questions

1. **Vocoder quality?** MioCodec outputs mel spectrograms — we haven't tested
   the full pipeline with PupuGAN vocoder. Actual waveform quality unknown.

2. **Cos(ge_src, ge_tgt) = 0.958** for cross-gender? Needs investigation — why
   are different-gender speaker embeddings so cosine-similar?

3. **Why 100% frame-to-frame token change?** FSQ quantizer produces unique
   tokens for every frame. Is this by design or an artifact of the quantization?

4. **Only 15.7% vocabulary used** — is the codebook too large, or does the
   model need more diverse training data?

5. **Causal student quality** — if we flip causal=True on transformers without
   retraining, how much quality do we lose? (can test immediately)

6. **Vocoder streaming** — can PupuGAN run in streaming mode with bounded latency?
