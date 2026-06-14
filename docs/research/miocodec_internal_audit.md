# MioCodec Internal Audit — Complete

**Date:** 2026-06-09
**Checkpoint:** `Aratako/MioCodec-25Hz-44.1kHz-v2`
**Status:** 구조 감사 완료, causality 불가 판정, teacher-student 분리 경로 확인

## 1. Module Structure

| Module | Input Shape | Output Shape | Rate | Causal? | Cacheable? |
|--------|------------|-------------|------|---------|------------|
| raw_audio | [B, T_audio] | — | 44100 Hz | — | — |
| WavLM SSL frontend | [B, T_audio] | [B, T_ssl, D_ssl] | ~50 Hz | **NO** (full-seq self-attn) | No |
| content_embedding | via SSL + local encoder | [T_content, 768] | 25 Hz | **NO** (window=125, ±62) | No |
| content_token_indices | via FSQ quantizer | [T_content] int64 | 25 Hz | **NO** | No |
| global_embedding | via ConvNeXt + AttnPool | [128] | global | **NO** (symmetric conv+pool) | **YES ★** |
| self_recon_mel | content + global → decoder | [T_mel] | ~440 Hz | **NO** (window=65, ±32) | No |
| vc_mel | source content + target global | [T_mel] | ~440 Hz | **NO** | No |
| ISTFT head | mel frames | [B, T_audio] | 44100 Hz | **NO** (overlap-add) | No |

### Config

```
sample_rate = 44100
hop_length = 98
n_fft = 392
downsample_factor = 2
n_mels = 100
use_wave_decoder = True
wave_upsampler_factors = (3, 3)
normalize_ssl_features = True (full-utterance mean/std)

local_encoder:  6 layers, dim 768, window_size 125, causal=False
wave_prenet:    6 layers, dim 768, window_size 65,  causal=False
wave_decoder:   8 layers, dim 512, window_size 65,  causal=False, AdaLN global conditioning
global_embedding_dim = 128
```

### Architectural Facts

1. **Content**: continuous embedding (768d) + discrete tokens (12579 classes via FSQ)
2. **Content rate**: 25Hz (1 frame = 40ms) — 2x faster than Mimi (12.5Hz)
3. **Global embedding**: 128d, pooled via AttentiveStatsPool from WavLM SSL features
4. **Decoder**: content tokens + global embedding → mel spectrogram → ISTFT → waveform
5. **Mel spectrogram → vocoder** (not in codec model, separate PupuGAN vocoder)
6. **WavLM frontend**: full-sequence self-attention → **NON-CAUSAL**
7. **Local transformer**: causal=False, window=125 → ~1.2s future context
8. **Wave decoder**: causal=False, window=65 → ~1.3s future context
9. **SSL normalization**: full-utterance mean/std → uses future frames
10. **Symmetric padding**: Conv1d with pad=(k-1)//2 on both sides → leaks future

### Non-Causality Verdict

```
MioCodec is ARCHITECTURALLY non-causal in both encoder and decoder.

Blockers:
1. WavLM SSL frontend — full-seq self-attention
2. Mio local/wave Transformers — causal=False
3. Symmetric local attention — ±62 frames (~1.2s) future context
4. Global embedding — AttentiveStatsPool over whole utterance
5. SSL normalization — full-utterance mean/std
6. Symmetric Conv1d/ConvTranspose/GroupNorm
7. ISTFT overlap-add synthesis

Cannot be patched to causal without full retraining.
→ Teacher-student distillation only viable path.
```

## 2. Representation Probe

10 speakers × 3 utterances = 30 samples.

| Representation | Speaker Acc | F0 Correlation | Verdict |
|---------------|-------------|---------------|---------|
| ContentEmbed (mean-pooled) | 16.7% (chance 10%) | 0.149 | Slight leak, mostly clean |
| GlobalEmbed | **96.7%** (chance 10%) | 0.035 | **STRONG speaker identity ★★★** |

**Comparison with Mimi:**

| Metric | Mimi | MioCodec |
|--------|------|----------|
| Content speaker acc | 8.3% (chance 5%) | 16.7% (chance 10%) |
| Speaker embedding acc | 81.7% | **96.7%** |
| Speaker F0 corr | — | 0.035 (very low) |

MioCodec's global embedding is dramatically better at speaker identification
(96.7% vs 81.7%) and has very low F0 correlation — it's capturing true speaker
identity, not pitch shortcut.

## 3. Speaker Embedding Stability

origin.mp3 variants tested against reference global embedding:

| Variant | cos(S_ref, S_variant) | Stable? |
|---------|----------------------|---------|
| 1s segment | 0.993 | STABLE |
| 3s segment | 1.000 | STABLE |
| loudness normalized | **0.999** | **STABLE ★** |
| lowpass 4kHz | 0.988 | STABLE |
| lowpass 8kHz | 1.000 | STABLE |

**Comparison with Mimi:**

| Variant | Mimi cos | MioCodec cos |
|---------|----------|-------------|
| loudnorm | **0.771** ✗ | **0.999** ★★★ |
| lowpass_4k | 0.816 | 0.988 |
| lowpass_8k | 0.915 | 1.000 |

**Verdict:** MioCodec's global embedding is **domain-invariant** — it does NOT
use loudness, EQ, or high-frequency shortcuts. This is a massive improvement
over Mimi's S embedding which collapsed under loudnorm (cos=0.77).

MioCodec's global embedding = **true speaker identity**, not spectral shortcut.

## 4. Upper Bound VC Quality

Using source content tokens + target global embedding:

```
Mel statistics:
  Self-recon: mean=0.000, std=0.055
  VC output:  mean=0.000, std=0.144 (2.6x wider variance)
  Mel delta L2: 48.2
  Mel cosine:   -0.0075 (near-orthogonal — completely different mel!)

The global embedding swap causes a MASSIVE mel change, not just speaker
characteristics. This suggests the decoder uses global embedding for more
than timbre — it may control the entire spectral envelope.

VC output was NOT vocoded to waveform — mel-only analysis.
Waveform quality TBD with vocoder path.
```

## 5. Causalization Strategy

### What CAN be cached offline

| Component | Cacheable? | Why |
|-----------|-----------|-----|
| Target global embedding | **YES ★** | 128d vector; 1-3s was experimentally stable, while production VoiceBank policy requires one continuous >=5s reference |
| Target speaker identity | **YES** | Can pre-compute from voicebank |

### What NEEDS causal student

| Component | Causal Student Needed? | Difficulty |
|-----------|----------------------|------------|
| Source content encoder | **YES** | Hard — replace WavLM + local transformer |
| Source content quantizer (FSQ) | **YES** | Medium — causal FSQ or RVQ |
| Wave decoder (mel → waveform) | **YES** | Hard — replace non-causal transformer + ISTFT |

### Proposed Streaming Architecture

```
OFFLINE (one-time per target speaker):
  target reference audio
    → MioCodec global encoder
    → S_target [128] cached

STREAMING (per source chunk):
  source audio chunk
    → Causal content encoder (student)
    → content tokens at 25Hz
    → MioCodec-style decoder (causal student, S_target conditioned)
    → causal vocoder (student)
    → streaming waveform
```

### Loss Candidates for Student

| Loss | Purpose |
|------|---------|
| Content distillation | Match MioCodec content tokens/embeddings |
| Mel reconstruction | Match MioCodec self-recon mel |
| Speaker similarity | Match target speaker characteristics |
| F0 consistency | Prevent octave errors |
| Jitter penalty | Temporal smoothness |
| Causal constraint | Streaming-compatible architecture |

## 6. Mimi vs MioCodec Comparison

| Aspect | Mimi | MioCodec |
|--------|------|----------|
| Content rate | 12.5Hz | 25Hz |
| Content type | Continuous 512d from RVQ sum | Continuous 768d + discrete FSQ tokens |
| Content speaker leak | 8.3% (very low) | 16.7% (low) |
| Speaker embedding dim | 512d (global) | 128d (global) |
| Speaker identity acc | 81.7% | **96.7% ★** |
| Speaker stability | **UNSTABLE** (loudnorm cos=0.77) | **STABLE** (loudnorm cos=0.999) |
| Decoder input | Single latent (B,D,T) | Content tokens + global embedding |
| Decoder output | Waveform (24kHz) | Mel → vocoder → waveform (44.1kHz) |
| Causality | Streaming-capable (80ms frame) | **FULLY NON-CAUSAL** |
| Streaming latency | ~110ms (CPU) | N/A (offline only) |
| VC quality | **UNUSABLE** (jitter 37.8%) | TBD (mel-only tested) |
| Speaker transfer mechanism | Source acoustic zeroing + target β injection | Target global embedding swap in decoder |
| Vocoder required | No (direct waveform decode) | **YES** (separate PupuGAN vocoder) |

## 7. Key Decisions

### What MioCodec does BETTER than Mimi

1. **Speaker embedding**: 96.7% acc, perfectly stable under augmentation
2. **Content/global separation**: Clean separation with minimal F0 leakage
3. **Output quality potential**: 44.1kHz with GAN vocoder

### What Mimi does BETTER than MioCodec

1. **Streaming**: Already streaming-capable at 110ms latency
2. **No external vocoder**: Direct waveform decode
3. **Proven codec architecture**: Simpler to modify

### Critical Unknowns

1. **MioCodec VC waveform quality**: Mel-only tested, vocoder path not verified
2. **Causal content encoder feasibility**: Can we train a causal student at 25Hz?
3. **Causal decoder feasibility**: Can the wave decoder be made causal?
4. **Vocoder streaming**: Can PupuGAN run in streaming mode?
5. **End-to-end latency**: If all causal students work, what's the total latency?

## 8. Recommended Next Steps

### Immediate (this branch)

1. Install MioCodec vocoder and test actual waveform VC quality
2. Probe content token entropy, per-speaker distribution, temporal patterns
3. Test MioCodec VC on p255→origin.mp3 with full vocoder path

### Medium-term

4. Design causal content encoder student architecture
5. Train content student to match MioCodec content tokens
6. Design causal decoder student architecture
7. Evaluate student vs teacher quality

### Long-term

8. Full causal student pipeline
9. Streaming latency measurement
10. Compare with Mimi Splitter VC baseline

## 9. Source Files

| File | Purpose |
|------|---------|
| `miocodec/model.py` | Main MioCodecModel class |
| `miocodec/module/ssl_extractor.py` | WavLM SSL frontend |
| `miocodec/module/transformer.py` | Mio local/wave transformer |
| `miocodec/module/global_encoder.py` | ConvNeXt + AttentiveStatsPool |
| `miocodec/module/fsq.py` | Finite Scalar Quantization |
| `miocodec/module/istft_head.py` | ISTFT waveform synthesis |
| `miocodec/module/convnext.py` | ConvNeXt blocks |
| `miocodec/pipeline.py` | MioCodec inference wrapper + vocoder |
| `miocodec/pupu/` | PupuGAN vocoder |
