# MioCodec Content Student — Research Log

> **Implementation note (2026-06-14):** The historical v3/v3-4k results below
> were produced with symmetrically padded Conv1d layers and utterance-level
> validation splits. The current `astrape/` implementation replaces them with
> left-padded causal convolutions, stateful streaming inference,
> speaker-disjoint validation, masked losses, and versioned checkpoints.
> Historical raw student checkpoints are now explicitly treated as legacy
> weights and require causal fine-tuning before new quality claims.
>
> The historical MPS backward failure is not reproducible with the current
> local PyTorch 2.12 runtime. Standard causal Conv1d is now the default;
> `train_xhigh.py --safe-convs` retains the unfold-based fallback for older
> runtimes.

> **Project**: btrv5 — Real-time Zero-Shot Voice Conversion via Teacher-Student Distillation  
> **Last updated**: 2026-06-14  
> **Author**: asill / stremtec

---

## 1. Overview

**Goal**: Build a real-time (<200ms) zero-shot voice conversion pipeline by distilling MioCodec's offline quality into a fully causal, streaming-capable system.

**Core idea**: MioCodec teacher (25Hz, 44.1kHz) provides upper-bound quality. Student learns to replicate content encoding and speaker conditioning in a causal manner.

```
Pipeline:
  Source Audio → Mel Spectrogram → [Content Student] → Content (768d @ 25Hz)
  Target Audio → MioCodec Encode → Global Embedding (128d)
  Content + Global → [Causal Mel Decoder] → Mel → [Wave Decoder] → Waveform
```

---

## 2. Model Architecture

### 2.1 Content Student (causal Transformer)

| Component | Detail |
|-----------|--------|
| Input | 80-bin log-mel spectrogram @ 50Hz (16kHz audio) |
| Stem | Conv1d(80→H) → GELU → Conv1d(H→H) → GELU, kernel=5, pad=2 |
| Positional Encoding | Sinusoidal, max_len=200 |
| Transformer Blocks | N layers, causal self-attention (lower-tri mask), FFN 4× expansion |
| Downsampling | Conv1d stride=2, 50Hz→25Hz |
| Output Head | Conv1d(H→768) → Content Embedding @ 25Hz |

### 2.2 Causal Mel Decoder

| Component | Detail |
|-----------|--------|
| Architecture | AdaLN-Zero Transformer decoder |
| Conditioning | Global embedding (128d) via FiLM/AdaLN |
| Causality | Causal self-attention (no future leakage) |
| Status | ✅ Trained, verified (target > source > other > zero) |

### 2.3 Wave Decoder (Teacher, non-causal)

| Component | Detail |
|-----------|--------|
| Current | MioCodec teacher wave decoder (ISTFT + non-causal Transformer) |
| Issue | ❌ Non-causal — blocks streaming |
| Plan | Causalize teacher decoder (causal=True flag + streaming ISTFT buffer) |
| Backup | Streaming HiFi-GAN if teacher causalization fails |

---

## 3. Training History

### 3.1 Content Student Evolution

| Version | Dim | Layers | Heads | Params | Data | val_cos | Notes |
|---------|-----|--------|-------|--------|------|---------|-------|
| v1 | 256 | TCN (4) | — | ~3M | 127 | 0.63 | Garbled English, mel+TCN |
| v2 | 256 | 4 | 4 | ~5M | 127 | 0.89 | Transformer, intelligible |
| v2-FINAL | 384 | 6 | 8 | 12.4M | 127 | 0.900 | Data ceiling reached |
| v3 | 384 | 6 | 8 | 12.4M | 1,090 | 0.923 | 109 speakers |
| **v3-4k** | 384 | 6 | 8 | 12.4M | 4,360 | 0.922 | 4× data, no improvement |

### 3.2 Key Findings

- **Data scaling alone doesn't help**: v3 (1090 samples) → v3-4k (4360 samples) plateaued at cos≈0.922
- **Architecture bottleneck**: 384dim with 6 layers likely saturated — need larger capacity
- **Teacher content self-cos = 1.0** — plenty of headroom remaining
- **MPS latency benchmark**: 1024dim=17.9ms → well within 40ms/frame real-time budget

### 3.3 Quality Tiers (defined, not trained)

| Tier | Dim | Layers | Heads | Params | MPS Latency |
|------|-----|--------|-------|--------|-------------|
| low | 512 | 6 | 8 | ~28M | ~5.6ms |
| medium | 768 | 8 | 12 | ~62M | ~12.1ms |
| **xhigh** | 1024 | 8 | 16 | ~110M | ~17.9ms |

Defined in: `tiers.py`

---

## 4. Dataset

| Item | Detail |
|------|--------|
| Source | VCTK (109 speakers) |
| Samples | 4,360 pre-extracted teacher content+global pairs |
| Location | `data/mio_4k/` (5.9GB, gitignored) |
| Mel cache | `data/mio_4k_mel/` (80-bin log-mel spectrograms) |
| Content labels | Teacher content embeddings (768d @ 25Hz) |
| Train/Val split | 85/15 (3,706 / 654 samples) |

Extraction script: `extract_4k.py`

---

## 5. Current Checkpoints

| File | Size | Description | Status |
|------|------|-------------|--------|
| `causal_student_v3_4k.pt` | 47MB | Content Student v3-4k (384dim) | ✅ Best so far |
| `content_student_v3_4k_causal.best.pt` | 47MB | Strict-causal conversion fine-tune, 1 epoch, speaker-disjoint val cosine 0.8666 | ✅ Initial causal checkpoint |
| `causal_mel_decoder.pt` | 56MB | Causal Mel Decoder (AdaLN-Zero) | ✅ Trained |
| `student_proj_out.pt` | 20KB | Legacy — FSQ calibration | ⚠️ Deprecated |

---

## 6. Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `train_v3_4k.py` | 101 | Full training (384dim, 4360 samples) |
| `train_v3_4k_mini.py` | 87 | Lightweight training (preloaded data) |
| `train_xhigh.py` | 132 | **xhigh tier training (1024dim)** — WIP |
| `bench_dim.py` | 52 | MPS latency benchmark across dimensions |
| `demo_v2.py` | 132 | Offline VC demo (teacher decoder) |
| `extract_4k.py` | 42 | Dataset extraction script |
| `tiers.py` | 23 | Quality tier config (low/medium/xhigh) |

---

## 7. MPS Compatibility

### 7.1 Issue

PyTorch 2.x MPS backend has a `convolution_backward` view/stride bug:
```
RuntimeError: view size is not compatible with input tensor's size and stride 
(at least one dimension spans across two contiguous subspaces).
Use .reshape(...) instead.
```

### 7.2 Attempted Fixes

| Approach | Result |
|----------|--------|
| `.contiguous()` everywhere | ❌ Forward OK, backward fails |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | ❌ View ops don't trigger fallback |
| `torch.compile(aot_eager)` | ❌ Same view error |
| `SafeConv1d` (unfold+matmul) | ✅ Backward OK, but im2col CPU fallback → slow |
| Custom `Conv1dMPS` autograd Function | 🧪 Testing |

### 7.3 Current Plan

- **Training**: CPU (reliable, ~110% CPU utilization)
- **Inference**: MPS (forward pass works at 17.9ms for 1024dim)
- **Future**: Wait for PyTorch MPS fix, or use custom autograd Function

---

## 8. Pipeline Status

| Stage | Causal? | Status |
|-------|---------|--------|
| 1. Mel Extraction | ✅ | Streaming (hop 320 samples = 20ms @ 16kHz) |
| 2. Content Student | ✅ | Trained (v3-4k), quality ceiling at cos=0.922 |
| 3. Global Embedding | N/A | From teacher encoder (not learned by student) |
| 4. Causal Mel Decoder | ✅ | Trained, conditioning verified |
| 5. Wave Decoder | ❌ | Teacher ISTFT+Transformer non-causal |
| 6. Full Pipeline Latency | — | Not measured yet |

### 8.1 Remaining Work

1. **Content quality**: Train xhigh (1024dim) to break cos=0.922 ceiling
2. **Vocoder causalization**: 
   - Option A: Causalize teacher wave decoder (causal=True + streaming ISTFT)
   - Option B: Streaming HiFi-GAN
3. **Streaming integration**: Frame-buffered pipeline
4. **Full pipeline latency measurement**: Target <200ms

---

## 9. Git History (key commits)

```
78d1ef1 Major cleaning
335dbf9 cleanup: remove failed experiments, keep Mio v3-4k pipeline
1b3f1a4 research: vocoder causalization analysis
eff2746 research: V3 — 1090 samples, val_cos=0.923
deaae55 research: V2-FINAL — 384dim, val_cos=0.900
d88de85 research: Transformer v2 demo — 7 speakers
```

---

## 10. Dead Ends / Lessons Learned

| Attempt | Why Failed |
|---------|------------|
| MimiSplitter-VC | Mimi lacks speaker-content factorization (jitter 37.8%) |
| FiLM adapter | Only works by zeroing source acoustic → OOD artifacts |
| LongCat codec | Token swap path unusable |
| TCN content student (v1) | cos=0.63, garbled English |
| 4× data scaling | No improvement over 1k samples (cos plateau) |
| MPS training (native Conv1d) | Backward view/stride bug |

---

## 11. References

- MioCodec: `Aratako/MioCodec-25Hz-44.1kHz-v2` (HuggingFace)
- FSQ levels: [8,8,8,5,5] = 12,800 codes
- Content rate: 25Hz (40ms/frame)
- Global embedding: 128dim
- Teacher content dimension: 768dim
- SR: 44.1kHz (teacher), 16kHz (mel input)
