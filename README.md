# Astrape-vc — Zero-Shot Voice Conversion

Zero-lookahead neural voice conversion at 44.1kHz.  
**Current best: cos768 = 0.934** (strict-causal, 16kHz WavLM + 7L Transformer).

## Architecture

```
Mic 44.1kHz → resample(44.1k→16k) → WavLM CNN L0-L4 (94M frozen, 10ms) → 200Hz
  → StridingAdapter(200→50Hz, 5ms, 87K) → Causal Depthwise Stem (1.4M, 8 blocks)
  → Downsample(2×) → 25Hz → ProjIn(320→512)
  → Causal Transformer 8L (15.8M, RoPE+SwiGLU, window=256)
  → Q2D2 (3M codes, Rhombic[9×9]³) → content 768d @ 25Hz
  → Decoder v4 (7.08M, causal streaming) → wav 44.1kHz
```

**Learnable: 24.3M | Frozen: 94.4M | Total algorithmic latency: ~17ms (encoder) + ~32ms (decoder) = ~49ms**

## Key Features

| Feature | Flag | Effect |
|---------|------|--------|
| 16kHz WavLM CNN | `--wavlm-frontend` | Replaces Mel. Exact 50Hz, proper kernel alignment. +0.02 cos |
| Time-Shift Distillation | `--time-shift 1` | student[t]↔teacher[t-1]. +0.004~0.014 |
| Depthwise Conv Stem | `--stem-block-type depthwise` | 8 blocks, 4.18s RF, -0.003 adaptation |
| Causal Mel (fallback) | `--center-false` | On-the-fly center=False mel |
| GRL Disentanglement | `--grl-weight 0.05` | Speaker stripping |
| Forecast Heads | `--forecast-weight 0.05` | t+1,t+2 prediction |
| Q2D2 Quantization | `--q2d2-levels 9,9,9,9,9,9` | 3M-code rhombic grid (ICML 2026) |

## Experiment Results

| Model | center | cos768 | usage | Notes |
|-------|--------|--------|-------|-------|
| **7L WavLM 16kHz** | **strict-causal** | **0.934** | 17% | ★ Best |
| 8L WavLM 44.1kHz | strict-causal | 0.902 | 22% | misaligned kernels |
| 8L Mel center=False | strict-causal | 0.907 | 46% | baseline |
| 6L Mel TS | strict-causal | 0.917 | 20% | time-shift |
| 4L Mel center=True | 23ms future | 0.911 | 38% | non-causal |

## Quick Start

```bash
# Train with 16kHz WavLM frontend (best config)
.venv/bin/python train_mcs_q2d2.py \
  --device mps --epochs 30 --steps-per-epoch 2000 --batch-size 2 \
  --n-layers 7 --trans-dim 512 --n-heads 8 --ffn-dim 1024 --window 256 \
  --rope --swiglu --stem-block-type depthwise \
  --q2d2-grid rhombic --q2d2-levels 9,9,9,9,9,9 --q2d2-dim 6 \
  --wavlm-frontend --time-shift 1 \
  --content-cos-weight 1.0 --content-l1-weight 0.5 --delta-weight 0.04 \
  --forecast-weight 0.05 --voiced-boost 1.5 --grl-weight 0.05 --grl-num-speakers 108 \
  --lr 1e-4 --mel-frames 200 --eval-mel-frames 300 \
  --val-fraction 0.05 --probe-samples 256 --log-every 100 --save-every-epoch \
  --out-dir checkpoints/my_run --run-name my_run

# Resume
.venv/bin/python train_mcs_q2d2.py \
  --resume-from checkpoints/my_run/my_run.best.pt --lr 2e-5 --epochs 60 ...

# VC Inference
.venv/bin/python3 -c "
from train_mcs_q2d2 import MCSTransQ2D2Config,MCSTransQ2D2
from eval_mcs_trans_audio import load_mio,load_wave
from astrape.voicebank import VoiceBank
...
"
```

## Prerequisites

- Extract caches (one-time):
  ```bash
  .venv/bin/python cache.py --what wavlm --limit 0          # WavLM L4 200Hz (encoder frontend)
  .venv/bin/python cache.py --what speakers --utts-per-speaker 8   # per-speaker centroids (decoder)
  ```
- Verify cache integrity:
  ```bash
  .venv/bin/python check_cache.py --wavlm-only --wavlm-dir wavlm_L4_200hz
  ```

## Project Structure

```
# Root — thin CLIs (+ encoder, packaged into astrape/ in Phase 2)
train_mcs_q2d2.py     ★ Encoder training (Q2D2 + RoPE + SwiGLU + GRL + StridingAdapter)
train_decoder.py      ★ Decoder v5 training (2-phase: recon warmup → MPD/MSD adversarial)
cache.py               WavLM + speaker-centroid caching  (--what wavlm | speakers)
build_voicebank.py     Build .astrape voicebank (chunked + energy-gated + averaged)
eval_q2d2_vc.py / eval_mcs_trans_audio.py   VC / listening-set evaluation
check_cache.py         Cache integrity checker
mcs_common.py          Encoder shared (dataset, conv blocks, losses)   [Phase 2: split]
mcs_q2d2.py            shim → astrape.quantizer

astrape/               Core library
  nn.py                Shared primitives (CausalConv1d, RoPE, SnakeBeta, AdaLN layer)
  decoder.py           Decoder v5 (~16M, strict-causal, NSF, iSTFT n_fft=1512)
  discriminators.py    MPD + MSD (adversarial; training-only, MPS-safe)
  quantizer.py         Q2D2 quantizer (ICML 2026)
  data.py              Decoder dataset (Phase0Dataset) + teacher Gaussian blur
  miocodec.py          MioCodec bridge (load_mio/load_wave/write_wave + speaker extraction)
  voicebank.py         .astrape VoiceBank format
tests/
  test_streaming_invariant.py  Streaming causality tests

data/mio_vctk_full_compact/   VCTK npz cache
data/mio_vctk_full_compact/wavlm_L4_200hz/   WavLM L4 200Hz cache (encoder frontend)
data/mio_vctk_full_compact/spk_centroids.npz  per-speaker MioCodec global centroids
checkpoints/ → /Volumes/UNTITLED/btrv5_checkpoints/
```

## Decoder v4

Ultra-lightweight streaming vocoder. Speaker conditioning via AdaLN-Zero.
STFT-domain teacher forcing with Gaussian-blurred targets (cdecoder.md).

```bash
# Phase 0: Train decoder with frozen encoder
.venv/bin/python train_decoder.py \
  --device mps --epochs 30 --steps-per-epoch 2000 --batch-size 2 --max-frames 50 \
  --lr 2e-5 --mrstft-weight 0.3 --mel-l1-weight 1.0 \
  --blur-sigma-ms 2.0 --nffts 512 1024 2048

# With Mamba SSM (CPU only, experimental)
.venv/bin/python train_decoder.py --use-mamba --device cpu --max-frames 25 ...
```

| Component | Params | Delay |
|-----------|--------|-------|
| Encoder (8L WavLM L4) | 24.3M | 17ms |
| Decoder (Conv) | 7.08M | 31.7ms |
| Decoder (Mamba) | 8.70M | 20.1ms |
| **E2E (Conv)** | **31.4M** | **48.7ms** |
| **E2E (Mamba)** | **33.0M** | **37.1ms** |

## References

- **Q2D2**: Shuster & Nachmani, "Two-Dimensional Quantization for Geometry-Aware Audio Coding", ICML 2026, arXiv:2512.01537
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023, arXiv:2312.00752
- **Hyena**: Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models", 2023, arXiv:2302.10866
- **MioCodec**: Aratako/MioCodec-25Hz-44.1kHz-v2 (HuggingFace)
- **WavLM**: Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", 2022
- **GRL**: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015
- **ConvNeXt**: Liu et al., "A ConvNet for the 2020s", CVPR 2022
- **WavTokenizer**: Ji et al., "WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer", 2024
- **Snake/BigVGAN**: Lee et al., "BigVGAN: A Universal Neural Vocoder", 2023, arXiv:2206.02944
- **ISTFT/Vocos**: Siuzdak et al., "Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders", 2024, arXiv:2306.00819
- **Predictive Coding**: Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018
- **APCodec**: Ai et al., "APCodec: A Neural Audio Codec", IEEE/ACM TASLP 2024, arXiv:2402.10533
