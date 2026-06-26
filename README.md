# Astrape-vc — Zero-Shot Voice Conversion

Zero-lookahead neural voice conversion at 44.1kHz.  
**Current best: cos768 = 0.934** (strict-causal, 16kHz WavLM + 7L Transformer).

## Architecture

```
Mic 44.1kHz → resample(44.1k→16k) → WavLM CNN (94M, frozen, pad=0) → 50Hz
  → Adapter(512→80, 764K) → Causal Depthwise Stem (1.4M, 8 blocks)
  → Downsample(2×) → 25Hz → ProjIn(320→512)
  → Causal Transformer 7L (13.8M, RoPE+SwiGLU, window=256)
  → Q2D2 (3M codes, Rhombic[9×9]³) → content 768d @ 25Hz
  → MioCodec Decoder (228M, frozen) → wav 44.1kHz
```

**Learnable: 22.3M | Frozen: 94.4M | Total algorithmic latency: ~27ms**

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

- Extract WavLM cache (one-time, ~60 min):
  ```bash
  .venv/bin/python cache_wavlm_16k.py
  ```
- Verify cache integrity:
  ```bash
  .venv/bin/python check_cache.py --wavlm-only
  ```

## Project Structure

```
train_mcs_q2d2.py         ★ Encoder training (Q2D2 + RoPE + SwiGLU + GRL + WavLM)
train_decoder.py           ★ Decoder Phase 0 training (MR-STFT + Mel L1)
mcs_common.py              CausalConv1d, DepthwiseResidualBlock, dataset, MR-STFT loss
mcs_q2d2.py                Q2D2 quantizer (ICML 2026)
cache_wavlm_16k.py         16kHz WavLM CNN cache extraction
cache_wavlm_cnn.py          44.1kHz WavLM CNN cache (legacy)
check_cache.py              Cache integrity checker
eval_q2d2_vc.py             VC evaluation
eval_mcs_trans_audio.py     MioCodec bridge utilities
astrape/                    Core library
  causal_decoder.py           Decoder v4 (7.08M, 31.9ms, Conv / Mamba)
  mamba_block.py              Mamba SSM block (experimental, CPU-only)
  encoder.py                  Causal Content Encoder
  fsq.py, data.py, audio.py   Supporting modules
  voicebank.py                VoiceBank management
  wave_decoder.py             CausalConv1d base
tests/                      Test scripts
  test_streaming_invariant.py  Streaming causality tests

data/mio_vctk_full_compact/   VCTK dataset (npz cache)
data/mio_vctk_full_compact/wavlm_16k/   16kHz WavLM CNN cache (14GB)
wavlm_16k_local/              Local copy of WavLM cache (for training stability)
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
| Encoder (7L WavLM) | 22.4M | 27ms |
| Decoder (Conv) | 7.08M | 31.9ms |
| Decoder (Mamba) | 8.70M | 20.3ms |
| **E2E (Conv)** | **29.4M** | **58.9ms** |
| **E2E (Mamba)** | **31.1M** | **47.3ms** |

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
