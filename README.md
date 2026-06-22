# MCS-Trans Q2D2 — Causal Voice Conversion Encoder

Causal neural content encoder for zero-shot voice conversion at 44.1kHz.
Mel → causal transformer → Q2D2 quantization → frozen MioCodec decoder.

**Current best**: cos768=0.9215 (GRL speaker disentanglement + decoder-in-loop)

## Architecture

```
50Hz log-mel (80-dim)
  → causal ConvNeXt frontend + skip connections
  → 25Hz downsample
  → 4-layer causal Transformer (dim=512, 8 heads, SwiGLU, RoPE, window=256)
  → Q2D2 rhombic grid [9,9,9,9,9,9] = 3,048,625 codes
  → proj_out (6→768)
  → frozen MioCodec decoder → 44.1kHz waveform
```

**Parameters**: 13.3M (SwiGLU), 11.2M (SiLU)

## Quick Start

```bash
# Train with best config (Q2D2 + RoPE + SwiGLU + GRL + decoder-loop)
.venv/bin/python train_mcs_q2d2.py \
  --device mps \
  --epochs 10 --steps-per-epoch 2000 --batch-size 4 \
  --n-layers 4 --trans-dim 512 --n-heads 8 --ffn-dim 1024 --window 256 \
  --rope --swiglu \
  --q2d2-grid rhombic --q2d2-levels 9,9,9,9,9,9 --q2d2-dim 6 \
  --content-cos-weight 1.0 --content-l1-weight 0.5 --delta-weight 0.04 \
  --decoder-wave-weight 0.15 --decoder-wave-prob 0.3 \
  --grl-weight 0.1 --val-fraction 0.05 \
  --lr 5e-5 --mel-frames 200 --eval-mel-frames 300 \
  --log-every 200 --save-every-epoch --seed 42 \
  --out-dir checkpoints/mcs_trans_q2d2 \
  --run-name mcs_trans_q2d2

# Resume from checkpoint
.venv/bin/python train_mcs_q2d2.py \
  --resume-from checkpoints/mcs_trans_q2d2/mcs_trans_q2d2.best.pt \
  --epochs 20 ...

# Init from existing checkpoint (loads conv+transformer, Q2D2 head fresh)
.venv/bin/python train_mcs_q2d2.py \
  --init-from checkpoints/mcs_trans_q2d2_rope_swiglu.best.pt \
  --rope --swiglu --grl-weight 0.1 ...
```

## Experiment Results

| Experiment | cos768 | Q2D2 usage | Key change |
|------------|--------|------------|------------|
| rope_swiglu (4L) | 0.9114 | 39.3% | RoPE + SwiGLU |
| + GRL | 0.9106 | 37.7% | Speaker disentanglement (grl_acc=0) |
| + decoder-loop | 0.9215+ | in progress | Waveform loss fine-tuning |

All trained on VCTK (43,885 utterances, 108 speakers), val_fraction=0.05.

## Project Structure

```
train_mcs_q2d2.py                   ★ Main training (Q2D2 + RoPE + SwiGLU + GRL + decoder-loop)
train_mcs_trans.py                  FSQ baseline (comparison)
mcs_q2d2.py                         Q2D2 quantizer (ICML 2026)
mcs_common.py                       Dataset, causal conv, checkpoint utilities
eval_mcs_trans_audio.py             VC evaluation + MioCodec loading
train_mcs_original_calibrator.py    MR-STFT loss function

astrape/
  voicebank.py                      .astrape VoiceBank format (v3)

docs/
  README.md                         This document

checkpoints/  → /Volumes/UNTITLED/btrv5_checkpoints/
data/         → /Volumes/UNTITLED/btrv5_data/
outputs/      → /Volumes/UNTITLED/btrv5_outputs/
```

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rope` | off | Rotary Position Embedding |
| `--swiglu` | off | SwiGLU FFN (gated SiLU) |
| `--grl-weight` | 0.0 | GRL speaker disentanglement (~0.1) |
| `--decoder-wave-weight` | 0.0 | MR-STFT waveform loss (~0.15) |
| `--decoder-wave-prob` | 0.5 | Fraction of steps with waveform loss |
| `--q2d2-grid` | rhombic | Grid: rhombic, hexagon, rectangle |
| `--q2d2-levels` | 7,7,7,7,7,7 | Per-dimension quantization levels |
| `--val-fraction` | 0.15 | Validation split (0.05 for more data) |

## Requirements

Python 3.11+, PyTorch 2.2+, torchaudio, soundfile, numpy.
MioCodec weights auto-downloaded from HuggingFace.

## References

### Core Architecture
- **MioCodec** — [Aratako/MioCodec-25Hz-44.1kHz-v2](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2) (HuggingFace)
- **Q2D2** — [Two-Dimensional Quantization for Geometry-Aware Audio Coding](https://arxiv.org/abs/2512.01537) (ICML 2026)
- **FSQ** — [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505) (2023)
- **WavTokenizer** — [arxiv.org/abs/2401.03078](https://arxiv.org/abs/2401.03078) (2024)

### Disentanglement & VC
- **Gradient Reversal (GRL)** — [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495) (ICML 2015)
- [arxiv.org/abs/2604.12456](https://arxiv.org/abs/2604.12456)
- [arxiv.org/abs/2602.00594](https://arxiv.org/abs/2602.00594)
- [aclanthology.org/2024.findings-acl.681/](https://aclanthology.org/2024.findings-acl.681/)

### Audio Generation & Codecs
- [arxiv.org/abs/2110.01900](https://arxiv.org/abs/2110.01900)
- [arxiv.org/abs/2202.01855](https://arxiv.org/abs/2202.01855)
