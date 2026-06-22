# MCS-Trans Q2D2 Documentation

## Active Training
- `train_mcs_q2d2.py` — Main training script (Q2D2 + RoPE + SwiGLU + GRL + decoder-loop)
- `mcs_q2d2.py` — Q2D2 quantizer implementation (ICML 2026)
- `mcs_common.py` — Shared utilities (dataset, causal conv, checkpoint)

## Key Results
| Experiment | cos768 | Notes |
|------------|--------|-------|
| rope_swiglu (4L) | 0.9114 | Baseline, RoPE + SwiGLU |
| + GRL | 0.9106 | Speaker disentanglement (grl_acc=0) |
| + decoder-loop | TBD | Waveform loss fine-tuning (in progress) |

## Architecture
- Encoder: causal conv + 4-layer causal transformer + Q2D2 rhombic [9,9,9,9,9,9]
- Decoder: frozen MioCodec (228M), decoder-in-loop fine-tuning
- 13.3M params (SwiGLU), 3,048,625 codebook (Q2D2)
