# Mimi Codec VC

Mimi 기반 음성 변환(Voice Conversion) 연구 — encoder/decoder transformer 분리, quantizer-space 변환, Kanade 스타일 disentanglement.

## 아키텍처 개요

```
                     ┌─────────────────────────────────┐
                     │        Mimi Codec (frozen)       │
                     │                                  │
  src audio ────────▶│ SEANet encoder                   │
                     │   ↓ (512-dim @ 25Hz)             │
                     │ Encoder Transformer (8 layers)   │
                     │   ├─ shallow [0-2]: content-rich │
                     │   └─ deep   [5-7]: speaker-rich  │
                     │   ↓ downsample (25→12.5Hz)       │
                     │ z (transformer space)            │
                     │   ↓                              │
                     │ RVQ Quantizer (8 levels)         │
                     │   ├─ LV0: semantic/content       │
                     │   └─ LV1-7: acoustic/speaker     │
                     │   ↓                              │
                     │ z_q (quantizer space)            │
                     │   ↓ upsample (12.5→25Hz)         │
                     │ Decoder Transformer (8 layers)   │
                     │   ├─ shallow [0-2]: content-ish  │
                     │   └─ deep   [5-7]: speaker-ish   │
                     │   ↓                              │
  vc audio ◀─────────│ SEANet decoder                   │
                     └─────────────────────────────────┘
```

## 핵심 인사이트

1. **Transformer space ≠ Quantizer space** (MSE=9.56, cos=0.0003)
   - Bypass quantizer → poor audio quality
   - Must operate in quantizer space (z_q) for proper decoding

2. **Token Swap (zero-shot, parallel text only)**
   - src LV0 + tgt LV1-7 → cos_tgt=0.85
   - Cross-text 불가능 (content leakage)

3. **Q-Space Converter (trained, parallel text)**
   - z_q_src + speaker_tgt → Δ=+0.74 (best so far)
   - Gradient flows through frozen decoder

4. **Encoder Kanade Splitter**
   - Shallow layers (0-2) → content (c_cos=0.996)
   - Deep layers (5-7) → speaker (s_cos=-0.39 improving)

5. **Decoder Transformer Kanade**
   - Zero-shot cross-text speaker transfer
   - But content preservation issue (sharp phase transition)

## 파일 구조

```
mimi_codec/
├── mimi_splitter.py          # KanadeSplitter variants (encoder multi-layer)
├── mimi_splitter_vc.py       # Clean splitter+converter architecture
├── mimi_qspace_vc.py         # Q-Space Converter (quantizer space)
├── mimi_vc_system.py         # End-to-end VC system
├── mimi_latent_converter.py  # Continuous latent converter
├── mimi_converter.py         # RVQ code-level converter
├── code_predictor.py         # Cross-text code predictor
│
├── train_mimi_splitter.py    # Splitter training
├── train_latent_converter.py # Latent converter training
├── train_code_predictor.py   # Code predictor training
├── train_cp_full.py          # Full MPS code predictor training
├── train_vc_system.py        # VC system training
│
├── test_converter.py         # Converter tests
├── test_final.py             # Final architecture test
├── test_fast.py              # Fast training test
├── test_improved.py          # Improved splitter test
├── test_resemblyzer.py       # Resemblyzer integration
├── test_res2.py / test_res3.py  # Resemblyzer VC tests
│
└── cached_dataset.py         # Mimi latent caching
```

## 결과 요약

| 방법 | Parallel Δ | Cross-text Δ | 학습 |
|------|:----------:|:------------:|:----:|
| Token swap (zero-shot) | +0.64 | ❌ | - |
| Q-Space Converter | **+0.74** | ❌ | 50 step |
| Decoder Kanade (α=0.5) | +0.70 | +0.80* | - |
| Resemblyzer + Kanade | +0.10 | -0.25 | 5 step |

*Decoder Kanade cross-text: speaker 전환되나 content 보존 미흡

## TODO

- [ ] Cross-text content preservation
- [ ] Text-independent speaker embedding (ECAPA/TitaNet)
- [ ] Full VCTK training (100+ speakers)
- [ ] Streaming/low-latency inference
- [ ] Objective evaluation (WER, MCD, MOS)

## References

- Mimi (Kyutai): https://github.com/kyutai-labs/moshi
- Kanade: Multi-layer feature separation for voice conversion
- Resemblyzer: https://github.com/resemble-ai/Resemblyzer
