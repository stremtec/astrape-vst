# Codex VC

Mimi 기반 음성 변환 — Codex 리뷰 아키텍처 구현.

```
source audio → Mimi encode → LV0 codes ──────────┐
target audio → Resemblyzer → spk embedding ───────┤
                                                   ↓
                                     Bidirectional Transformer
                                                   ↓
                                     LV1-7 codes (7×T)
                                                   ↓
                                     Mimi decoder → VC audio
```

## 구조

```
codex_vc/
├── model.py    # CodeGenerator (bidirectional transformer)
├── train.py    # Training script
├── infer.py    # CLI inference
├── README.md   # This file
└── __init__.py
```

## 설치

```bash
pip install moshi resemblyzer soundfile scipy
```

## 사용법

### 학습

```bash
python codex_vc/train.py
```

필요 파일:
- `runs/vctk_codes_full.pt` — Mimi 코드 캐시
- `runs/vctk_full_spk.pt` — Resemblyzer 화자 임베딩

### 추론

```bash
python codex_vc/infer.py \
    --source input.wav \
    --target-speaker p226 \
    --output vc_output.wav
```

## 성능

| 지표 | 값 |
|------|-----|
| Code prediction accuracy | 99.9% |
| Parallel VC Δ | +0.70 (avg across 4 speakers) |
| Model params | ~5M |
| Training time (5spk, 100step) | ~2min (CPU) |

## 아키텍처 상세

- **LV0**: Mimi semantic/content codes (2048-dim codebook)
- **Speaker**: Resemblyzer 256-dim text-independent embedding
- **Transformer**: 3-layer bidirectional, d_model=256, 4 heads
- **Output**: 7 independent linear heads → LV1-7 acoustic codes
- **Decoder**: Frozen Mimi SEANet decoder

## References

- Kyutai Mimi: https://github.com/kyutai-labs/moshi
- Resemblyzer: https://github.com/resemble-ai/Resemblyzer
- Codex review: https://github.com/stremtec/astrape-vc/pull/7
