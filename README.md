# Astrape VC

**Real-time zero-shot voice conversion via MioCodec teacher-student distillation.**

> **Αστραπή** (Astrape) — Greek for "lightning"

## Architecture

```
Source (44.1kHz)
  → Causal Content Student (Transformer, 25Hz)
  → Content Embedding [T, 768]
  → Causal AdaLN-Zero Mel Decoder
  + Cached Target Global Embedding [128]
  → Teacher Wave Decoder / Vocoder
  → Target Voice (44.1kHz)
```

| Component | Params | Description |
|-----------|:------:|-------------|
| Causal Content Student v2 | ~4M | Causal Transformer encoder, mel→content embedding distillation |
| Content Student v1 (baseline) | ~1M | Mel+TCN encoder, FSQ 5-dim target |
| Causal Mel Decoder | ~3M | AdaLN-Zero speaker-conditioned, self-attention |
| MioCodec Teacher (offline) | ~200M | WavLM + Transformer, non-causal, quality upper bound |
| **Total Student** | **~7M** | Fully causal, streaming-capable |

## Key Features

- **Zero-shot**: Target speaker from 1-3s reference audio, cached as 128d global embedding
- **Causal-first**: All convolutions and attention are strictly causal — no future leakage
- **Teacher-student**: MioCodec provides quality upper bound; student distilled for streaming
- **Usable quality**: Jitter lower than teacher, content intelligibility confirmed
- **44.1kHz native**: No bandwidth extension needed
- **AdaLN-Zero conditioning**: Clean speaker/content separation

## Research Status

| Milestone | Status |
|-----------|--------|
| MioCodec teacher VC quality confirmed | ✅ jitter 7.1%, crest 6.0 |
| Target global embedding stable + cacheable | ✅ cos >0.99 under augmentation |
| Causal Content Student v1 (TCN) | ✅ functional, cos 0.63, garbled content |
| Causal Content Student v2 (Transformer) | ✅ cos 0.90 val, intelligible |
| Causal Mel Decoder | ✅ content-driven, global works |
| Global conditioning verified | ✅ tgt > src > other > zero |
| Streaming vocoder | ⏳ not started |
| End-to-end latency benchmark | ⏳ pending |

## Quick Start

```bash
# Load pretrained student + teacher
python3 -c "
from miocodec.model import MioCodecModel
import torch

# Teacher (quality reference, non-causal)
teacher = MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')

# Student models (causal, streaming)
# checkpoints/causal_student_v2_final.pt  — 384dim Transformer
# checkpoints/causal_mel_decoder.pt       — causal decoder
# checkpoints/student_proj_out.pt         — content projection
"
```

## Documentation

```
docs/research/
├── mimi_splitter_vc_summary.md       # Mimi VC experiments (negative result)
├── miocodec_internal_audit.md        # MioCodec deep structure analysis
├── miocodec_causality_audit.md       # Non-causality root causes
├── miocodec_vc_confirmed.md          # Teacher VC quality confirmation
├── mio_causal_student_status.md      # Current student pipeline status
└── mio_student_content_bottleneck.md # Content intelligibility diagnosis
```

## Branches

- `main` — FlowVC / F³ architecture (legacy)
- `research_mio` — **Active**: MioCodec causal student distillation

## License

MIT
