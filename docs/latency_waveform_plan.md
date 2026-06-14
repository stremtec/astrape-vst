# Latency And Causal Waveform Plan

Date: 2026-06-14

## Current Training Run

The 768x10 FSQ curriculum is running with:

- 93 teacher-training speakers and 16 held-out speakers
- 3,720 teacher clips and 640 validation clips
- 37,632 full original VCTK utterances
- batch size 2 on MPS
- 5 original epochs, 10 blend epochs, 30 teacher epochs
- two seconds of bounded past attention and no future attention
- hard teacher-content cosine target 0.99

The first original epoch completed at about 0.43 seconds per step:

- CTC loss: 2.9263
- held-out hard teacher frame cosine: 0.4590
- held-out p05 cosine: -0.0355
- exact FSQ token accuracy: 0.000

This is expected to be a language-only phase. Teacher FSQ supervision starts
during the blend phase.

## Collection Latency

The current 16 kHz frontend uses:

- causal STFT: `n_fft=512`, or 32 ms of collected waveform
- mel hop: 320 samples, or 20 ms
- content output rate: 25 Hz

The stride-2 student used to wait for two mel frames before emitting its first
content frame. That wait was unnecessary because the first causal downsampled
output depends only on mel frame zero and left padding.

The streaming schedule now:

1. processes the first mel frame immediately;
2. buffers one subsequent mel frame;
3. processes later mel frames in pairs.

This moves first content availability from 52 ms to 32 ms while preserving the
29 ms per 40 ms steady-state compute schedule. Full and chunked outputs still
match to about `1e-6`.

Audio callback granularity adds up to one callback period. A 5 ms callback
therefore makes the practical collection bound about 35 ms with the 512-sample
window.

## Shorter Window Experiment

Thirty-two VCTK utterances were measured against the 512-sample frontend.
Framewise mel cosine was computed after removing each frame's mean. The existing
causal student was also run without any adaptation.

| Window | Collection | Mel cosine | Mel p05 | Existing student output cosine |
|---:|---:|---:|---:|---:|
| 512 | 32 ms | 1.0000 | 1.0000 | 1.0000 |
| 400 | 25 ms | 0.9371 | 0.8109 | 0.9878 |
| 320 | 20 ms | 0.8504 | 0.6249 | 0.9691 |
| 256 | 16 ms | 0.7560 | 0.4452 | 0.9474 |

`n_fft=400` is the next candidate. It saves 7 ms and causes only a small output
shift before retraining. `n_fft=320` is a larger distribution change and should
not be adopted without a separate full-quality run. `n_fft=256` also produces
empty filters with 80 mel bands and is rejected.

Reproduce with:

```bash
.venv/bin/python analyze_frontend_latency.py \
  --audio-root /path/to/VCTK/wav48_silence_trimmed \
  --checkpoint checkpoints/content_student_v3_4k_causal.best.pt
```

## Frontend Hypotheses

### H1: immediate first-frame emission

Confirmed. It removes 20 ms with no model or target change.

### H2: 400-sample frontend adaptation

A short teacher-heavy fine-tune using `n_fft=400` should recover almost all
teacher cosine because the unadapted student output already has 0.9878 cosine
to its 512-window output.

Acceptance gate:

- held-out teacher frame cosine loses no more than 0.002;
- p05 cosine loses no more than 0.005;
- no regression in exact FSQ token accuracy;
- first collection latency falls from 32 ms to 25 ms.

If this gate fails, retain the 512-sample frontend. The 32 ms collection plus
about 30 ms content compute still leaves roughly 38 ms inside the 100 ms target
for waveform generation.

### H3: 320-sample frontend

This likely requires training from the original phase, not a short adaptation.
The risk is loss of stable low-frequency and unvoiced-boundary evidence. It is
only worth testing after the 400-sample experiment.

## Waveform Decoder Decision

Use a direct `content + global -> 44.1 kHz waveform` decoder. Keep the current
causal mel decoder as an auxiliary target and diagnostic, not as a mandatory
runtime stage.

The old `student_v2_global_final.pt` is not a production causal decoder:

- its `GroupNorm` observes the time axis;
- its transposed convolutions use symmetric overlap;
- it has no persistent streaming state;
- its ISTFT is full-sequence fold-based.

It is useful only as an offline quality baseline or partial weight source.

The direct decoder is now implemented in `astrape/wave_decoder.py`. Its default
runtime profile has 8.78M parameters. With MPS training paused, saturated
one-frame streaming measured 7.86 ms p50 and 9.79 ms p95. The larger 22.7M
parallel-MRF configuration remains available through
`WaveDecoderConfig.quality_profile()`.

The preferred body is a causal multi-dilation waveform generator:

```text
hard Mio FSQ content, 768d at 25 Hz
  -> global-conditioned input projection
  -> nearest-neighbor causal upsample stages
     factors 7, 7, 3, 3, 2, 2 = 1764
  -> channel-only normalization
  -> Snake activations
  -> causal dilated residual blocks
  -> causal waveform projection
  -> exactly 1764 samples per content frame
```

Nearest-neighbor expansion followed by cached causal convolution is preferred
over transposed convolution because its streaming alignment is explicit.

## Implementation Phases

### Phase W0: streaming primitives

- `astrape/wave_decoder.py` added.
- Stateful causal convolution and upsample caches implemented.
- Exactly 1,764 samples are returned for every content frame.
- Prefix-invariance and irregular-chunk equivalence tests added.
- Only channel-wise normalization is used.

Status: implemented; test error is below `2e-6`.

### Phase W1: teacher-forced decoder bootstrap

- Build an offline cache of Mio reconstructions from cached token indices and
  global embeddings.
- Feed exact teacher content and teacher global embeddings.
- Train on one-second crops with waveform L1, multi-resolution STFT, mel, and
  energy losses.
- Keep adversarial training disabled.

Exit gate: clean intelligible reconstruction with stable energy and no periodic
clicks at 40 ms boundaries.

### Phase W2: student-content adaptation

- Mix teacher content and the new student's hard FSQ content.
- Ramp student-content probability from 0 to 1.
- Keep the content student frozen initially.
- Use the current causal mel decoder output as an auxiliary acoustic loss.

Exit gate: student-content audio remains close to teacher-content audio and
does not introduce token-boundary clicks.

### Phase W3: quality polish

- Restore two-second crops and dense spectral losses.
- Add feature matching and adversarial loss only after reconstruction is stable.
- Optionally mix a small amount of original waveform target with Mio
  reconstruction targets.

Exit gate: held-out speaker quality improves without source-speaker leakage.

### Phase W4: runtime integration

- Precompute target global embeddings in the versioned VoiceBank.
- Connect streaming PCM, log-mel, content student, and waveform decoder through
  `StreamingVoiceConverter`.
- Bridge 40 ms model output to device callbacks with `OutputRingBuffer`.
- Measure callback-to-first-sample and saturated p50/p95 latency.
- Test one-frame, irregular-chunk, and long-running bounded-memory streams.

Status: software integration and full/chunk equivalence tests implemented.
Live-device timing and audio-quality sign-off require trained checkpoints.

Runtime gates:

- no future lookahead;
- bounded state;
- first output below 100 ms;
- steady compute below 40 ms per content frame;
- no discontinuity at chunk boundaries.

## Expected Latency Budget

With the current 512-sample frontend:

| Component | Budget |
|---|---:|
| waveform collection | 32-35 ms |
| content student p95 | about 31 ms |
| waveform decoder p95 | about 10 ms |
| callback and transfer margin | 4 ms |
| total | about 77 ms |

The 400-sample frontend would recover about 7 ms of safety margin if its quality
gate passes.

## VoiceBank Policy

The production VoiceBank is built from one continuous reference recording of
at least five seconds. There is no fixed upper duration: users may provide
10 seconds, 30 seconds, or one minute when they prefer more reference evidence.

Multiple references are neither required nor assumed. The VoiceBank is still
zero-shot because no target-specific model training or fine-tuning occurs.
