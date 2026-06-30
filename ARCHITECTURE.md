# Astrape VC — 16kHz Streaming Architecture

## Full Pipeline (Training)

```
VCTK wav 44.1kHz
  │
  ▼ resample(44.1k→16k)
  │
  ▼ WavLM CNN L0-L4 (94M frozen, layers 0-4 only, raw 200Hz output)
  │   delay: 160 samples @16kHz = 10ms
  │   → (T, 512) float32 @ 200Hz
  │
  │  [cached to wavlm_L4_200hz/]
  │
  ▼ StridingAdapter (learned, ~87K)
  │   CausalConv1d(512→256, k=4, s=4, groups=256) + Linear(256→80)
  │   rate: 200Hz → 50Hz, k=s ⇒ full 4-frame coverage (no dropped frames)
  │   causal ⇒ 0 look-ahead; 3-frame backward state @200Hz
  │   → (T, 80) @ 50Hz
  │
  ▼ Causal Depthwise Stem (1.4M)
  │   8 depthwise blocks, dilations 1-16
  │   → (T, 320) @ 50Hz
  │
  ▼ CellDownsample(2×)
  │   → (T/2, 320) @ 25Hz
  │
  ▼ ProjIn(320→512) → Transformer 8L (15.8M)
  │   RoPE + SwiGLU, window=256
  │   → (T/2, 512) @ 25Hz
  │
  ▼ Q2D2 (8.5K, 3M codes)
  │   → content (T/2, 768) @ 25Hz
  │
  ▼ MioCodec Decoder (228M, frozen)
  │   → wav 44.1kHz
```

> **✅ Resolved (StridingAdapter):** the decimating conv is now `kernel=4,
> stride=4`, so every 200 Hz frame reaches exactly one output (was `kernel=2`,
> which dropped ~50 %). See **Architecture Review Notes → #1**.

## Full Pipeline (Streaming Inference)

```
Mic input 44.1kHz
  │
  ▼ Polyphase resampler (44.1k→16k)
  │   delay: ~2ms
  │
  ▼ WavLM CNN L0-L4 — state-carry per conv layer
  │   5 causal convs, padding=0, raw 200Hz
  │   algorithmic delay: 160 samples @16kHz = 10ms
  │   compute: ~0.3ms/frame (CPU)
  │
  ▼ StridingAdapter — learned k=4 stride-4 conv (200→50Hz)
  │   0 look-ahead (causal); 3-frame backward state, ~0.05ms compute
  │
  ▼ Stem + Downsample — state-carry, ~0.3ms
  │   output: 1 frame @ 25Hz per 640 samples
  │
  ▼ Transformer — KV-cache per layer
  │   causal windowed attention (backlog only)
  │   compute: ~0.5ms/frame (CPU)
  │
  ▼ Q2D2 — per-frame, ~0.01ms
  │
  ▼ Decoder → audio output 44.1kHz
```

## Latency Budget

Three distinct quantities — keep them separate (they were conflated in earlier
revisions):

- **Look-ahead** — future input the model must wait for. Strict-causal ⇒ **0**.
- **Algorithmic latency** — group/processing delay with *no* future access
  (e.g. the group delay of a frozen valid-conv extractor). Real, sometimes tunable.
- **Backward RF** — past context carried as state. **Free** (no latency).

| Component | Look-ahead | Algorithmic latency | Backward RF | Compute |
|-----------|-----------|---------------------|-------------|---------|
| Resampler 44.1k→16k | 0 | ~2ms (anti-alias FIR) | — | ~0.1ms |
| WavLM CNN L0-L4 (5 convs, frozen) | 0 | **~10ms** (160-sample RF) | — | 0.3ms |
| StridingAdapter (200→50Hz) | 0 | 0 (replicate/pool/conv) | 3 frames | 0.05ms |
|   ReplicatePool (α=0.8, fixed) | 0 | 0 (5ms boundary RF) | 3 frames | 0.05ms |
|     α-decaying replicate-pad + avg_pool(k=4). | | | | |
|     Merges avg_pool stability with causal 5ms delay. | | | | |
|     pad = [α³x₀, α²x₀, α¹x₀] → Bayesian boundary prior. | | | | |
| Stem + Downsample | 0 | 0 | ~212 frames | 0.3ms |
| Transformer 8L (causal window) | 0 | 0 | window=256 | 0.6ms |
| Q2D2 | 0 | 0 | — | 0.01ms |
| **Encoder Total** | **0** | **~12ms** | (state, free) | **~1.4ms** |

The only real encoder latency is the **frozen WavLM CNN (~10ms) + resampler
(~2ms)** — fixed, not part of the learnable design. Everything learnable
(adapter, stem, 8L Transformer, Q2D2) is strictly causal: 0 look-ahead, 0
algorithmic latency, pure backward state.

## State Carry (per component)

| Module | State | Size |
|--------|-------|------|
| Resampler | polyphase filter state | ~1KB |
| WavLM CNN L0 | last 9 input samples | 9×float32 |
| WavLM CNN L1-L6 | last k-1 frames per layer | ~4KB |
| Stem convs | last k-1 frames per block | ~10KB |
| Transformer | KV-cache (window=256, 7L) | ~7MB |
| **Total state** | | **~7MB** |

## Training vs 44.1kHz Pipeline

| | 44.1kHz (old) | 16kHz (new) |
|---|---------------|-------------|
| CNN input rate | 44.1kHz | **16kHz** |
| CNN output rate | 137.8Hz → pool → 46Hz | **50Hz (native)** |
| avg_pool | 필요 | **불필요** |
| interpolation | 46→50Hz 보간 | **불필요** |
| 8% temporal warp | 있음 | **없음** |
| CNN compute | 44.1k samples/s | **16k samples/s (2.7× faster)** |
| WavLM kernel alignment | misaligned (44.1k→16k mismatch) | **perfectly aligned** |

## Decoder v4 — Ultra-Lightweight Streaming Vocoder (7.08M, 0 look-ahead, ~8.6ms algorithmic)

### Decoder Pipeline (Training)

```
Q2D2 content (B, T, 768) @ 25Hz      Speaker (B, 128)
  │
  ▼ Phase 0: Input Norm (0ms)
  │   x = x × 1.21  (Q2D2 std→Teacher std)
  │
  ▼ Phase 1: Causal Prenet (0ms)
  │   Pointwise 768→384
  │   2L Causal Transformer (dim=384, 8heads, AdaLN-Zero)
  │     KV-cache, window=50, RoPE + SwiGLU
  │   → (B, T, 384) @ 25Hz
  │
  ▼ Phase 2: Zero-Delay Upsample (0ms)
  │   repeat_interleave(7×) @ 25→175Hz
  │   Pointwise Conv1d(k=1) 384→384
  │   trim to stft_len (~172Hz)
  │   → (B, T_stft, 384) @ 172Hz
  │
  ▼ Phase 3: Shallow ConvNet (0ms latency)
  │   2× ShallowConvBlock(k=3 depthwise + AdaLN-Zero)
  │     causal depthwise conv → SiLU → pointwise → +residual×gate
  │     4-frame BACKWARD RF @172Hz (state-carry) — 0 algorithmic latency
  │   → (B, T_stft, 384) @ 172Hz
  │
  ▼ Phase 4: ISTFT Head (8.6ms algorithmic latency)
  │   Pointwise 384→512
  │   ISTFTHead(Linear→mag/phase→iSTFT, n_fft=1008, hop=252)
  │     delay: pad=378 samples @ 44.1kHz = 8.5ms
  │   → (B, T_audio) @ 44.1kHz
  │
  ▼ MR-STFT loss vs original audio (Gaussian blurred, σ=2ms)
```

### Decoder Pipeline (Streaming Inference)

```
Q2D2 content (1 frame @ 25Hz)        Speaker (1, 128)
  │
  ▼ Phase 0: norm (×1.21) — 0ms
  │
  ▼ Phase 1: Pointwise 768→384 — 0ms
  │   Transformer KV-cache inference (1 frame, 0ms)
  │     cache: window=50 frames @ 25Hz = 2.0 sec backlog
  │   → (1, 384) @ 25Hz
  │
  ▼ Phase 2: repeat_interleave(7×) @ 25→175Hz — 0ms
  │   Pointwise(k=1) — 0ms
  │   append to ring buffer, emit 1 frame @ 172Hz when ready
  │
  ▼ Phase 3: Shallow ConvNet — 0ms latency
  │   2 blocks, state-carry per causal conv
  │   4-frame BACKWARD RF @172Hz = state, NOT delay (verified: future→0.00e+00)
  │   state: 2 conv input caches = ~3KB
  │
  ▼ Phase 4: Pointwise 384→512 — 0ms
  │   ISTFTHead — 8.6ms algorithmic latency (group delay, no future)
  │     state: ring buffer (384 samples) + Hann window overlap
  │     emit: 256 samples @ 44.1kHz = 5.8ms audio chunk
  │
  ▼ Audio output 44.1kHz (256 samples per frame)
```

### Decoder Latency Budget

Same three-way split as the encoder (look-ahead / algorithmic latency / backward RF):

| Component | Look-ahead | Algorithmic latency | Backward RF | Compute |
|-----------|-----------|---------------------|-------------|---------|
| Phase 0 (norm) | 0 | 0 | — | ~0ms |
| Phase 1 (Pointwise + 2L causal Trans) | 0 | 0 | window=50 | ~0.2ms |
| Phase 2 (repeat + pointwise) | 0 | 0 | — | ~0.01ms |
| Phase 3 (2× causal conv k=3) | 0 | 0 | 4 frames | ~0.05ms |
| Phase 4 (pointwise + iSTFT) | 0 | **~8.6ms** (n_fft=1008) | — | ~0.2ms |
| **Decoder Total** | **0** | **~8.6ms** | (state, free) | **~0.5ms** |

**Correction (2026-06-27):** the decoder's *only* algorithmic latency is the
iSTFT overlap-add group delay, ≈ (n_fft−hop)/2 = (1008−252)/2 = 378 smp = **8.6ms**.
Phase 3's "23.2ms" in earlier revisions was the 4-frame *backward* receptive
field (state-carry), not latency — empirically verified: perturbing future
content frames changes earlier audio by 0.00e+00. The iSTFT figure is the one
knob where latency trades for spectral quality (see "Decoder v5" below).

| Stage | Look-ahead | Algorithmic latency | Compute |
|-------|-----------|---------------------|---------|
| Encoder (WavLM + resampler, fixed) | 0 | ~12ms | ~1.4ms |
| Decoder (iSTFT) | 0 | ~8.6ms | ~0.5ms |
| **E2E** | **0** | **~21ms** (+~3ms compute ≈ **~24ms**) | |

**Budget = ~50ms E2E ⇒ ~25ms of headroom** to spend on decoder quality —
none of it on look-ahead (forbidden), all of it on capacity (free) + a larger
iSTFT window (algorithmic latency that buys frequency resolution).

### Decoder State Carry (per component)

| Module | State | Size |
|--------|-------|------|
| Transformer 2L | KV-cache (window=50, 2L, dim=384) | ~1.5MB |
| Upsample ring buffer | 7-frame FIFO @ 25Hz | ~3KB |
| Conv Block 0 | last 2 input frames @ 172Hz | ~1KB |
| Conv Block 1 | last 2 input frames @ 172Hz | ~1KB |
| ISTFT | ring buffer (384 samples) + overlap | ~2KB |
| **Total state** | | **~1.5MB** |

### Training vs Streaming Comparison

| | Training | Streaming |
|---|---|---|
| Content input | batch (B, T, 768) | 1 frame (1, 1, 768) |
| Transformer mode | full attention (O(T²)) | KV-cache (O(1)/frame) |
| Upsample | repeat_interleave full sequence | ring buffer + emit |
| Conv mode | full sequence | state-carry (causal pad) |
| ISTFT | full overlap-add | ring buffer + emit 256 samples |
| Loss | MR-STFT(σ=2ms blur) + Mel L1 | — |

### Design Philosophy

- Encoder handles ALL long-range context (7L Transformer + Q2D2). Decoder only: speaker conditioning + waveform rendering.
- 25Hz domain: ZERO temporal convs. Transformer (KV-cache) = 0ms delay.
- 172Hz domain: 2-block shallow conv (k=3). Forces precise phase prediction; deeper convs cause phase blurring.
- Phase 3 can be swapped for Mamba×2 (k=2) → 20.3ms delay, unlimited RF via SSM state.

---

## Architecture Review Notes (2026-06-27)

Improvements possible *within the current constraints* (streaming-causal,
44.1 kHz I/O, frozen WavLM + MioCodec). Each gives the issue, the proposal, and
the reasoning. **All three are now implemented** (2026-06-27) — see `devlog.md`
for the change list and verification. (Two separate training bugs found in the
same review were also fixed.)

### #1 — StridingAdapter decimation dropped ~50 % of the 200 Hz WavLM frames  ✅ Implemented

- **Issue identified.** The 200 Hz→50 Hz adapter was
  `CausalConv1d(512→256, kernel=2, stride=4, groups=256)`
  (`WavLMFrontendAdapter` in `astrape/encoder.py`). With `kernel < stride`, each
  output frame `j` read only input frames `4j-1, 4j`; frames `4j+1, 4j+2`
  contributed to *no* output. Roughly half of the cached self-supervised
  features were silently discarded before the encoder ever saw them. (WavLM's
  own conv layers 5–6, which this adapter replaces, are two stacked
  `k=2,s=2` convs and *do* cover all 4 frames.)
- **Proposal (done).** Set `kernel_size = stride` (`k=4, s=4`, exact
  non-overlapping window), keeping the causal left-padding and depthwise
  grouping. `k=5` would add slight overlap. +~1 K params, +2 frames of state.
- **Reasoning behind it.** A decimating filter should integrate its whole
  decimation window; dropping half the inputs is pure information loss that
  most plausibly caps cos768. A `k=4` learned conv can also represent the fixed
  `avg_pool(4)` baseline (equal weights) **and** improve on it, so the 200 Hz
  raw cache finally pays off — whereas `k=2` couldn't even see the frames the
  pool averages. Crucially, the conv is **causal** (left-padded only), so the
  wider kernel extends *backward* (state-carried) context and adds **zero
  look-ahead latency** — the "15 ms → 5 ms" change recorded in git reduced
  backward context, not real-time delay, so full coverage is recoverable at no
  streaming look-ahead cost (see the † footnote on the latency budget).

### #2 — Auxiliary losses (forecast, GRL) ignored the padding mask  ✅ Implemented

- **Issue identified.** In `q2d2_losses` the forecast heads used
  `F.mse_loss(..., reduction="mean")` over *all* frames including right-padding,
  and the GRL speaker classifier pooled `content.mean(dim=-1)` over padded frames
  too. The primary content losses already mask correctly; only these auxiliaries
  didn't.
- **Proposal (done).** Forecast MSE now uses a per-frame mask that requires both
  the source frame and the (t+1 / t+2) future target frame to be valid; the GRL
  `SpeakerClassifier` does a mask-weighted temporal mean
  (`(content*mask).sum / mask.sum`) over the valid loss region.
- **Reasoning behind it.** Padded positions are zeros, not signal. Averaging
  them into the forecast target dilutes the gradient by a batch-dependent factor,
  and pooling them into the speaker embedding injects a constant "padding token"
  that weakens the adversarial disentanglement. The mask was already computed, so
  the fix is near-free and removes a length-dependent bias from training.

### #3 — `astrape/check_cache.py` integrity guard pointed at the wrong cache  ✅ Implemented

- **Issue identified.** `astrape/check_cache.py` exists to catch caches that get
  corrupted/truncated *after* extraction — a valid and active purpose. But it
  hard-coded the `wavlm_cnn` directory and a 44.1 kHz / 3×-avg-pool repair recipe,
  both of which predate the current `wavlm_16k` / `wavlm_L4` / `wavlm_L4_200hz`
  caches. So `--wavlm-only` reported "directory missing" and (worse) `--repair`
  would have regenerated features with the wrong recipe — the exact corruption it
  is meant to prevent could slip through on the cache actually in use.
- **Proposal (done).** Added `--wavlm-dir` (default `wavlm_L4_200hz`) so the
  rate-agnostic integrity check (exists / loads / 2-D / 512-channel / not
  truncated) covers the active cache, and rewrote `--repair` to use the L4-raw
  200 Hz extraction (resample→16 kHz, 5 conv layers, save `(T, 512)`) matching
  `astrape.cache --what wavlm`.
- **Reasoning behind it.** The tool's purpose is right; only its target and
  repair recipe had drifted. Re-pointing it makes the corruption guard actually
  cover the cache the StridingAdapter encoder trains on, and makes `--repair`
  regenerate correct features instead of legacy-pipeline ones.
- **Resolved (2026-06-27 packaging).** The dead FSQ `astrape/encoder.py`
  (+ `wave_decoder.py`, `fsq.py`, `data.py`, `audio.py`, `mamba_block.py`,
  `causal_decoder.py`) was deleted in the Phase-1 packaging; the codebase was
  consolidated into the `astrape/` package (see `devlog.md`).

---

# Decoder v5 — Max-Quality Causal Design (2026-06-27)

**Constraints:** strict causal (**0 look-ahead**), **~50ms E2E budget** (encoder
fixed at ~12ms → **~25ms of decoder headroom**, vs ~8.6ms used today), no Mamba
(selective-scan is unstable/absent on MPS; the pure-torch fallback is an O(T)
Python loop). Within these, the goal is to maximize naturalness.

## What the decoder is actually solving

The encoder hands the decoder only:

| Signal | Rate | Real information |
|--------|------|------------------|
| Content (768-d, frozen linear proj of the 6-d Q2D2 latent) | 25 Hz | log2(3,048,625) = **21.5 bits/frame ≈ 0.54 kbps** |
| Speaker (global) | static | timbre / mean-pitch, time-invariant |

So the decoder is a **1764× generative upsampler**: it expands 0.54 kbps + a
static timbre into 44.1 kHz. Pitch periods, phase, and excitation are **not** in
the input — they must be synthesized. `p(waveform | content, speaker)` is
**multimodal**, so an L1/MR-STFT regressor converges to the conditional *mean* =
spectral over-smoothing = the current buzzy/phasey timbre. **Distribution-matching
(adversarial) training is therefore the dominant lever, not a tweak.**

## The three budgets (spend each correctly)

| Lever type | Cost | Examples |
|-----------|------|----------|
| **Free** (compute + backward RF only) | 0 algorithmic latency | depth/width, dilation, learned upsampler, SnakeBeta, NSF, **adversarial training** |
| **Algorithmic latency** | group delay, no future | iSTFT `n_fft` (frequency resolution) |
| **Forbidden** | — | any future-frame access |

Priority: **exhaust the free levers first**; spend latency on `n_fft` only if
spectral resolution is still the limiter.

## Architecture

```
content 25Hz (768) ─┬─► Phase 1: causal windowed self-attention, KV-cache  (DEEPEN: free)
speaker 128 (static)┘     + dilated causal TCN @25Hz, AdaLN-Zero speaker
                          │
                  Phase 2: LEARNED causal upsampler 25→175Hz
                          │   causal ConvTranspose + SnakeBeta  (replaces repeat_interleave ZOH)
                          │
                  Phase 2b: causal F0 head ─► sine+noise excitation (NSF)
                          │                       │ harmonic comb at F0
                  Phase 3: deeper causal SnakeBeta conv stack @175Hz  (DEEPEN: free backward RF)
                          │   ◄── excitation spectrum injected as extra iSTFT-head channels
                  Phase 4: iSTFT head, n_fft=1512 hop=252 (mag+phase, "same") → 14.3ms
                          ▼
                     wav 44.1kHz
```

### Component changes vs v4 (each: lever type)

1. **Phase 2 — learned causal upsampler [FREE].** Replace `repeat_interleave`
   (zero-order hold → 175 Hz staircase the later stages must clean up) with a
   causal `ConvTranspose1d` + `SnakeBeta`. Biggest single architectural win, 0 latency.
2. **NSF excitation [FREE].** A small causal F0 head (F0 *is* recoverable — the
   MioCodec teacher resynthesizes correct prosody from the same content) drives a
   sine-plus-noise harmonic comb (cf. `pupu/.../sine_excitation.py`), fed into the
   head so it predicts only the spectral *envelope* — directly attacks buzziness.
3. **SnakeBeta everywhere [FREE].** Periodic activation (BigVGAN) for harmonic
   extrapolation; replaces SiLU in Phases 2/3.
4. **Deepen Phase 1 + Phase 3 [FREE].** The "Phase 3 = 23.2 ms" myth hid that
   backward RF is free — make the conv stack deeper/wider/more-dilated (more local
   modeling than MioCodec's *centered* ResNet) at **0** added latency. Grow total
   params from ~7M toward ~15–25M (35 ms compute budget is barely touched at ~1 ms).
5. **iSTFT `n_fft` 1008→`1512` [ALGORITHMIC LATENCY: +5.7 ms] — COMMITTED.**
   22.9→34.3 ms window, 43.8→29.2 Hz bins, group delay (1512−252)/2 = 630 smp =
   **14.3 ms**. Chosen for better low-F0/harmonic resolution within the 50 ms budget.
6. **Speaker:** keep AdaLN-Zero in Phase 1 + Phase 3 (already richer than MioCodec,
   which conditions only its decoder transformer).

### Training — 2-phase adversarial curriculum (the real lever — 0 inference cost)

| Phase | Epochs | Objective |
|-------|--------|-----------|
| **A — warmup** | `0 .. warmup-1` | reconstruction only: MR-STFT (CPU, grad-safe) + Mel-L1 on σ=2 ms blurred target. Gets a stable spectral baseline before the GAN. |
| **B — adversarial** | `warmup .. end` | + **MPD + MSD** (LSGAN) + feature matching; recon kept as anchor. (Enable NSF here with `--use-nsf`.) |

- **Discriminators: MPD + MSD** (both time-domain → **MPS-safe**; `torch.stft` is
  unstable on MPS, so a spectral MR-STFT-D is deferred to CUDA). Makes the decoder
  *sample* `p(x|c,s)` instead of averaging it. Training-only.
- MR-STFT + Mel-L1 (gradient-detachment bug fixed — see `devlog.md`) stay as the
  reconstruction anchor, weighted below the adversarial loss.

### v5 latency budget (n_fft = 1512)

| | Look-ahead | Algorithmic latency | Compute |
|--|-----------|---------------------|---------|
| Phases 1–3 (causal attn + upsampler + convs + NSF) | 0 | 0 | ~1–2.5ms |
| Phase 4 iSTFT (n_fft 1512) | 0 | **14.3ms** | ~0.3ms |
| **Decoder total** | **0** | **14.3ms** | **~2–3ms** |
| **+ Encoder (fixed)** | 0 | ~12ms | ~1.4ms |
| **E2E** | **0** | **~26ms** | ~4ms → **~29ms** |

Comfortably inside the ~50 ms budget with ~20 ms to spare.

### Implementation status (2026-06-27) — built & smoke-tested

| File | What | Verified |
|------|------|----------|
| `astrape/decoder.py` | `CausalDecoderV5` (+SnakeBeta, learned causal upsampler, dilated SnakeBeta conv stack, optional `HarmonicComb` NSF) | **15.9 M** params (16.7 M w/ NSF); **strictly causal (future→0.00e+00)**; iSTFT 14.3 ms |
| `astrape/discriminators.py` | MPD(2,3,5,7,11) + MSD(×3), LSGAN + feature-matching losses | 70.7 M (training-only); fwd/bwd OK, MPS-safe |
| `astrape/train_decoder.py` | 2-phase curriculum (warmup→adversarial), reuses frozen encoder + `Phase0Dataset` | both phases run fwd/bwd with G-gradient |

The naturalness gain comes almost entirely from **adversarial training + NSF +
learned upsampler** (all 0 latency) — the latency budget was never the binding
constraint; "no future" and "model the distribution" are.
