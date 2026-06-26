# Astrape VC — 16kHz Streaming Architecture

## Full Pipeline (Training)

```
VCTK wav 44.1kHz
  │
  ▼ resample(44.1k→16k)  ← MioCodec ssl.resampler와 동일
  │
  ▼ WavLM CNN (94M, frozen, stride=320)
  │   16,000 / 320 = 50.0 Hz (정확)
  │   → (T, 512) float32 @ 50Hz
  │
  │  [cached to wavlm_16k/s_XXXXX.npy, 14GB]
  │
  ▼ WavLMFrontendAdapter (764K, learned)
  │   Linear(512→256)→GELU→Linear(256→80)
  │   → (T, 80) @ 50Hz
  │
  ▼ Causal Depthwise Stem (1.4M)
  │   8 depthwise blocks, dilations 1-16
  │   → (T, 320) @ 50Hz
  │
  ▼ CellDownsample(2×)
  │   → (T/2, 320) @ 25Hz
  │
  ▼ ProjIn(320→512) → Transformer 7L (13.8M)
  │   RoPE + SwiGLU, window=256
  │   → (T/2, 512) @ 25Hz
  │
  ▼ Q2D2 (8.5K, 3M codes)
  │   → content (T/2, 768) @ 25Hz
  │
  ▼ MioCodec Decoder (228M, frozen)
  │   → wav 44.1kHz
```

## Full Pipeline (Streaming Inference)

```
Mic input 44.1kHz
  │
  ▼ Polyphase resampler (44.1k→16k)
  │   delay: ~2ms
  │
  ▼ WavLM CNN — state-carry per conv layer
  │   7 causal convs, padding=0
  │   output: 1 frame @ 50Hz per 320 samples
  │   algorithmic delay: 400 samples @ 16kHz = 25ms
  │   compute: ~0.4ms/frame (CPU)
  │
  ▼ Adapter — per-frame, 0ms delay
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

| Component | Algorithmic | Compute (CPU) |
|-----------|-------------|---------------|
| Resampler 44.1k→16k | ~2ms | ~0.1ms |
| WavLM CNN RF | 25ms | 0.4ms |
| Adapter | 0ms | 0.01ms |
| Stem + Downsample | 0ms | 0.3ms |
| Transformer 7L | 0ms | 0.5ms |
| Q2D2 | 0ms | 0.01ms |
| **Encoder Total** | **~27ms** | **~1.3ms** |

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

## Decoder v4 — Ultra-Lightweight Streaming Vocoder (7.08M, 31.9ms)

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
  ▼ Phase 3: Shallow ConvNet (23.2ms)
  │   2× ShallowConvBlock(k=3 depthwise + AdaLN-Zero)
  │     causal depthwise conv → SiLU → pointwise → +residual×gate
  │     delay: 4 frames @ 172Hz = 23.2ms
  │   → (B, T_stft, 384) @ 172Hz
  │
  ▼ Phase 4: ISTFT Head (8.7ms)
  │   Pointwise 384→512
  │   ISTFTHead(Linear→mag/phase→iSTFT, n_fft=1024, hop=256)
  │     delay: pad=384 samples @ 44.1kHz = 8.7ms
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
  ▼ Phase 3: Shallow ConvNet — 23.2ms
  │   2 blocks, state-carry per causal conv
  │   delay: 4 frames @ 172Hz = 23.2ms
  │   state: 2 conv input caches = ~3KB
  │
  ▼ Phase 4: Pointwise 384→512 — 0ms
  │   ISTFTHead — 8.7ms
  │     state: ring buffer (384 samples) + Hann window overlap
  │     emit: 256 samples @ 44.1kHz = 5.8ms audio chunk
  │
  ▼ Audio output 44.1kHz (256 samples per frame)
```

### Decoder Latency Budget

| Component | Algorithmic | Compute (CPU) |
|-----------|-------------|---------------|
| Phase 0 (norm) | 0ms | ~0ms |
| Phase 1 (Pointwise+2L Trans) | 0ms | ~0.2ms |
| Phase 2 (repeat+pointwise) | 0ms | ~0.01ms |
| Phase 3 (2× Conv k=3) | 23.2ms | ~0.05ms |
| Phase 4 (pointwise+iSTFT) | 8.7ms | ~0.2ms |
| **Decoder Total** | **31.9ms** | **~0.5ms** |
| + Encoder | 27ms | ~1.3ms |
| **E2E Total** | **58.9ms** | **~1.8ms** |

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
