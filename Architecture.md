# Astrape VC Architecture

## Purpose

Real-time (<100ms end-to-end latency), zero-shot voice conversion at 44.1kHz output.

The system converts arbitrary source speech into a target speaker's voice without paired training data. It requires only a 5+ second reference recording from the target speaker.

## Hard Constraints

Every design decision is downstream of these constraints:

1. **Strict causality.** Every output sample depends only on current and past inputs. No lookahead, no bidirectional attention, no centered convolutions. This is non-negotiable for real-time streaming.

2. **<100ms end-to-end latency.** From the moment a PCM sample enters the pipeline to the moment the corresponding output sample is available. This budget covers buffering, compute, and audio I/O.

3. **Speaker separation through information bottleneck.** The encoder must destroy speaker identity while preserving phonetic content. This is achieved via a 5-dimensional FSQ quantization layer (12,800 discrete codes). Content survives; speaker timbre does not.

4. **Zero-shot operation.** The decoder is conditioned on a single 128d speaker embedding extracted from a reference recording by MioCodec. No fine-tuning, no enrollment, no adaptation.

5. **44.1kHz output at CD quality.** The decoder must synthesize broadcast-quality audio, not 16kHz telephone speech.

---

## System Overview

```
16kHz PCM ─┐
           │
     StreamingLogMel (n_fft=512, hop=320)
           │
           ▼
     80-bin log-mel @ 50Hz
           │
     CausalContentEncoder (~12.3M params)
           │
           ▼
     768d content @ 25Hz  ←─── speaker-stripped via FSQ bottleneck
           │
           │     VoiceBank (.astrape file)
           │         │
           │     128d speaker embedding (from MioCodec global encoder)
           │         │
           ▼         ▼
     CausalSynthesisDecoder (~42M params)
           │
           ▼
     44.1kHz PCM (mono, float32, tanh-bounded)
```

---

## Latency Budget

| Stage | Buffering | Reason |
|-------|-----------|--------|
| Log-mel frontend | 2 frames = 640 samples = 40ms | STFT requires n_fft samples; hop=320 means first frame at 512 samples, second at 832 |
| Encoder frame-pair accumulation | +1 frame = 320 samples = 20ms | Downsample from 50Hz to 25Hz concatenates two adjacent mel frames |
| Encoder compute | ~3-5ms | Single frame through ConvNeXt + transformer (MPS) |
| Decoder compute | ~5-8ms | Single frame through transformer + upsample + waveform chain (MPS) |
| **Total algorithmic** | **~60-65ms** | Well within the 100ms budget |

The remaining ~35ms margin absorbs audio driver latency and scheduling jitter.

---

## CausalContentEncoder

**File:** `astrape/encoder.py`  
**Config:** `EncoderConfig`  
**Parameters:** 12,271,493

### Why This Architecture

The encoder must map 80-bin log-mel at 50Hz into a 768d content representation at 25Hz that:
- Matches the teacher's (MioCodec / WavLM) content space in cosine similarity
- Contains zero speaker identity information
- Is strictly causal
- Is small enough to run in <5ms per frame on Apple MPS

The FSQ bottleneck is the key mechanism: by forcing all information through 5 quantized dimensions (8x8x8x5x5 = 12,800 codes), speaker-specific continuous variation is destroyed while discrete phonetic distinctions survive. This is the same principle that makes VQ-VAE and neural codecs work for speaker-independent representations.

### Stage-by-Stage

#### Stage 1: Frontend (50Hz)

```
Input: (B, 80, T) log-mel spectrogram at 50Hz

CausalConv1d(80 → 384, kernel=5)     [153,984 params]
LayerNorm(384)                         [768 params]
GELU activation
2x CausalConvNeXtBlock(384, kernel=7) [2,370,816 params]
```

**CausalConvNeXtBlock:** depthwise causal conv → LayerNorm → pointwise up (384→1536) → GELU → pointwise down (1536→384) → residual add. This extracts local spectral patterns (formants, harmonics, transitions) with 7-frame receptive field per block (140ms after 2 blocks).

**Why ConvNeXt:** Higher parameter efficiency than plain convolutions. The depthwise-separable structure + inverted bottleneck captures local patterns without quadratic attention cost. At 50Hz, each frame is 20ms — purely local patterns dominate (phoneme boundaries, formant transitions).

**Why causal conv:** Left-padding only. The kernel sees current frame + 4 past frames (kernel=5) or + 6 past frames (kernel=7). No future information leaks in.

#### Stage 2: Downsample (50Hz → 25Hz)

```
Concatenate adjacent frame pairs: (B, T, 384) → (B, T/2, 768)
Linear(768 → 384)                     [295,296 params]
```

**Why 25Hz:** Phoneme duration averages 60-120ms. At 25Hz (40ms/frame), each phoneme spans 1.5-3 frames — sufficient temporal resolution for content while halving compute for all downstream stages. Also matches MioCodec's native content rate, so targets align without resampling.

**Why concat-then-project (not strided conv):** Deterministic, no learnable downsampling artifacts, trivially causal (just pairs sequential frames), and creates a clean break between the 50Hz local-feature stage and the 25Hz sequence-modeling stage.

#### Stage 3: Causal Transformer (25Hz)

```
4 layers, dim=384, 6 heads, head_dim=64  [9,444,096 params]
RoPE positional encoding (theta=10000)
SwiGLU FFN (ff_mult=4 → hidden=1536)
Sliding window attention (window=64 frames = 2.56 seconds)
```

**Why transformer here:** After ConvNeXt extracts local features, the transformer provides cross-frame reasoning — coarticulation, prosody continuity, and disambiguation of spectrally similar phones based on context. At 25Hz with window=64, each frame attends to 2.56s of history.

**Why sliding window (not full causal):** Unbounded causal attention has O(T^2) cost and unbounded KV cache growth. Window=64 at 25Hz gives 2.56s context — more than enough for phonetic disambiguation (which rarely requires >1s of history). The circular KV cache in streaming mode is fixed at `window * layers * 2 * head_dim * heads * dtype_size` regardless of utterance length.

**Why RoPE:** Relative position encoding that generalizes to unseen lengths. No learned position embeddings that cap sequence length.

**Why SwiGLU (not vanilla FFN):** `SiLU(W1·x) * W3·x` followed by `W2·...` gives better gradient flow and representation capacity per parameter than ReLU/GELU FFN at the same hidden size.

#### Stage 4: FSQ Bottleneck

```
Linear(384 → 5)     proj_in    [1,925 params]
  → bound (tanh, scaled to level range)
  → quantize (round with straight-through estimator)
  → normalize (divide by half-levels)
Linear(5 → 768)     proj_out   [4,608 params]
```

**Levels:** (8, 8, 8, 5, 5) = 12,800 total codes

**Why FSQ (not VQ-VAE):** FSQ (Finite Scalar Quantization) avoids codebook collapse, requires no EMA updates or commitment losses, and is trivially differentiable via straight-through. Each of the 5 axes is independently quantized — no shared codebook to manage.

**Why (8,8,8,5,5):** 12,800 codes provides sufficient capacity for English phoneme inventory (~44 phones × position variants × coarticulation contexts ≈ hundreds of distinct states) while being small enough that continuous speaker variation cannot be encoded. The asymmetric levels (8,8,8,5,5 rather than uniform) were determined by probing which axes carry more phonetic information.

**Why 5 dimensions:** This is the critical bottleneck width. At 5d, the representation has 5 × log2(average_level) ≈ 14.6 bits per frame. This is enough for ~16k phonetic states but far too little to encode speaker formant frequencies (which require continuous precision). Wider bottlenecks (8d, 10d) were tested and leaked speaker identity; narrower (3d) lost phonetic distinctions.

**Why Linear(5→768) for proj_out:** The training target (WavLM layer 6+9 average, or teacher MioCodec ce_768) lives in 768d space. The projection maps discrete FSQ codes into this target space. During inference, this is the last operation — the 768d output goes directly to the decoder.

#### Output

```
ContentOutput:
  content:       (B, 768, T_25hz)   — main output, channel-first
  pre_fsq:       (B, 5, T_25hz)     — pre-quantization values (for loss computation)
  hidden_384:    (B, 384, T_25hz)   — transformer output (for auxiliary losses)
  token_indices: (B, T_25hz)        — flat FSQ code indices (for codebook analysis)
```

---

## CausalSynthesisDecoder

**File:** `astrape/decoder.py`  
**Config:** `SynthesisDecoderConfig`  
**Parameters:** 41,991,041

### Why This Architecture

The decoder must synthesize 44.1kHz audio from:
- 768d content at 25Hz (speaker-stripped)
- 128d global speaker embedding (from VoiceBank)

This is a 1764x temporal upsampling problem (25Hz → 44.1kHz) that must:
- Produce natural speech with the target speaker's timbre
- Maintain strict causality for streaming
- Run in <8ms per content frame on MPS

The design separates the problem into sequence modeling (transformer at 25Hz for prosody and phoneme-to-acoustic mapping) and signal synthesis (progressive upsampling with Multi-Receptive-Field fusion for waveform quality).

### Stage-by-Stage

#### Stage 1: Input Projection (25Hz)

```
Linear(768 → 512)                      [393,728 params]
```

Maps content from the encoder's 768d representation to the decoder's internal dimension. 512d was chosen as the minimum width that doesn't bottleneck synthesis quality (verified by ablation).

#### Stage 2: Synthesis Transformer (25Hz)

```
4 layers, dim=512, 8 heads, head_dim=64   [15,414,784 params]
AdaLN-Zero conditioning (128d speaker → scale/shift/gate per sublayer)
RoPE positional encoding (theta=10000)
SwiGLU FFN (ff_mult=3 → hidden=1536)
Sliding window attention (window=32 frames = 1.28 seconds)
```

**Why transformer at 25Hz (not 50Hz):** The transformer is the most expensive component per frame. Running it at 25Hz instead of 50Hz halves its FLOPs with no quality loss — the content representation is already at 25Hz, so running the transformer at 50Hz on duplicated frames wastes compute processing redundant information. The upsampling to 50Hz happens after the transformer completes its sequence reasoning.

**Why AdaLN-Zero (not FiLM or cross-attention):** AdaLN-Zero provides per-sublayer conditioning with a learned gate that starts at zero (identity initialization). This means the model initially ignores speaker conditioning and learns content-to-acoustic mapping first, then gradually incorporates speaker style. FiLM (scale+shift only) lacks the gate. Cross-attention would require a sequence of speaker tokens rather than a single global vector.

**AdaLN-Zero mechanism:**
```
condition (128d) → SiLU → Linear → (gamma, beta, alpha)  [3 * dim per sublayer]
normalized = LayerNorm(x) * (1 + gamma) + beta           [scale and shift]
output = x + alpha * sublayer_output                      [gated residual]
```
The `alpha` gate (initialized to zero) means at init, each sublayer is a no-op. Training gradually opens the gates.

**Why window=32 (not 64 like encoder):** The decoder processes content that has already been contextualized by the encoder's 64-frame window. A shorter decoder window (32 frames = 1.28s) is sufficient for local prosody decisions (pitch contour, timing, energy envelope) without redundantly re-attending to distant context.

#### Stage 3: Upsample (25Hz → 50Hz)

```
repeat_interleave(2)  — duplicate each frame
CausalConv1d(512, 512, kernel=5)       [1,311,232 params]
ChannelLayerNorm(512)                  [1,024 params]
GELU activation
```

**Why repeat-then-smooth (not transposed conv):** Transposed convolutions create checkerboard artifacts. Nearest-neighbor upsampling (repeat_interleave) followed by a causal smoothing conv produces clean interpolation without artifacts. The conv learns to distribute the duplicated information across the 2x time expansion.

**Why 50Hz intermediate:** The waveform chain expects 50Hz input (matching the original log-mel frame rate). 50Hz → 44.1kHz requires 882x upsampling, factored as 7×7×3×3×2 across 5 stages.

#### Stage 4: Local Refinement ResNet (50Hz)

```
2x CausalResNetBlock(512, kernel=5)    [5,249,024 params]
  dilations: (1, 3)
  residual_scale: 0.3
```

Each block:
```
x → ChannelLayerNorm → GELU → CausalConv1d(dilation=d) → ChannelLayerNorm → GELU → CausalConv1d(dilation=1) → scale(0.3) → + x
```

**Why ResNet here (between transformer and waveform chain):** The transformer operates in a semantic space (content + speaker = acoustic plan). The waveform chain operates in a signal space (samples). The ResNet bridges these — it refines the 50Hz representation with local context (dilations 1 and 3 give receptive fields of 5 and 13 frames = 100ms and 260ms) before the signal-level upsampling begins.

**Why residual_scale=0.3:** Prevents training instability from large residual magnitudes in deep networks. The scaled residual means early training is dominated by the identity path; the refinement branch contributes gradually as its parameters learn useful transformations.

#### Stage 5: Waveform Upsample Chain (50Hz → 44.1kHz)

```
5 CausalUpsampleStages:                [19,620,608 params total]

  Stage A: 512 → 384, factor 7    (50Hz → 350Hz)
  Stage B: 384 → 256, factor 7    (350Hz → 2450Hz)
  Stage C: 256 → 128, factor 3    (2450Hz → 7350Hz)
  Stage D: 128 → 96,  factor 3    (7350Hz → 22050Hz)
  Stage E: 96  → 64,  factor 2    (22050Hz → 44100Hz)

Product: 7 × 7 × 3 × 3 × 2 = 882x
```

Each CausalUpsampleStage:
```
x → repeat_interleave(factor) → CausalConv1d(kernel=2*factor+1) → ChannelLayerNorm → Snake1d → CausalMRF
```

**CausalMRF (Multi-Receptive-Field fusion):**
```
For each kernel_size in (3, 7):
  branch: x → [Norm → Snake → DilatedCausalConv → Norm → Snake → CausalConv] × 3 dilations (1,3,5)
Mean of all branches → ChannelLayerNorm → FiLM(speaker_embedding)
```

**Why progressive upsampling (not single-step):** A single 882x upsample would require either a massive transposed convolution or generate severe aliasing. Progressive stages let each level focus on its frequency band: early stages (7x) generate pitch/formant structure, later stages (3x, 2x) add fine harmonic detail and noise characteristics.

**Why these specific factors (7,7,3,3,2):** Chosen so that intermediate rates correspond to meaningful signal bands. 350Hz captures fundamental frequency; 2450Hz captures first formants; 7350Hz captures most consonant energy; 22050Hz is the Nyquist for 44.1kHz. The factorization 882 = 7×7×3×3×2 minimizes the maximum single-step factor (7) while hitting exact 44100Hz.

**Why Snake1d activation:** `x + sin(alpha*x)^2 / alpha` — a periodic activation that naturally models voiced speech harmonics. Learned `alpha` per channel adapts the activation period to different frequency bands at each stage. Standard activations (ReLU, GELU) have no periodic inductive bias and require more parameters to learn harmonic structure.

**Why FiLM conditioning at each stage:** Speaker identity affects all frequency bands (fundamental frequency, formant positions, spectral tilt, breathiness). FiLM (Feature-wise Linear Modulation: `x * (1 + scale) + shift`) from the 128d speaker embedding is the cheapest way to inject global conditioning without expanding the channel dimension.

**Why channel reduction (512→384→256→128→96→64):** Higher-rate stages process more samples per content frame. Reducing channels compensates — total FLOPs per content frame stay roughly constant across stages rather than exploding at high sample rates.

#### Stage 6: Output Head

```
ChannelLayerNorm(64)
Snake1d(64)
CausalConv1d(64 → 1, kernel=7)        [641 params]
tanh()
```

**Why tanh output:** Bounds output to [-1, 1] (standard PCM float range). The output conv is initialized near-zero (std=1e-3) so the model starts outputting near-silence and gradually learns to synthesize audio — prevents early-training instability from large random waveforms.

---

## VoiceBank

**File:** `astrape/voicebank.py`  
**Format:** `.astrape` (binary, 48-byte header + 512-byte embedding + JSON metadata)

The VoiceBank stores a 128d speaker embedding extracted by MioCodec's global encoder from a 5+ second reference recording. At inference time, this embedding is the only information about the target speaker.

**Why 128d:** This is MioCodec's native global embedding dimension. It captures speaker identity (timbre, pitch range, speaking style) in a space where cosine distance correlates with perceptual speaker similarity.

**Why .astrape format (not just .npz):** Fixed-offset binary allows mmap-based zero-copy reads for the embedding (48 bytes → seek → 512 bytes). The streaming pipeline reads only the embedding at startup; metadata (quality metrics, provenance) is parsed lazily. NPZ requires decompressing the entire ZIP container.

---

## FSQ (Finite Scalar Quantization)

**File:** `astrape/fsq.py`  
**Levels:** (8, 8, 8, 5, 5) — 12,800 codes

### How It Works

1. `proj_in` maps transformer output (384d) to 5 continuous values
2. `bound`: tanh scaling constrains each axis to its level range (e.g., [-3.5, 3.5] for level 8)
3. `quantize`: round to nearest integer with straight-through gradient (gradient flows through the round as if it were identity)
4. `normalize`: divide by half-level to produce values in [-1, 1]
5. `proj_out` maps the 5d normalized code to 768d content space

### Why This Is The Speaker Separation Mechanism

Speaker identity lives in continuous subspaces of acoustic features — formant frequencies differ by 50-200Hz between speakers, pitch varies continuously, spectral tilt is a scalar. None of these can survive quantization to 12,800 discrete codes spread across 5 axes.

Phonetic content, conversely, is inherently categorical — /b/ vs /p/ vs /d/ are discrete distinctions. A 12,800-code space has far more capacity than needed for the phonetic inventory, so phonetic information passes through unharmed.

This is the same insight behind VQ-VAE and neural audio codecs: discrete bottlenecks destroy speaker while preserving content.

---

## Streaming Pipeline

**File:** `astrape/streaming_pipeline.py`

### StreamingVoiceConverter

Stateful wrapper that chains:
1. `StreamingLogMel.forward_stream()` — PCM → mel, maintaining waveform buffer
2. `CausalContentEncoder.forward_stream()` — mel → content, maintaining conv caches + KV cache + pending frame
3. `CausalSynthesisDecoder.forward_stream()` — content → PCM, maintaining conv caches + KV cache + waveform stage states

Each component maintains its own state dataclass. The pipeline can process arbitrarily small PCM chunks (even single samples) and will emit output audio as soon as sufficient input has accumulated.

### OutputRingBuffer

CPU-side circular buffer that bridges the model's output (variable-size chunks aligned to content frames = 1764 samples each) to the audio driver's callback (fixed-size requests, typically 256-1024 samples). Handles underrun/overrun counting for diagnostics.

---

## Training Pipeline

### Phase 0: Decoder Pretraining

**Script:** `train_decoder_phase0.py`  
**Trains:** CausalSynthesisDecoder only  
**Input:** Ground-truth teacher content embeddings (768d @ 25Hz) + speaker embedding + raw waveform  
**Loss:** Multi-scale mel spectrogram loss (L1 on mel at FFT sizes 512, 1024, 2048)

**Why train decoder first:** The decoder can be trained against ground-truth content embeddings from the teacher (MioCodec). This decouples decoder quality from encoder quality — the decoder learns synthesis before the encoder exists.

### Phase 1: Encoder Warm-Up

**Script:** `train_encoder_phase1.py`  
**Trains:** CausalContentEncoder only  
**Input:** Log-mel spectrograms  
**Target:** Teacher MioCodec content embeddings (768d @ 25Hz)  
**Loss:** MSE + cosine distance (weighted 1.0 + 0.1)

**Why warm-start:** Gives the encoder a reasonable initialization before the harder SSL training phase. The FSQ bottleneck needs its input distribution to be in a learnable regime before switching to the WavLM target.

### Phase SSL: Encoder Main Training

**Script:** `train_encoder_ssl.py`  
**Trains:** CausalContentEncoder only  
**Input:** Log-mel spectrograms  
**Target:** WavLM layers 6+9 averaged, pooled to 25Hz  
**Loss:** Cosine distance + 0.3 * L1  
**Prerequisite:** `extract_wavlm_targets.py` (pre-extracts WavLM features to disk)

**Why WavLM[6,9] as target (not teacher MioCodec):** WavLM's intermediate layers (6 and 9) are empirically content-rich and speaker-poor — this has been established by probing studies across the SSL literature. Training the encoder to predict these features through the FSQ bottleneck forces it to learn content representations that are inherently speaker-independent, rather than imitating the teacher's potentially speaker-contaminated space.

**Why pre-extract to disk:** WavLM is 95M parameters and non-causal — running it per batch during training would dominate compute cost and require the full WavLM model in GPU memory alongside the encoder. Pre-extraction runs once and training reads cheap .npz files.

---

## File Map

```
astrape/
  __init__.py            Package exports
  encoder.py             CausalContentEncoder, EncoderConfig, ContentOutput
  decoder.py             CausalSynthesisDecoder, SynthesisDecoderConfig
  wave_decoder.py        CausalConv1d, CausalMRF, CausalUpsampleStage, Snake1d, FiLM, DirectWaveDecoder (legacy)
  fsq.py                 FSQ helpers, DEFAULT_LEVELS, fit_fsq_projection
  audio.py               StreamingLogMel
  voicebank.py           VoiceBank (.astrape format), reference quality analysis
  data.py                MioContentDataset, speaker_disjoint_split
  streaming_pipeline.py  StreamingVoiceConverter, OutputRingBuffer

train_encoder_phase1.py  Phase 1: encoder warm-up against teacher
train_encoder_ssl.py     SSL phase: encoder against WavLM[6,9]
train_decoder_phase0.py  Phase 0: decoder against ground-truth content
extract_wavlm_targets.py Pre-extract WavLM features to NPZ cache

external/MioCodec/       Teacher codec (200M params, non-causal, 25Hz, 44.1kHz)
voicebanks/              Speaker embedding files (.astrape)
checkpoints/             Model checkpoints (.pt)
data/                    Training data cache (mio_vctk_full_compact/)
```
