# Astrape VC

Research code for a causal, zero-shot voice-conversion pipeline distilled from
MioCodec.

## Current Pipeline

```text
16 kHz source PCM
  -> streaming log-mel, 50 Hz
  -> strictly causal ContentStudent
  -> content embedding, 768d at 25 Hz
  -> CausalMelDecoder + cached target global embedding, 128d
  -> mel, 80 bins at 25 Hz
```

The source-to-mel path supports stateful streaming and has regression tests
that compare chunked output against full-sequence output. Waveform synthesis is
still a research dependency: `demo_v2.py` can use MioCodec's offline teacher
decoder, while a production causal vocoder has not yet been trained.

## Models

- `astrape.model.ContentStudent`: left-padded causal convolutions, causal
  attention, aligned 50 Hz to 25 Hz downsampling, streaming state.
- `astrape.mel_decoder.CausalMelDecoder`: source-restored AdaLN-Zero decoder
  matching `checkpoints/causal_mel_decoder.pt`.
- `astrape.audio.StreamingLogMel`: exact `center=False` full/chunked log-mel
  extraction for 16 kHz PCM.

Existing `checkpoints/causal_student_v3_4k.pt` was trained by the old
symmetrically padded architecture. It is therefore treated as a legacy weight
file and requires `--allow-legacy` or `--import-legacy`. Fine-tune it with the
new causal architecture before reporting causal quality.

## Training

```bash
# Standard 384d model
.venv/bin/python train_v3_4k.py

# Short run with a separate checkpoint name
.venv/bin/python train_v3_4k_mini.py

# Configured capacity tier
.venv/bin/python train_xhigh.py --tier xhigh --device mps

# Causal mel decoder
.venv/bin/python train_mel_decoder.py --target-mode teacher
```

Training uses speaker-disjoint validation, aligned even-frame crops, masked
variable-length losses, deterministic seeds, full validation, versioned
checkpoints, and separate `.best.pt`/`.last.pt` files.

To import the historical student weights:

```bash
.venv/bin/python train_v3_4k.py \
  --import-legacy checkpoints/causal_student_v3_4k.pt
```

## Extraction

```bash
.venv/bin/python extract_4k.py \
  --vctk-root /path/to/VCTK/wav48_silence_trimmed
```

Extraction randomly samples utterances per speaker with a fixed seed and stores
speaker names, utterance IDs, and source paths in `meta.npz`. MioCodec is an
optional external dependency required for extraction and teacher decoding.

## Inference And Benchmarking

```bash
# Incremental content + mel inference on cached data
.venv/bin/python stream_infer.py \
  --mel data/mio_4k_mel/m_00000.npz \
  --target data/mio_4k/s_00001.npz \
  --checkpoint checkpoints/content_student_v3_4k_causal.best.pt \
  --mel-decoder checkpoints/causal_mel_decoder.pt

# Synchronized accelerator benchmark
.venv/bin/python bench_dim.py --device mps

# Offline waveform comparison through the MioCodec teacher decoder
.venv/bin/python demo_v2.py \
  --source source.wav --reference target.wav
```

Benchmark timings synchronize MPS/CUDA before and after every measurement and
report full-sequence latency, streaming latency per 25 Hz content frame, and
real-time factor.

## Tests

```bash
.venv/bin/python -m unittest discover -v
```

The suite covers causal prefix invariance, streaming equivalence, log-mel
streaming, speaker-disjoint splitting, crop alignment, padding masks,
checkpoint compatibility, decoder loading, and tier construction.
