# Astrape VC

Real-time (<100ms), zero-shot voice conversion at 44.1kHz. Strictly causal encoder-decoder architecture distilled from MioCodec.

## Pipeline

```
16kHz source PCM
  → StreamingLogMel (50Hz, 80-bin)
  → CausalContentEncoder (50Hz → 25Hz, FSQ bottleneck, 12.3M params)
  → 768d content @ 25Hz (speaker-stripped)
  → CausalSynthesisDecoder (25Hz → 44.1kHz, speaker-conditioned, 42M params)
  → 44.1kHz PCM
```

Speaker identity comes from a 128d VoiceBank embedding extracted once from a 5+ second reference recording using MioCodec's global encoder.

## Architecture

See [Architecture.md](Architecture.md) for full structural documentation with parameter counts, stage-by-stage rationale, and design constraints.

## Training

Three-phase training pipeline. Each phase trains one component in isolation:

```bash
# Phase 0: Decoder learns synthesis from ground-truth teacher content
.venv/bin/python train_decoder_phase0.py --device mps

# Phase 1: Encoder warm-up against teacher MioCodec embeddings
.venv/bin/python train_encoder_phase1.py --device mps

# Pre-extract WavLM targets (requires external drive with VCTK audio)
.venv/bin/python extract_wavlm_targets.py \
  --audio-root /Volumes/UNTITLED/asill-storage/research2/datasets/vctk \
  --resume

# SSL phase: Encoder predicts WavLM[6,9] through FSQ bottleneck
.venv/bin/python train_encoder_ssl.py --device mps
```

Training data: 43,885 VCTK samples cached in `data/mio_vctk_full_compact/`.

## Project Structure

```
astrape/
  encoder.py           CausalContentEncoder (mel → FSQ → 768d content)
  decoder.py           CausalSynthesisDecoder (content + speaker → 44.1kHz)
  wave_decoder.py      Shared building blocks (CausalConv1d, MRF, Snake1d, FiLM)
  fsq.py               Finite Scalar Quantization (8,8,8,5,5 = 12,800 codes)
  audio.py             StreamingLogMel
  voicebank.py         VoiceBank format (.astrape / legacy .npz)
  data.py              Dataset and speaker-disjoint splits
  streaming_pipeline.py  End-to-end streaming runtime

external/MioCodec/     Teacher codec (200M params, non-causal)
checkpoints/           Model weights
data/                  Training cache
voicebanks/            Speaker profiles (.astrape)
```

## VoiceBank

A VoiceBank stores a single 128d speaker embedding from a 5+ second reference recording. Format v3 (`.astrape`) uses a fixed 48-byte binary header + raw float32 embedding + JSON metadata. Legacy `.npz` files remain loadable.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Requirements

Python 3.10+, PyTorch 2.2+, torchaudio, numpy, scipy, soundfile.

MioCodec teacher inference requires the separate `.venv-mio` environment.
