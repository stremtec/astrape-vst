"""
FlowVC streaming inference with overlap-add for smooth chunk boundaries.
"""

from __future__ import annotations
import argparse, time, torch, torch.nn.functional as F, torchaudio

from .encoder import F3Encoder, make_encoder
from .decoder import F3Decoder
from .converter import VectorFieldNet, solve_cfm_euler, make_vector_field_net
from .speaker import SpeakerEncoder, make_speaker_encoder
from .prosody import FCPEProsodyExtractor, make_prosody_extractor
from .config import DecoderConfig


class FlowVCInference:
    """Streaming VC with ring buffer + overlap-add."""

    def __init__(self, encoder, decoder, vfn, speaker_enc, prosody_ext,
                 device="cpu", chunk_ms=80, left_ctx_ms=320, overlap_ms=40, ode_steps=4):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self.vfn = vfn.to(self.device).eval()
        self.speaker_enc = speaker_enc.to(self.device).eval()
        self.prosody_ext = prosody_ext.to(self.device).eval()

        sr = 44100
        self.chunk_samples = int(sr * chunk_ms / 1000)
        self.left_ctx_samples = int(sr * left_ctx_ms / 1000)
        self.overlap_samples = int(sr * overlap_ms / 1000)
        self.ode_steps = ode_steps

        # ring buffer: stores past audio for left context
        self.buffer = torch.zeros(1, 1, self.left_ctx_samples, device=self.device)

        # speaker state
        self.speaker_emb = None
        self.prompt_tokens = None

        # overlap state
        self.prev_tail = None  # tail of previous output for crossfade

        self.stats = {"chunks": 0, "total_ms": 0.0}

    def set_target_speaker(self, ref_audio: torch.Tensor):
        if ref_audio.dim() == 2:
            ref_audio = ref_audio.unsqueeze(0)
        ref_audio = ref_audio.to(self.device)
        with torch.no_grad():
            self.speaker_emb, self.prompt_tokens = self.speaker_enc(ref_audio)
        print(f"  speaker set: emb={self.speaker_emb.shape}")

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.speaker_emb is None:
            raise RuntimeError("call set_target_speaker() first")

        t0 = time.time()

        chunk = chunk.to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Cold-start: on first chunk, fill buffer with audio to avoid silence context
        if self.stats["chunks"] == 0:
            self.buffer = chunk.repeat(1, 1, max(1, self.buffer.shape[2] // max(chunk.shape[2], 1)))[:, :, :self.buffer.shape[2]]

        # Prepend left context from ring buffer
        ctx = torch.cat([self.buffer, chunk], dim=-1)  # (1, 1, ctx+T)

        with torch.no_grad():
            # Encode with context
            z = self.encoder.encode(ctx)

            # Update ring buffer: append chunk, truncate to left_ctx_samples
            new_buf = torch.cat([self.buffer, chunk], dim=-1)
            self.buffer = new_buf[:, :, -self.left_ctx_samples:]

            # Prosody on context window
            prosody = self.prosody_ext(ctx)
            if prosody.size(1) != z.size(1):
                n = min(z.size(1), prosody.size(1))
                z, prosody = z[:, :n, :], prosody[:, :n, :]

            # CFM ODE
            z_tgt = solve_cfm_euler(self.vfn, z, self.speaker_emb, self.prompt_tokens, prosody, self.ode_steps)

            # Decode
            out = self.decoder(z_tgt, self.speaker_emb)  # (1, 1, T_out)

        # Extract only the chunk portion (discard context tail)
        chunk_out_len = self.chunk_samples
        if out.shape[2] > chunk_out_len:
            out = out[:, :, -chunk_out_len:]

        # Overlap-add with previous tail
        result = out.squeeze(0).squeeze(0)  # (T_out,)
        if self.prev_tail is not None and self.overlap_samples > 0:
            ov = self.overlap_samples
            # crossfade: linear ramp
            ramp_out = torch.linspace(0, 1, ov, device=self.device)
            ramp_prev = torch.linspace(1, 0, ov, device=self.device)
            result[:ov] = result[:ov] * ramp_out + self.prev_tail[-ov:] * ramp_prev

        # Save tail for next chunk
        if self.overlap_samples > 0:
            self.prev_tail = result[-self.overlap_samples:].clone()

        elapsed = (time.time() - t0) * 1000
        self.stats["chunks"] += 1
        self.stats["total_ms"] += elapsed

        return result

    def process_stream(self, audio: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, audio.shape[-1], self.chunk_samples):
            end = min(start + self.chunk_samples, audio.shape[-1])
            chunk = audio[start:end]
            if chunk.shape[-1] < self.chunk_samples:
                chunk = F.pad(chunk, (0, self.chunk_samples - chunk.shape[-1]))
            out = self.process_chunk(chunk)
            outputs.append(out)
        result = torch.cat(outputs, dim=-1)
        return result[:audio.shape[-1]]

    @property
    def avg_latency_ms(self):
        return self.stats["total_ms"] / max(self.stats["chunks"], 1)

    @property
    def rtf(self):
        if self.stats["chunks"] == 0:
            return 0.0
        audio_ms = self.stats["chunks"] * self.chunk_samples / 44100 * 1000
        return self.stats["total_ms"] / audio_ms


def load_models(checkpoint_path: str, device: str):
    """Load all models from checkpoint with explicit error on missing keys."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    required = ["encoder", "decoder"]
    for key in required:
        if key not in ckpt:
            raise KeyError(f"Missing '{key}' in checkpoint. Available: {list(ckpt.keys())}")

    encoder = make_encoder().to(device)
    decoder = F3Decoder(DecoderConfig()).to(device)
    vfn = make_vector_field_net().to(device)
    speaker_enc = make_speaker_encoder().to(device)
    prosody = make_prosody_extractor(device=device).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    vfn.load_state_dict(ckpt.get("vfn", vfn.state_dict()))  # fallback to random if missing
    speaker_enc.load_state_dict(ckpt.get("speaker_enc", speaker_enc.state_dict()))
    prosody.load_state_dict(ckpt.get("prosody", prosody.state_dict()))
    return encoder, decoder, vfn, speaker_enc, prosody


def convert_file(checkpoint: str, src: str, ref: str, out: str, device="cpu", ode_steps=4):
    encoder, decoder, vfn, speaker_enc, prosody = load_models(checkpoint, device)
    engine = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody, device=device, ode_steps=ode_steps)

    src_wav, sr = torchaudio.load(src)
    ref_wav, rsr = torchaudio.load(ref)
    if sr != 44100:
        src_wav = torchaudio.functional.resample(src_wav, sr, 44100)
    if rsr != 44100:
        ref_wav = torchaudio.functional.resample(ref_wav, rsr, 44100)
    if src_wav.shape[0] > 1:
        src_wav = src_wav.mean(dim=0, keepdim=True)
    if ref_wav.shape[0] > 1:
        ref_wav = ref_wav.mean(dim=0, keepdim=True)

    engine.set_target_speaker(ref_wav)
    out_wav = engine.process_stream(src_wav.squeeze(0))
    torchaudio.save(out, out_wav.unsqueeze(0).cpu(), 44100)
    print(f"  saved: {out}  chunks={engine.stats['chunks']}  avg_lat={engine.avg_latency_ms:.1f}ms  RTF={engine.rtf:.3f}")


def profile(device="cpu", chunk_ms=80, ode_steps=4):
    dev = torch.device(device)
    chunk_samples = int(44100 * chunk_ms / 1000)
    from .converter import make_vector_field_net
    from .speaker import make_speaker_encoder
    from .prosody import make_prosody_extractor
    encoder = make_encoder().to(dev).eval()
    decoder = F3Decoder(DecoderConfig()).to(dev).eval()
    vfn = make_vector_field_net().to(dev).eval()
    speaker_enc = make_speaker_encoder().to(dev).eval()
    prosody = make_prosody_extractor(device=str(dev)).to(dev).eval()

    wav = torch.randn(1, 1, chunk_samples + 320 * 44100 // 1000, device=dev)
    ref = torch.randn(1, 1, 44100, device=dev)
    with torch.no_grad():
        spk_emb, prompt = speaker_enc(ref)

    # Warmup
    for _ in range(30):
        with torch.no_grad():
            z = encoder.encode(wav)
            p = prosody(wav)
            _ = solve_cfm_euler(vfn, z, spk_emb, prompt, p, n_steps=ode_steps)
            _ = decoder(z, spk_emb)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    times = {}
    N = 50
    for name, fn in [
        ("encoder", lambda: encoder.encode(wav)),
        ("prosody", lambda: prosody(wav)),
        ("cfm_ode", lambda: solve_cfm_euler(vfn, z, spk_emb, prompt, p, n_steps=ode_steps)),
        ("decoder", lambda: decoder(z, spk_emb)),
    ]:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times[name] = (time.time() - t0) / N * 1000

    total = sum(times.values())
    print(f"\n  === FlowVC profile ({device}, {chunk_ms}ms, {ode_steps}-step ODE) ===")
    for name, t in times.items():
        pct = t / total * 100
        print(f"  {name:12s} {t:6.1f} ms  ({pct:4.1f}%)  {'█' * int(pct/2)}")
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':12s} {total:6.1f} ms")
    print(f"  RTF: {total/chunk_ms:.3f}")


def main():
    p = argparse.ArgumentParser(description="FlowVC inference")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--src", type=str, default="")
    p.add_argument("--ref", type=str, default="")
    p.add_argument("--out", type=str, default="output.wav")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--chunk-ms", type=int, default=80)
    p.add_argument("--ode-steps", type=int, default=4)
    args = p.parse_args()

    if args.profile:
        profile(args.device, args.chunk_ms, args.ode_steps)
    elif args.checkpoint and args.src and args.ref:
        convert_file(args.checkpoint, args.src, args.ref, args.out, args.device, args.ode_steps)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
