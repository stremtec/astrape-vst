"""Quick VC listening test for MCS-Trans Q2D2 model."""
import sys, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
sys.path.insert(0, "external/MioCodec/src")

import torch, torchaudio, argparse
import numpy as np
from pathlib import Path

from astrape.miocodec import load_mio, load_wave, SAMPLE_RATE
from train_mcs_q2d2 import MCSTransQ2D2Config, MCSTransQ2D2

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEL_HOP = 882  # 50Hz at 44.1kHz


def compute_mel(wav: torch.Tensor) -> torch.Tensor:
    """Match the compact cache mel extraction pipeline."""
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=MEL_HOP,
        n_mels=80,
        f_min=0.0,
        f_max=SAMPLE_RATE / 2.0,
        power=1,
        center=True,
    ).to(wav.device)
    mel = mel_t(wav)
    return torch.log(torch.clamp(mel, min=1e-5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="Source audio (wav/flac/mp3)")
    parser.add_argument("--target", required=True, help="Target voicebank (.astrape) or ref audio (.wav)")
    parser.add_argument("--output", default="/tmp/vc_output.wav")
    parser.add_argument("--max-seconds", type=float, default=6.0)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {})
    config = MCSTransQ2D2Config(**{
        k: tuple(v) if isinstance(v, list) else v
        for k, v in saved_cfg.items()
    })
    model = MCSTransQ2D2(config).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    print(f"Loading MioCodec ...", flush=True)
    mio = load_mio(DEVICE)
    mio.eval()

    # Load source audio
    src_path = Path(args.source)
    print(f"Source: {src_path}", flush=True)
    src_wav = load_wave(src_path, SAMPLE_RATE, args.max_seconds).to(DEVICE)

    # Encode through MCS — prefer cache mel if available
    mel = None
    cache_data_dir = Path("data/mio_vctk_full_compact")
    if cache_data_dir.exists():
        meta = np.load(cache_data_dir / "meta.npz", allow_pickle=False)
        source_files = meta["source_files"].astype(str)
        for i, sf_p in enumerate(source_files):
            if Path(sf_p).resolve() == src_path.resolve():
                npz_path = cache_data_dir / f"s_{i:05d}.npz"
                if npz_path.exists():
                    data = np.load(npz_path, allow_pickle=False)
                    mel = torch.from_numpy(data["logmel"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                    print(f"Using cache mel: {npz_path.name} ({mel.shape[2]} frames)", flush=True)
                    T_content = min(mel.shape[2] // 2, int(args.max_seconds * 25))
                    mel = mel[:, :, :T_content * 2]
                    break

    if mel is None:
        mel = compute_mel(src_wav.unsqueeze(0))
        T_content = min(mel.shape[2] // 2, int(args.max_seconds * 25))
        mel = mel[:, :, :T_content * 2]
        print(f"Computed mel: {mel.shape}", flush=True)

    mask = torch.ones(1, T_content, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        out = model(mel, padding_mask=mask)
    content = out["projected"]  # (1, 768, T)
    print(f"Content: {content.shape}", flush=True)

    # Load target voicebank
    tgt_path = Path(args.target)
    if tgt_path.suffix == ".astrape":
        from astrape.voicebank import VoiceBank
        vb = VoiceBank.load(tgt_path)
        global_emb = vb.global_embedding.float().to(DEVICE)
        print(f"VoiceBank: {vb.source_path or tgt_path.name}", flush=True)
    else:
        tgt_wav = load_wave(tgt_path, SAMPLE_RATE, 5.0).to(DEVICE)
        with torch.no_grad():
            feats = mio.encode(tgt_wav.unsqueeze(0), return_content=False, return_global=True)
        global_emb = feats.global_embedding
        print(f"Target from audio: {tgt_wav.shape[0]/SAMPLE_RATE:.1f}s", flush=True)

    # Decode
    content_t = content.transpose(1, 2)  # (1, T, 768)
    stft_len = mio._calculate_target_stft_length(src_wav.numel())
    with torch.no_grad():
        output_wav = mio.forward_wave(
            content_t, global_emb.unsqueeze(0),
            stft_length=stft_len,
        ).squeeze(0).cpu()
    print(f"Output: {output_wav.shape} ({output_wav.shape[0]/SAMPLE_RATE:.1f}s)", flush=True)

    # Save
    import soundfile as sf
    sf.write(args.output, output_wav.cpu().numpy(), SAMPLE_RATE)
    src_out = args.output.replace(".wav", "_source.wav")
    src_np = src_wav.cpu().numpy()
    sf.write(src_out, src_np, SAMPLE_RATE)
    print(f"Saved: {args.output}", flush=True)
    print(f"Source ref: {src_out}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
