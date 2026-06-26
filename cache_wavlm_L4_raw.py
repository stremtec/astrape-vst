"""Cache WavLM CNN features at L4 (5 layers, raw 200Hz, NO pooling).

Saves 512d @ 200Hz. Downstream adapter handles rate conversion.
Delay: 160 samples @16kHz = 10ms.

Usage: .venv/bin/python cache_wavlm_L4_raw.py --limit 0
"""
import sys, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
sys.path.insert(0, 'external/MioCodec/src')

import torch, torchaudio, numpy as np, argparse
from pathlib import Path
from eval_mcs_trans_audio import load_mio, load_wave, SAMPLE_RATE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/mio_vctk_full_compact')
    ap.add_argument('--max-s', type=float, default=6.0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    meta = np.load(Path(args.data_dir) / 'meta.npz', allow_pickle=False)
    n = int(meta['n_samples'])
    srcs = meta['source_files'][:n].astype(str)
    end = n if args.limit == 0 else min(n, args.start + args.limit)
    print(f'Caching WavLM L4 raw (200Hz): {args.start}..{end-1} ({end-args.start} total)')

    mio = load_mio('cpu').eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor
    out_dir = Path(args.data_dir) / 'wavlm_L4_200hz'
    out_dir.mkdir(exist_ok=True)

    for i in range(args.start, end):
        out_path = out_dir / f's_{i:05d}.npy'
        if out_path.exists(): continue
        wav = load_wave(Path(str(srcs[i])), SAMPLE_RATE, max_seconds=args.max_s)
        wav_16 = torchaudio.functional.resample(wav.unsqueeze(0), SAMPLE_RATE, 16000).squeeze(0)

        with torch.no_grad():
            x = wav_16.unsqueeze(0)
            for layer_idx in range(5):
                layer = fe.conv_layers[layer_idx]
                x = layer.conv(x)
                if hasattr(layer, 'layer_norm') and layer.layer_norm is not None:
                    x = layer.layer_norm(x.unsqueeze(0)).squeeze(0) if x.dim()==2 else layer.layer_norm(x)
                x = torch.nn.functional.gelu(x)

        cnn = x.squeeze(0)  # (512, T_200Hz)
        cnn = cnn.transpose(0, 1)  # (T_200Hz, 512)
        cnn_np = cnn.cpu().numpy().astype(np.float32)
        np.save(out_path, cnn_np)

        if (i - args.start) % 100 == 0:
            print(f'{i - args.start}/{end - args.start}')

    print(f'Done: {end - args.start} samples')

if __name__ == '__main__':
    main()
