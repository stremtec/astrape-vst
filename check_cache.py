"""Cache integrity checker + auto-repair for btrv5 datasets.

Checks all NPZ and WavLM CNN cache files. Reports issues.
With --repair, auto-regenerates broken/missing files from source audio.

Usage:
  .venv/bin/python check_cache.py           # check only
  .venv/bin/python check_cache.py --repair  # check + fix broken files
  .venv/bin/python check_cache.py --wavlm-only  # check only WavLM CNN cache
"""
import sys, warnings, logging, argparse, time
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np
from pathlib import Path

def check_npz(data_dir, repair=False, srcs=None, meta=None):
    """Check all s_XXXXX.npz files for required keys."""
    expected = {'logmel', 'ce_768', 'ct'}
    n = int(meta['n_samples']) if meta else 43885
    broken, missing_keys = [], []
    
    for i in range(n):
        path = data_dir / f's_{i:05d}.npz'
        if not path.exists():
            broken.append((i, 'file missing'))
            continue
        try:
            data = np.load(path, allow_pickle=False)
            missing = expected - set(data.keys())
            if missing:
                missing_keys.append((i, ', '.join(missing)))
                if path.stat().st_size < 100:
                    broken.append((i, f'too small ({path.stat().st_size}B)'))
        except Exception as e:
            broken.append((i, str(e)))
            if repair and srcs is not None:
                path.unlink(missing_ok=True)
    
    return broken, missing_keys

def check_wavlm(data_dir, repair=False, srcs=None, meta=None, subdir='wavlm_L4_200hz'):
    """Check all <subdir>/s_XXXXX.npy WavLM cache files.

    The integrity check (exists / loads / 2-D / 512-channel / not truncated) is
    rate-agnostic, so it covers any of wavlm_16k, wavlm_L4, wavlm_L4_200hz.
    Pass --wavlm-dir to point it at the cache the encoder actually trains on.
    """
    wavlm_dir = Path(subdir) if Path(subdir).is_absolute() else data_dir / subdir
    if not wavlm_dir.exists():
        return [(0, f'{subdir} directory missing')], []
    n = int(meta['n_samples']) if meta else 43885
    broken, missing = [], []
    
    for i in range(n):
        path = wavlm_dir / f's_{i:05d}.npy'
        if not path.exists():
            missing.append(i)
            continue
        try:
            d = np.load(path, allow_pickle=False)
            if d.ndim != 2 or d.shape[1] != 512:
                broken.append((i, f'bad shape {d.shape}'))
                if repair: path.unlink(missing_ok=True)
            if path.stat().st_size < 100:
                broken.append((i, f'too small ({path.stat().st_size}B)'))
                if repair: path.unlink(missing_ok=True)
        except Exception as e:
            broken.append((i, str(e)))
            if repair: path.unlink(missing_ok=True)
    
    return broken, missing

def repair_files(broken, missing, data_dir, srcs, subdir='wavlm_L4_200hz'):
    """Auto-repair broken WavLM cache files (L4 raw 200Hz recipe).

    Regenerates with the SAME extraction as cache_wavlm_L4_raw.py — resample to
    16kHz, run the first 5 conv layers (L4, 200Hz), save (T, 512).  This matches
    the wavlm_L4_200hz cache the StridingAdapter encoder consumes (the previous
    recipe fed 44.1kHz audio through the full CNN + 3× avg-pool, which is wrong
    for every current cache).
    """
    if not (broken or missing):
        return
    print(f'\nRepairing {len(broken)} broken + {len(missing)} missing files...')
    from astrape.miocodec import load_mio, load_wave, SAMPLE_RATE
    import torch, torchaudio
    import torch.nn.functional as F

    mio = load_mio('cpu').eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor
    wavlm_dir = Path(subdir) if Path(subdir).is_absolute() else data_dir / subdir
    wavlm_dir.mkdir(parents=True, exist_ok=True)

    to_fix = set()
    for i, _ in broken:
        to_fix.add(i)
    for i in missing:
        to_fix.add(i)

    fixed = 0
    for i in sorted(to_fix):
        try:
            wav = load_wave(Path(str(srcs[i])), SAMPLE_RATE, max_seconds=6.0)
            wav_16 = torchaudio.functional.resample(
                wav.unsqueeze(0), SAMPLE_RATE, 16000).squeeze(0)
            with torch.no_grad():
                x = wav_16.unsqueeze(0)
                for layer_idx in range(5):
                    layer = fe.conv_layers[layer_idx]
                    x = layer.conv(x)
                    if hasattr(layer, 'layer_norm') and layer.layer_norm is not None:
                        x = (layer.layer_norm(x.unsqueeze(0)).squeeze(0)
                             if x.dim() == 2 else layer.layer_norm(x))
                    x = F.gelu(x)
            cnn = x.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)  # (T, 512)
            np.save(wavlm_dir / f's_{i:05d}.npy', cnn)
            fixed += 1
        except Exception as e:
            print(f'  FAILED s_{i:05d}: {e}')
        if fixed % 100 == 0:
            print(f'  {fixed}/{len(to_fix)}')

    print(f'  Repaired {fixed}/{len(to_fix)} files')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/mio_vctk_full_compact')
    ap.add_argument('--repair', action='store_true', help='Auto-repair broken files')
    ap.add_argument('--wavlm-only', action='store_true')
    ap.add_argument('--npz-only', action='store_true')
    ap.add_argument('--wavlm-dir', default='wavlm_L4_200hz',
                    help='WavLM cache subdir to check (relative to data-dir or absolute). '
                         'e.g. wavlm_L4_200hz (StridingAdapter), wavlm_L4, wavlm_16k')
    args = ap.parse_args()
    
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / 'meta.npz', allow_pickle=False)
    n = int(meta['n_samples'])
    srcs = meta['source_files'][:n].astype(str) if args.repair else None
    
    t0 = time.time()
    print(f'Checking {n} files at {data_dir}...')
    
    if not args.wavlm_only:
        broken_npz, missing_npz = check_npz(data_dir, args.repair, srcs, meta)
        print(f'  NPZ broken: {len(broken_npz)}')
        if broken_npz:
            for i, err in broken_npz[:5]:
                print(f'    s_{i:05d}: {err}')
            if len(broken_npz) > 5: print(f'    ... and {len(broken_npz) - 5} more')
        print(f'  NPZ missing keys: {len(missing_npz)}')
    else:
        broken_npz, missing_npz = [], []
    
    if not args.npz_only:
        broken_wl, missing_wl = check_wavlm(data_dir, args.repair, srcs, meta, args.wavlm_dir)
        print(f'  WavLM ({args.wavlm_dir}) broken: {len(broken_wl)}')
        if broken_wl:
            for i, err in broken_wl[:5]:
                print(f'    s_{i:05d}: {err}')
            if len(broken_wl) > 5: print(f'    ... and {len(broken_wl) - 5} more')
        print(f'  WavLM missing: {len(missing_wl)}')
        if missing_wl:
            print(f'    Range: s_{min(missing_wl):05d} .. s_{max(missing_wl):05d}')

        if args.repair and (broken_wl or missing_wl):
            repair_files(broken_wl, missing_wl, data_dir, srcs, args.wavlm_dir)
    else:
        broken_wl, missing_wl = [], []
    
    total = len(broken_npz) + len(broken_wl) + len(missing_wl)
    elapsed = time.time() - t0
    status = '✅ All clean' if total == 0 else f'⚠️ {total} issues'
    print(f'\n{status} ({elapsed:.1f}s)')
    return 1 if total > 0 else 0

if __name__ == '__main__':
    sys.exit(main())
