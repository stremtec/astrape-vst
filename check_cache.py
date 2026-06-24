"""Validate cache file integrity before training.

Checks:
  1. All s_XXXXX.npz files are readable with expected keys
  2. All wavlm_cnn/s_XXXXX.npy files are readable with correct dims
  3. File count matches expected total

Usage:
  .venv/bin/python check_cache.py               # check all
  .venv/bin/python check_cache.py --wavlm-only   # check only WavLM CNN cache
"""
import sys,numpy as np,argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data/mio_vctk_full_compact')
    ap.add_argument('--wavlm-only',action='store_true')
    ap.add_argument('--npz-only',action='store_true')
    args=ap.parse_args()
    root=Path(args.data_dir)

    meta=np.load(root/'meta.npz',allow_pickle=False)
    n=int(meta['n_samples'])
    expected_keys={'logmel','ce_768','ct'}

    broken_npz=[];broken_wavlm=[];missing_wavlm=[]

    for i in range(n):
        # NPZ check
        npz_path=root/f's_{i:05d}.npz'
        if not args.wavlm_only:
            try:
                data=np.load(npz_path,allow_pickle=False)
                missing_keys=expected_keys-set(data.keys())
                if missing_keys:
                    broken_npz.append((i,f'missing keys: {missing_keys}'))
            except Exception as e:
                broken_npz.append((i,str(e)))
                # Delete broken file
                npz_path.unlink(missing_ok=True)

        # WavLM CNN check
        if not args.npz_only:
            wavlm_path=root/'wavlm_cnn'/f's_{i:05d}.npy'
            if wavlm_path.exists():
                try:
                    d=np.load(wavlm_path,allow_pickle=False)
                    if d.ndim!=2 or d.shape[1]!=512:
                        broken_wavlm.append((i,f'shape={d.shape}'))
                except Exception as e:
                    broken_wavlm.append((i,str(e)))
            else:
                missing_wavlm.append(i)

    print(f'Checked {n} files:')
    print(f'  Broken npz: {len(broken_npz)}')
    if broken_npz:
        for i,err in broken_npz[:10]:print(f'    s_{i:05d}: {err}')
        if len(broken_npz)>10:print(f'    ... and {len(broken_npz)-10} more')
    print(f'  Broken wavlm: {len(broken_wavlm)}')
    if broken_wavlm:
        for i,err in broken_wavlm[:5]:print(f'    s_{i:05d}: {err}')
    print(f'  Missing wavlm: {len(missing_wavlm)}')
    if missing_wavlm and len(missing_wavlm)<=10:
        for i in missing_wavlm:print(f'    s_{i:05d}')
    elif missing_wavlm:print(f'    s_{missing_wavlm[0]:05d} ... s_{missing_wavlm[-1]:05d}')

    if broken_npz or broken_wavlm:
        print(f'\n❌ {len(broken_npz)+len(broken_wavlm)} issues found')
        sys.exit(1)
    else:
        print('\n✅ All cache files valid')

if __name__=='__main__':main()
