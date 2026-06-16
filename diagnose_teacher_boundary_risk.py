#!/usr/bin/env python3
"""Measure teacher latent temporal self-similarity against frame lag,
to evaluate the hypothesis "boundary inconsistency on random 300-frame crops
is a major noise source limiting convergence."
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path('/Users/asill/btrv5/data/mio_vctk_full_compact')
N_FILES = 500
KEYS = ('pre_fsq_768', 'ce_768')
LAGS = (0, 1, 2, 5, 10, 25, 50, 75, 100, 150, 200)


def autocorr_curve(x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    x = x - x.mean(0, keepdims=True)
    f_full = np.fft.rfft(x, axis=0)
    spec = f_full * np.conj(f_full)
    R = np.fft.irfft(spec, axis=0).real / (n * d)
    return R.sum(axis=-1)  # (n, )


def main() -> None:
    files = sorted(OUT_DIR.glob('s_*.npz'))[:N_FILES]
    curves = {key: defaultdict(list) for key in KEYS}

    for f in files:
        try:
            with np.load(f) as d:
                for key in KEYS:
                    if key not in d.files:
                        continue
                    arr = d[key].astype(np.float32)
                    T = arr.shape[0]
                    if T < max(LAGS) + 5:
                        continue
                    R = autocorr_curve(arr)
                    norm = R[0]
                    if norm < 1e-6:
                        continue
                    for lag in LAGS:
                        if lag < T:
                            curves[key][lag].append(R[lag] / norm)
        except Exception:
            continue

    print(f'{"lag":>5}  {"pre_fsq":>8}  {"ce_768":>8}    (ms @ 25Hz = 40 ms / frame)')
    for lag in LAGS:
        row = []
        for key in KEYS:
            arr = np.asarray(curves[key][lag])
            row.append('     n/a' if arr.size == 0 else f'{arr.mean():8.4f}')
        print(f'{lag:5d}  ' + '  '.join(row) + f'    {lag * 40}ms')


if __name__ == '__main__':
    main()
