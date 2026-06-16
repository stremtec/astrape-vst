#!/usr/bin/env python3
"""Amortized per-chunk latency for the streaming pipeline.

The naive per-call perf_counter() measurement is dominated by MPS kernel-launch
latency and the synchronisation round-trip. In the production streaming loop
CPU-side enqueue overlaps with GPU execution -- what matters is **wall-clock
throughput** measured across many forward_stream calls bracketed by a single
torch.mps.synchronize().

We replicate that here:

    synchronize
    for k in range(K):
        forward_stream(mel_chunk)
    synchronize
    elapsed = ...

Then per_chunk = elapsed / K, p95 = upper-95%-of-K-blocks of mini-batch wall.

We also probe a single forward_stream per bracket to capture *cold* launch
overhead separately.

Run:
  .venv-mio/bin/python bench_latency_amortized.py --device mps --inner 50 --blocks 40
"""
from __future__ import annotations

import argparse
import json
import os
import math
import statistics
from dataclasses import dataclass

import torch

from astrape.token_student import (
    TokenStudentConfig,
    TokenStreamingState,
    TokenSynchronousStudent,
)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, q):
    vs = sorted(values)
    if not vs:
        return float("nan")
    return vs[min(len(vs) - 1, math.ceil(len(vs) * q) - 1)]


@dataclass
class Preset:
    label: str
    cfg: TokenStudentConfig


PRESETS = [
    Preset("A_current", TokenStudentConfig(
        edge_dim=384, core_dim=512, edge_layers=4, core_layers=8,
        n_heads=8, ff_hidden=1536, attention_context=50,
        core_kernel=9, edge_kernel=5, content_dim=768,
    )),
    Preset("B_wider768", TokenStudentConfig(
        edge_dim=384, core_dim=768, edge_layers=4, core_layers=8,
        n_heads=12, ff_hidden=2304, attention_context=50,
        core_kernel=9, edge_kernel=5, content_dim=768,
    )),
    Preset("C_wider768_ctx200", TokenStudentConfig(
        edge_dim=384, core_dim=768, edge_layers=4, core_layers=8,
        n_heads=12, ff_hidden=2304, attention_context=200,
        core_kernel=9, edge_kernel=5, content_dim=768,
    )),
    Preset("D_xhigh1024", TokenStudentConfig(
        edge_dim=512, core_dim=1024, edge_layers=4, core_layers=8,
        n_heads=16, ff_hidden=3072, attention_context=50,
        core_kernel=9, edge_kernel=5, content_dim=768,
    )),
    Preset("E_deeper12", TokenStudentConfig(
        edge_dim=384, core_dim=512, edge_layers=4, core_layers=12,
        n_heads=8, ff_hidden=1536, attention_context=50,
        core_kernel=9, edge_kernel=5, content_dim=768,
    )),
    Preset("F_deeper14", TokenStudentConfig(
        edge_dim=384, core_dim=512, edge_layers=4, core_layers=14,
        n_heads=8, ff_hidden=1536, attention_context=50,
        core_kernel=9, edge_kernel=5, content_dim=768,
    )),
]


def profile_preset(p: Preset, device, inner=50, blocks=40, warmup=20):
    model = TokenSynchronousStudent(p.cfg).to(device).eval()
    n_params_M = sum(x.numel() for x in model.parameters()) / 1e6
    n_attn = sum(1 for b in model.core_blocks if b.attention is not None)
    n_conv = len(model.core_blocks) - n_attn

    @torch.inference_mode()
    def warm(state):
        for _ in range(warmup):
            mel_buf.copy_(torch.randn(1, p.cfg.in_dim, 2, device=device))
            _, state[0] = model.forward_stream(mel_buf, state[0])
    state = [None]
    mel_buf = torch.empty(1, p.cfg.in_dim, 2, device=device)
    warm(state)

    @torch.inference_mode()
    def block_run(state, k):
        for _ in range(k):
            mel_buf.copy_(torch.randn(1, p.cfg.in_dim, 2, device=device))
            _, state[0] = model.forward_stream(mel_buf, state[0])

    mel_buf = torch.empty(1, p.cfg.in_dim, 2, device=device)

    # ------ 1. block-amortized per-chunk wall time (single sync per block) ------
    block_per_chunk: list[float] = []
    for _ in range(blocks):
        # decay state small amount each block so cache remains warm but ok
        synchronize(device)
        t0 = time.perf_counter()
        block_run(state, inner)
        synchronize(device)
        block_per_chunk.append((time.perf_counter() - t0) * 1000.0 / inner)

    # ------ 2. single-step latency (one sync per call) — measures kernel-launch ------
    @torch.inference_mode()
    def one_run(state):
        mel_buf.copy_(torch.randn(1, p.cfg.in_dim, 2, device=device))
        _, state[0] = model.forward_stream(mel_buf, state[0])

    single = []
    for _ in range(blocks):
        synchronize(device)
        t0 = time.perf_counter()
        one_run(state)
        synchronize(device)
        single.append((time.perf_counter() - t0) * 1000.0)

    # ------ 3. ground-truth: pipeline run on 80 chunks at once ------
    @torch.inference_mode()
    def long_run(state, k):
        for _ in range(k):
            mel_buf.copy_(torch.randn(1, p.cfg.in_dim, 2, device=device))
            _, state[0] = model.forward_stream(mel_buf, state[0])

    k_long = 80
    synchronize(device)
    t0 = time.perf_counter()
    long_run(state, k_long)
    synchronize(device)
    long_per_chunk = (time.perf_counter() - t0) * 1000.0 / k_long

    return {
        "label": p.label,
        "params_M": n_params_M,
        "n_attn": n_attn,
        "n_conv": n_conv,
        "core_dim": p.cfg.core_dim,
        "core_layers": p.cfg.core_layers,
        "ff_hidden": p.cfg.ff_hidden,
        "attention_context": p.cfg.attention_context,

        "block_p50_per_chunk_ms": percentile(block_per_chunk, 0.50),
        "block_p95_per_chunk_ms": percentile(block_per_chunk, 0.95),
        "block_p99_per_chunk_ms": percentile(block_per_chunk, 0.99),
        "block_mean_per_chunk_ms": statistics.fmean(block_per_chunk),

        "single_p50_ms": percentile(single, 0.50),
        "single_p95_ms": percentile(single, 0.95),
        "single_mean_ms": statistics.fmean(single),

        "long_run_per_chunk_ms": long_per_chunk,
        "inner_per_block": inner,
    }


import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inner", type=int, default=50,
                        help="chunks per amortized measurement block")
    parser.add_argument("--blocks", type=int, default=40,
                        help="number of amortized measurement blocks")
    parser.add_argument("--output", default="docs/research/latency_amortized.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"device={device} inner={args.inner} blocks={args.blocks}")
    rows = []
    for p in PRESETS:
        print(f"... {p.label}")
        rows.append(profile_preset(p, device, args.inner, args.blocks))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)

    print()
    print(f"{'preset':<20} {'Params':>8} {'dim':>5} {'lyrs':>5} "
          f"{'block p50':>11} {'block p95':>11} {'block p99':>11} "
          f"{'single p50':>11} {'single p95':>11} "
          f"{'long80':>8}")
    for r in rows:
        print(f"{r['label']:<20} {r['params_M']:7.2f}M {r['core_dim']:>5} {r['core_layers']:>5} "
              f"{r['block_p50_per_chunk_ms']:10.2f}ms {r['block_p95_per_chunk_ms']:10.2f}ms "
              f"{r['block_p99_per_chunk_ms']:10.2f}ms "
              f"{r['single_p50_ms']:10.2f}ms {r['single_p95_ms']:10.2f}ms "
              f"{r['long_run_per_chunk_ms']:7.2f}ms")

    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
