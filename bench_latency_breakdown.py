#!/usr/bin/env python3
"""Forensic end-to-end latency accounting for the Astrape pipeline.

This measures per-component cost on the actual current architecture
(TokenSynchronousStudent, current_quality preset) on the requested device,
then projects per-frame costs for candidate upgrades that we've been
discussing but never measured quantitatively in the latency budget:

   A. current_quality      (existing: 4 edge + 8 core, core_dim 512, ctx 50)
   B. wider_core           (core_dim 768, ff 2304, ctx 50)
   C. wider_core_larger_ctx(core_dim 768, ctx 200)
   D. xhigh_dim            (core_dim 1024, ff 3072, ctx 50)
   E. deeper_core          (4 edge + 12 core, core_dim 512)
   F. deeper_core2         (4 edge + 14 core, core_dim 512)

It also measures the *streaming step* (forward_stream on 2 mel frames = 40ms)
end-to-end vs the equivalent two mel-frame sync forward, the wall-clock cost
of a single core GRU cell, and the cost blanket-estimates for: multi-scale
(2 calls/40ms), deeper GRU (1-layer cost * L), and the wave_decoder call.

The output is one JSON whose keys are component-level p50/p95 estimates and
preset-level total-latency projections, plus an instantaneous text summary
suitable as the response.

Run inside .venv-mio:
   .venv-mio/bin/python bench_latency_breakdown.py --device mps --repeats 80
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

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

def time_fn(fn: Callable[[], None], device: torch.device, repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def percentile(values: list[float], q: float) -> float:
    vs = sorted(values)
    if not vs:
        return float("nan")
    return vs[min(len(vs) - 1, math.ceil(len(vs) * q) - 1)]


def summarize(values: list[float]) -> dict[str, float]:
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": n,
        "mean": statistics.fmean(vs),
        "p50": vs[min(n - 1, math.ceil(n * 0.50) - 1)],
        "p95": vs[min(n - 1, math.ceil(n * 0.95) - 1)],
        "p99": vs[min(n - 1, math.ceil(n * 0.99) - 1)],
        "min": vs[0],
        "max": vs[-1],
    }


def profile_preset(
    label: str,
    config: TokenStudentConfig,
    device: torch.device,
    chunk_mel_frames: int = 2,
    repeats: int = 80,
    warmup: int = 12,
) -> dict:
    """End-to-end + per-component breakdown for a single preset."""
    model = TokenSynchronousStudent(config).to(device).eval()

    # trivial warmup (also primes any runtime caches)
    @torch.inference_mode()
    def streaming_chunk(state: list[Optional[TokenStreamingState]]) -> None:
        mel = torch.randn(1, config.in_dim, chunk_mel_frames, device=device)
        _, state[0] = model.forward_stream(mel, state[0])

    state: list[Optional[TokenStreamingState]] = [None]
    for _ in range(warmup):
        streaming_chunk(state)

    # ---- (1) full streaming step ----
    step_times = time_fn(lambda: streaming_chunk(state), device, repeats)

    # ---- (2) per-component costs using isolated forward on primed state ----
    # Prime state once then probe each component.
    @torch.inference_mode()
    def prime(step_count: int = 3) -> TokenStreamingState:
        s: Optional[TokenStreamingState] = None
        for _ in range(step_count):
            mel = torch.randn(1, config.in_dim, chunk_mel_frames, device=device)
            _, s = model.forward_stream(mel, s)
        return s

    primed = prime(3)

    # 2a: input_project (norm + linear)
    @torch.inference_mode()
    def fn_input() -> None:
        mel = torch.randn(1, config.in_dim, chunk_mel_frames, device=device)
        _ = model.input_projection(model.input_norm(mel.transpose(1, 2)))
    input_times = time_fn(fn_input, device, repeats)

    # 2b: streaming the four edge blocks (sequential, with cache)
    @torch.inference_mode()
    def fn_edges() -> None:
        mel = torch.randn(1, config.in_dim, chunk_mel_frames, device=device)
        local_state = TokenStreamingState(
            edge_caches=[None] * len(model.edge_blocks),
            core_caches=[None] * len(model.core_blocks),
            attention_histories=[None] * len(model.core_blocks),
        )
        hidden = model.input_projection(model.input_norm(mel.transpose(1, 2)))
        for block_idx, block in enumerate(model.edge_blocks):
            hidden, local_state.edge_caches[block_idx] = block.forward_stream(
                hidden, local_state.edge_caches[block_idx]
            )
    edge_times = time_fn(fn_edges, device, repeats)

    # 2c: core_projection + recurrent(GRU)
    @torch.inference_mode()
    def fn_gru() -> None:
        x = torch.randn(1, chunk_mel_frames, config.core_dim, device=device)
        h0 = torch.randn(1, 1, config.core_dim, device=device)
        _y, _ = model.recurrent(x, h0)
    gru_times = time_fn(fn_gru, device, repeats)

    # 2d: a single core conv block (no attention)
    conv_block = next((b for b in model.core_blocks if b.attention is None), None)
    @torch.inference_mode()
    def fn_conv() -> None:
        x = torch.randn(1, chunk_mel_frames, config.core_dim, device=device)
        _ = conv_block(x, None) if conv_block is not None else x
    conv_times = time_fn(fn_conv, device, repeats) if conv_block else [0.0]

    # 2e: a single core attn block
    attn_block = next((b for b in model.core_blocks if b.attention is not None), None)
    @torch.inference_mode()
    def fn_attn() -> None:
        x = torch.randn(1, chunk_mel_frames, config.core_dim, device=device)
        _ = attn_block(x, None) if attn_block is not None else x
    attn_times = time_fn(fn_attn, device, repeats) if attn_block else [0.0]

    # 2f: heads (output_norm + code/ordinal/future/text/fsq_proj)
    @torch.inference_mode()
    def fn_heads() -> None:
        h = torch.randn(1, chunk_mel_frames, config.core_dim, device=device)
        _ = model._heads(h, None)
    heads_times = time_fn(fn_heads, device, repeats)

    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    n_attn_blocks = sum(1 for b in model.core_blocks if b.attention is not None)
    n_conv_blocks = len(model.core_blocks) - n_attn_blocks

    # also: how many forward_stream calls happen per *content* frame?
    #  current 25 Hz content @ 50 Hz mel => 1 forward per content frame already
    #  multi-scale proposal => 2 forwards per 40ms mel chunk (12.5 Hz + 25 Hz)
    #  multi-scale dual => 4 forwards per 40ms

    # also 2-mel-frame batch (40ms of audio, no shift)
    @torch.inference_mode()
    def fn_two_frames() -> None:
        mel = torch.randn(1, config.in_dim, 4, device=device)
        x = mel[:, :, :2]
        y = mel[:, :, 2:]
        nonlocal_state = primed
        _, ns = model.forward_stream(x, nonlocal_state)
        _, ns = model.forward_stream(y, ns)
    two_frame_times = time_fn(fn_two_frames, device, repeats)

    # ---- assemble ----
    return {
        "label": label,
        "config": asdict(config),
        "params_M": n_params_M,
        "n_attn_blocks": n_attn_blocks,
        "n_conv_blocks": n_conv_blocks,
        "stream_p50": summarize(step_times)["p50"],
        "stream_p95": summarize(step_times)["p95"],
        "stream_p99": summarize(step_times)["p99"],
        "stream_mean": summarize(step_times)["mean"],
        "two_frame_p50": summarize(two_frame_times)["p50"],
        "two_frame_p95": summarize(two_frame_times)["p95"],
        "two_frame_mean": summarize(two_frame_times)["mean"],
        "components": {
            "input_proj_" + label: summarize(input_times),
            "edge_4x_" + label: summarize(edge_times),
            "gru_" + label: summarize(gru_times),
            "core_conv_block_" + label: summarize(conv_times),
            "core_attn_block_" + label: summarize(attn_times),
            "heads_" + label: summarize(heads_times),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--output", default="docs/research/latency_budget_v3.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device} repeats={args.repeats} warmup={args.warmup}")

    base_cfg = TokenStudentConfig(
        edge_dim=384, core_dim=512, edge_layers=4, core_layers=8,
        n_heads=8, ff_hidden=1536, attention_context=50,
        core_kernel=9, edge_kernel=5, content_dim=768,
    )
    presets = [
        ("A_current_quality", base_cfg),
        ("B_wider_core_768", TokenStudentConfig(
            edge_dim=384, core_dim=768, edge_layers=4, core_layers=8,
            n_heads=12, ff_hidden=2304, attention_context=50,
            core_kernel=9, edge_kernel=5, content_dim=768,
        )),
        ("C_wider_core_larger_ctx_200", TokenStudentConfig(
            edge_dim=384, core_dim=768, edge_layers=4, core_layers=8,
            n_heads=12, ff_hidden=2304, attention_context=200,
            core_kernel=9, edge_kernel=5, content_dim=768,
        )),
        ("D_xhigh_dim_1024", TokenStudentConfig(
            edge_dim=512, core_dim=1024, edge_layers=4, core_layers=8,
            n_heads=16, ff_hidden=3072, attention_context=50,
            core_kernel=9, edge_kernel=5, content_dim=768,
        )),
        ("E_deeper_core_12", TokenStudentConfig(
            edge_dim=384, core_dim=512, edge_layers=4, core_layers=12,
            n_heads=8, ff_hidden=1536, attention_context=50,
            core_kernel=9, edge_kernel=5, content_dim=768,
        )),
        ("F_deeper_core_14", TokenStudentConfig(
            edge_dim=384, core_dim=512, edge_layers=4, core_layers=14,
            n_heads=8, ff_hidden=1536, attention_context=50,
            core_kernel=9, edge_kernel=5, content_dim=768,
        )),
    ]

    results: list[dict] = []
    for name, cfg in presets:
        print(f"... preset {name}")
        r = profile_preset(name, cfg, device=device, repeats=args.repeats, warmup=args.warmup)
        results.append(r)

    out_path = args.output
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- Console summary (one-line per preset) ----
    print()
    print(f"{'preset':<28} {'Params':>9} {'blocks':>10} "
          f"{'stream p50':>11} {'stream p95':>11} {'2fr p95':>10}")
    for r in results:
        blocks = f"{r['n_attn_blocks']}A+{r['n_conv_blocks']}C"
        print(f"{r['label']:<28} {r['params_M']:7.2f}M {blocks:>10} "
              f"{r['stream_p50']:10.2f}ms {r['stream_p95']:10.2f}ms "
              f"{r['two_frame_p95']:9.2f}ms")

    print()
    print("Component breakdown (per microservice -- mean ms):")
    print(f"{'preset':<28} {'input':>8} {'edges4':>8} {'gru':>8} "
          f"{'conv-blk':>9} {'attn-blk':>9} {'heads':>8}")
    for r in results:
        c = r["components"]
        print(
            f"{r['label']:<28} "
            f"{c['input_proj_' + r['label']]['mean']:7.2f} "
            f"{c['edge_4x_' + r['label']]['mean']:7.2f} "
            f"{c['gru_' + r['label']]['mean']:7.2f} "
            f"{c['core_conv_block_' + r['label']]['mean']:8.2f} "
            f"{c['core_attn_block_' + r['label']]['mean']:8.2f} "
            f"{c['heads_' + r['label']]['mean']:7.2f}"
        )
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
