"""Training script for Codex VC."""

from __future__ import annotations

import os
import random
import time

import torch
import torch.nn as nn
from moshi.models import loaders
from pathlib import Path

from codex_vc.model import CodeGenerator, compute_loss

# ── Config ──────────────────────────────────────────────────────────────
CACHE_PATH = "runs/vctk_codes_full.pt"
SPK_PATH = "runs/vctk_full_spk.pt"
MODEL_PATH = "runs/codex_model.pt"

BATCH_SIZE = 32
STEPS = 300
LR = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
# ─────────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    spk_emb = torch.load(SPK_PATH, weights_only=True)
    cache = torch.load(CACHE_PATH, weights_only=True)
    print(f"  {len(spk_emb)} speaker embeddings, {len(cache)} cached codes")

    # Normalize time dimension
    T = min(c.shape[2] for c in cache.values())
    cache = {k: v[:, :, :T] for k, v in cache.items()}
    print(f"  T = {T}")

    # Available speakers
    speakers = sorted({k[0] for k in cache} & set(spk_emb.keys()))
    print(f"  {len(speakers)} speakers with both codes + embeddings")

    # Build training pairs (same text, different speakers)
    pairs = []
    all_utts = sorted({k[1] for k in cache})
    for u in all_utts:
        spks_with = [s for s in speakers if (s, u) in cache]
        if len(spks_with) < 2:
            continue
        for s in spks_with:
            for t in spks_with:
                if s != t:
                    pairs.append((s, t, u))
    random.shuffle(pairs)
    print(f"  {len(pairs)} training pairs")

    # Model
    model = CodeGenerator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    print()
    print(f"Training {STEPS} steps (batch={BATCH_SIZE})...")
    t0 = time.time()
    best_acc = 0.0

    for step in range(STEPS):
        random.shuffle(pairs)
        step_pairs = pairs[: min(len(pairs), 2000)]  # limit per step
        total_loss = 0.0
        total_acc = 0.0

        for b_start in range(0, len(step_pairs), BATCH_SIZE):
            batch = step_pairs[b_start : b_start + BATCH_SIZE]
            B = len(batch)

            # Prepare batch tensors
            lv0_list = []
            lv1_7_list = []
            spk_list = []
            for s, t, u in batch:
                lv0_list.append(cache[(s, u)][0, 0])       # (T,)
                lv1_7_list.append(cache[(t, u)][0, 1:])    # (7, T)
                spk_list.append(spk_emb[t])                 # (256,)

            lv0 = torch.stack(lv0_list).to(device).long()
            lv1_7 = torch.stack(lv1_7_list).to(device).long()
            spk = torch.stack(spk_list).to(device)

            # Forward + backward
            loss = compute_loss(model, lv0, lv1_7, spk, criterion)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item() * B
            with torch.no_grad():
                pred = model.predict(lv0, spk)
                total_acc += (pred == lv1_7).float().mean().item() * B

        # Epoch metrics
        N = len(step_pairs)
        avg_loss = total_loss / N
        avg_acc = total_acc / N

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), MODEL_PATH)

        if step % 30 == 0 or step == STEPS - 1:
            elapsed = time.time() - t0
            print(
                f"  step {step:4d}: loss={avg_loss:.4f} acc={avg_acc:.4f} "
                f"best={best_acc:.4f} [{elapsed:.0f}s]"
            )

    print()
    print(f"Done. Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
