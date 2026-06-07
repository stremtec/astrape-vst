"""
Cached AudioDec dataset for fast training (btrv3lite-style).

Loads pre-computed AudioDec latents instead of encoding in real-time.
"""

from __future__ import annotations
import os, glob, random, torch
from pathlib import Path
from collections import defaultdict


class CachedAudioDecDataset(torch.utils.data.Dataset):
    """Loads pre-cached AudioDec latents with speaker grouping."""

    def __init__(self, cache_dir: str, device: str = "cpu"):
        self.cache_dir = cache_dir
        self.device = device
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        
        # Group by speaker
        self.speaker_files = defaultdict(list)
        for f in self.files:
            try:
                data = torch.load(f, map_location="cpu", weights_only=True)
                spk = data.get("speaker_id", "unknown")
                self.speaker_files[spk].append(f)
            except Exception:
                pass
        
        self.speakers = list(self.speaker_files.keys())
        print(f"  CachedDataset: {len(self.files)} files, {len(self.speakers)} speakers")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 50% same speaker, 50% cross speaker (btrv3lite-style)
        src_path = self.files[idx]
        src_data = torch.load(src_path, map_location="cpu", weights_only=True)
        src_spk = src_data.get("speaker_id", "unknown")
        
        if random.random() < 0.5:
            # Same speaker (identity anchor)
            tgt_path = random.choice([f for f in self.speaker_files[src_spk] if f != src_path])
            tgt_data = torch.load(tgt_path, map_location="cpu", weights_only=True)
            is_cross = False
        else:
            # Cross speaker
            tgt_spk = random.choice([s for s in self.speakers if s != src_spk])
            tgt_path = random.choice(self.speaker_files[tgt_spk])
            tgt_data = torch.load(tgt_path, map_location="cpu", weights_only=True)
            is_cross = True
        
        return {
            "z_src": src_data["z"].squeeze(0),  # (T, 64)
            "z_tgt": tgt_data["z"].squeeze(0),
            "is_cross": is_cross,
        }


def collate_cached(batch):
    """Pad to max length."""
    max_len = max(b["z_src"].size(0) for b in batch)
    z_src = torch.stack([
        torch.nn.functional.pad(b["z_src"], (0, 0, 0, max_len - b["z_src"].size(0)))
        for b in batch
    ])
    z_tgt = torch.stack([
        torch.nn.functional.pad(b["z_tgt"], (0, 0, 0, max_len - b["z_tgt"].size(0)))
        for b in batch
    ])
    return {"z_src": z_src, "z_tgt": z_tgt}
