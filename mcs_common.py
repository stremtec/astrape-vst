from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_DATA_DIR = Path("data/mio_vctk_full_compact")
DEFAULT_PROJECTION = Path("checkpoints/teacher_fsq_proj_out.pt")
FSQ_LEVELS = (8, 8, 8, 5, 5)


@dataclass
class Batch:
    mel: torch.Tensor            # (B, 80, M)
    content: torch.Tensor        # (B, L, 768)  teacher target
    tokens: torch.Tensor         # (B, L)        integer codes
    mask: torch.Tensor           # (B, L)        bool
    speakers: list[str]
    indices: torch.Tensor
    crop_starts: torch.Tensor
    ssl_L0: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # (B, Lx2, 768)
    ssl_L4: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    ssl_L8: torch.Tensor = field(default_factory=lambda: torch.empty(0))


class MioCompactDataset(Dataset):
    def __init__(self, root: Path, indices: np.ndarray, speakers: np.ndarray):
        self.root = root
        self.indices = [int(i) for i in indices.tolist()]
        self.speakers = speakers

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        idx = self.indices[item]
        with np.load(self.root / f"s_{idx:05d}.npz", allow_pickle=False) as data:
            ssl0 = data.get("ssl_L0")
            ssl4 = data.get("ssl_L4")
            ssl8 = data.get("ssl_L8")
            return {
                "idx": idx,
                "speaker": str(self.speakers[idx]),
                "mel": torch.from_numpy(data["logmel"].astype(np.float32)),
                "content": torch.from_numpy(data["ce_768"].astype(np.float32)),
                "tokens": torch.from_numpy(data["ct"].astype(np.int64)),
                "ssl_L0": torch.from_numpy(ssl0.astype(np.float32)) if ssl0 is not None else torch.empty(0,768),
                "ssl_L4": torch.from_numpy(ssl4.astype(np.float32)) if ssl4 is not None else torch.empty(0,768),
                "ssl_L8": torch.from_numpy(ssl8.astype(np.float32)) if ssl8 is not None else torch.empty(0,768),
            }


class ContentCollator:
    def __init__(self, mel_frames: int | None, seed: int, pad_mel_multiple: int = 2,
                 frames_per_token: int = 2):
        self.mel_frames = mel_frames
        self.rng = random.Random(seed)
        self.pad_mel_multiple = pad_mel_multiple
        # Frontend ("mel") frames per teacher token (25Hz content).  Mel and the
        # 50Hz WavLM cache are 2:1; the 200Hz L4 raw cache is 8:1.  Crops must use
        # this ratio to keep the cropped frontend window aligned with the cropped
        # content/token window (a hard-coded 2 overruns and mis-pairs at 200Hz).
        self.frames_per_token = frames_per_token

    def _crop(self, sample: dict) -> tuple:
        mel = sample["mel"]
        content = sample["content"]
        tokens = sample["tokens"]
        ssl0 = sample.get("ssl_L0", torch.empty(0,768))
        ssl4 = sample.get("ssl_L4", torch.empty(0,768))
        ssl8 = sample.get("ssl_L8", torch.empty(0,768))
        idx = int(sample["idx"])
        if self.mel_frames is None or mel.shape[1] <= self.mel_frames:
            return mel, content, tokens, ssl0, ssl4, ssl8, 0, idx

        R = self.frames_per_token
        max_start = mel.shape[1] - self.mel_frames
        start = self.rng.randint(0, max_start)
        start -= start % R                       # align crop to a token boundary
        mel = mel[:, start : start + self.mel_frames]
        token_start = start // R
        token_len = math.ceil(mel.shape[1] / R)
        ssl_start = token_start * 2              # SSL features are 50Hz = 2× token rate
        ssl_len = token_len * 2
        return (
            mel,
            content[token_start : token_start + token_len],
            tokens[token_start : token_start + token_len],
            ssl0[ssl_start : ssl_start + ssl_len] if ssl0.numel()>0 else ssl0,
            ssl4[ssl_start : ssl_start + ssl_len] if ssl4.numel()>0 else ssl4,
            ssl8[ssl_start : ssl_start + ssl_len] if ssl8.numel()>0 else ssl8,
            start,
            idx,
        )

    def __call__(self, samples: list[dict]) -> Batch:
        cropped = [self._crop(sample) for sample in samples]
        max_mel = max(mel.shape[1] for mel, _, _, _, _, _, _, _ in cropped)
        if self.pad_mel_multiple > 1:
            max_mel = ((max_mel + self.pad_mel_multiple - 1) // self.pad_mel_multiple) * self.pad_mel_multiple
        max_tokens = max(tokens.shape[0] for _, _, tokens, _, _, _, _, _ in cropped)
        max_ssl = max_tokens * 2

        mels, contents, tokens_out, masks = [], [], [], []
        ssl0s, ssl4s, ssl8s = [], [], []
        crop_starts, indices = [], []
        for mel, content, tokens, ssl0, ssl4, ssl8, crop_start, idx in cropped:
            token_len = min(tokens.shape[0], content.shape[0])
            mels.append(F.pad(mel, (0, max_mel - mel.shape[1])))
            contents.append(F.pad(content[:token_len], (0, 0, 0, max_tokens - token_len)))
            tokens_out.append(F.pad(tokens[:token_len], (0, max_tokens - token_len)))
            mask = torch.zeros(max_tokens, dtype=torch.bool)
            mask[:token_len] = True
            masks.append(mask)
            # SSL features: pad to max_ssl
            if ssl0.numel() > 0:
                sl = min(ssl0.shape[0], max_ssl)
                ssl0s.append(F.pad(ssl0[:sl], (0,0,0,max_ssl-sl)))
                ssl4s.append(F.pad(ssl4[:sl], (0,0,0,max_ssl-sl)))
                ssl8s.append(F.pad(ssl8[:sl], (0,0,0,max_ssl-sl)))
            else:
                ssl0s.append(torch.zeros(max_ssl, 768))
                ssl4s.append(torch.zeros(max_ssl, 768))
                ssl8s.append(torch.zeros(max_ssl, 768))
            crop_starts.append(crop_start)
            indices.append(idx)

        return Batch(
            mel=torch.stack(mels),
            content=torch.stack(contents),
            tokens=torch.stack(tokens_out),
            mask=torch.stack(masks),
            ssl_L0=torch.stack(ssl0s),
            ssl_L4=torch.stack(ssl4s),
            ssl_L8=torch.stack(ssl8s),
            speakers=[sample["speaker"] for sample in samples],
            indices=torch.tensor(indices, dtype=torch.long),
            crop_starts=torch.tensor(crop_starts, dtype=torch.long),
        )


def split_by_speaker(speakers: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(speakers.astype(str).tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_speakers = set(str(s) for s in unique[:n_val].tolist())
    train_idx, val_idx = [], []
    for idx, speaker in enumerate(speakers.astype(str).tolist()):
        (val_idx if speaker in val_speakers else train_idx).append(idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def speaker_balanced_subset(indices: np.ndarray, speakers: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    by_speaker: dict[str, list[int]] = {}
    for idx in indices.tolist():
        by_speaker.setdefault(str(speakers[idx]), []).append(int(idx))
    for values in by_speaker.values():
        rng.shuffle(values)
    selected: list[int] = []
    names = sorted(by_speaker)
    cursor = 0
    while len(selected) < min(n, len(indices)) and names:
        name = names[cursor % len(names)]
        if by_speaker[name]:
            selected.append(by_speaker[name].pop())
        else:
            names.remove(name)
            cursor -= 1
        cursor += 1
    return np.asarray(selected, dtype=np.int64)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    @property
    def left_context(self) -> int:
        return self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_context:
            x = F.pad(x, (self.left_context, 0))
        return super().forward(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = CausalConv1d(dim, dim, kernel, dilation=dilation)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = F.silu(self.conv(h))
        return residual + F.dropout(h, self.dropout, self.training)



class DepthwiseResidualBlock(nn.Module):
    """Causal depthwise separable conv block (WavTokenizer/ConvNeXt style).

    LayerNorm → DepthwiseConv1d(groups=dim) → SiLU → PointwiseConv1d → residual.
    Much more parameter-efficient than standard conv, enabling deeper stems.
    """
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = CausalConv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.depthwise(h)
        h = F.silu(h)
        h = self.pointwise(h)
        return residual + F.dropout(h, self.dropout, self.training)


class CellDownsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2:
            x = F.pad(x, (0, 1))
        return self.conv(x)


def indices_to_level_indices(indices: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    basis = []
    product = 1
    for level in levels:
        basis.append(product)
        product *= level
    basis_t = torch.tensor(basis, device=indices.device, dtype=torch.long)
    levels_t = torch.tensor(levels, device=indices.device, dtype=torch.long)
    return (indices.unsqueeze(-1) // basis_t) % levels_t


def indices_to_codes(indices: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    level_indices = indices_to_level_indices(indices, levels)
    half_width = torch.tensor([level // 2 for level in levels], device=indices.device, dtype=torch.float32)
    return (level_indices.float() - half_width) / half_width


def quantize_normalized_codes(codes: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    half_width = torch.tensor([level // 2 for level in levels], device=codes.device, dtype=codes.dtype)
    max_index = torch.tensor([level - 1 for level in levels], device=codes.device, dtype=codes.dtype)
    levels_f = (codes * half_width + half_width).round().clamp(min=0)
    levels_f = torch.minimum(levels_f, max_index)
    quantized = (levels_f - half_width) / half_width
    return codes + (quantized - codes).detach()


def _voiced_weights(mel: torch.Tensor, length: int, voiced_boost: float) -> torch.Tensor:
    if voiced_boost <= 1.0:
        return mel.new_ones(mel.shape[0], length)
    t_mel = mel.shape[2]
    # Frontend frames per content frame.  Mel and the 50Hz WavLM cache run at
    # 2× the content rate (factor=2); the 200Hz L4 raw cache runs at 8×.
    # Deriving the factor from the actual lengths keeps the voiced mask
    # time-aligned with the content frames for every frontend rate.  (Was
    # hard-coded to 2, which mis-mapped — and silently dropped 3/4 of — the
    # utterance when the 200Hz StridingAdapter frontend was used.)
    factor = max(1, int(round(t_mel / length)))
    t_tok = min(length, t_mel // factor)
    mel_groups = mel[:, :, : t_tok * factor].reshape(
        mel.shape[0], mel.shape[1], t_tok, factor
    )
    rms = mel_groups.pow(2).mean(dim=(1, 3)).sqrt()
    threshold = rms.mean(dim=1, keepdim=True).clamp(min=1e-5)
    voiced = (rms > threshold * 0.5).float()
    weights = 1.0 + (voiced_boost - 1.0) * voiced
    if t_tok < length:
        weights = torch.cat([weights, weights.new_ones(weights.shape[0], length - t_tok)], dim=1)
    return weights


def _adjacent_ce(logits: torch.Tensor, target: torch.Tensor, level: int) -> torch.Tensor:
    idx = torch.arange(logits.shape[1], device=logits.device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float()
    w_matrix = torch.where(dist == 1, torch.tensor(0.5, device=logits.device), torch.ones_like(dist))
    log_p = F.log_softmax(logits, dim=1)
    target_w = w_matrix[target]
    return -(target_w * log_p.transpose(1, 2)).sum(-1) / w_matrix[target].sum(-1).clamp(min=1e-5)


def _weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mask.any():
        return pred.sum() * 0.0
    per_frame = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    if weights is None:
        weights = torch.ones_like(per_frame)
    weighted = per_frame * weights * mask.float()
    return weighted.sum() / (weights * mask.float()).sum().clamp(min=1)


def masked_losses(
    output: dict,
    batch: Batch,
    levels: tuple[int, ...],
    args: argparse.Namespace,
    ordinal_weight: float | None = None,
) -> tuple[torch.Tensor, dict]:
    codes = output["codes"]
    projected = output["projected"]
    ordinal = output["ordinal"]
    assert isinstance(codes, torch.Tensor)
    assert isinstance(projected, torch.Tensor)

    ow = ordinal_weight if ordinal_weight is not None else args.ordinal_weight
    voiced_boost = getattr(args, "voiced_boost", 1.0)
    ord_mode = getattr(args, "ord_mode", "ce")

    length = min(codes.shape[1], batch.tokens.shape[1], batch.mask.shape[1], batch.content.shape[1])
    mask = batch.mask[:, :length]
    pred = codes[:, :length]
    target_codes = indices_to_codes(batch.tokens[:, :length], levels).to(pred.dtype)

    vw = _voiced_weights(batch.mel, length, voiced_boost)
    weighted_mask_sum = (vw * mask.float()).sum().clamp(min=1)

    cos = F.cosine_similarity(pred, target_codes, dim=-1)
    cos_loss = ((1.0 - cos) * vw * mask.float()).sum() / weighted_mask_sum
    axis_weights = torch.tensor(args.axis_weights, device=pred.device, dtype=pred.dtype)
    code_l1 = (((pred - target_codes).abs() * axis_weights.view(1, 1, -1)).mean(-1) * vw * mask.float()).sum() / weighted_mask_sum

    targets = indices_to_level_indices(batch.tokens[:, :length], levels)
    if ord_mode == "none" or ow == 0.0:
        half_w = torch.tensor([lv // 2 for lv in levels], device=pred.device, dtype=pred.dtype)
        pred_bins = (pred * half_w + half_w).round().clamp(0).long()
        pred_bins = torch.min(pred_bins, torch.tensor([lv - 1 for lv in levels], device=pred.device))
        accs = [(pred_bins[:, :, ax][mask] == targets[:, :, ax][mask]).float().mean() for ax in range(len(levels))]
        ord_acc = torch.stack(accs).mean()
        exact_acc = (pred_bins == targets).all(dim=-1)[mask].float().mean()
        ord_loss = codes.sum() * 0.0
    elif ord_mode == "unified":
        half_w = torch.tensor([lv // 2 for lv in levels], device=pred.device, dtype=pred.dtype)
        ce_losses, accs, preds = [], [], []
        for ax, lv in enumerate(levels):
            hw = half_w[ax]
            bin_centers = (torch.arange(lv, device=pred.device).float() - hw) / hw
            logits_ax = -(pred[:, :, ax].unsqueeze(-1) - bin_centers).pow(2) * 10.0
            logits_ax = logits_ax.transpose(1, 2)
            axis_target = targets[:, :, ax]
            pred_ax = logits_ax.argmax(dim=1)
            preds.append(pred_ax)
            ce_per = F.cross_entropy(logits_ax, axis_target, reduction="none")
            ce_losses.append((ce_per * vw * mask.float()).sum() / weighted_mask_sum)
            accs.append((pred_ax[mask] == axis_target[mask]).float().mean())
        ord_loss = torch.stack(ce_losses).mean()
        ord_acc = torch.stack(accs).mean()
        exact_acc = (torch.stack(preds, dim=-1) == targets).all(dim=-1)[mask].float().mean()
    else:
        assert isinstance(ordinal, tuple) and len(ordinal) > 0
        ce_losses, accs, preds = [], [], []
        for ax, (logits, lv) in enumerate(zip(ordinal, levels)):
            logits = logits[:, :, :length]
            axis_target = targets[:, :, ax]
            pred_ax = logits.argmax(dim=1)
            preds.append(pred_ax)
            ce_per = _adjacent_ce(logits, axis_target, lv) if ord_mode == "adjacent" else F.cross_entropy(logits, axis_target, reduction="none")
            ce_losses.append((ce_per * vw * mask.float()).sum() / weighted_mask_sum)
            accs.append((pred_ax[mask] == axis_target[mask]).float().mean())
        ord_loss = torch.stack(ce_losses).mean()
        ord_acc = torch.stack(accs).mean()
        exact_acc = (torch.stack(preds, dim=-1) == targets).all(dim=-1)[mask].float().mean()

    if length >= 2:
        delta_mask = mask[:, 1:] & mask[:, :-1]
        pred_delta = pred[:, 1:] - pred[:, :-1]
        target_delta = target_codes[:, 1:] - target_codes[:, :-1]
        delta_weights = 0.5 * (vw[:, 1:] + vw[:, :-1])
        delta = _weighted_smooth_l1(pred_delta, target_delta, delta_mask, delta_weights)
    else:
        delta = pred.sum() * 0

    pred_768 = projected[:, :, :length].transpose(1, 2)
    tgt_768 = batch.content[:, :length]
    cos768 = F.cosine_similarity(pred_768[mask], tgt_768[mask], dim=-1)

    if length >= 3:
        delta2_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        pred_delta2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        target_delta2 = target_codes[:, 2:] - 2 * target_codes[:, 1:-1] + target_codes[:, :-2]
        delta2 = _weighted_smooth_l1(pred_delta2, target_delta2, delta2_mask, vw[:, 1:-1])
    else:
        delta2 = pred.sum() * 0

    if length >= 2:
        content_delta_mask = mask[:, 1:] & mask[:, :-1]
        pred_content_delta = pred_768[:, 1:] - pred_768[:, :-1]
        target_content_delta = tgt_768[:, 1:] - tgt_768[:, :-1]
        content_delta_weights = 0.5 * (vw[:, 1:] + vw[:, :-1])
        content_delta = _weighted_smooth_l1(
            pred_content_delta, target_content_delta, content_delta_mask, content_delta_weights
        )
    else:
        content_delta = pred.sum() * 0

    if length >= 3:
        content_delta2_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        pred_content_delta2 = pred_768[:, 2:] - 2 * pred_768[:, 1:-1] + pred_768[:, :-2]
        target_content_delta2 = tgt_768[:, 2:] - 2 * tgt_768[:, 1:-1] + tgt_768[:, :-2]
        content_delta2 = _weighted_smooth_l1(
            pred_content_delta2, target_content_delta2, content_delta2_mask, vw[:, 1:-1]
        )
    else:
        content_delta2 = pred.sum() * 0

    loss = cos_loss + args.code_l1_weight * code_l1 + ow * ord_loss + args.delta_weight * delta
    loss = loss + getattr(args, "delta2_weight", 0.0) * delta2
    loss = loss + getattr(args, "content_delta_weight", 0.0) * content_delta
    loss = loss + getattr(args, "content_delta2_weight", 0.0) * content_delta2
    if args.content_aux_weight > 0:
        loss = loss + args.content_aux_weight * (1.0 - cos768).mean()

    return loss, {
        "loss": float(loss.detach().cpu()),
        "cos5": float(cos[mask].mean().detach().cpu()),
        "code_l1": float(code_l1.detach().cpu()),
        "ord_loss": float(ord_loss.detach().cpu()),
        "ord_acc": float(ord_acc.detach().cpu()),
        "exact_acc": float(exact_acc.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "delta2": float(delta2.detach().cpu()),
        "content_delta": float(content_delta.detach().cpu()),
        "content_delta2": float(content_delta2.detach().cpu()),
        "cos768": float(cos768.mean().detach().cpu()),
    }


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        tokens=batch.tokens.to(device),
        mask=batch.mask.to(device),
        speakers=batch.speakers,
        indices=batch.indices.to(device),
        crop_starts=batch.crop_starts.to(device),
        ssl_L0=batch.ssl_L0.to(device),
        ssl_L4=batch.ssl_L4.to(device),
        ssl_L8=batch.ssl_L8.to(device),
    )


def parse_axis_weights(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if len(values) != len(FSQ_LEVELS):
        raise argparse.ArgumentTypeError(f"expected {len(FSQ_LEVELS)} comma-separated weights")
    return values


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    best_cos: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model.__class__.__name__,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }
    if "Q2D2" in model.__class__.__name__:
        payload["best_probe_cos768"] = best_cos
    else:
        payload["best_probe_cos5"] = best_cos
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ── STFT / MR-STFT loss (shared by decoder-in-loop) ──

def stft_mag(wave: torch.Tensor, n_fft: int) -> torch.Tensor:
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(
        wave, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )
    return spec.abs().clamp_min(1e-7)


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_ffts: tuple[int, ...],
) -> torch.Tensor:
    pred = pred.squeeze(0) if pred.dim() == 2 else pred
    target = target.squeeze(0) if target.dim() == 2 else target
    length = min(pred.shape[-1], target.shape[-1])
    pred = pred[:length]
    target = target[:length]
    losses = []
    for n_fft in n_ffts:
        pred_mag = stft_mag(pred, n_fft)
        target_mag = stft_mag(target, n_fft)
        spectral_convergence = torch.linalg.vector_norm(pred_mag - target_mag) / (
            torch.linalg.vector_norm(target_mag).clamp_min(1e-7))
        log_mag = F.l1_loss(pred_mag.log(), target_mag.log())
        losses.append(spectral_convergence + log_mag)
    return torch.stack(losses).mean()
