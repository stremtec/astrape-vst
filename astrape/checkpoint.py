from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

from .model import ContentStudent, ContentStudentConfig


FORMAT_VERSION = 2


def save_checkpoint(
    path: str | Path,
    model: ContentStudent,
    *,
    epoch: int,
    metrics: dict[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_type": "content_student",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _infer_legacy_config(state: dict[str, torch.Tensor]) -> ContentStudentConfig:
    hidden = state["stem.0.weight"].shape[0]
    layer_ids = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    if hidden % 16 == 0 and hidden >= 1024:
        heads = 16
    elif hidden % 12 == 0 and hidden >= 768:
        heads = 12
    else:
        heads = 8 if hidden % 8 == 0 else 4
    return ContentStudentConfig(
        hidden=hidden,
        n_layers=max(layer_ids) + 1,
        n_heads=heads,
        kernel_size=state["stem.0.weight"].shape[-1],
        content_dim=state["content_head.weight"].shape[0],
        auxiliary_prefsq="prefsq_head.weight" in state,
    )


def _normalize_legacy_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = dict(state)
    normalized.pop("pos_enc.pe", None)
    for key in list(normalized):
        if ".ff.3." in key:
            normalized[key.replace(".ff.3.", ".ff.2.")] = normalized.pop(key)
    return normalized


def load_content_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    allow_legacy: bool = False,
    safe_convs: Optional[bool] = None,
) -> tuple[ContentStudent, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    is_versioned = (
        isinstance(payload, dict)
        and payload.get("model_type") == "content_student"
        and "state_dict" in payload
    )
    if is_versioned:
        metadata = payload
        config_data = dict(payload["config"])
        if safe_convs is not None:
            config_data["safe_convs"] = safe_convs
        config = ContentStudentConfig(**config_data)
        state = payload["state_dict"]
    else:
        if not allow_legacy:
            raise ValueError(
                "Legacy raw state_dict detected. Pass allow_legacy=True to load weights "
                "trained with symmetric convolution padding into the causal architecture."
            )
        state = payload
        config = _infer_legacy_config(state)
        if safe_convs is not None:
            config = ContentStudentConfig(
                **{**asdict(config), "safe_convs": safe_convs}
            )
        metadata = {
            "format_version": 1,
            "model_type": "legacy_content_student",
            "config": asdict(config),
        }
        warnings.warn(
            "Loading legacy symmetric-padding weights into the causal architecture. "
            "Fine-tune before treating this model as a validated causal checkpoint.",
            stacklevel=2,
        )
        state = _normalize_legacy_state(state)
    model = ContentStudent(config).to(device)
    model.load_state_dict(state, strict=True)
    return model, metadata
