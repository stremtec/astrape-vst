#!/usr/bin/env python3
"""Quality tier definitions shared by training and benchmarking."""

from dataclasses import dataclass

from astrape.model import ContentStudentConfig


@dataclass(frozen=True)
class Tier:
    model: ContentStudentConfig
    learning_rate: float
    epochs: int
    label: str


TIERS = {
    "low": Tier(
        model=ContentStudentConfig(hidden=512, n_layers=6, n_heads=8),
        learning_rate=3e-4,
        epochs=30,
        label="low (512dim)",
    ),
    "medium": Tier(
        model=ContentStudentConfig(hidden=768, n_layers=8, n_heads=12),
        learning_rate=3e-4,
        epochs=30,
        label="medium (768dim)",
    ),
    "xhigh": Tier(
        model=ContentStudentConfig(
            hidden=1024,
            n_layers=8,
            n_heads=16,
        ),
        learning_rate=2e-4,
        epochs=30,
        label="xhigh (1024dim)",
    ),
}


def get_tier(name: str) -> Tier:
    try:
        return TIERS[name]
    except KeyError as error:
        choices = ", ".join(sorted(TIERS))
        raise ValueError(f"Unknown tier {name!r}. Choose one of: {choices}") from error
