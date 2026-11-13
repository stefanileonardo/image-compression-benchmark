"""Configuration and path resolution.

Every path in this repository is derived from the location of this file, so the code
runs unchanged on Windows, macOS and Linux and from any working directory. Nothing here
hardcodes a user directory — that was the single biggest portability defect in the
original notebooks (see docs/AUDIT.md, P1).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# src/config.py -> src/ -> repository root
ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "config" / "experiment.yaml"

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
RECONSTRUCTIONS = RESULTS / "reconstructions"
SMOE_PARAMS = RESULTS / "smoe_params"
FIGURES = RESULTS / "figures"

PAPER = ROOT / "paper"


@dataclass(frozen=True)
class Experiment:
    """Typed view over ``config/experiment.yaml``."""

    images: list[str]
    jpeg: dict[str, Any] = field(default_factory=dict)
    jpeg2000: dict[str, Any] = field(default_factory=dict)
    avif: dict[str, Any] = field(default_factory=dict)
    smoe: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def image_path(self, name: str) -> Path:
        """Absolute path to a processed benchmark image."""
        return DATA_PROCESSED / f"{name}.png"

    @property
    def ssim_window(self) -> str:
        return self.metrics.get("ssim_window", "gaussian")

    @property
    def smoe_budgets(self) -> list[int]:
        return list(self.smoe.get("kernel_budgets", []))


@functools.lru_cache(maxsize=1)
def load(path: Path | None = None) -> Experiment:
    """Load and cache the experiment configuration."""
    p = Path(path) if path is not None else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Experiment(
        images=raw["images"],
        jpeg=raw.get("jpeg", {}),
        jpeg2000=raw.get("jpeg2000", {}),
        avif=raw.get("avif", {}),
        smoe=raw.get("smoe", {}),
        metrics=raw.get("metrics", {}),
    )


def ensure_dirs() -> None:
    """Create the output directories that scripts write into."""
    for d in (METRICS, RECONSTRUCTIONS, FIGURES, SMOE_PARAMS):
        d.mkdir(parents=True, exist_ok=True)
    for codec in ("jpeg", "jpeg2000", "avif", "smoe"):
        (RECONSTRUCTIONS / codec).mkdir(parents=True, exist_ok=True)
