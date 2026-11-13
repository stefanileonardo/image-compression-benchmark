"""Every figure in the paper, regenerated from ``results/metrics/*.csv``.

Nothing here hardcodes a measurement. The original ``GraphsFinal.ipynb`` embedded roughly
350 lines of metrics as Python literals plus an inline fallback copy of a timing table,
so the figures could drift silently from the code that claimed to produce them
(docs/AUDIT.md m6). Here the CSVs are the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: works in CI and over SSH
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

from src.analysis.aggregate import _lossy  # noqa: E402

STYLES = {
    "JPEG":     {"color": "#2E86AB", "marker": "o", "linestyle": "--"},
    "JPEG2000": {"color": "#A23B72", "marker": "s", "linestyle": "-."},
    "AVIF":     {"color": "#F18F01", "marker": "^", "linestyle": ":"},
    "SMoE":     {"color": "#C73E1D", "marker": "D", "linestyle": "-"},
}

UNCOMPRESSED_BPP = 8.0


def _style(codec: str) -> dict:
    return STYLES.get(codec, {"color": "gray", "marker": ".", "linestyle": "-"})


def rate_distortion(df: pd.DataFrame, out_dir: Path, metric: str = "psnr") -> list[Path]:
    """PSNR (or SSIM) versus bits per pixel, one panel per image.

    This is the headline figure the original study lacked: a single physically
    meaningful axis on which all four codecs can be compared.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lossy = _lossy(df)
    written = []

    for image, group in lossy.groupby("image"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for codec, sub in group.groupby("codec"):
            sub = sub.sort_values("bpp")
            ax.plot(sub["bpp"], sub[metric], label=codec, linewidth=2,
                    markersize=6, **_style(codec))

        ax.axvline(UNCOMPRESSED_BPP, color="gray", linestyle=":", linewidth=1)
        ax.text(UNCOMPRESSED_BPP, ax.get_ylim()[0], " uncompressed (8 bpp)",
                fontsize=8, color="gray", va="bottom")

        ax.set_xscale("log")
        ax.set_xlabel("Rate (bits per pixel)")
        ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "SSIM")
        ax.set_title(f"Rate-distortion - {image}")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        path = out_dir / f"rd_{metric}_{image}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def quality_vs_time(df: pd.DataFrame, out_dir: Path, image: str,
                    metric: str = "psnr") -> Path:
    """Quality against encode time on a log axis, for one image.

    Reproduces the paper's Fig. 3. Note in the caption that SMoE ran on a Tesla P100 GPU
    while the classical codecs ran on an i7-1165G7 CPU (docs/AUDIT.md M3).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    group = _lossy(df)
    group = group[group["image"] == image]

    fig, ax = plt.subplots(figsize=(7, 5))
    for codec, sub in group.groupby("codec"):
        sub = sub.sort_values("encode_seconds")
        ax.plot(sub["encode_seconds"], sub[metric], label=codec, linewidth=2,
                markersize=6, **_style(codec))

    ax.set_xscale("log")
    ax.set_xlabel("Encode time (s, log scale)")
    ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "SSIM")
    ax.set_title(f"Quality vs encode time - {image}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    path = out_dir / f"time_{metric}_{image}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def aggregate_bars(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Min / mean / max PSNR and SSIM per codec, over lossy points only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (lo, mid, hi, label) in zip(
        axes,
        [("psnr_min", "psnr_mean", "psnr_max", "PSNR (dB)"),
         ("ssim_min", "ssim_mean", "ssim_max", "SSIM")],
    ):
        x = np.arange(len(summary))
        lower = summary[mid] - summary[lo]
        upper = summary[hi] - summary[mid]
        colors = [_style(c)["color"] for c in summary["codec"]]

        ax.errorbar(x, summary[mid], yerr=[lower, upper], fmt="none",
                    ecolor="gray", capsize=6, linewidth=1.5)
        ax.scatter(x, summary[mid], c=colors, s=90, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["codec"])
        ax.set_ylabel(label)
        ax.set_title(f"{label}: mean with min/max range")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    fig.suptitle("Aggregate quality across all images (lossless points excluded)",
                 fontsize=11)
    fig.tight_layout()
    path = out_dir / "aggregate_quality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def spatial_information(df: pd.DataFrame, out_dir: Path, image: str) -> Path:
    """Spatial Information against rate, for one image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    group = _lossy(df)
    group = group[group["image"] == image]

    fig, ax = plt.subplots(figsize=(7, 5))
    for codec, sub in group.groupby("codec"):
        sub = sub.sort_values("bpp")
        ax.plot(sub["bpp"], sub["si"], label=codec, linewidth=2,
                markersize=6, **_style(codec))

    ax.set_xscale("log")
    ax.set_xlabel("Rate (bits per pixel)")
    ax.set_ylabel("Spatial Information (ITU-T P.910)")
    ax.set_title(f"Retained structural detail - {image}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    path = out_dir / f"si_{image}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
