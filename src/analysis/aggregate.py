"""Turn per-operating-point metrics into the tables the paper reports.

The aggregation policy for lossless operating points is stated explicitly here rather
than left implicit. This is the fix for docs/AUDIT.md **B4**: JPEG 2000 in reversible mode is
genuinely lossless, so its PSNR is ``inf``, and the original code fed that straight into
``np.mean`` / ``np.max`` with no guard. Any mean or max over that array is ``inf``, yet
the paper reported a finite 42.7 dB — a number the committed code could not produce.

Policy adopted here: **lossless points are excluded from PSNR and SSIM aggregates and
reported separately.** A codec that reaches mathematical losslessness has left the
rate-distortion curve; averaging ``inf`` into a fidelity summary is meaningless, and
silently substituting a large finite number is worse because it hides the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["LOSSLESS_POLICY", "summarise", "rate_distortion_table", "lossless_report"]

LOSSLESS_POLICY = (
    "Lossless operating points (PSNR = inf) are excluded from PSNR/SSIM aggregates "
    "and reported separately in lossless_report()."
)


def _lossy(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that sit on the rate-distortion curve: finite PSNR, not flagged lossless."""
    out = df[np.isfinite(df["psnr"])]
    if "lossless" in out.columns:
        out = out[~out["lossless"].astype(bool)]
    return out


def summarise(df: pd.DataFrame, by: str = "codec") -> pd.DataFrame:
    """Min / mean / max / std of PSNR and SSIM per codec, over lossy points only."""
    lossy = _lossy(df)
    if lossy.empty:
        raise ValueError("no finite operating points to aggregate")

    out = lossy.groupby(by).agg(
        n_points=("psnr", "size"),
        psnr_min=("psnr", "min"),
        psnr_mean=("psnr", "mean"),
        psnr_max=("psnr", "max"),
        psnr_std=("psnr", "std"),
        ssim_min=("ssim", "min"),
        ssim_mean=("ssim", "mean"),
        ssim_max=("ssim", "max"),
        ssim_std=("ssim", "std"),
        bpp_min=("bpp", "min"),
        bpp_max=("bpp", "max"),
    )
    return out.round(4).reset_index()


def rate_distortion_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every lossy operating point on a common bits-per-pixel axis.

    This is the comparison the original study was missing entirely (docs/AUDIT.md M1): it
    plotted quality against each codec's own setting, which is not comparable across
    codecs. bpp is.
    """
    lossy = _lossy(df)
    cols = ["codec", "image", "quality", "bpp", "psnr", "ssim", "si", "encode_seconds"]
    cols = [c for c in cols if c in lossy.columns]
    return lossy[cols].sort_values(["image", "codec", "bpp"]).reset_index(drop=True)


def lossless_report(df: pd.DataFrame) -> pd.DataFrame:
    """The lossless operating points, kept visible instead of averaged away."""
    if "lossless" in df.columns:
        rows = df[df["lossless"].astype(bool) | ~np.isfinite(df["psnr"])]
    else:
        rows = df[~np.isfinite(df["psnr"])]
    cols = [c for c in ["codec", "image", "quality", "bpp", "n_bytes"] if c in rows.columns]
    return rows[cols].reset_index(drop=True)


def compare_at_matched_rate(df: pd.DataFrame, tolerance: float = 0.15) -> pd.DataFrame:
    """Pair each SMoE point with the classical points at a comparable rate.

    Answers the question the paper should have asked: at the *same* number of bits, who
    wins? ``tolerance`` is the relative bpp window for a match.
    """
    lossy = _lossy(df)
    smoe = lossy[lossy["codec"].str.upper() == "SMOE"]
    other = lossy[lossy["codec"].str.upper() != "SMOE"]

    rows = []
    for _, s in smoe.iterrows():
        lo, hi = s["bpp"] * (1 - tolerance), s["bpp"] * (1 + tolerance)
        near = other[(other["image"] == s["image"]) & other["bpp"].between(lo, hi)]
        for _, o in near.iterrows():
            rows.append({
                "image": s["image"],
                "smoe_quality": s["quality"],
                "smoe_bpp": round(s["bpp"], 3),
                "smoe_psnr": round(s["psnr"], 2),
                "rival": o["codec"],
                "rival_quality": o["quality"],
                "rival_bpp": round(o["bpp"], 3),
                "rival_psnr": round(o["psnr"], 2),
                "psnr_gap_db": round(o["psnr"] - s["psnr"], 2),
            })
    return pd.DataFrame(rows)
