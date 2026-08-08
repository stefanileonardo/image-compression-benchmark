#!/usr/bin/env python3
"""Regenerate every figure from results/metrics/, into results/figures/."""

import _bootstrap  # noqa: F401
import pandas as pd

from src import config
from src.analysis import aggregate as A
from src.analysis import figures as F


def main() -> None:
    path = config.METRICS / "all_operating_points.csv"
    if not path.exists():
        raise SystemExit("run scripts/05_aggregate.py first")

    df = pd.read_csv(path)
    out = config.FIGURES
    written = []

    written += F.rate_distortion(df, out, metric="psnr")
    written += F.rate_distortion(df, out, metric="ssim")
    written.append(F.aggregate_bars(A.summarise(df), out))

    for image in df["image"].unique():
        written.append(F.quality_vs_time(df, out, image, metric="psnr"))
        written.append(F.spatial_information(df, out, image))

    for p in written:
        print(f"  {p.relative_to(config.ROOT)}")
    print(f"\n{len(written)} figures -> {out}")


if __name__ == "__main__":
    main()
