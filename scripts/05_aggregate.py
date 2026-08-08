#!/usr/bin/env python3
"""Combine classical.csv + smoe.csv into the tables the paper reports."""

import _bootstrap  # noqa: F401
import pandas as pd

from src import config
from src.analysis import aggregate as A


def main() -> None:
    frames = []
    for fname in ("classical.csv", "smoe.csv"):
        path = config.METRICS / fname
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"  ! {fname} missing - run the earlier scripts first")
    if not frames:
        raise SystemExit("nothing to aggregate")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(config.METRICS / "all_operating_points.csv", index=False)

    print("\nAggregate quality per codec")
    print(f"({A.LOSSLESS_POLICY})\n")
    summary = A.summarise(df)
    print(summary.to_string(index=False))
    summary.to_csv(config.METRICS / "summary_by_codec.csv", index=False)

    A.rate_distortion_table(df).to_csv(
        config.METRICS / "rate_distortion.csv", index=False)

    lossless = A.lossless_report(df)
    if not lossless.empty:
        print("\nLossless operating points (excluded from the aggregate above)")
        print(lossless.to_string(index=False))
        lossless.to_csv(config.METRICS / "lossless_points.csv", index=False)

    matched = A.compare_at_matched_rate(df)
    if not matched.empty:
        print("\nSMoE vs classical codecs at comparable rate")
        print(matched.to_string(index=False))
        matched.to_csv(config.METRICS / "matched_rate_comparison.csv", index=False)

    print(f"\nTables written to {config.METRICS}")


if __name__ == "__main__":
    main()
