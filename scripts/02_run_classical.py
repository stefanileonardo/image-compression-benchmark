#!/usr/bin/env python3
"""Sweep JPEG, JPEG 2000 and AVIF; write results/metrics/classical.csv.

Records the encoded byte count and therefore bits per pixel at EVERY operating point,
which is what makes a rate-distortion comparison possible (AUDIT.md M1). Grayscale
sources are encoded natively as single channel (AUDIT.md M2).
"""

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from src import config
from src.codecs import JPEG, JPEG2000, AVIF


def build_codecs(exp, repeats):
    codecs = [
        JPEG(qualities=exp.jpeg["qualities"],
             subsampling=exp.jpeg.get("subsampling", 0),
             optimize=exp.jpeg.get("optimize", True),
             repeats=repeats),
        JPEG2000(labels=exp.jpeg2000["labels"],
                 psnr_targets=exp.jpeg2000["psnr_targets"],
                 repeats=repeats),
    ]
    try:
        codecs.append(AVIF(qualities=exp.avif["qualities"],
                           speed=exp.avif.get("speed", 6),
                           repeats=repeats))
    except ImportError:
        print("  ! pillow-heif not installed; skipping AVIF")
    return codecs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=None,
                    help="encodes per operating point; median is reported")
    ap.add_argument("--images", nargs="+", default=None,
                    help="subset of images; default is all of them")
    ap.add_argument("--codecs", nargs="+", default=None,
                    choices=["JPEG", "JPEG2000", "AVIF"],
                    help="subset of codecs; default is all of them")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing classical.csv instead of overwriting, so a "
                         "long sweep can be split across several invocations")
    args = ap.parse_args()

    exp = config.load()
    config.ensure_dirs()
    repeats = args.repeats or exp.metrics.get("timing_repeats", 5)
    images = args.images or exp.images

    rows = []
    for codec in build_codecs(exp, repeats):
        if args.codecs and codec.name not in args.codecs:
            continue
        for name in images:
            path = exp.image_path(name)
            if not path.exists():
                print(f"  ! {path.name} missing - run 01_preprocess.py first")
                continue
            print(f"  {codec.name:9s} {name}")
            for result in codec.sweep(path, name, ssim_window=exp.ssim_window):
                rows.append(result.as_row())

    out = config.METRICS / "classical.csv"
    df = pd.DataFrame(rows)
    if args.append and out.exists():
        prior = pd.read_csv(out)
        df = pd.concat([prior, df], ignore_index=True)
        df = df.drop_duplicates(subset=["codec", "image", "quality"], keep="last")
    df.to_csv(out, index=False)
    print(f"\n{len(rows)} new operating points, {len(df)} total -> {out}")


if __name__ == "__main__":
    main()
