#!/usr/bin/env python3
"""Build the benchmark set: data/raw -> data/processed (square, 8-bit grayscale)."""

import _bootstrap  # noqa: F401
from src import config
from src.preprocess import to_square_gray

RAW_TO_PROCESSED = {
    "Gradient.png":   "Gradient_sq_gray.png",
    "Grass.png":      "Grass_sq_gray.png",
    "Mixed.png":      "Mixed_sq_gray.png",
    "Nature.png":     "Nature_sq_gray.png",
    "SharpEdges.png": "SharpEdges_sq_gray.png",
}


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in RAW_TO_PROCESSED.items():
        src = config.DATA_RAW / src_name
        if not src.exists():
            print(f"  skip {src_name}: not found in data/raw/")
            continue
        size = to_square_gray(src, config.DATA_PROCESSED / dst_name)
        print(f"  {src_name:18s} -> {dst_name:24s} {size[0]}x{size[1]} grayscale")
    print(f"\nBenchmark set written to {config.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
