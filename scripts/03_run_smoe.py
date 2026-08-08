#!/usr/bin/env python3
"""Train SMoE on every image at every kernel budget.

WARNING: this is the expensive step. Expect 8 minutes to 3.5 hours per image depending
on the kernel budget, and a GPU with roughly 16 GB. The repository already ships the
fitted parameters under results/smoe_params/, so you only need this to reproduce them
from scratch or to add a budget.
"""

import argparse

import _bootstrap  # noqa: F401

from src import config
from src.smoe.tiling import run_image


def main() -> None:
    exp = config.load()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernels", type=int, nargs="+", default=exp.smoe_budgets,
                    help="kernels per dimension, e.g. --kernels 16 32 96")
    ap.add_argument("--images", nargs="+", default=exp.images)
    args = ap.parse_args()

    config.ensure_dirs()
    for budget in args.kernels:
        out_root = config.RESULTS / "smoe_runs" / f"K{budget:03d}"
        out_root.mkdir(parents=True, exist_ok=True)
        for name in args.images:
            path = exp.image_path(name)
            if not path.exists():
                print(f"  ! {path.name} missing - run 01_preprocess.py first")
                continue
            run_image(
                path, out_root,
                kernels_per_dim=budget,
                tile_size=exp.smoe["tile_size"],
                overlap=exp.smoe["overlap"],
                iterations=exp.smoe["iterations"],
                batches=exp.smoe["batches"],
            )


if __name__ == "__main__":
    main()
