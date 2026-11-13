#!/usr/bin/env python3
"""Measure SMoE rate and distortion; write results/metrics/smoe.csv.

This is the fix for docs/AUDIT.md B1. Rate comes from the PARAMETER COUNT, not from the size
of the reconstructed PNG. It also re-verifies that each results folder really holds the
kernel budget its name claims, which is how the original mislabelling was caught.
"""

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image

from src import config, metrics as M
from src.smoe.rate import implied_budget, measure_image


def main() -> None:
    exp = config.load()
    config.ensure_dirs()
    rows, warnings = [], []

    for budget in exp.smoe_budgets:
        label = f"K{budget:03d}"
        base = config.SMOE_PARAMS / label
        if not base.is_dir():
            print(f"  ! {label}: no parameters under results/smoe_params/")
            continue

        for name in exp.images:
            params_dir = base / name
            if not params_dir.is_dir():
                warnings.append(f"{label}/{name}: no parameters (run incomplete)")
                continue

            # --- sanity check: do the stored parameters match the folder label? ---
            implied = implied_budget(params_dir)
            if abs(implied - budget) > 0.05 * budget:
                warnings.append(
                    f"{label}/{name}: parameters imply K={implied:.1f}, "
                    f"folder claims K={budget} - MISLABELLED"
                )

            orig_path = exp.image_path(name)
            orig = np.array(Image.open(orig_path).convert("L"), dtype=np.uint8)
            h, w = orig.shape
            rate = measure_image(params_dir, name, budget, w, h)

            recon_path = config.RECONSTRUCTIONS / "smoe" / f"{name}_{label}.png"
            if recon_path.exists():
                recon = np.array(Image.open(recon_path).convert("L"), dtype=np.uint8)
                psnr = M.psnr(orig, recon)
                ssim = M.ssim(orig, recon, window=exp.ssim_window)
                si = M.spatial_information(recon)
            else:
                warnings.append(f"{label}/{name}: no stitched reconstruction")
                psnr = ssim = si = float("nan")

            row = rate.as_row()
            row.update({"codec": "SMoE", "quality": label,
                        "psnr": psnr, "ssim": ssim, "si": si,
                        "bpp": rate.bpp_fp32, "n_bytes": rate.bytes_fp32,
                        "lossless": False, "encode_seconds": float("nan")})
            rows.append(row)
            print(f"  {label} {name:20s} {rate.tiles:2d} tiles  "
                  f"{rate.kernels:8,d} kernels  {rate.bpp_fp32:6.2f} bpp fp32  "
                  f"({rate.bpp_q8:5.2f} bpp @8-bit)  PSNR {psnr:.2f}")

    out = config.METRICS / "smoe.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)} SMoE operating points -> {out}")
    print("Reference: uncompressed 8-bit grayscale is 8.00 bpp.")

    if warnings:
        print("\nWARNINGS:")
        for wmsg in warnings:
            print(f"  - {wmsg}")


if __name__ == "__main__":
    main()
