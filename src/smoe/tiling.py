r"""Tiled SMoE training and reconstruction.

A full-resolution SMoE fit needs roughly 70 GB of RAM at 1280x1280, so each image is
split into overlapping tiles, one model is trained per tile, and the reconstructions are
recombined with a separable Hann window:

.. math::

    \hat{I}(\mathbf{x}) =
    \frac{\sum_{t} w_t(\mathbf{x})\,\hat{I}_t(\mathbf{x})}
         {\sum_{t} w_t(\mathbf{x}) + \varepsilon},
    \qquad w(u,v) = w_1(u)\,w_1(v).

This is a cleaned-up port of the tiling cell in the original ``SMoE.ipynb``. The tiling
and blending logic there was **correct** (edge clamping, de-duplicated coordinates,
normalisation by accumulated weight) and is preserved. What changed:

* no Kaggle paths, no shell magics, no runtime source patching — the vendored ``tf-smoe``
  under ``src/smoe/vendor/`` is already patched for TF 1.x compat mode (AUDIT.md P2, P3);
* paths come from :mod:`src.config`, so it runs on any OS (AUDIT.md P1);
* the kernel budget is recorded in the output directory name, so a results folder can
  never disagree with its label the way the originals did (AUDIT.md B2).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VENDOR = Path(__file__).resolve().parent / "vendor" / "tf_smoe"

__all__ = ["tile_coordinates", "blend_tiles", "run_image"]


def tile_coordinates(height: int, width: int, tile_size: int, overlap: int):
    """Top-left coordinates of every tile, clamped to stay inside the image."""
    step = tile_size - overlap
    nx = math.ceil((width - overlap) / step)
    ny = math.ceil((height - overlap) / step)

    coords = []
    for j in range(ny):
        top = min(j * step, height - tile_size)
        for i in range(nx):
            left = min(i * step, width - tile_size)
            coords.append((max(0, top), max(0, left), j, i))
    # identical coordinates can repeat once clamped at the right/bottom edge
    return list(dict.fromkeys(coords))


def blend_tiles(shape, pieces, tile_size: int, eps: float = 1e-8) -> np.ndarray:
    """Recombine ``(top, left, array)`` pieces with Hann-window feathering."""
    win1d = np.hanning(tile_size)
    win2d = np.outer(win1d, win1d).astype(np.float32)

    accum = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    for top, left, arr in pieces:
        a = np.asarray(arr, dtype=np.float32)
        accum[top:top + tile_size, left:left + tile_size] += a * win2d
        weight[top:top + tile_size, left:left + tile_size] += win2d

    return np.clip(accum / (weight + eps), 0, 255).astype(np.uint8)


def run_image(
    image_path: Path,
    out_root: Path,
    kernels_per_dim: int,
    tile_size: int = 512,
    overlap: int = 64,
    iterations: int = 100,
    batches: int = 64,
    python: str | None = None,
) -> Path:
    """Train SMoE on one image, tile by tile, and write the stitched reconstruction.

    Returns the path to the reconstruction. Expect 8 minutes to 3.5 hours per image
    depending on ``kernels_per_dim``; a GPU with ~16 GB is strongly recommended.
    """
    image_path = Path(image_path)
    name = image_path.stem
    python = python or sys.executable

    tiles_dir = out_root / "tiles" / name
    results_dir = out_root / "tile_results" / name
    for d in (tiles_dir, results_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    height, width = img.shape
    if height != width:
        raise ValueError(f"expected a square image, got {img.shape} for {image_path}")

    coords = tile_coordinates(height, width, tile_size, overlap)
    print(f"[{name}] K={kernels_per_dim}  {len(coords)} tiles of {tile_size}px "
          f"(overlap {overlap})")

    pieces = []
    for top, left, ty, tx in coords:
        tile_name = f"tile_r{ty}_c{tx}"
        tile_path = tiles_dir / f"{tile_name}.png"
        Image.fromarray(img[top:top + tile_size, left:left + tile_size]).save(tile_path)

        tile_out = results_dir / tile_name
        tile_out.mkdir(parents=True, exist_ok=True)

        proc = subprocess.run(
            [
                python, str(VENDOR / "train_model.py"),
                "--image_path", str(tile_path),
                "--results_path", str(tile_out),
                "--kernels_per_dim", str(kernels_per_dim),
                "--iterations", str(iterations),
                "--batches", str(batches),
                "--quiet", "1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=str(VENDOR),
        )
        recon = tile_out / "reconstruction.png"
        if not recon.exists():
            tail = "\n".join(proc.stdout.splitlines()[-15:])
            raise RuntimeError(f"[{name}] {tile_name} produced no reconstruction:\n{tail}")

        arr = np.array(Image.open(recon).convert("L"), dtype=np.float32)
        if arr.shape != (tile_size, tile_size):
            raise RuntimeError(
                f"[{name}] {tile_name}: got {arr.shape}, expected "
                f"({tile_size}, {tile_size})"
            )
        pieces.append((top, left, arr))

    stitched = blend_tiles((height, width), pieces, tile_size)
    out_path = out_root / f"{name}_K{kernels_per_dim:03d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stitched).save(out_path)
    print(f"[{name}] wrote {out_path}")
    return out_path
