r"""SMoE rate measurement — from model parameters, not from the decoded image.

This module exists because of docs/AUDIT.md **B1**, the most consequential error in the
original study: the reported "SMoE file size" was the byte size of
``<image>_reconstructed.png``, i.e. the **decoder output**. A PNG of a decoded image
measures the entropy of a blurry picture; it is not what a SMoE decoder would have to
receive. Using it made SMoE look competitive on rate when it is not.

The transmitted payload is the fitted model. Each surviving Gaussian kernel carries

===================  =======  ==========================================================
symbol               values   meaning
===================  =======  ==========================================================
:math:`\pi_j`        1        gating weight
:math:`\mu_j`        2        kernel centre in normalised image coordinates
:math:`A_j`          4        steering matrix, :math:`\Sigma_j^{-1} = A_j^\top A_j`
:math:`\gamma_{e,j}` 2        linear expert slope
:math:`\nu_{e,j}`    1        linear expert offset
===================  =======  ==========================================================

so **10 values per kernel**, stored as ``float32`` by the reference implementation.

.. math::

    \mathrm{bpp} = \frac{b \cdot 10 \cdot \sum_{t} K_t}{W \times H}

where :math:`K_t` is the number of kernels surviving :math:`\ell_1` pruning in tile
:math:`t` and :math:`b` is the bits per stored value.

Two rates are reported:

``bpp_fp32``
    What the pipeline actually produces (:math:`b = 32`). This is the honest measured
    rate, because **no quantisation or entropy coding of SMoE parameters is implemented
    anywhere in this project**.
``bpp_q8``
    An optimistic lower bound assuming 8-bit quantisation with no entropy coding
    (:math:`b = 8`). Not achieved — reported only for context.

For reference, an 8-bit grayscale source is 8.00 bpp uncompressed.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["VALUES_PER_KERNEL", "SMoERate", "count_kernels", "measure_image"]

#: pis(1) + musX(2) + A(4) + gamma_e(2) + nu_e(1)
VALUES_PER_KERNEL = 10


@dataclass
class SMoERate:
    """Measured rate for one image at one kernel budget."""

    image: str
    budget: int              # kernels per dimension, K
    tiles: int
    kernels: int             # total surviving kernels across all tiles
    n_values: int            # kernels * VALUES_PER_KERNEL
    width: int
    height: int
    bpp_fp32: float
    bpp_q8: float
    bytes_fp32: int

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def count_kernels(params_pickle: Path) -> int:
    """Number of kernels surviving :math:`\\ell_1` pruning in one tile.

    Reads ``params_best.pkl`` as written by the reference ``utils.save_model``. The
    length of the ``pis`` vector is the surviving kernel count: an unpruned run gives
    exactly :math:`K^2` (256 for K=16, 1024 for K=32), which is how the folder
    mislabelling in docs/AUDIT.md B2 was detected.
    """
    with open(params_pickle, "rb") as fh:
        blob = pickle.load(fh)
    params = blob["params"] if isinstance(blob, dict) and "params" in blob else blob
    return int(np.asarray(params["pis"]).size)


def measure_image(
    params_dir: Path,
    image: str,
    budget: int,
    width: int,
    height: int,
) -> SMoERate:
    """Aggregate the rate over every tile of one image.

    ``params_dir`` is the directory holding one ``*.pkl`` per tile, e.g.
    ``results/smoe_params/K096/Nature_sq_gray/``.
    """
    tiles = sorted(Path(params_dir).glob("*.pkl"))
    if not tiles:
        raise FileNotFoundError(f"no tile parameter files in {params_dir}")

    kernels = sum(count_kernels(t) for t in tiles)
    n_values = kernels * VALUES_PER_KERNEL
    pixels = width * height

    return SMoERate(
        image=image,
        budget=budget,
        tiles=len(tiles),
        kernels=kernels,
        n_values=n_values,
        width=width,
        height=height,
        bpp_fp32=32.0 * n_values / pixels,
        bpp_q8=8.0 * n_values / pixels,
        bytes_fp32=4 * n_values,
    )


def implied_budget(params_dir: Path) -> float:
    """Kernels per dimension implied by the stored parameters of the first tile.

    Use this to verify that a results folder holds what its name claims. An exact square
    (256, 1024, 4096) means no pruning occurred and the budget is unambiguous.
    """
    tiles = sorted(Path(params_dir).glob("*.pkl"))
    if not tiles:
        raise FileNotFoundError(f"no tile parameter files in {params_dir}")
    return float(np.sqrt(count_kernels(tiles[0])))
