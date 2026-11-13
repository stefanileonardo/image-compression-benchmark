"""Tests for the SMoE rate measurement and for the shipped parameter files.

These guard the correction at the centre of this repository (docs/AUDIT.md B1): rate must come
from the model, not from the size of the decoded image. They also re-verify the
configuration labelling (B2), so a mislabelled directory can never silently reappear.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from src import config
from src.smoe import rate as R


# --------------------------------------------------------------------------------------
# the rate formula itself
# --------------------------------------------------------------------------------------

def test_values_per_kernel_is_ten():
    """pis(1) + musX(2) + A(4) + gamma_e(2) + nu_e(1). If upstream changes, rates change."""
    assert R.VALUES_PER_KERNEL == 10


def test_rate_formula_closed_form(tmp_path):
    """1000 kernels over a 100x100 image at fp32 = 10*32*1000/10000 = 32 bpp."""
    d = tmp_path / "img"
    d.mkdir()
    params = {
        "pis": np.zeros(1000, dtype=np.float32),
        "musX": np.zeros((1000, 2), dtype=np.float32),
        "A": np.zeros((1000, 2, 2), dtype=np.float32),
        "gamma_e": np.zeros((1000, 2), dtype=np.float32),
        "nu_e": np.zeros(1000, dtype=np.float32),
    }
    with open(d / "tile_r0_c0.pkl", "wb") as fh:
        pickle.dump({"params": params}, fh)

    r = R.measure_image(d, "img", budget=32, width=100, height=100)
    assert r.kernels == 1000
    assert r.n_values == 10_000
    assert r.bpp_fp32 == pytest.approx(32.0)
    assert r.bpp_q8 == pytest.approx(8.0)
    assert r.bytes_fp32 == 40_000


def test_missing_params_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.measure_image(tmp_path / "nope", "img", 16, 64, 64)


# --------------------------------------------------------------------------------------
# the parameters actually shipped in this repository
# --------------------------------------------------------------------------------------

#: K^2, the number of kernels a tile starts with before l1 pruning.
INITIAL_KERNELS_PER_TILE = {16: 256, 32: 1024, 96: 9216}


@pytest.mark.parametrize("budget", [16, 32, 96])
def test_shipped_params_match_their_label(budget):
    """Regression guard for docs/AUDIT.md B2, where two folders held the wrong budget."""
    base = config.SMOE_PARAMS / f"K{budget:03d}"
    if not base.is_dir():
        pytest.skip(f"K{budget:03d} parameters not present")

    for image_dir in sorted(base.iterdir()):
        if not image_dir.is_dir():
            continue
        implied = R.implied_budget(image_dir)
        # l1 pruning only ever removes kernels, so the implied budget is <= the nominal
        # one and should stay within a few percent of it.
        assert implied <= budget * 1.001, (
            f"{image_dir.name} implies K={implied:.1f} but is labelled K={budget}"
        )
        assert implied >= budget * 0.90, (
            f"{image_dir.name} implies K={implied:.1f}, far below its K={budget} label"
        )


@pytest.mark.parametrize("budget,initial", sorted(INITIAL_KERNELS_PER_TILE.items()))
def test_kernel_counts_never_exceed_the_budget(budget, initial):
    """l1 pruning only ever removes kernels, so a tile can never hold more than K^2.

    A count *above* the nominal budget is the signature of the mislabelling in
    docs/AUDIT.md B2, where a folder named ``64k`` actually held K=32 results.
    """
    base = config.SMOE_PARAMS / f"K{budget:03d}"
    if not base.is_dir():
        pytest.skip(f"K{budget:03d} parameters not present")

    for image_dir in sorted(base.iterdir()):
        if not image_dir.is_dir():
            continue
        for tile in sorted(image_dir.glob("*.pkl")):
            n = R.count_kernels(tile)
            assert n <= initial, f"{tile.name}: {n} kernels exceeds K^2 = {initial}"
            # measured floor across the shipped runs is 0.748 (K=96, heavy pruning)
            assert n >= 0.70 * initial, (
                f"{tile.name}: {n} kernels is far below K^2 = {initial}; "
                "the folder label is probably wrong"
            )


def test_smoe_rate_exceeds_uncompressed_at_largest_budget():
    """The headline correction: at K=96 the model costs more than the raw image.

    If this ever stops holding, the README and the paper both need revisiting.
    """
    base = config.SMOE_PARAMS / "K096"
    if not (base / "Nature_sq_gray").is_dir():
        pytest.skip("K096 Nature parameters not present")

    r = R.measure_image(base / "Nature_sq_gray", "Nature_sq_gray", 96, 1200, 1200)
    assert r.bpp_fp32 > 8.0, "K=96 should exceed the 8 bpp uncompressed reference"
    assert r.bpp_fp32 == pytest.approx(14.92, abs=0.05)
