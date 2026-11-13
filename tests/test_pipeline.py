"""End-to-end smoke tests: the codec sweep interface and the aggregation policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import config
from src.analysis import aggregate as A
from src.codecs import JPEG, JPEG2000


@pytest.fixture(scope="module")
def small_image(tmp_path_factory):
    """A small real-ish image, so codec tests stay fast."""
    src = config.DATA_PROCESSED / "Nature_sq_gray.png"
    if not src.exists():
        pytest.skip("benchmark images not present")
    img = Image.open(src).convert("L").resize((160, 160))
    p = tmp_path_factory.mktemp("img") / "small.png"
    img.save(p)
    return p


def test_config_paths_are_repo_relative():
    """Nothing may depend on a user directory (docs/AUDIT.md P1)."""
    assert config.ROOT.is_dir()
    assert (config.ROOT / "src" / "config.py").exists()
    for p in (config.DATA_PROCESSED, config.RESULTS, config.METRICS):
        assert config.ROOT in p.parents or p == config.ROOT


def test_sweep_records_rate_at_every_point(small_image):
    """docs/AUDIT.md M1: every operating point must carry a byte count and a bpp."""
    results = list(JPEG(qualities=[30, 60, 90], repeats=1)
                   .sweep(small_image, "small", ssim_window="uniform"))
    assert len(results) == 3
    for r in results:
        assert r.n_bytes > 0
        assert r.bpp > 0
        assert np.isfinite(r.psnr)
        assert 0 < r.ssim <= 1
        assert r.encode_seconds > 0


def test_higher_jpeg_quality_costs_more_bits(small_image):
    results = list(JPEG(qualities=[20, 50, 90], repeats=1)
                   .sweep(small_image, "small", ssim_window="uniform"))
    bpps = [r.bpp for r in results]
    psnrs = [r.psnr for r in results]
    assert bpps == sorted(bpps), "rate must increase with quality"
    assert psnrs == sorted(psnrs), "PSNR must increase with quality"


def test_grayscale_source_stays_single_channel(small_image):
    """docs/AUDIT.md M2: encoding grayscale as RGB inflated every classical rate."""
    img = Image.open(small_image)
    assert img.mode == "L"
    codec = JPEG(qualities=[75], repeats=1)
    _, decoded, _ = codec.encode_once(img, 75)
    assert decoded.ndim == 2, "decoded array should stay 2-D for a grayscale source"


def test_jpeg2000_lossless_point_is_flagged(small_image):
    codec = JPEG2000(labels=[50, 100], psnr_targets=[40], repeats=1)
    results = list(codec.sweep(small_image, "small", ssim_window="uniform"))
    lossless = [r for r in results if r.lossless]
    assert len(lossless) == 1
    assert lossless[0].psnr == float("inf"), "reversible mode must be bit-exact"


# --------------------------------------------------------------------------------------
# aggregation policy (docs/AUDIT.md B4)
# --------------------------------------------------------------------------------------

def _frame():
    return pd.DataFrame([
        {"codec": "JPEG", "image": "a", "quality": 50, "bpp": 1.0,
         "psnr": 30.0, "ssim": 0.9, "si": 10.0, "encode_seconds": 0.1, "lossless": False},
        {"codec": "JPEG", "image": "a", "quality": 90, "bpp": 2.0,
         "psnr": 40.0, "ssim": 0.95, "si": 12.0, "encode_seconds": 0.1, "lossless": False},
        {"codec": "JPEG2000", "image": "a", "quality": 100, "bpp": 5.0,
         "psnr": np.inf, "ssim": 1.0, "si": 13.0, "encode_seconds": 0.2, "lossless": True},
    ])


def test_lossless_points_excluded_from_aggregate():
    """An unguarded np.mean over an inf gives inf. That was the original bug."""
    s = A.summarise(_frame())
    assert np.isfinite(s["psnr_mean"]).all()
    assert np.isfinite(s["psnr_max"]).all()
    assert "JPEG2000" not in set(s["codec"]), "lossless-only codec must not be averaged"
    jpeg = s[s["codec"] == "JPEG"].iloc[0]
    assert jpeg["psnr_mean"] == pytest.approx(35.0)


def test_lossless_points_still_reported():
    rep = A.lossless_report(_frame())
    assert len(rep) == 1
    assert rep.iloc[0]["codec"] == "JPEG2000"


def test_rate_distortion_table_is_finite_and_sorted():
    rd = A.rate_distortion_table(_frame())
    assert np.isfinite(rd["psnr"]).all()
    assert list(rd["bpp"]) == sorted(rd["bpp"])
