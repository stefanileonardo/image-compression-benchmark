"""Correctness tests for the metric implementations.

The reference implementations that the original project rolled by hand were wrong in
ways that were invisible without a cross-check (AUDIT.md m1). These tests pin the
behaviour against ``scikit-image`` and against closed-form cases, so a regression is
caught rather than silently absorbed into a figure.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from src import metrics as M


@pytest.fixture(scope="module")
def pair():
    """A deterministic reference image and a noisy version of it."""
    rng = np.random.default_rng(20260808)
    ref = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    # smooth it a little so it is not pure noise, which SSIM handles degenerately
    ref = M._box_filter(ref.astype(float), 5).clip(0, 255).astype(np.uint8)
    noise = rng.normal(0, 10, ref.shape)
    test = np.clip(ref.astype(float) + noise, 0, 255).astype(np.uint8)
    return ref, test


# --------------------------------------------------------------------------------------
# box filter -- the helper the original project got wrong
# --------------------------------------------------------------------------------------

def test_box_filter_preserves_shape():
    """The original helper returned (H-1, W-1). This one must not."""
    a = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    for w in (3, 5, 11):
        assert M._box_filter(a, w).shape == a.shape


def test_box_filter_is_centred():
    """Interior values must equal the true centred mean, with no one-pixel shift."""
    a = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    out = M._box_filter(a, 11)
    expected = a[20 - 5:20 + 6, 20 - 5:20 + 6].mean()
    assert out[20, 20] == pytest.approx(expected)


def test_box_filter_constant_image():
    a = np.full((32, 32), 7.5)
    assert M._box_filter(a, 11) == pytest.approx(7.5)


def test_box_filter_rejects_even_window():
    with pytest.raises(ValueError):
        M._box_filter(np.zeros((16, 16)), 10)


# --------------------------------------------------------------------------------------
# PSNR
# --------------------------------------------------------------------------------------

def test_psnr_matches_skimage(pair):
    ref, test = pair
    assert M.psnr(ref, test) == pytest.approx(sk_psnr(ref, test, data_range=255), rel=1e-12)


def test_psnr_identical_is_infinite(pair):
    """A lossless codec must produce inf, not a large finite number (AUDIT.md B4)."""
    ref, _ = pair
    assert M.psnr(ref, ref) == float("inf")


def test_psnr_known_value():
    """A uniform offset of 1 LSB gives MSE = 1, so PSNR = 20*log10(255)."""
    ref = np.full((16, 16), 100, dtype=np.uint8)
    test = np.full((16, 16), 101, dtype=np.uint8)
    assert M.psnr(ref, test) == pytest.approx(20 * np.log10(255.0))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.psnr(np.zeros((8, 8)), np.zeros((8, 9)))


# --------------------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------------------

def test_ssim_gaussian_matches_skimage(pair):
    ref, test = pair
    ours = M.ssim(ref, test, window="gaussian")
    theirs = sk_ssim(ref, test, data_range=255, gaussian_weights=True,
                     sigma=1.5, use_sample_covariance=False)
    assert ours == pytest.approx(theirs, abs=1e-10)


def test_ssim_uniform_matches_skimage(pair):
    ref, test = pair
    ours = M.ssim(ref, test, window="uniform", win_size=11)
    theirs = sk_ssim(ref, test, data_range=255, win_size=11, use_sample_covariance=False)
    assert ours == pytest.approx(theirs, abs=1e-10)


def test_ssim_identical_is_one(pair):
    ref, _ = pair
    assert M.ssim(ref, ref) == pytest.approx(1.0, abs=1e-12)


def test_ssim_decreases_with_noise(pair):
    """Monotonicity: more distortion must not score higher (AUDIT.md M4)."""
    ref, _ = pair
    rng = np.random.default_rng(1)
    scores = []
    for sigma in (2, 8, 20, 40):
        noisy = np.clip(ref + rng.normal(0, sigma, ref.shape), 0, 255).astype(np.uint8)
        scores.append(M.ssim(ref, noisy))
    assert scores == sorted(scores, reverse=True)


def test_ssim_rejects_unknown_window(pair):
    ref, test = pair
    with pytest.raises(ValueError):
        M.ssim(ref, test, window="triangular")


def test_ssim_handles_multichannel(pair):
    """Three identical channels must score the same as one (AUDIT.md M2 rationale)."""
    ref, test = pair
    ref3 = np.stack([ref] * 3, axis=-1)
    test3 = np.stack([test] * 3, axis=-1)
    assert M.ssim(ref3, test3) == pytest.approx(M.ssim(ref, test), abs=1e-12)


# --------------------------------------------------------------------------------------
# Spatial Information
# --------------------------------------------------------------------------------------

def test_si_of_flat_image_is_zero():
    assert M.spatial_information(np.full((32, 32), 128, dtype=np.uint8)) == pytest.approx(0.0)


def test_si_of_linear_ramp_is_near_zero():
    """SI measures the *variability* of the gradient, not its magnitude.

    A linear ramp has a constant Sobel response, so its standard deviation is ~0 even
    though the image is far from flat. This matters when reading the results: the
    ``Gradient`` benchmark image scores low SI by construction, and a low SI there does
    not mean detail was lost the way it does on textured content.
    """
    ramp = np.tile(np.linspace(0, 255, 256), (256, 1)).astype(np.uint8)
    assert M.spatial_information(ramp) < 1.0


def test_si_of_perfect_checkerboard_is_zero():
    """A known blind spot of SI, worth pinning because the paper uses it as a proxy.

    A single-pixel checkerboard is maximally detailed, yet every pixel has the *same*
    Sobel magnitude, so the standard deviation is exactly zero. SI rewards *varied*
    gradient structure, not high-frequency content per se. Conclusions drawn from SI
    should therefore always be corroborated by PSNR/SSIM, as they are here.
    """
    checker = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    assert M.spatial_information(checker) == pytest.approx(0.0, abs=1e-9)


def test_si_increases_with_varied_structure():
    flat = np.full((64, 64), 128, dtype=np.uint8)
    blobs = (128 + 60 * np.sin(np.linspace(0, 6 * np.pi, 64))[None, :]
             * np.cos(np.linspace(0, 6 * np.pi, 64))[:, None]).astype(np.uint8)
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    si = [M.spatial_information(x) for x in (flat, blobs, noise)]
    assert si[0] < si[1] < si[2]


def test_si_blurring_reduces_detail():
    """The property SI is used for: a blurrier reconstruction must score lower."""
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 256, size=(128, 128)).astype(np.uint8)
    blurred = M._box_filter(sharp.astype(float), 11).astype(np.uint8)
    assert M.spatial_information(blurred) < M.spatial_information(sharp)


# --------------------------------------------------------------------------------------
# Rate
# --------------------------------------------------------------------------------------

def test_bpp_uncompressed_8bit_is_8():
    """The sanity anchor for every rate figure in the study."""
    w = h = 512
    assert M.bits_per_pixel(w * h, w, h) == pytest.approx(8.0)


def test_bpp_known_value():
    assert M.bits_per_pixel(855 * 1024, 1200, 1200) == pytest.approx(4.864, abs=1e-3)


def test_bpp_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        M.bits_per_pixel(1000, 0, 100)
