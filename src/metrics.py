"""Objective quality and rate metrics.

All functions accept 2-D grayscale or 3-D multi-channel ``uint8``/float arrays and
assume an 8-bit dynamic range unless ``data_range`` says otherwise.

Notes on correctness (see docs/AUDIT.md):

* ``_box_filter`` here is the *corrected* version. The original project helper
  (``_box_filter_same``) returned an ``(H-1, W-1)`` map with the window shifted by one
  pixel, because it used ``cumsum[w:] - cumsum[:-w]`` without prepending a zero row.
  This module prepends the zero row, so windows are exact and centred and the output
  keeps the input shape.
* :func:`ssim` defaults to the **Gaussian** ``sigma=1.5`` window of Wang et al. (2004),
  which is what ``scikit-image`` and the published literature use. The original project
  used a uniform window, which reads roughly +0.003 to +0.008 high. Pass
  ``window="uniform"`` to reproduce the legacy numbers.
* :func:`psnr` returns ``inf`` for a lossless reconstruction. Callers must decide how to
  aggregate that; see :func:`src.analysis.aggregate.summarise` for the project policy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["psnr", "ssim", "spatial_information", "bits_per_pixel", "mse"]


# --------------------------------------------------------------------------------------
# local windows
# --------------------------------------------------------------------------------------

def _box_filter(img: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average over a ``w x w`` window, preserving shape."""
    if w % 2 == 0:
        raise ValueError("window size must be odd")
    pad = w // 2
    p = np.pad(img, pad, mode="reflect")

    c = np.zeros((p.shape[0] + 1, p.shape[1]), dtype=np.float64)
    c[1:] = np.cumsum(p, axis=0)
    t = c[w:] - c[:-w]                      # -> (H, W + 2*pad)

    c2 = np.zeros((t.shape[0], t.shape[1] + 1), dtype=np.float64)
    c2[:, 1:] = np.cumsum(t, axis=1)
    r = c2[:, w:] - c2[:, :-w]              # -> (H, W)

    return r / float(w * w)


def _conv1d(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Vectorised 1-D correlation along ``axis``, 'valid' mode.

    Uses a strided sliding-window view plus one BLAS-backed contraction. This is roughly
    two orders of magnitude faster than ``np.apply_along_axis`` on a 2300x2300 image,
    which matters because the largest benchmark image is exactly that size.
    """
    windows = np.lib.stride_tricks.sliding_window_view(a, k.size, axis=axis)
    return np.tensordot(windows, k, axes=([-1], [0]))


def _gaussian_filter(img: np.ndarray, sigma: float, truncate: float = 3.5) -> np.ndarray:
    """Separable Gaussian blur, reflect-padded, shape preserving."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()

    p = np.pad(np.asarray(img, dtype=np.float64), radius, mode="reflect")
    return _conv1d(_conv1d(p, k, axis=0), k, axis=1)


def _filter(img: np.ndarray, window: str, win_size: int, sigma: float) -> np.ndarray:
    if window == "uniform":
        return _box_filter(img, win_size)
    if window == "gaussian":
        return _gaussian_filter(img, sigma)
    raise ValueError(f"unknown window {window!r}; use 'gaussian' or 'uniform'")


# --------------------------------------------------------------------------------------
# distortion metrics
# --------------------------------------------------------------------------------------

def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean squared error between two images of identical shape."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    r"""Peak signal-to-noise ratio in dB.

    .. math:: \mathrm{PSNR} = 10 \log_{10}\!\left(\frac{L^2}{\mathrm{MSE}}\right)

    Returns ``inf`` when the reconstruction is bit-exact (MSE = 0), which happens for
    JPEG 2000 in reversible mode and for any lossless codec.
    """
    m = mse(reference, test)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / m))


def ssim(
    reference: np.ndarray,
    test: np.ndarray,
    data_range: float = 255.0,
    window: str = "gaussian",
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    crop_border: bool = True,
) -> float:
    r"""Mean structural similarity index.

    .. math::
        \mathrm{SSIM}(\mathbf{x}) =
        \frac{(2\mu_I\mu_{\hat I} + C_1)(2\sigma_{I\hat I} + C_2)}
             {(\mu_I^2 + \mu_{\hat I}^2 + C_1)(\sigma_I^2 + \sigma_{\hat I}^2 + C_2)}

    Multi-channel input is handled by averaging the per-channel SSIM.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    if a.ndim == 3:
        return float(np.mean([
            ssim(a[..., c], b[..., c], data_range, window, win_size, sigma, K1, K2,
                 crop_border)
            for c in range(a.shape[2])
        ]))

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_a = _filter(a, window, win_size, sigma)
    mu_b = _filter(b, window, win_size, sigma)
    saa = _filter(a * a, window, win_size, sigma) - mu_a * mu_a
    sbb = _filter(b * b, window, win_size, sigma) - mu_b * mu_b
    sab = _filter(a * b, window, win_size, sigma) - mu_a * mu_b

    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    smap = num / den

    if crop_border:
        # Discard the border where the window overlapped padded pixels. This is what
        # scikit-image and the reference implementation do, so values here are directly
        # comparable to published SSIM numbers.
        pad = (win_size - 1) // 2 if window == "uniform" else int(3.5 * sigma + 0.5)
        if smap.shape[0] > 2 * pad and smap.shape[1] > 2 * pad:
            smap = smap[pad:-pad, pad:-pad]

    return float(np.mean(smap))


def spatial_information(img: np.ndarray) -> float:
    r"""Spatial Information, ITU-T Rec. P.910.

    Standard deviation of the Sobel gradient magnitude. Higher values indicate more
    retained structural detail; blurring a reconstruction lowers SI.

    **Caveat.** SI rewards *varied* gradient structure, not high-frequency content as
    such. Two degenerate cases score zero despite being far from flat: a linear ramp
    (constant gradient) and a single-pixel checkerboard (constant gradient magnitude).
    Treat SI as corroborating evidence alongside PSNR/SSIM, never on its own. See
    ``tests/test_metrics.py`` for both cases.

    .. math:: \mathrm{SI} = \operatorname{std}\sqrt{(S_h * I)^2 + (S_v * I)^2}
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    p = np.pad(a, 1, mode="reflect")
    gx = (
        -1 * p[:-2, :-2] + 1 * p[:-2, 2:]
        - 2 * p[1:-1, :-2] + 2 * p[1:-1, 2:]
        - 1 * p[2:, :-2] + 1 * p[2:, 2:]
    )
    gy = (
        -1 * p[:-2, :-2] - 2 * p[:-2, 1:-1] - 1 * p[:-2, 2:]
        + 1 * p[2:, :-2] + 2 * p[2:, 1:-1] + 1 * p[2:, 2:]
    )
    return float(np.std(np.sqrt(gx ** 2 + gy ** 2)))


# --------------------------------------------------------------------------------------
# rate
# --------------------------------------------------------------------------------------

def bits_per_pixel(n_bytes: int, width: int, height: int) -> float:
    r"""Compressed rate in bits per pixel.

    .. math:: \mathrm{bpp} = \frac{8 \cdot \text{bytes}}{W \times H}

    An 8-bit grayscale source is 8.00 bpp uncompressed; that is the reference against
    which any codec must be judged.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return float(8.0 * n_bytes / (width * height))
