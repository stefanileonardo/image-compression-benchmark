"""Shared interface for the classical codecs.

Every codec exposes the same contract: given a PIL image and a quality setting, encode
it in memory, decode it back, and report the encoded size, the decoded array and the
elapsed encode time.

Two corrections relative to the original notebooks (see AUDIT.md):

* **M1** — the encoded byte count is recorded at *every* sweep point. The originals only
  measured the single top-quality file written to disk, which is why no rate-distortion
  curve could be produced.
* **M2** — grayscale sources are encoded as single-channel ``mode='L'``. The originals
  called ``.convert('RGB')`` on images that were already grayscale, inflating the
  compressed size by 16-73 % and biasing the size comparison against the classical
  codecs. PSNR/SSIM were unaffected, since the three channels were identical.
* **M3** — encode time is the median of ``repeats`` runs after a warm-up, rather than a
  single unwarmed measurement at the 40 ms scale.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
from PIL import Image

__all__ = ["EncodeResult", "CodecSweep", "timed_encode"]


@dataclass
class EncodeResult:
    """One operating point on a codec's rate-distortion curve."""

    codec: str
    image: str
    quality: Any            # the codec's own setting; not comparable across codecs
    n_bytes: int
    bpp: float
    psnr: float             # may be +inf for a lossless operating point
    ssim: float
    si: float
    encode_seconds: float
    lossless: bool = False

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def timed_encode(save_fn, repeats: int = 5, warmup: int = 1) -> tuple[bytes, float]:
    """Run ``save_fn(buffer)`` repeatedly and return the payload and the median time.

    ``save_fn`` must write the encoded image into the buffer it is given.
    """
    for _ in range(max(0, warmup)):
        save_fn(io.BytesIO())

    times: list[float] = []
    payload = b""
    for _ in range(max(1, repeats)):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        save_fn(buf)
        times.append(time.perf_counter() - t0)
        payload = buf.getvalue()

    return payload, float(np.median(times))


class CodecSweep:
    """Base class: subclasses supply ``name``, ``qualities`` and ``_save``."""

    name: str = "base"

    def __init__(self, repeats: int = 5, **options: Any) -> None:
        self.repeats = repeats
        self.options = options

    # -- to be implemented by subclasses ------------------------------------------------

    def qualities(self) -> Iterable[Any]:
        raise NotImplementedError

    def _save(self, image: Image.Image, buf: io.BytesIO, quality: Any) -> None:
        raise NotImplementedError

    def is_lossless(self, quality: Any) -> bool:
        return False

    # -- shared machinery ---------------------------------------------------------------

    def encode_once(self, image: Image.Image, quality: Any) -> tuple[bytes, np.ndarray, float]:
        """Encode at one quality and decode back. Returns (payload, decoded, seconds)."""
        payload, seconds = timed_encode(
            lambda b: self._save(image, b, quality), repeats=self.repeats
        )
        decoded = Image.open(io.BytesIO(payload))
        # keep the source's channel layout so metrics compare like with like
        if image.mode == "L":
            decoded = decoded.convert("L")
        return payload, np.array(decoded, dtype=np.uint8), seconds

    def sweep(self, image_path, image_name: str, ssim_window: str = "gaussian"):
        """Yield an :class:`EncodeResult` for every quality setting."""
        from src import metrics as M

        image = Image.open(image_path)
        if image.mode not in ("L", "RGB"):
            image = image.convert("L" if image.mode in ("1", "I;16", "LA") else "RGB")
        original = np.array(image, dtype=np.uint8)
        height, width = original.shape[:2]

        for q in self.qualities():
            try:
                payload, decoded, seconds = self.encode_once(image, q)
            except Exception as exc:                     # noqa: BLE001 - report and continue
                print(f"  [{self.name}] {image_name} q={q}: encode failed ({exc})")
                continue

            yield EncodeResult(
                codec=self.name,
                image=image_name,
                quality=q,
                n_bytes=len(payload),
                bpp=M.bits_per_pixel(len(payload), width, height),
                psnr=M.psnr(original, decoded),
                ssim=M.ssim(original, decoded, window=ssim_window),
                si=M.spatial_information(decoded),
                encode_seconds=seconds,
                lossless=self.is_lossless(q),
            )
