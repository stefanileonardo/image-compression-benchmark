"""AVIF (AV1 Image File Format) via ``pillow-heif`` (libheif + libaom).

Two fixes relative to the original notebook (AUDIT.md m2, m3):

* the encoder ``speed`` setting is actually passed to ``save()``. The original defined
  ``speed = 6`` with a comment about the speed/quality tradeoff and then never used it,
  so every encode ran at the library default and the timings did not reflect the
  documented configuration;
* the original markdown claimed ``pillow-avif-plugin`` while the code used
  ``pillow-heif``. Only ``pillow-heif`` is used here.

Output is version-sensitive: a different libaom build produces different bytes at the
same nominal quality. Pin the version (see requirements.txt) when reporting numbers.
"""

from __future__ import annotations

import io
from typing import Iterable

from PIL import Image

from .base import CodecSweep


def register() -> None:
    """Register AVIF/HEIF support with Pillow. Safe to call more than once."""
    from pillow_heif import register_heif_opener

    register_heif_opener()


class AVIF(CodecSweep):
    """AVIF sweep over the libaom quality scale.

    ``quality=100`` is *near*-lossless, not lossless — ``pillow-heif`` reserves
    ``quality=-1`` for true lossless. The ladder here stays lossy throughout so that
    every point is a genuine rate-distortion operating point.
    """

    name = "AVIF"

    def __init__(
        self,
        qualities: Iterable[int] = range(10, 101, 10),
        speed: int = 6,
        repeats: int = 5,
    ) -> None:
        super().__init__(repeats=repeats)
        self._qualities = list(qualities)
        self.speed = speed
        register()

    def qualities(self) -> list[int]:
        return self._qualities

    def _save(self, image: Image.Image, buf: io.BytesIO, quality: int) -> None:
        image.save(buf, format="AVIF", quality=int(quality), speed=self.speed)

    def is_lossless(self, quality: int) -> bool:
        return False
