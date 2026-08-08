"""JPEG (ISO/IEC 10918) via Pillow's libjpeg binding."""

from __future__ import annotations

import io
from typing import Iterable

from PIL import Image

from .base import CodecSweep


class JPEG(CodecSweep):
    """Baseline JPEG sweep over the libjpeg quality factor.

    Chroma subsampling is disabled (4:4:4) so colour handling never confounds the
    comparison. For a grayscale (``mode='L'``) source there is no chroma at all, which
    is the point of encoding it natively rather than as RGB.
    """

    name = "JPEG"

    def __init__(
        self,
        qualities: Iterable[int] = range(10, 101, 10),
        subsampling: int = 0,
        optimize: bool = True,
        repeats: int = 5,
    ) -> None:
        super().__init__(repeats=repeats)
        self._qualities = list(qualities)
        self.subsampling = subsampling
        self.optimize = optimize

    def qualities(self) -> list[int]:
        return self._qualities

    def _save(self, image: Image.Image, buf: io.BytesIO, quality: int) -> None:
        image.save(
            buf,
            format="JPEG",
            quality=int(quality),
            subsampling=self.subsampling,
            optimize=self.optimize,
        )

    def is_lossless(self, quality: int) -> bool:
        # Baseline JPEG is lossy at every quality factor, including 100.
        return False
