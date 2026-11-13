"""JPEG 2000 (ISO/IEC 15444) via Pillow's OpenJPEG binding.

Pillow exposes rate control through ``quality_mode='dB'`` with ``quality_layers`` giving
target PSNR values. The top of the ladder uses the reversible 5/3 wavelet with no rate
targeting, which is **genuinely lossless**: measured MSE is exactly 0, so PSNR is
``inf``.

That is not a bug, but it must be handled explicitly rather than fed into ``np.mean``.
The original project hardcoded ``float('inf')`` into its metric tables and then
aggregated with an unguarded ``np.mean`` / ``np.max`` (docs/AUDIT.md, B4). Here the operating
point is flagged via ``lossless=True`` so downstream aggregation can apply a stated
policy.
"""

from __future__ import annotations

import io
from typing import Iterable

from PIL import Image

from .base import CodecSweep


class JPEG2000(CodecSweep):
    """JPEG 2000 sweep.

    ``labels`` are arbitrary ladder positions kept for continuity with the original
    notebooks; ``psnr_targets`` are what actually drives the encoder. Neither is
    comparable to another codec's quality number — compare on **bpp**.
    """

    name = "JPEG2000"

    def __init__(
        self,
        labels: Iterable[int] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        psnr_targets: Iterable[int] = (28, 32, 36, 40, 44, 48, 52, 56, 60),
        repeats: int = 5,
    ) -> None:
        super().__init__(repeats=repeats)
        self._labels = list(labels)
        self._targets = list(psnr_targets)
        if len(self._targets) != len(self._labels) - 1:
            raise ValueError(
                "expected one PSNR target per label except the last, "
                "which is the lossless point"
            )
        self._lossless_label = self._labels[-1]

    def qualities(self) -> list[int]:
        return self._labels

    def _target_for(self, label: int) -> int:
        return self._targets[self._labels.index(label)]

    def _save(self, image: Image.Image, buf: io.BytesIO, quality: int) -> None:
        if self.is_lossless(quality):
            image.save(buf, format="JPEG2000", irreversible=False)
        else:
            image.save(
                buf,
                format="JPEG2000",
                quality_mode="dB",
                quality_layers=[self._target_for(quality)],
            )

    def is_lossless(self, quality: int) -> bool:
        return quality == self._lossless_label
