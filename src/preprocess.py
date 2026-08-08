"""Turn raw captures into the square grayscale benchmark set.

The reference SMoE implementation needs square, single-channel input, so every image is
centre-cropped to a square and converted to 8-bit grayscale. All four codecs then see
exactly the same pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["to_square_gray"]


def to_square_gray(src: Path, dst: Path) -> tuple[int, int]:
    """Centre-crop to a square, convert to 8-bit grayscale, write PNG.

    Returns the output ``(width, height)``.
    """
    img = Image.open(src)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("L")

    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG", optimize=True)
    return img.size
