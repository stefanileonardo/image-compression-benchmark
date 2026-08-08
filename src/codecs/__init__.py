"""Classical codec wrappers with a common sweep interface."""

from .base import CodecSweep, EncodeResult, timed_encode
from .jpeg import JPEG
from .jpeg2000 import JPEG2000
from .avif import AVIF

__all__ = ["CodecSweep", "EncodeResult", "timed_encode", "JPEG", "JPEG2000", "AVIF"]
