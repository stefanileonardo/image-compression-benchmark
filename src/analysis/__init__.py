"""Aggregation and figure generation, driven entirely by results/metrics/*.csv."""
from .aggregate import LOSSLESS_POLICY, summarise, rate_distortion_table, lossless_report

__all__ = ["LOSSLESS_POLICY", "summarise", "rate_distortion_table", "lossless_report"]
