"""Exponentially bucketed, exponentially decaying histograms (the VPA recommender's core).

Bucket ``i`` covers ``[start(i), start(i + 1))`` with
``start(i) = first * (ratio**i - 1) / (ratio - 1)``, so bucket width grows by ``ratio`` per
bucket: fine resolution for small values, bounded relative error (``ratio - 1``) for large
ones, and a fixed bucket count independent of the number of samples.

Each sample's weight is multiplied by ``2 ** ((t - reference) / half_life)``, so a sample one
half-life older counts half as much. Percentiles are scale-invariant, which lets costlens put
the reference at the newest sample (all decay factors <= 1, no overflow) instead of shifting
the reference periodically as the Go implementation does.

Complexity: adding ``n`` samples is ``O(n)`` (vectorised bucketing + ``bincount``); a
percentile query is ``O(B)`` for ``B`` buckets (a few hundred); memory is ``O(B)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HistogramOptions:
    """Bucket layout; the defaults mirror the VPA recommender's CPU and memory options."""

    first_bucket: float
    ratio: float = 1.05
    max_value: float = 1000.0

    def __post_init__(self) -> None:
        if self.first_bucket <= 0 or self.ratio <= 1 or self.max_value <= self.first_bucket:
            raise ValueError("need first_bucket > 0, ratio > 1 and max_value > first_bucket")

    @property
    def n_buckets(self) -> int:
        """Number of buckets needed to reach ``max_value``."""
        return (
            math.ceil(
                math.log(self.max_value * (self.ratio - 1) / self.first_bucket + 1)
                / math.log(self.ratio)
            )
            + 1
        )

    def bucket_start(self, index: int | npt.NDArray[np.int64]) -> FloatArray | float:
        """Lower edge of bucket ``index``."""
        return self.first_bucket * (np.power(self.ratio, index) - 1) / (self.ratio - 1)

    def find_bucket(self, values: FloatArray) -> npt.NDArray[np.int64]:
        """Bucket index of each value (values beyond ``max_value`` land in the last bucket)."""
        clipped = np.maximum(values, 0.0)
        raw = np.floor(
            np.log1p(clipped * (self.ratio - 1) / self.first_bucket) / math.log(self.ratio)
        )
        idx = np.clip(raw, 0, self.n_buckets - 1).astype(np.int64)
        # log/exp rounding differs across libm builds; a value sitting on a bucket edge could
        # land one bucket low or high. Snap to the bucket whose [start, next start) holds it.
        starts = np.asarray(self.bucket_start(idx))
        idx = np.where(
            (idx < self.n_buckets - 1) & (clipped >= self.bucket_start(idx + 1)), idx + 1, idx
        )
        idx = np.where(
            (idx > 0) & (clipped < starts) & (clipped < self.bucket_start(idx)), idx - 1, idx
        )
        return idx.astype(np.int64)


CPU_OPTIONS = HistogramOptions(first_bucket=0.01, ratio=1.05, max_value=1000.0)
MEMORY_OPTIONS = HistogramOptions(first_bucket=1e7, ratio=1.05, max_value=1e12)


@dataclass
class DecayingHistogram:
    """A weighted histogram whose sample weights halve every ``half_life_seconds``."""

    options: HistogramOptions
    half_life_seconds: float
    reference: float = 0.0
    weights: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        if self.half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        self.weights = np.zeros(self.options.n_buckets)

    @property
    def total_weight(self) -> float:
        """Sum of (decayed) sample weights."""
        return float(self.weights.sum())

    def is_empty(self) -> bool:
        """True when no sample with positive weight was added."""
        return self.total_weight <= 0.0

    def add_samples(
        self, values: FloatArray, timestamps: FloatArray, weights: FloatArray | None = None
    ) -> None:
        """Add samples (NaN values are ignored); weights default to 1."""
        values = np.asarray(values, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64)
        keep = ~np.isnan(values) & (w > 0)
        if not np.any(keep):
            return
        values, timestamps, w = values[keep], timestamps[keep], w[keep]
        newest = float(timestamps.max())
        if newest > self.reference:
            # Re-anchor on the newest sample: rescale existing weights so ratios are kept.
            if self.total_weight > 0:
                self.weights *= np.exp2((self.reference - newest) / self.half_life_seconds)
            self.reference = newest
        decay = np.exp2((timestamps - self.reference) / self.half_life_seconds)
        buckets = self.options.find_bucket(values)
        self.weights += np.bincount(buckets, weights=w * decay, minlength=self.options.n_buckets)

    def merge(self, other: DecayingHistogram) -> None:
        """Add another histogram with the same options and half-life into this one."""
        if other.options != self.options or other.half_life_seconds != self.half_life_seconds:
            raise ValueError("cannot merge histograms with different options")
        if other.is_empty():
            return
        reference = max(self.reference, other.reference)
        mine = self.weights * np.exp2((self.reference - reference) / self.half_life_seconds)
        theirs = other.weights * np.exp2((other.reference - reference) / self.half_life_seconds)
        self.weights = mine + theirs
        self.reference = reference

    def percentile(self, p: float) -> float:
        """Upper edge of the bucket holding the ``p``-th weighted percentile (0 if empty).

        Like the VPA, the *end* of the bucket is returned, so the estimate errs on the side of
        more resources by at most one bucket width (``ratio - 1`` relative).
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError("percentile must be within [0, 1]")
        total = self.total_weight
        if total <= 0:
            return 0.0
        cumulative = np.cumsum(self.weights)
        threshold = p * total
        index = int(np.searchsorted(cumulative, threshold - 1e-12 * total, side="left"))
        index = min(index, self.options.n_buckets - 1)
        if p >= 1.0:
            index = int(np.flatnonzero(self.weights > 0)[-1])
        return float(self.options.bucket_start(index + 1))
