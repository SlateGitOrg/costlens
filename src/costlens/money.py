"""Exact money arithmetic: integer micro-dollars and largest-remainder apportionment.

Cost allocation is a conservation problem: every micro-dollar a node costs must land on exactly
one tenant (or on the explicit idle/shared buckets). Doing the split in floating point leaves
the books off by rounding noise that grows with the number of splits, so costlens represents
money as ``int64`` micro-dollars and splits integers with Hamilton's largest-remainder method.
That makes ``sum(parts) == total`` an exact identity, which the property tests assert with
``==`` rather than ``approx``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from costlens.errors import ConfigError

MICROS_PER_DOLLAR = 1_000_000

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


def to_micros(dollars: float) -> int:
    """Convert dollars to the nearest integer micro-dollar."""
    return round(dollars * MICROS_PER_DOLLAR)


def to_dollars(micros: int | np.integer[npt.NBitBase] | float) -> float:
    """Convert micro-dollars to (float) dollars, for display only."""
    return float(micros) / MICROS_PER_DOLLAR


def apportion_rows(totals: IntArray, weights: FloatArray, fallback_col: int = -1) -> IntArray:
    """Split each integer ``totals[r]`` across the columns of ``weights[r]`` exactly.

    The result ``out`` satisfies, for every row ``r``:

    * ``out[r].sum() == totals[r]`` (exact conservation),
    * ``|out[r, k] - totals[r] * w[r, k] / w[r].sum()| < 1`` (each part is within one unit of
      its exact proportional share),
    * zero-weight columns receive nothing, and
    * the result is deterministic (ties go to the lower column index).

    Rows whose weights sum to zero send their whole total to ``fallback_col``.
    Complexity is ``O(R * K log K)`` time (one row-wise argsort) and ``O(R * K)`` space.

    Args:
        totals: shape ``(R,)`` non-negative integers to split.
        weights: shape ``(R, K)`` non-negative finite weights.
        fallback_col: column receiving the total of rows with no positive weight.
    """
    totals = np.asarray(totals, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or totals.ndim != 1 or weights.shape[0] != totals.shape[0]:
        raise ValueError(
            f"shape mismatch: totals {totals.shape} vs weights {weights.shape}",
        )
    rows, cols = weights.shape
    if cols == 0:
        if np.any(totals != 0):
            raise ValueError("cannot apportion a non-zero total across zero columns")
        return np.zeros((rows, 0), dtype=np.int64)
    if np.any(totals < 0):
        raise ValueError("totals must be non-negative")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")

    weights = weights.copy()
    row_sum = weights.sum(axis=1)
    empty = row_sum <= 0
    if np.any(empty):
        weights[empty, :] = 0.0
        weights[empty, fallback_col] = 1.0
        row_sum = weights.sum(axis=1)

    exact = weights * (totals.astype(np.float64) / row_sum)[:, None]
    base = np.floor(exact).astype(np.int64)
    # Floating error can push floor() one unit too high/low; clamp to the zero-weight rule.
    base[weights == 0] = 0
    frac = exact - base
    remainder = totals - base.sum(axis=1)

    positive = remainder > 0
    if np.any(positive):
        eligible = weights > 0
        # Hand the leftover units to the eligible columns with the largest fractional parts
        # (stable sort: ties go to the lower index). The remainder is below the number of
        # eligible columns for well-formed input; divmod keeps pathological float cases exact.
        n_eligible = np.maximum(eligible.sum(axis=1), 1)
        elig_rank = _eligible_ranks(frac, eligible)
        add = (elig_rank < (remainder % n_eligible)[:, None]) & eligible & positive[:, None]
        base += add.astype(np.int64)
        base += ((remainder // n_eligible) * positive)[:, None] * eligible

    negative = remainder < 0
    if np.any(negative):
        # Take units back from the columns with the smallest fractional parts that hold units.
        holders = base > 0
        neg_order = np.argsort(frac + np.where(holders, 0.0, 2.0), axis=1, kind="stable")
        neg_ranks = np.empty_like(neg_order)
        np.put_along_axis(
            neg_ranks, neg_order, np.arange(cols)[None, :].repeat(rows, axis=0), axis=1
        )
        take = (neg_ranks < (-remainder)[:, None]) & negative[:, None] & holders
        base -= take.astype(np.int64)

    return base


def _eligible_ranks(frac: FloatArray, eligible: npt.NDArray[np.bool_]) -> IntArray:
    """Rank of each eligible column by descending fractional part (ineligible -> large)."""
    rows, cols = frac.shape
    keyed = np.where(eligible, -frac, np.inf)
    order = np.argsort(keyed, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(cols)[None, :].repeat(rows, axis=0), axis=1)
    return ranks.astype(np.int64)


def apportion(total: int, weights: FloatArray | list[float], fallback_col: int = -1) -> IntArray:
    """Split a single integer total across ``weights`` (see :func:`apportion_rows`)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    rows = apportion_rows(np.array([total], dtype=np.int64), w[None, :], fallback_col)
    out: IntArray = rows[0]
    return out


def spread_evenly(total: int, slots: int) -> IntArray:
    """Split ``total`` into ``slots`` integers differing by at most one (earlier slots larger)."""
    if slots <= 0:
        raise ConfigError("cannot spread a total over zero slots")
    if total < 0:
        raise ConfigError("cannot spread a negative total")
    base, extra = divmod(total, slots)
    out = np.full(slots, base, dtype=np.int64)
    out[:extra] += 1
    return out
