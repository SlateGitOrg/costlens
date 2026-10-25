"""Decaying exponential histogram behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from costlens.rightsizing.histogram import (
    CPU_OPTIONS,
    MEMORY_OPTIONS,
    DecayingHistogram,
    HistogramOptions,
)

DAY = 86400.0


def test_bucket_layout_matches_vpa_cpu_options() -> None:
    opts = CPU_OPTIONS
    assert opts.bucket_start(0) == 0.0
    assert opts.bucket_start(1) == pytest.approx(0.01)
    assert opts.bucket_start(2) == pytest.approx(0.01 + 0.0105)
    assert opts.bucket_start(opts.n_buckets - 1) <= opts.max_value * 1.06
    assert opts.find_bucket(np.array([0.0, 0.0099, 0.01, 5000.0])).tolist() == [
        0,
        0,
        1,
        opts.n_buckets - 1,
    ]
    # both span five decades at 5% per bucket
    assert MEMORY_OPTIONS.n_buckets == CPU_OPTIONS.n_buckets == 176


def test_invalid_options() -> None:
    with pytest.raises(ValueError):
        HistogramOptions(first_bucket=0)
    with pytest.raises(ValueError):
        HistogramOptions(first_bucket=1, ratio=1.0)
    with pytest.raises(ValueError):
        DecayingHistogram(CPU_OPTIONS, 0)


def test_percentile_of_uniform_samples() -> None:
    hist = DecayingHistogram(CPU_OPTIONS, half_life_seconds=1e12)  # effectively no decay
    values = np.linspace(0.1, 1.0, 1000)
    hist.add_samples(values, np.zeros(1000))
    for p in (0.5, 0.9, 0.99):
        est = hist.percentile(p)
        exact = float(np.quantile(values, p))
        # bucket upper edge: never below the true quantile, at most one bucket (5%) above
        assert exact <= est <= exact * 1.05 + 0.011
    assert hist.percentile(1.0) >= 1.0
    assert hist.percentile(0.0) > 0


def test_empty_histogram_and_nan_samples() -> None:
    hist = DecayingHistogram(CPU_OPTIONS, DAY)
    assert hist.is_empty()
    assert hist.percentile(0.9) == 0.0
    hist.add_samples(np.array([np.nan]), np.array([0.0]))
    hist.add_samples(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    assert hist.is_empty()
    with pytest.raises(ValueError):
        hist.percentile(1.5)


def test_decay_halves_weight_per_half_life() -> None:
    hist = DecayingHistogram(CPU_OPTIONS, DAY)
    hist.add_samples(np.array([0.1]), np.array([0.0]))
    hist.add_samples(np.array([1.0]), np.array([DAY]))
    w = hist.weights[hist.weights > 0]
    assert w.tolist() == pytest.approx([0.5, 1.0])
    # Newer, larger samples dominate the median once old ones have decayed.
    assert hist.percentile(0.5) > 0.9


def test_merge_equals_adding_all_samples() -> None:
    rng = np.random.default_rng(1)
    a_vals, b_vals = rng.random(50), rng.random(70) * 3
    a_ts, b_ts = rng.random(50) * 5 * DAY, rng.random(70) * 5 * DAY
    a = DecayingHistogram(CPU_OPTIONS, DAY)
    b = DecayingHistogram(CPU_OPTIONS, DAY)
    both = DecayingHistogram(CPU_OPTIONS, DAY)
    a.add_samples(a_vals, a_ts)
    b.add_samples(b_vals, b_ts)
    both.add_samples(np.concatenate([a_vals, b_vals]), np.concatenate([a_ts, b_ts]))
    a.merge(b)
    a.merge(DecayingHistogram(CPU_OPTIONS, DAY))
    scale = both.total_weight / a.total_weight
    assert np.allclose(a.weights * scale, both.weights)
    for p in (0.5, 0.9):
        assert a.percentile(p) == both.percentile(p)
    with pytest.raises(ValueError):
        a.merge(DecayingHistogram(MEMORY_OPTIONS, DAY))


@given(
    st.lists(st.floats(0.0, 50.0, allow_nan=False), min_size=1, max_size=200),
    st.floats(0.01, 1.0),
)
@settings(max_examples=150, deadline=None)
def test_percentile_is_monotone_and_bounds_samples(values: list[float], p: float) -> None:
    hist = DecayingHistogram(CPU_OPTIONS, DAY)
    hist.add_samples(np.array(values), np.zeros(len(values)))
    lower = hist.percentile(p * 0.5)
    upper = hist.percentile(p)
    assert lower <= upper
    assert upper >= min(values)
    assert hist.percentile(1.0) >= max(values)


def test_incremental_add_is_order_independent() -> None:
    rng = np.random.default_rng(2)
    vals, ts = rng.random(100), np.sort(rng.random(100) * 10 * DAY)
    one = DecayingHistogram(CPU_OPTIONS, DAY)
    one.add_samples(vals, ts)
    chunked = DecayingHistogram(CPU_OPTIONS, DAY)
    for k in range(0, 100, 7):
        chunked.add_samples(vals[k : k + 7], ts[k : k + 7])
    assert np.allclose(one.weights, chunked.weights)
