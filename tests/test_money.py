"""Exactness properties of integer apportionment."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from costlens.errors import ConfigError
from costlens.money import (
    apportion,
    apportion_rows,
    spread_evenly,
    to_dollars,
    to_nanos,
)


@st.composite
def problems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    rows = draw(st.integers(min_value=1, max_value=20))
    cols = draw(st.integers(min_value=1, max_value=12))
    totals = draw(
        arrays(np.int64, rows, elements=st.integers(min_value=0, max_value=10**12)),
    )
    weights = draw(
        arrays(
            np.float64,
            (rows, cols),
            elements=st.one_of(
                st.just(0.0),
                st.floats(min_value=1e-9, max_value=1e9, allow_nan=False),
            ),
        ),
    )
    return totals, weights


@given(problems())
@settings(max_examples=300, deadline=None)
def test_apportion_conserves_every_row_exactly(problem: tuple[np.ndarray, np.ndarray]) -> None:
    totals, weights = problem
    out = apportion_rows(totals, weights)
    assert out.dtype == np.int64
    assert np.array_equal(out.sum(axis=1), totals)
    assert np.all(out >= 0)


@given(problems())
@settings(max_examples=300, deadline=None)
def test_apportion_parts_within_one_unit_of_exact_share(
    problem: tuple[np.ndarray, np.ndarray],
) -> None:
    totals, weights = problem
    out = apportion_rows(totals, weights)
    sums = weights.sum(axis=1)
    for r in range(len(totals)):
        if sums[r] <= 0:
            assert out[r, -1] == totals[r]
            continue
        exact = totals[r] * (weights[r] / sums[r])
        # one unit of rounding plus float slack for totals near 1e12
        slack = 1.0 + 1e-9 * float(totals[r])
        assert np.all(np.abs(out[r] - exact) <= slack)
        assert np.all(out[r][weights[r] == 0] == 0)


def test_apportion_known_values_and_tie_break() -> None:
    assert apportion(10, [1, 1, 1]).tolist() == [4, 3, 3]
    # 3.5 / 1.75 / 1.75 -> floors 3,1,1; two leftover units go to the largest fractions
    # (0.75, 0.75 beat 0.5), ties resolved by column order.
    assert apportion(7, [0.5, 0.25, 0.25]).tolist() == [3, 2, 2]
    assert apportion(100, [0, 0, 0]).tolist() == [0, 0, 100]
    assert apportion(100, [0, 0, 0], fallback_col=0).tolist() == [100, 0, 0]
    assert apportion(0, [1, 2]).tolist() == [0, 0]


def test_apportion_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    totals = rng.integers(0, 10**6, 50)
    weights = rng.random((50, 6))
    assert np.array_equal(apportion_rows(totals, weights), apportion_rows(totals, weights))


def test_apportion_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        apportion_rows(np.array([1, 2]), np.ones((3, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        apportion_rows(np.array([-1]), np.ones((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        apportion_rows(np.array([1]), np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="zero columns"):
        apportion_rows(np.array([1]), np.ones((1, 0)))
    assert apportion_rows(np.array([0]), np.ones((1, 0))).shape == (1, 0)
    with pytest.raises(ValueError, match="one-dimensional"):
        apportion(1, np.ones((2, 2)))


@given(st.integers(min_value=0, max_value=10**15), st.integers(min_value=1, max_value=500))
def test_spread_evenly(total: int, slots: int) -> None:
    out = spread_evenly(total, slots)
    assert out.sum() == total
    assert out.max() - out.min() <= 1


def test_spread_evenly_rejects_bad_input() -> None:
    with pytest.raises(ConfigError):
        spread_evenly(1, 0)
    with pytest.raises(ConfigError):
        spread_evenly(-1, 2)


def test_nano_conversion_round_trip() -> None:
    assert to_nanos(1.2345678905) == 1234567890 or to_nanos(1.2345678905) == 1234567891
    assert to_nanos(73.0) == 73_000_000_000
    assert to_dollars(1_234_567_890) == pytest.approx(1.23456789)
