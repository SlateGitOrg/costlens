"""Quantity and duration parsing."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from costlens.errors import UnitParseError
from costlens.units import (
    GIB,
    MIB,
    format_cpu,
    format_duration,
    format_memory,
    parse_cpu,
    parse_duration,
    parse_memory,
)


@pytest.mark.parametrize(
    ("text", "cores"),
    [("250m", 0.25), ("1", 1.0), ("1.5", 1.5), (" 100m ", 0.1), (2, 2.0), ("1e1", 10.0)],
)
def test_parse_cpu(text: str | int, cores: float) -> None:
    assert parse_cpu(text) == pytest.approx(cores)


@pytest.mark.parametrize("text", ["", "abc", "1Gi", "-1", "1 m m"])
def test_parse_cpu_rejects(text: str) -> None:
    with pytest.raises(UnitParseError):
        parse_cpu(text)


@pytest.mark.parametrize(
    ("text", "num_bytes"),
    [("512Mi", 512 * MIB), ("2Gi", 2 * GIB), ("1G", 1e9), ("1000", 1000.0), ("1k", 1000.0)],
)
def test_parse_memory(text: str, num_bytes: float) -> None:
    assert parse_memory(text) == pytest.approx(num_bytes)


@pytest.mark.parametrize("text", ["1Xi", "lots", "-5Mi"])
def test_parse_memory_rejects(text: str) -> None:
    with pytest.raises(UnitParseError):
        parse_memory(text)
    with pytest.raises(UnitParseError):
        parse_memory(-1)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("1h30m", 5400.0), ("7d", 604800.0), ("15m", 900.0), ("90", 90.0), (60, 60.0), ("1.5h", 5400)],
)
def test_parse_duration(text: str | int, seconds: float) -> None:
    assert parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "1h30", "abc", "0s", "h1"])
def test_parse_duration_rejects(text: str) -> None:
    with pytest.raises(UnitParseError):
        parse_duration(text)


@given(st.integers(min_value=1, max_value=10**7))
def test_duration_round_trip(seconds: int) -> None:
    assert parse_duration(format_duration(seconds)) == seconds


def test_formatters() -> None:
    assert format_cpu(0.25) == "250m"
    assert format_cpu(12.0) == "12.00"
    assert format_memory(512 * MIB) == "512Mi"
    assert format_memory(3 * GIB) == "3.00Gi"
    assert format_duration(0) == "0s"
