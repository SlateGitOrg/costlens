"""Parsing and formatting of Kubernetes-style quantities and durations.

Only the subset that cost tooling actually meets is supported: CPU quantities (``250m``,
``1.5``), memory quantities with binary or decimal suffixes (``512Mi``, ``2G``) and Go-style
durations made of ``d``/``h``/``m``/``s`` components (``1h30m``, ``7d``).
"""

from __future__ import annotations

import re

from costlens.errors import UnitParseError

GIB = 1024**3
MIB = 1024**2

_BINARY = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4, "Pi": 1024**5}
_DECIMAL = {"k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}
_QUANTITY_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([A-Za-z]{0,2})\s*$")
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(d|h|m|s)")
_DURATION_SECONDS = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def parse_cpu(text: str | float | int) -> float:
    """Parse a CPU quantity into cores (``"250m"`` -> ``0.25``)."""
    if isinstance(text, int | float):
        return _non_negative(float(text), str(text))
    match = _QUANTITY_RE.match(text)
    if match is None:
        raise UnitParseError(f"invalid CPU quantity: {text!r}")
    number, suffix = float(match.group(1)), match.group(2)
    if suffix == "":
        return _non_negative(number, text)
    if suffix == "m":
        return _non_negative(number / 1000.0, text)
    raise UnitParseError(f"invalid CPU suffix {suffix!r} in {text!r}")


def parse_memory(text: str | float | int) -> float:
    """Parse a memory quantity into bytes (``"512Mi"`` -> ``536870912``)."""
    if isinstance(text, int | float):
        return _non_negative(float(text), str(text))
    match = _QUANTITY_RE.match(text)
    if match is None:
        raise UnitParseError(f"invalid memory quantity: {text!r}")
    number, suffix = float(match.group(1)), match.group(2)
    if suffix == "":
        return _non_negative(number, text)
    if suffix in _BINARY:
        return _non_negative(number * _BINARY[suffix], text)
    if suffix in _DECIMAL:
        return _non_negative(number * _DECIMAL[suffix], text)
    raise UnitParseError(f"invalid memory suffix {suffix!r} in {text!r}")


def parse_duration(text: str | float | int) -> float:
    """Parse a Go-style duration (``"1h30m"``) or a number of seconds into seconds."""
    if isinstance(text, int | float):
        return _positive(float(text), str(text))
    stripped = text.strip()
    if not stripped:
        raise UnitParseError("empty duration")
    pos = 0
    total = 0.0
    for match in _DURATION_RE.finditer(stripped):
        if match.start() != pos:
            raise UnitParseError(f"invalid duration: {text!r}")
        total += float(match.group(1)) * _DURATION_SECONDS[match.group(2)]
        pos = match.end()
    if pos != len(stripped):
        try:
            return _positive(float(stripped), text)
        except ValueError as exc:
            raise UnitParseError(f"invalid duration: {text!r}") from exc
    return _positive(total, text)


def format_cpu(cores: float) -> str:
    """Format cores the way a manifest would carry them (millicores below 10 cores)."""
    if cores < 10:
        return f"{round(cores * 1000):d}m"
    return f"{cores:.2f}"


def format_memory(num_bytes: float) -> str:
    """Format bytes with the largest binary suffix that keeps the value >= 1."""
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.2f}Gi"
    return f"{num_bytes / MIB:.0f}Mi"


def format_duration(seconds: float) -> str:
    """Format seconds as a compact duration (``5400`` -> ``1h30m``)."""
    remaining = round(seconds)
    if remaining <= 0:
        return "0s"
    parts: list[str] = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, remaining = divmod(remaining, size)
        if count:
            parts.append(f"{count}{unit}")
    return "".join(parts)


def _non_negative(value: float, text: str) -> float:
    if value < 0:
        raise UnitParseError(f"quantity must be non-negative: {text!r}")
    return value


def _positive(value: float, text: str) -> float:
    if value <= 0:
        raise UnitParseError(f"duration must be positive: {text!r}")
    return value
