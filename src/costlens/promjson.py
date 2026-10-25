"""Reader and writer for Prometheus HTTP API range-query (``/api/v1/query_range``) responses.

The wire format is::

    {"status": "success",
     "data": {"resultType": "matrix",
              "result": [{"metric": {"namespace": "a", ...},
                          "values": [[1717200000, "0.25"], ...]}]}}

Sample values are strings so that ``NaN`` / ``+Inf`` survive JSON; they are parsed to float64.
Files may be plain ``.json`` or gzip-compressed ``.json.gz``.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from costlens.errors import ExportFormatError


@dataclass(frozen=True)
class Series:
    """One labelled time series from a matrix result."""

    labels: Mapping[str, str]
    timestamps: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

    def label(self, name: str, default: str = "") -> str:
        """Return a label value, or ``default`` when the label is absent."""
        return self.labels.get(name, default)


def parse_range_response(payload: Any, source: str = "<memory>") -> list[Series]:
    """Parse a decoded range-query response into :class:`Series` objects.

    Raises:
        ExportFormatError: when the payload is not a successful ``matrix`` response.
    """
    if not isinstance(payload, dict):
        raise ExportFormatError(f"{source}: top-level JSON value must be an object")
    status = payload.get("status")
    if status != "success":
        error = payload.get("error", "no error message")
        raise ExportFormatError(f"{source}: response status is {status!r} ({error})")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ExportFormatError(f"{source}: missing 'data' object")
    result_type = data.get("resultType")
    if result_type != "matrix":
        raise ExportFormatError(
            f"{source}: resultType is {result_type!r}; export with /api/v1/query_range",
        )
    result = data.get("result")
    if not isinstance(result, list):
        raise ExportFormatError(f"{source}: 'data.result' must be a list")
    return [_parse_series(item, source, idx) for idx, item in enumerate(result)]


def _parse_series(item: Any, source: str, idx: int) -> Series:
    where = f"{source}: result[{idx}]"
    if not isinstance(item, dict):
        raise ExportFormatError(f"{where} must be an object")
    metric = item.get("metric", {})
    if not isinstance(metric, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metric.items()
    ):
        raise ExportFormatError(f"{where}: 'metric' must map strings to strings")
    values = item.get("values")
    if not isinstance(values, list):
        raise ExportFormatError(f"{where}: 'values' must be a list (is this an instant query?)")
    timestamps = np.empty(len(values), dtype=np.float64)
    samples = np.empty(len(values), dtype=np.float64)
    for j, pair in enumerate(values):
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ExportFormatError(f"{where}: sample {j} is not a [timestamp, value] pair")
        ts, raw = pair
        if not isinstance(ts, int | float) or isinstance(ts, bool):
            raise ExportFormatError(f"{where}: sample {j} has a non-numeric timestamp")
        if not isinstance(raw, str):
            raise ExportFormatError(f"{where}: sample {j} value must be a string")
        try:
            samples[j] = float(raw)
        except ValueError as exc:
            raise ExportFormatError(f"{where}: sample {j} value {raw!r} is not a number") from exc
        timestamps[j] = float(ts)
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ExportFormatError(f"{where}: timestamps must be strictly increasing")
    return Series(labels=dict(metric), timestamps=timestamps, values=samples)


def load_file(path: Path) -> list[Series]:
    """Load one export file (``.json`` or ``.json.gz``)."""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
    except FileNotFoundError as exc:
        raise ExportFormatError(f"{path}: file not found") from exc
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, EOFError) as exc:
        raise ExportFormatError(f"{path}: cannot decode JSON ({exc})") from exc
    return parse_range_response(payload, source=path.name)


def format_value(value: float) -> str:
    """Format a sample the way Prometheus does (shortest round-trip repr, NaN/+Inf)."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


def build_response(series: Iterable[tuple[Mapping[str, str], Iterable[tuple[float, float]]]]) -> (
    dict[str, Any]
):
    """Build a range-query response object from ``(labels, [(ts, value), ...])`` pairs."""
    result = []
    for labels, points in series:
        values = [[_ts(ts), format_value(v)] for ts, v in points]
        if values:
            result.append({"metric": dict(labels), "values": values})
    return {"status": "success", "data": {"resultType": "matrix", "result": result}}


def _ts(ts: float) -> int | float:
    return int(ts) if float(ts).is_integer() else float(ts)


def write_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a response; gzip (with a fixed mtime, so output is reproducible) for ``.gz``."""
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as gz:
            gz.write(text.encode("utf-8"))
    else:
        path.write_text(text, encoding="utf-8")
