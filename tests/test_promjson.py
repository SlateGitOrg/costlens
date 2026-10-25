"""Prometheus range-query JSON parsing and export manifest handling."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

from costlens.catalog import (
    METRICS,
    Manifest,
    export_script,
    load_manifest,
    render_promql,
    write_manifest,
)
from costlens.errors import ExportFormatError, ManifestError
from costlens.promjson import (
    build_response,
    format_value,
    load_file,
    parse_range_response,
    write_file,
)


def _ok(result: list[object]) -> dict[str, object]:
    return {"status": "success", "data": {"resultType": "matrix", "result": result}}


def test_parse_valid_matrix() -> None:
    payload = _ok(
        [{"metric": {"node": "n1"}, "values": [[100, "1.5"], [160.0, "NaN"], [220, "+Inf"]]}],
    )
    (series,) = parse_range_response(payload)
    assert series.label("node") == "n1"
    assert series.label("missing", "d") == "d"
    assert series.timestamps.tolist() == [100.0, 160.0, 220.0]
    assert series.values[0] == 1.5
    assert math.isnan(series.values[1])
    assert math.isinf(series.values[2])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "object"),
        ({"status": "error", "error": "boom"}, "boom"),
        ({"status": "success"}, "data"),
        ({"status": "success", "data": {"resultType": "vector", "result": []}}, "query_range"),
        ({"status": "success", "data": {"resultType": "matrix", "result": {}}}, "list"),
        (_ok(["x"]), "object"),
        (_ok([{"metric": {"a": 1}, "values": []}]), "strings"),
        (_ok([{"metric": {}, "value": [1, "2"]}]), "instant"),
        (_ok([{"metric": {}, "values": [[1]]}]), "pair"),
        (_ok([{"metric": {}, "values": [["1", "2"]]}]), "timestamp"),
        (_ok([{"metric": {}, "values": [[1, 2]]}]), "string"),
        (_ok([{"metric": {}, "values": [[1, "abc"]]}]), "not a number"),
        (_ok([{"metric": {}, "values": [[2, "1"], [1, "1"]]}]), "increasing"),
    ],
)
def test_parse_rejects_malformed(payload: object, message: str) -> None:
    with pytest.raises(ExportFormatError, match=message):
        parse_range_response(payload)


def test_round_trip_plain_and_gzip(tmp_path: Path) -> None:
    response = build_response(
        [({"node": "a"}, [(0, 1.0), (60, 2.5)]), ({"node": "b"}, [])],
    )
    assert len(response["data"]["result"]) == 1  # type: ignore[index]
    for name in ("x.json", "x.json.gz"):
        path = tmp_path / name
        write_file(path, response)
        (series,) = load_file(path)
        assert series.values.tolist() == [1.0, 2.5]
    first = (tmp_path / "x.json.gz").read_bytes()
    write_file(tmp_path / "x.json.gz", response)
    assert (tmp_path / "x.json.gz").read_bytes() == first, "gzip output must be reproducible"


def test_load_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ExportFormatError, match="not found"):
        load_file(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ExportFormatError, match="decode"):
        load_file(bad)
    bad_gz = tmp_path / "bad.json.gz"
    bad_gz.write_bytes(gzip.compress(b"[1,2"))
    with pytest.raises(ExportFormatError, match="decode"):
        load_file(bad_gz)


def test_format_value() -> None:
    assert format_value(float("nan")) == "NaN"
    assert format_value(float("inf")) == "+Inf"
    assert format_value(float("-inf")) == "-Inf"
    assert float(format_value(0.1 + 0.2)) == 0.1 + 0.2


def _manifest_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "format": "costlens-export/v1",
        "start": 0,
        "end": 3600,
        "step_seconds": 900,
        "cluster": "c",
        "metrics": {m.name: [f"{m.name}.json"] for m in METRICS.values() if m.required},
    }
    base.update(overrides)
    return base


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(0, 3600, 900, "c", {"cpu_usage_cores": ["a.json"]}, {"seed": 1})
    write_manifest(tmp_path, manifest)
    raw = json.loads((tmp_path / "manifest.json").read_text())
    assert raw["extra"] == {"seed": 1}
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest_dict()))
    loaded = load_manifest(tmp_path)
    assert loaded.n_steps == 4
    assert loaded.cluster == "c"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_manifest_dict(format="v0"), "unsupported"),
        (_manifest_dict(step_seconds=0), "multiple"),
        (_manifest_dict(end=1000), "multiple"),
        (_manifest_dict(metrics=[]), "object"),
        (_manifest_dict(metrics={"bogus": ["x"]}), "unknown"),
        (_manifest_dict(metrics={"cpu_usage_cores": [1]}), "list file"),
        (_manifest_dict(metrics={"cpu_usage_cores": ["../x"]}), "relative"),
        (_manifest_dict(metrics={"cpu_usage_cores": "x.json"}), "required metrics"),
        ({"format": "costlens-export/v1"}, "missing"),
        ([1], "object"),
    ],
)
def test_manifest_rejects(tmp_path: Path, raw: object, message: str) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match=message):
        load_manifest(tmp_path)


def test_manifest_missing_or_invalid(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no manifest"):
        load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("{")
    with pytest.raises(ManifestError, match="invalid JSON"):
        load_manifest(tmp_path)


def test_promql_and_export_script() -> None:
    query = render_promql(METRICS["memory_working_set_bytes"], 900)
    assert "[900s]" in query and "$step" not in query
    script = export_script("http://prometheus:9090", 300, 10**9)
    assert "CHUNK=3300000" in script, "chunk must respect the 11k points limit"
    assert script.count("query_range") == len(METRICS)
