"""Ingestion of export directories into aligned arrays."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from costlens.catalog import METRICS, Manifest, write_manifest
from costlens.dataset import load_dataset, workload_name
from costlens.errors import DataQualityError, ExportFormatError
from costlens.promjson import build_response, write_file

STEP = 60
START = 1_000_020  # divisible by 60 so the grid is clean


def _write(
    directory: Path, metric: str, series: list[tuple[dict[str, str], list[tuple[float, float]]]]
) -> str:
    name = f"{metric}.json"
    write_file(directory / name, build_response(series))
    return name


def _minimal(
    tmp_path: Path,
    *,
    usage_ts_offset: float = 0.0,
    extra_node: bool = False,
    orphan_node: str | None = None,
    split_node: bool = False,
) -> Path:
    ts = [START + i * STEP for i in range(4)]
    node = {"node": "n1"}
    c = {"namespace": "ns", "pod": "web-5d8f7c9b4-abcde", "container": "app", "node": "n1"}
    files: dict[str, list[str]] = {}

    def put(metric: str, series: list[tuple[dict[str, str], list[tuple[float, float]]]]) -> None:
        files[metric] = [_write(tmp_path, metric, series)]

    put("node_cpu_capacity_cores", [(node, [(t, 4.0) for t in ts])])
    put("node_memory_capacity_bytes", [(node, [(t, 8e9) for t in ts[:3]])])
    labels = [
        (
            {"node": "n1", "label_node_kubernetes_io_instance_type": "m5.xlarge"},
            [(t, 1.0) for t in ts],
        )
    ]
    if extra_node:
        labels.append(
            (
                {
                    "node": "n2",
                    "label_node_kubernetes_io_instance_type": "m5.xlarge",
                    "label_karpenter_sh_capacity_type": "spot",
                },
                [(ts[0], 1.0)],
            )
        )
    put("node_labels", labels)
    usage_series = [(c, [(t + usage_ts_offset, 0.5) for t in ts[1:]])]
    if split_node:
        usage_series.append(({**c, "node": "n2"}, [(ts[0], 0.5)]))
    put("cpu_usage_cores", usage_series)
    put("memory_working_set_bytes", [(c, [(t, 1e9) for t in ts])])
    put("cpu_request_cores", [(c, [(t, 1.0) for t in ts])])
    put("memory_request_bytes", [(c, [(t, 2e9) for t in ts])])
    if orphan_node:
        o = {**c, "pod": "lost", "node": orphan_node}
        files["memory_request_bytes"].append(_write(tmp_path, "orphan", [(o, [(ts[0], 1.0)])]))
    put(
        "pod_owner",
        [
            (
                {
                    "namespace": "ns",
                    "pod": c["pod"],
                    "owner_kind": "ReplicaSet",
                    "owner_name": "web-5d8f7c9b4",
                },
                [(ts[0], 1.0)],
            )
        ],
    )
    put(
        "pod_labels",
        [
            (
                {
                    "namespace": "ns",
                    "pod": c["pod"],
                    "label_app": "web",
                    "label_team": "",
                },
                [(ts[0], 1.0)],
            )
        ],
    )
    put("namespace_labels", [({"namespace": "ns", "label_team": "red"}, [(ts[0], 1.0)])])
    put("oom_events", [({"namespace": "ns", "pod": c["pod"], "container": "app"}, [(ts[2], 1.0)])])
    write_manifest(tmp_path, Manifest(START, START + 4 * STEP, STEP, "t", files))
    return tmp_path


def test_minimal_export_loads(tmp_path: Path) -> None:
    ds = load_dataset(_minimal(tmp_path))
    assert ds.n_steps == 4
    assert ds.timestamps.tolist() == [START + i * STEP for i in range(4)]
    assert ds.end == START + 4 * STEP
    assert ds.step_hours == pytest.approx(1 / 60)
    (c,) = ds.containers
    assert c.key == "ns/web-5d8f7c9b4-abcde/app"
    assert c.workload == "Deployment/web"
    assert c.pod_labels == {"app": "web"}
    assert ds.label_of(c, "team") == "red"
    assert ds.label_of(c, "app") == "web"
    # usage missing at step 0 while the container is present -> 0, not NaN
    assert ds.cpu_usage[:, 0].tolist()[:1] == [0.0]
    assert ds.oom_events[:, 0].tolist() == [0.0, 0.0, 1.0, 0.0]
    # node memory missing at the last step -> whole node-step dropped, container too
    assert ds.node_present[:, 0].tolist() == [True, True, True, False]
    assert not ds.present[3, 0]
    assert any("only one of cpu/memory" in w for w in ds.warnings)
    assert any("without capacity" in w for w in ds.warnings)


def test_slice_steps(tmp_path: Path) -> None:
    ds = load_dataset(_minimal(tmp_path))
    part = ds.slice_steps(1, 3)
    assert part.n_steps == 2
    assert part.start == START + STEP
    with pytest.raises(ValueError):
        ds.slice_steps(3, 3)


def test_misaligned_timestamps_rejected(tmp_path: Path) -> None:
    with pytest.raises(ExportFormatError, match="aligned"):
        load_dataset(_minimal(tmp_path, usage_ts_offset=7.0))


def test_small_jitter_is_tolerated(tmp_path: Path) -> None:
    ds = load_dataset(_minimal(tmp_path, usage_ts_offset=0.3))
    assert ds.cpu_usage[1, 0] == 0.5


def test_unknown_node_and_split_node_warnings(tmp_path: Path) -> None:
    ds = load_dataset(_minimal(tmp_path, orphan_node="ghost", split_node=True, extra_node=True))
    assert any("unknown node 'ghost'" in w for w in ds.warnings)
    assert any("several nodes" in w for w in ds.warnings)
    assert [c.pod for c in ds.containers] == ["web-5d8f7c9b4-abcde"]


def test_capacity_without_labels_rejected(tmp_path: Path) -> None:
    path = _minimal(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    name = _write(
        path, "nolabels", [({"node": "n1", "label_node_kubernetes_io_instance_type": "x"}, [])]
    )
    manifest["metrics"]["node_labels"] = [name]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataQualityError, match="no node_labels"):
        load_dataset(path)


def test_missing_required_label_rejected(tmp_path: Path) -> None:
    path = _minimal(tmp_path)
    write_file(
        path / "cpu_usage_cores.json",
        build_response([({"namespace": "ns", "pod": "p"}, [(START, 1.0)])]),
    )
    with pytest.raises(ExportFormatError, match="required labels"):
        load_dataset(path)


def test_conflicting_overlap_warns(tmp_path: Path) -> None:
    path = _minimal(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    c = {"namespace": "ns", "pod": "web-5d8f7c9b4-abcde", "container": "app", "node": "n1"}
    extra = _write(path, "dup", [(c, [(START + STEP, 0.9), (START + 99 * STEP, 1.0)])])
    manifest["metrics"]["cpu_usage_cores"].append(extra)
    (path / "manifest.json").write_text(json.dumps(manifest))
    ds = load_dataset(path)
    assert ds.cpu_usage[1, 0] == 0.9
    assert any("disagreed" in w for w in ds.warnings)
    assert any("outside the manifest window" in w for w in ds.warnings)


@pytest.mark.parametrize(
    ("pod", "kind", "owner", "expected"),
    [
        ("api-7d9f8c6b5-x2k9p", "ReplicaSet", "api-7d9f8c6b5", "Deployment/api"),
        ("weird", "ReplicaSet", "x", "ReplicaSet/x"),
        ("etl-28455-abcde", "Job", "etl-28455120", "CronJob/etl"),
        ("once-abcde", "Job", "once", "Job/once"),
        ("db-0", "StatefulSet", "db", "StatefulSet/db"),
        ("bare", "", "", "Pod/bare"),
    ],
)
def test_workload_name(pod: str, kind: str, owner: str, expected: str) -> None:
    assert workload_name(pod, kind, owner) == expected


def test_simulated_export_round_trips(small_export: Path, tmp_path: Path) -> None:
    ds = load_dataset(small_export)
    assert ds.warnings == []
    assert len(ds.nodes) == 7
    assert ds.n_steps == 14 * 24
    kinds = {c.workload.split("/")[0] for c in ds.containers}
    assert kinds == {"Deployment", "StatefulSet", "CronJob", "DaemonSet"}
    assert np.nansum(ds.oom_events) > 0
    # copying the export elsewhere keeps it loadable (manifest paths are relative)
    copy = tmp_path / "copy"
    shutil.copytree(small_export, copy)
    assert load_dataset(copy).n_steps == ds.n_steps
    assert set(METRICS) >= set(json.loads((copy / "manifest.json").read_text())["metrics"])
