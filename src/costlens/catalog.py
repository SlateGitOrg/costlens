"""The export contract: which PromQL queries costlens consumes and how an export is described.

An *export directory* holds one ``manifest.json`` plus one or more range-query response files
per logical metric. Several files per metric are allowed because Prometheus refuses range
queries that would return more than 11,000 points per series, so long periods at fine steps
must be exported in chunks (``costlens queries`` prints a chunked ``curl`` loop).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from costlens.errors import ManifestError

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = "costlens-export/v1"

CONTAINER_LABELS = ("namespace", "pod", "container", "node")


@dataclass(frozen=True)
class MetricSpec:
    """A logical metric: its PromQL, the labels it must carry, and whether it is required."""

    name: str
    promql: str
    required_labels: tuple[str, ...]
    required: bool
    description: str


_EXCLUDE = 'container!="", container!="POD"'

METRICS: dict[str, MetricSpec] = {
    spec.name: spec
    for spec in (
        MetricSpec(
            "cpu_usage_cores",
            "sum by (namespace, pod, container, node) "
            f"(rate(container_cpu_usage_seconds_total{{{_EXCLUDE}}}[5m]))",
            CONTAINER_LABELS,
            True,
            "CPU used per container, in cores.",
        ),
        MetricSpec(
            "memory_working_set_bytes",
            "max by (namespace, pod, container, node) "
            f"(max_over_time(container_memory_working_set_bytes{{{_EXCLUDE}}}[$step]))",
            CONTAINER_LABELS,
            True,
            "Peak working-set memory per container within each step, in bytes.",
        ),
        MetricSpec(
            "cpu_request_cores",
            'sum by (namespace, pod, container, node) '
            '(kube_pod_container_resource_requests{resource="cpu"})',
            CONTAINER_LABELS,
            True,
            "CPU request per container, in cores.",
        ),
        MetricSpec(
            "memory_request_bytes",
            'sum by (namespace, pod, container, node) '
            '(kube_pod_container_resource_requests{resource="memory"})',
            CONTAINER_LABELS,
            True,
            "Memory request per container, in bytes.",
        ),
        MetricSpec(
            "cpu_limit_cores",
            'sum by (namespace, pod, container, node) '
            '(kube_pod_container_resource_limits{resource="cpu"})',
            CONTAINER_LABELS,
            False,
            "CPU limit per container, in cores (absent = unlimited).",
        ),
        MetricSpec(
            "memory_limit_bytes",
            'sum by (namespace, pod, container, node) '
            '(kube_pod_container_resource_limits{resource="memory"})',
            CONTAINER_LABELS,
            False,
            "Memory limit per container, in bytes (absent = unlimited).",
        ),
        MetricSpec(
            "node_cpu_capacity_cores",
            'max by (node) (kube_node_status_capacity{resource="cpu"})',
            ("node",),
            True,
            "CPU capacity per node; a node exists at a step iff this has a sample.",
        ),
        MetricSpec(
            "node_memory_capacity_bytes",
            'max by (node) (kube_node_status_capacity{resource="memory"})',
            ("node",),
            True,
            "Memory capacity per node, in bytes.",
        ),
        MetricSpec(
            "node_labels",
            "max by (node, label_node_kubernetes_io_instance_type, "
            "label_karpenter_sh_capacity_type, label_topology_kubernetes_io_zone) "
            "(kube_node_labels)",
            ("node", "label_node_kubernetes_io_instance_type"),
            True,
            "Instance type / capacity type / zone used to price each node.",
        ),
        MetricSpec(
            "pod_labels",
            "max by (namespace, pod, label_app, label_team, label_cost_center) (kube_pod_labels)",
            ("namespace", "pod"),
            False,
            "Pod labels (label_* names) for label-based aggregation.",
        ),
        MetricSpec(
            "namespace_labels",
            "max by (namespace, label_team, label_cost_center) (kube_namespace_labels)",
            ("namespace",),
            False,
            "Namespace labels; the team label normally lives here.",
        ),
        MetricSpec(
            "pod_owner",
            "max by (namespace, pod, owner_kind, owner_name) (kube_pod_owner)",
            ("namespace", "pod", "owner_kind", "owner_name"),
            False,
            "Pod controller, used to group pods into workloads for rightsizing.",
        ),
        MetricSpec(
            "oom_events",
            "sum by (namespace, pod, container) "
            f"(increase(container_oom_events_total{{{_EXCLUDE}}}[$step]))",
            ("namespace", "pod", "container"),
            False,
            "OOM kills per container per step (drives the OOM-aware memory bump).",
        ),
    )
}


@dataclass(frozen=True)
class Manifest:
    """Describes one export: the step grid and the files holding each logical metric."""

    start: int
    end: int
    step_seconds: int
    cluster: str
    files: dict[str, list[str]]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Number of grid points; sample ``t`` covers ``[t, t + step)`` and ``end`` is excluded."""
        return (self.end - self.start) // self.step_seconds

    def to_json(self) -> dict[str, Any]:
        """Serialise to the on-disk JSON structure."""
        return {
            "format": FORMAT_VERSION,
            "cluster": self.cluster,
            "start": self.start,
            "end": self.end,
            "step_seconds": self.step_seconds,
            "metrics": {name: list(paths) for name, paths in sorted(self.files.items())},
            **({"extra": self.extra} if self.extra else {}),
        }


def load_manifest(directory: Path) -> Manifest:
    """Read and validate ``manifest.json`` in an export directory."""
    path = directory / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"{path}: no manifest (is this an export directory?)") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: manifest must be a JSON object")
    if raw.get("format") != FORMAT_VERSION:
        raise ManifestError(f"{path}: unsupported format {raw.get('format')!r}")
    try:
        start, end, step = int(raw["start"]), int(raw["end"]), int(raw["step_seconds"])
        metrics = raw["metrics"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"{path}: missing or invalid field ({exc})") from exc
    if step <= 0 or end <= start or (end - start) % step != 0:
        raise ManifestError(
            f"{path}: need step > 0 and (end - start) a positive multiple of step",
        )
    if not isinstance(metrics, dict):
        raise ManifestError(f"{path}: 'metrics' must be an object")
    files: dict[str, list[str]] = {}
    for name, value in metrics.items():
        if name not in METRICS:
            raise ManifestError(f"{path}: unknown metric {name!r}")
        paths = [value] if isinstance(value, str) else value
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ManifestError(f"{path}: metric {name!r} must list file names")
        for rel in paths:
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ManifestError(f"{path}: file {rel!r} must be relative to the export dir")
        files[name] = list(paths)
    missing = [m.name for m in METRICS.values() if m.required and m.name not in files]
    if missing:
        raise ManifestError(f"{path}: required metrics missing: {', '.join(missing)}")
    extra = raw.get("extra", {})
    return Manifest(
        start=start,
        end=end,
        step_seconds=step,
        cluster=str(raw.get("cluster", "cluster")),
        files=files,
        extra=extra if isinstance(extra, dict) else {},
    )


def write_manifest(directory: Path, manifest: Manifest) -> Path:
    """Write ``manifest.json`` into ``directory``."""
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True) + "\n")
    return path


def render_promql(spec: MetricSpec, step_seconds: int) -> str:
    """Substitute the step into a query template (``$step`` -> ``900s``)."""
    return spec.promql.replace("$step", f"{step_seconds}s")


def export_script(prometheus_url: str, step_seconds: int, chunk_seconds: int) -> str:
    """Return a POSIX shell script that exports every metric in chunks with ``curl``.

    The script only *prints* commands for the operator to review and run; costlens never talks
    to a live Prometheus itself.
    """
    max_chunk = step_seconds * 11000
    chunk = min(chunk_seconds, max_chunk)
    lines = [
        "#!/bin/sh",
        "# Export the metrics costlens needs from a Prometheus server.",
        "# Usage: START=<unix> END=<unix> OUT=<dir> sh export.sh",
        "set -eu",
        f'PROM="${{PROM:-{prometheus_url}}}"',
        f"STEP={step_seconds}",
        f"CHUNK={chunk}  # <= 11000 points per series per request",
        'mkdir -p "$OUT"',
        'i=0; t="$START"',
        'while [ "$t" -lt "$END" ]; do',
        '  e=$((t + CHUNK)); [ "$e" -gt "$END" ] && e="$END"',
    ]
    for spec in METRICS.values():
        query = render_promql(spec, step_seconds).replace("'", "'\"'\"'")
        lines.append(
            f"  curl -fsS --get \"$PROM/api/v1/query_range\" --data-urlencode '"
            f"query={query}' --data-urlencode \"start=$t\" --data-urlencode "
            f"\"end=$((e - STEP))\" --data-urlencode \"step=$STEP\" "
            f"| gzip -n > \"$OUT/{spec.name}.$i.json.gz\"",
        )
    lines += [
        "  i=$((i + 1)); t=$e",
        "done",
        'echo "now write $OUT/manifest.json listing the chunk files per metric"',
    ]
    return "\n".join(lines) + "\n"
