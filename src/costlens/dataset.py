"""Turn a Prometheus export directory into dense, aligned numpy arrays.

The :class:`ClusterDataset` is the single in-memory model every analysis consumes:

* a regular time grid of ``S`` steps (sample ``t`` covers ``[t, t + step)``),
* ``N`` nodes with per-step capacity (``NaN`` where the node did not exist),
* ``C`` containers (one per ``namespace/pod/container``) with per-step usage, requests and
  limits, a presence mask, the node each pod ran on, its labels and its workload.

Ingestion is strict about structure (misaligned timestamps or missing required labels raise)
and lenient about gaps (missing samples become "absent"); every repair it performs is recorded
in :attr:`ClusterDataset.warnings` so reports can surface data-quality issues.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from costlens.catalog import METRICS, Manifest, load_manifest
from costlens.errors import DataQualityError, ExportFormatError
from costlens.promjson import Series, load_file

log = logging.getLogger(__name__)

FloatMatrix = npt.NDArray[np.float64]
FloatArray = npt.NDArray[np.float64]
BoolMatrix = npt.NDArray[np.bool_]

_RS_HASH = re.compile(r"^(?P<base>.+)-[a-z0-9]{5,10}$")
_CRONJOB_SUFFIX = re.compile(r"^(?P<base>.+)-\d{6,}$")
_ALIGN_TOLERANCE = 0.01  # fraction of a step a timestamp may deviate from the grid


@dataclass(frozen=True)
class NodeInfo:
    """Static node attributes used for pricing."""

    name: str
    instance_type: str
    capacity_type: str
    zone: str


@dataclass(frozen=True)
class ContainerInfo:
    """Identity and metadata of one container."""

    namespace: str
    pod: str
    container: str
    node_index: int
    workload: str
    pod_labels: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    @property
    def key(self) -> str:
        """``namespace/pod/container``."""
        return f"{self.namespace}/{self.pod}/{self.container}"


@dataclass(frozen=True)
class ClusterDataset:
    """Aligned per-step arrays for one cluster export (shape ``(S, N)`` or ``(S, C)``)."""

    cluster: str
    start: int
    step_seconds: int
    nodes: list[NodeInfo]
    node_cpu: FloatMatrix
    node_memory: FloatMatrix
    containers: list[ContainerInfo]
    present: BoolMatrix
    cpu_usage: FloatMatrix
    memory_usage: FloatMatrix
    cpu_request: FloatMatrix
    memory_request: FloatMatrix
    cpu_limit: FloatMatrix
    memory_limit: FloatMatrix
    oom_events: FloatMatrix
    namespace_labels: dict[str, dict[str, str]]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        """Number of grid steps ``S``."""
        return int(self.node_cpu.shape[0])

    @property
    def step_hours(self) -> float:
        """Length of one step in hours."""
        return self.step_seconds / 3600.0

    @property
    def end(self) -> int:
        """Exclusive end timestamp of the grid."""
        return self.start + self.n_steps * self.step_seconds

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        """Start timestamp of every step."""
        return self.start + self.step_seconds * np.arange(self.n_steps, dtype=np.int64)

    @property
    def node_present(self) -> BoolMatrix:
        """``(S, N)`` mask of steps at which each node existed."""
        return ~np.isnan(self.node_cpu)

    def container_node_indices(self) -> npt.NDArray[np.int64]:
        """``(C,)`` node index of every container."""
        return np.array([c.node_index for c in self.containers], dtype=np.int64)

    def slice_steps(self, begin: int, stop: int) -> ClusterDataset:
        """Return the sub-dataset covering steps ``[begin, stop)`` (same nodes/containers)."""
        if not 0 <= begin < stop <= self.n_steps:
            raise ValueError(f"invalid step range [{begin}, {stop}) for {self.n_steps} steps")
        sl = slice(begin, stop)
        return replace(
            self,
            start=self.start + begin * self.step_seconds,
            node_cpu=self.node_cpu[sl],
            node_memory=self.node_memory[sl],
            present=self.present[sl],
            cpu_usage=self.cpu_usage[sl],
            memory_usage=self.memory_usage[sl],
            cpu_request=self.cpu_request[sl],
            memory_request=self.memory_request[sl],
            cpu_limit=self.cpu_limit[sl],
            memory_limit=self.memory_limit[sl],
            oom_events=self.oom_events[sl],
            warnings=list(self.warnings),
        )

    def label_of(self, container: ContainerInfo, label: str) -> str:
        """Resolve a label for a container: pod labels first, then namespace labels."""
        if label in container.pod_labels:
            return container.pod_labels[label]
        return self.namespace_labels.get(container.namespace, {}).get(label, "")


def workload_name(pod: str, owner_kind: str, owner_name: str) -> str:
    """Derive the stable workload name of a pod from its owner reference.

    ReplicaSets are named ``<deployment>-<pod-template-hash>`` and CronJob-spawned Jobs are
    named ``<cronjob>-<scheduled-minute>``; both suffixes change on every rollout / run, so they
    are stripped to obtain the Deployment / CronJob that a rightsizing recommendation targets.
    """
    if owner_kind == "ReplicaSet":
        match = _RS_HASH.match(owner_name)
        return f"Deployment/{match.group('base')}" if match else f"ReplicaSet/{owner_name}"
    if owner_kind == "Job":
        match = _CRONJOB_SUFFIX.match(owner_name)
        return f"CronJob/{match.group('base')}" if match else f"Job/{owner_name}"
    if owner_kind:
        return f"{owner_kind}/{owner_name}"
    return f"Pod/{pod}"


class _Grid:
    """Maps sample timestamps onto step indices."""

    def __init__(self, manifest: Manifest) -> None:
        self.start = manifest.start
        self.step = manifest.step_seconds
        self.n = manifest.n_steps
        self.dropped_outside = 0

    def indices(self, series: Series, source: str) -> tuple[npt.NDArray[np.int64], FloatArray]:
        pos = (series.timestamps - self.start) / self.step
        idx = np.rint(pos)
        if np.any(np.abs(pos - idx) > _ALIGN_TOLERANCE):
            raise ExportFormatError(
                f"{source}: timestamps are not aligned to start={self.start} "
                f"step={self.step}s (export with the manifest's start and step)",
            )
        keep = (idx >= 0) & (idx < self.n)
        self.dropped_outside += int((~keep).sum())
        return idx[keep].astype(np.int64), series.values[keep]



def _load_metric(directory: Path, manifest: Manifest, name: str) -> list[tuple[str, Series]]:
    out: list[tuple[str, Series]] = []
    spec = METRICS[name]
    for rel in manifest.files.get(name, []):
        path = directory / rel
        for series in load_file(path):
            missing = [lbl for lbl in spec.required_labels if not series.label(lbl)]
            if missing:
                raise ExportFormatError(
                    f"{rel}: series {dict(series.labels)} lacks required labels {missing}",
                )
            out.append((rel, series))
    return out


def load_dataset(directory: Path) -> ClusterDataset:
    """Load and align every metric of an export directory."""
    manifest = load_manifest(directory)
    grid = _Grid(manifest)
    warnings: list[str] = []
    s = manifest.n_steps

    loaded = {name: _load_metric(directory, manifest, name) for name in METRICS}

    # ---- nodes ---------------------------------------------------------------------------
    node_meta: dict[str, NodeInfo] = {}
    for _, series in loaded["node_labels"]:
        name = series.label("node")
        node_meta[name] = NodeInfo(
            name=name,
            instance_type=series.label("label_node_kubernetes_io_instance_type"),
            capacity_type=series.label("label_karpenter_sh_capacity_type", "on-demand")
            or "on-demand",
            zone=series.label("label_topology_kubernetes_io_zone"),
        )
    cap_nodes = sorted(
        {series.label("node") for _, series in loaded["node_cpu_capacity_cores"]},
    )
    for name in cap_nodes:
        if name not in node_meta:
            raise DataQualityError(f"node {name!r} has capacity samples but no node_labels")
    node_index = {name: i for i, name in enumerate(cap_nodes)}
    nodes = [node_meta[n] for n in cap_nodes]
    node_cpu = _fill(grid, loaded["node_cpu_capacity_cores"], node_index, s, _node_key, warnings)
    node_memory = _fill(
        grid, loaded["node_memory_capacity_bytes"], node_index, s, _node_key, warnings
    )
    mismatch = np.isnan(node_cpu) != np.isnan(node_memory)
    if np.any(mismatch):
        warnings.append(
            f"{int(mismatch.sum())} node-steps had only one of cpu/memory capacity; dropped",
        )
        node_cpu[mismatch] = np.nan
        node_memory[mismatch] = np.nan

    # ---- containers ----------------------------------------------------------------------
    container_metrics = (
        "cpu_usage_cores",
        "memory_working_set_bytes",
        "cpu_request_cores",
        "memory_request_bytes",
        "cpu_limit_cores",
        "memory_limit_bytes",
    )
    node_votes: dict[tuple[str, str, str], Counter[str]] = {}
    for metric in container_metrics:
        for _, series in loaded[metric]:
            key = _container_key(series)
            node_votes.setdefault(key, Counter())[series.label("node")] += len(series.values)

    keys: list[tuple[str, str, str]] = []
    container_node: list[int] = []
    for key in sorted(node_votes):
        votes = node_votes[key]
        node, _ = votes.most_common(1)[0]
        if len(votes) > 1:
            warnings.append(f"container {'/'.join(key)} reported on several nodes; using {node}")
        if node not in node_index:
            warnings.append(f"container {'/'.join(key)} ran on unknown node {node!r}; skipped")
            continue
        keys.append(key)
        container_node.append(node_index[node])
    col = {key: i for i, key in enumerate(keys)}

    def fill(metric: str) -> FloatMatrix:
        return _fill(grid, loaded[metric], col, s, _container_key, warnings)

    cpu_usage = fill("cpu_usage_cores")
    memory_usage = fill("memory_working_set_bytes")
    cpu_request = fill("cpu_request_cores")
    memory_request = fill("memory_request_bytes")
    cpu_limit = fill("cpu_limit_cores")
    memory_limit = fill("memory_limit_bytes")
    oom_by_key = _fill(grid, loaded["oom_events"], col, s, _container_key, warnings)
    oom_events = np.nan_to_num(oom_by_key, nan=0.0)

    present = ~(np.isnan(cpu_usage) & np.isnan(memory_usage))
    present |= ~np.isnan(cpu_request) | ~np.isnan(memory_request)
    node_idx_arr = np.array(container_node, dtype=np.int64)
    if keys:
        node_alive = ~np.isnan(node_cpu[:, node_idx_arr])
        orphan = present & ~node_alive
        if np.any(orphan):
            warnings.append(
                f"{int(orphan.sum())} container-steps on a node without capacity data; dropped",
            )
            present &= node_alive
    for arr in (cpu_usage, memory_usage, cpu_request, memory_request):
        arr[~present] = np.nan
        arr[present & np.isnan(arr)] = 0.0
    for arr in (cpu_limit, memory_limit):
        arr[~present] = np.nan

    # ---- metadata ------------------------------------------------------------------------
    pod_labels: dict[tuple[str, str], dict[str, str]] = {}
    for _, series in loaded["pod_labels"]:
        pod_labels[(series.label("namespace"), series.label("pod"))] = _strip_label_prefix(
            series.labels,
        )
    owners: dict[tuple[str, str], tuple[str, str]] = {}
    for _, series in loaded["pod_owner"]:
        owners[(series.label("namespace"), series.label("pod"))] = (
            series.label("owner_kind"),
            series.label("owner_name"),
        )
    ns_labels: dict[str, dict[str, str]] = {}
    for _, series in loaded["namespace_labels"]:
        ns_labels[series.label("namespace")] = _strip_label_prefix(series.labels)

    containers = []
    for (namespace, pod, container), node_i in zip(keys, container_node, strict=True):
        kind, owner = owners.get((namespace, pod), ("", ""))
        containers.append(
            ContainerInfo(
                namespace=namespace,
                pod=pod,
                container=container,
                node_index=node_i,
                workload=workload_name(pod, kind, owner),
                pod_labels=pod_labels.get((namespace, pod), {}),
            ),
        )
    if grid.dropped_outside:
        warnings.append(f"{grid.dropped_outside} samples outside the manifest window ignored")
    for message in warnings:
        log.warning("%s", message)
    return ClusterDataset(
        cluster=manifest.cluster,
        start=manifest.start,
        step_seconds=manifest.step_seconds,
        nodes=nodes,
        node_cpu=node_cpu,
        node_memory=node_memory,
        containers=containers,
        present=present,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        cpu_request=cpu_request,
        memory_request=memory_request,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        oom_events=oom_events,
        namespace_labels=ns_labels,
        warnings=warnings,
    )


def _node_key(series: Series) -> str:
    return series.label("node")


def _container_key(series: Series) -> tuple[str, str, str]:
    return (series.label("namespace"), series.label("pod"), series.label("container"))


def _fill[K](
    grid: _Grid,
    items: Iterable[tuple[str, Series]],
    index: dict[K, int],
    n_steps: int,
    key_of: Callable[[Series], K],
    warnings: list[str],
) -> FloatMatrix:
    """Scatter series samples into an ``(S, len(index))`` matrix (NaN = no sample)."""
    out = np.full((n_steps, len(index)), np.nan, dtype=np.float64)
    conflicts = 0
    for source, series in items:
        column = index.get(key_of(series))
        if column is None:
            continue
        idx, values = grid.indices(series, source)
        current = out[idx, column]
        clash = ~np.isnan(current) & (current != values)
        conflicts += int(clash.sum())
        out[idx, column] = values
    if conflicts:
        warnings.append(f"{conflicts} overlapping samples disagreed; later file wins")
    return out


def _strip_label_prefix(labels: Mapping[str, str]) -> dict[str, str]:
    return {
        k.removeprefix("label_"): v
        for k, v in labels.items()
        if k.startswith("label_") and v != ""
    }
