"""Write a :class:`SimulatedCluster` as a costlens export directory.

The output is exactly what ``costlens queries`` would fetch from a real Prometheus: one
range-query response per metric per chunk (gzip), a ``manifest.json``, plus two files that a
real export would not have — ``pricing.toml`` (the catalog the simulated nodes were billed
from) and ``ground_truth.json`` (see :mod:`costlens.synth.truth`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from costlens.catalog import Manifest, write_manifest
from costlens.promjson import build_response, write_file
from costlens.synth.generator import SimContainer, SimulatedCluster
from costlens.synth.truth import ground_truth
from costlens.units import MIB

SeriesPoints = tuple[dict[str, str], list[tuple[float, float]]]


def _points(
    timestamps: np.ndarray, values: np.ndarray, a: int, b: int
) -> list[tuple[float, float]]:
    seg_t = timestamps[a:b]
    seg_v = values[a:b]
    keep = ~np.isnan(seg_v)
    return list(zip(seg_t[keep].tolist(), seg_v[keep].tolist(), strict=True))


def _container_labels(sim: SimulatedCluster, c: SimContainer) -> dict[str, str]:
    return {
        "namespace": c.namespace,
        "pod": c.pod,
        "container": c.container,
        "node": sim.nodes[c.node].name,
    }


def _constant(c: SimContainer, value: float | None, n_steps: int) -> np.ndarray:
    out = np.full(n_steps, np.nan)
    if value is not None:
        out[c.start : c.stop] = value
    return out


def _metric_series(sim: SimulatedCluster, a: int, b: int) -> Iterator[tuple[str, SeriesPoints]]:
    ts = sim.timestamps
    n = sim.n_steps
    for node in sim.nodes:
        alive = np.full(n, np.nan)
        alive[node.alive_from : node.alive_to] = 1.0
        labels = {"node": node.name}
        yield "node_cpu_capacity_cores", (labels, _points(ts, alive * node.cores, a, b))
        yield "node_memory_capacity_bytes", (labels, _points(ts, alive * node.memory_bytes, a, b))
        yield (
            "node_labels",
            (
                {
                    "node": node.name,
                    "label_node_kubernetes_io_instance_type": node.pool.instance_type,
                    "label_karpenter_sh_capacity_type": node.pool.capacity_type,
                    "label_topology_kubernetes_io_zone": node.zone,
                },
                _points(ts, alive, a, b),
            ),
        )
    seen_pods: set[str] = set()
    for i, c in enumerate(sim.containers):
        labels = _container_labels(sim, c)
        spec = c.spec
        yield "cpu_usage_cores", (labels, _points(ts, sim.cpu_usage[:, i], a, b))
        yield "memory_working_set_bytes", (labels, _points(ts, sim.memory_usage[:, i], a, b))
        yield "cpu_request_cores", (labels, _points(ts, _constant(c, spec.cpu_request, n), a, b))
        yield (
            "memory_request_bytes",
            (labels, _points(ts, _constant(c, spec.mem_request_mib * MIB, n), a, b)),
        )
        if spec.cpu_limit is not None:
            yield "cpu_limit_cores", (labels, _points(ts, _constant(c, spec.cpu_limit, n), a, b))
        if spec.mem_limit_mib is not None:
            yield (
                "memory_limit_bytes",
                (labels, _points(ts, _constant(c, spec.mem_limit_mib * MIB, n), a, b)),
            )
        oom_labels = {k: labels[k] for k in ("namespace", "pod", "container")}
        yield "oom_events", (oom_labels, _points(ts, sim.oom_events[:, i], a, b))
        pod_key = f"{c.namespace}/{c.pod}"
        if pod_key in seen_pods:
            continue
        seen_pods.add(pod_key)
        alive = _constant(c, 1.0, n)
        pod_labels = {"namespace": c.namespace, "pod": c.pod}
        pod_labels.update({f"label_{k}": v for k, v in sorted(c.pod_labels.items())})
        yield "pod_labels", (pod_labels, _points(ts, alive, a, b))
        yield (
            "pod_owner",
            (
                {
                    "namespace": c.namespace,
                    "pod": c.pod,
                    "owner_kind": c.owner_kind,
                    "owner_name": c.owner_name,
                },
                _points(ts, alive, a, b),
            ),
        )
    for ns in sim.spec.namespaces:
        labels = {"namespace": ns.name}
        if ns.team:
            labels["label_team"] = ns.team
        if ns.cost_center:
            labels["label_cost_center"] = ns.cost_center
        yield "namespace_labels", (labels, _points(ts, np.ones(n), a, b))


TRUTH_METRICS = ("true_cpu_demand_cores", "true_memory_demand_bytes")


def _truth_series(sim: SimulatedCluster, a: int, b: int) -> Iterator[tuple[str, SeriesPoints]]:
    """Uncensored demand per container (simulation-only; not something Prometheus has)."""
    ts = sim.timestamps
    for i, c in enumerate(sim.containers):
        labels = _container_labels(sim, c)
        yield "true_cpu_demand_cores", (labels, _points(ts, sim.cpu_demand[:, i], a, b))
        yield "true_memory_demand_bytes", (labels, _points(ts, sim.memory_demand[:, i], a, b))


def pricing_toml(sim: SimulatedCluster) -> str:
    """The pricing catalog matching the simulated node pools."""
    lines = [
        "# Pricing catalog for the simulated cluster.",
        "# Prices are illustrative approximations of public on-demand list prices; spot",
        "# prices are synthetic. Verify against your provider before relying on them.",
        'currency = "USD"',
        f"cpu_to_memory_cost_ratio = {sim.spec.cpu_to_memory_cost_ratio}",
        "default_spot_discount = 0.65",
        "",
    ]
    seen: set[str] = set()
    for pool in sim.spec.pools:
        if pool.instance_type in seen:
            continue
        seen.add(pool.instance_type)
        lines += [
            "[[instance]]",
            f'type = "{pool.instance_type}"',
            f"vcpu = {pool.cores:g}",
            f"memory_gib = {pool.memory_gib:g}",
        ]
        if pool.capacity_type == "spot":
            lines += [
                f"on_demand_hourly = {pool.hourly_usd / 0.35:.4f}",
                f"spot_hourly = {pool.hourly_usd}",
            ]
        else:
            lines.append(f"on_demand_hourly = {pool.hourly_usd}")
        lines.append("")
    return "\n".join(lines)


def write_export(sim: SimulatedCluster, out_dir: Path, chunk_days: float = 7.0) -> Manifest:
    """Write the export, pricing catalog and ground truth into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = max(1, int(chunk_days * 86400 // sim.step_seconds))
    files: dict[str, list[str]] = defaultdict(list)
    truth_files: dict[str, list[str]] = defaultdict(list)
    for chunk_index, a in enumerate(range(0, sim.n_steps, chunk)):
        b = min(sim.n_steps, a + chunk)
        for source, target in ((_metric_series, files), (_truth_series, truth_files)):
            grouped: dict[str, list[SeriesPoints]] = defaultdict(list)
            for metric, series in source(sim, a, b):
                grouped[metric].append(series)
            for metric, series_list in sorted(grouped.items()):
                name = f"{metric}.{chunk_index:02d}.json.gz"
                write_file(out_dir / name, build_response(series_list))
                target[metric].append(name)
    manifest = Manifest(
        start=sim.start,
        end=sim.start + sim.n_steps * sim.step_seconds,
        step_seconds=sim.step_seconds,
        cluster=sim.spec.name,
        files=dict(files),
        extra={
            "generator": "costlens.synth",
            "seed": sim.seed,
            "truth_files": dict(sorted(truth_files.items())),
        },
    )
    write_manifest(out_dir, manifest)
    (out_dir / "pricing.toml").write_text(pricing_toml(sim), encoding="utf-8")
    truth = ground_truth(sim)
    (out_dir / "ground_truth.json").write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
