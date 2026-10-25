"""Shared fixtures: hand-built datasets and a small simulated export."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from costlens.dataset import ClusterDataset, ContainerInfo, NodeInfo
from costlens.synth.export import write_export
from costlens.synth.generator import simulate
from costlens.synth.spec import default_spec
from costlens.units import GIB

START = 1772409600  # 2026-03-02T00:00:00Z, a Monday


def make_dataset(
    node_caps: Sequence[tuple[float, float]],
    containers: Sequence[tuple[str, str, int, dict[str, str]]],
    cpu_request: np.ndarray,
    memory_request: np.ndarray,
    cpu_usage: np.ndarray | None = None,
    memory_usage: np.ndarray | None = None,
    node_present: np.ndarray | None = None,
    present: np.ndarray | None = None,
    step_seconds: int = 3600,
    namespace_labels: dict[str, dict[str, str]] | None = None,
    instance_type: str = "m5.xlarge",
) -> ClusterDataset:
    """Build a dataset directly from arrays (memory given in GiB for readability)."""
    s = cpu_request.shape[0]
    n = len(node_caps)
    node_cpu = np.tile(np.array([c for c, _ in node_caps], dtype=float), (s, 1))
    node_mem = np.tile(np.array([m * GIB for _, m in node_caps], dtype=float), (s, 1))
    if node_present is not None:
        node_cpu[~node_present] = np.nan
        node_mem[~node_present] = np.nan
    cpu_usage = cpu_request.copy() if cpu_usage is None else cpu_usage
    memory_usage = memory_request.copy() if memory_usage is None else memory_usage
    present = np.ones(cpu_request.shape, dtype=bool) if present is None else present
    infos = [
        ContainerInfo(
            namespace=ns,
            pod=f"{ns}-pod-{i}",
            container=name,
            node_index=node,
            workload=f"Deployment/{name}",
            pod_labels=labels,
        )
        for i, (ns, name, node, labels) in enumerate(containers)
    ]

    def masked(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
        out = arr.astype(float) * scale
        out[~present] = np.nan
        return out

    return ClusterDataset(
        cluster="test",
        start=START,
        step_seconds=step_seconds,
        nodes=[NodeInfo(f"node-{j}", instance_type, "on-demand", "z") for j in range(n)],
        node_cpu=node_cpu,
        node_memory=node_mem,
        containers=infos,
        present=present,
        cpu_usage=masked(cpu_usage),
        memory_usage=masked(memory_usage, GIB),
        cpu_request=masked(cpu_request),
        memory_request=masked(memory_request, GIB),
        cpu_limit=np.full(cpu_request.shape, np.nan),
        memory_limit=np.full(cpu_request.shape, np.nan),
        oom_events=np.zeros(cpu_request.shape),
        namespace_labels=namespace_labels or {},
    )


@pytest.fixture(scope="session")
def small_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two simulated weeks of the default cluster at 1h steps, exported to disk."""
    out = tmp_path_factory.mktemp("export")
    sim = simulate(default_spec(), start=START, weeks=2, step_seconds=3600, seed=11)
    write_export(sim, out, chunk_days=4)
    return out
