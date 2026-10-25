"""Timing benchmark: the bundled fixture end to end, plus synthetic scale points.

Usage: uv run python scripts/bench.py [examples/demo-cluster]
Numbers depend on the machine; the README quotes one run and names the machine class.
"""

from __future__ import annotations

import platform
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from costlens.allocation import allocate
from costlens.config import AllocationConfig, IdleStrategy
from costlens.dataset import ClusterDataset, ContainerInfo, NodeInfo, load_dataset
from costlens.pricing import catalog_from_dict, load_catalog, price_nodes
from costlens.rightsizing.backtest import run_backtest
from costlens.rightsizing.recommender import recommend
from costlens.units import GIB


def timed[T](label: str, fn: Callable[[], T]) -> T:
    t0 = time.perf_counter()
    out = fn()
    print(f"| {label} | {(time.perf_counter() - t0) * 1000:,.0f} ms |")
    return out


def synthetic(n_containers: int, n_steps: int, n_nodes: int, seed: int = 0) -> ClusterDataset:
    """Random dense cluster: every container present at every step."""
    rng = np.random.default_rng(seed)
    node_of = rng.integers(0, n_nodes, n_containers)
    per_node = np.bincount(node_of, minlength=n_nodes).max()
    cores = float(per_node)  # ~1 core requested per container
    req_cpu = np.tile(rng.uniform(0.1, 1.0, n_containers), (n_steps, 1))
    req_mem = np.tile(rng.uniform(0.1, 2.0, n_containers) * GIB, (n_steps, 1))
    use_cpu = req_cpu * rng.uniform(0.1, 1.2, (n_steps, n_containers))
    use_mem = req_mem * rng.uniform(0.3, 1.1, (n_steps, n_containers))
    nan = np.full((n_steps, n_containers), np.nan)
    return ClusterDataset(
        cluster="bench",
        start=0,
        step_seconds=300,
        nodes=[NodeInfo(f"n{j}", "m5.xlarge", "on-demand", "z") for j in range(n_nodes)],
        node_cpu=np.full((n_steps, n_nodes), cores),
        node_memory=np.full((n_steps, n_nodes), cores * 4 * GIB),
        containers=[
            ContainerInfo(f"ns{i % 40}", f"p{i}", "c", int(node_of[i]), f"Deployment/w{i % 300}")
            for i in range(n_containers)
        ],
        present=np.ones((n_steps, n_containers), dtype=bool),
        cpu_usage=use_cpu,
        memory_usage=use_mem,
        cpu_request=req_cpu,
        memory_request=req_mem,
        cpu_limit=nan,
        memory_limit=nan.copy(),
        oom_events=np.zeros((n_steps, n_containers)),
        namespace_labels={},
    )


def main(fixture: Path) -> None:
    print(
        f"python {platform.python_version()}, numpy {np.__version__}, "
        f"{platform.machine()} {platform.system()}"
    )
    print("\n| fixture stage | time |\n| :--- | ---: |")
    ds = timed("load_dataset (78 gzip JSON files)", lambda: load_dataset(fixture))
    costs = timed("price_nodes", lambda: price_nodes(ds, load_catalog(fixture / "pricing.toml")))
    for idle in IdleStrategy:
        timed(
            f"allocate idle={idle.value}",
            lambda idle=idle: allocate(ds, costs, AllocationConfig(group_by="team", idle=idle)),
        )
    timed("recommend (all workloads)", lambda: recommend(ds))
    timed("backtest (7 policies, val + test)", lambda: run_backtest(ds, costs))

    catalog = catalog_from_dict({"fallback": {"cpu_hourly": 0.03, "memory_gib_hourly": 0.004}})
    print("\n| synthetic scale (1 week @ 5 min) | allocate | recommend |\n| :--- | ---: | ---: |")
    for n_containers, n_nodes in ((250, 10), (1000, 40), (2000, 80)):
        big = synthetic(n_containers, 2016, n_nodes)
        nc = price_nodes(big, catalog)
        t0 = time.perf_counter()
        allocate(big, nc, AllocationConfig(idle=IdleStrategy.PER_NODE))
        t_alloc = time.perf_counter() - t0
        t0 = time.perf_counter()
        recommend(big)
        t_rec = time.perf_counter() - t0
        print(
            f"| {n_containers} containers x {n_nodes} nodes | {t_alloc * 1000:,.0f} ms | "
            f"{t_rec * 1000:,.0f} ms |"
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024
    print(f"\npeak RSS: {rss_mb:,.0f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "examples/demo-cluster"))
