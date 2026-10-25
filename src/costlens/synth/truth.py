"""Ground-truth cost allocation computed straight from the simulation.

This is a deliberately *independent* reference implementation: it works from the generator's
own objects (pool prices, the team each workload was declared for, the scheduler's node
choice, the raw demand arrays), uses plain float arithmetic with no apportionment, and never
touches the exported JSON or the costlens ingestion/allocation code. Agreement between the two
paths (within rounding to nano-dollars) is what the end-to-end ground-truth tests assert.

Scenarios cover the two charge bases (``request``, ``max_request_usage``), the ``keep`` and
``proportional`` idle strategies, grouping by team and by namespace, and one scenario with
``kube-system``/``monitoring`` treated as shared and redistributed proportionally.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from costlens.synth.generator import SimulatedCluster
from costlens.units import GIB, MIB

UNLABELLED = "__unlabelled__"
SHARED_NAMESPACES = ("kube-system", "monitoring")


def _rates(sim: SimulatedCluster) -> tuple[np.ndarray, np.ndarray]:
    ratio = sim.spec.cpu_to_memory_cost_ratio
    cpu_rate = np.zeros(len(sim.nodes))
    mem_rate = np.zeros(len(sim.nodes))  # per GiB-hour
    for j, node in enumerate(sim.nodes):
        price = node.pool.hourly_usd
        mem_rate[j] = price / (ratio * node.pool.cores + node.pool.memory_gib)
        cpu_rate[j] = ratio * mem_rate[j]
    return cpu_rate, mem_rate


def _charged(sim: SimulatedCluster, basis: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-step cores and GiB each container is charged for (0 when absent)."""
    s, c = sim.cpu_usage.shape
    cpu = np.zeros((s, c))
    mem = np.zeros((s, c))
    for i, cont in enumerate(sim.containers):
        sl = slice(cont.start, cont.stop)
        req_cpu = cont.spec.cpu_request
        req_mem = cont.spec.mem_request_mib * MIB / GIB
        if basis == "request":
            cpu[sl, i] = req_cpu
            mem[sl, i] = req_mem
        else:
            cpu[sl, i] = np.maximum(req_cpu, sim.cpu_usage[sl, i])
            mem[sl, i] = np.maximum(req_mem, sim.memory_usage[sl, i] / GIB)
    return cpu, mem


def _scenario(
    sim: SimulatedCluster, basis: str, idle: str, group: str, shared: tuple[str, ...]
) -> dict[str, Any]:
    cpu_rate, mem_rate = _rates(sim)
    h = sim.step_hours
    cpu, mem = _charged(sim, basis)
    n_steps = sim.n_steps
    node_of = np.array([c.node for c in sim.containers], dtype=np.int64)

    cost = np.zeros_like(cpu)
    idle_per_step = np.zeros(n_steps)
    for j, node in enumerate(sim.nodes):
        cols = node_of == j
        alive = np.zeros(n_steps, dtype=bool)
        alive[node.alive_from : node.alive_to] = True
        for amount, capacity, rate in (
            (cpu, node.pool.cores, cpu_rate[j]),
            (mem, node.pool.memory_gib, mem_rate[j]),
        ):
            used = amount[:, cols].sum(axis=1)
            scale = np.where(used > capacity, capacity / np.maximum(used, 1e-300), 1.0)
            cost[:, cols] += amount[:, cols] * scale[:, None] * rate * h
            idle_per_step += np.where(alive, np.maximum(capacity - used, 0.0) * rate * h, 0.0)

    def key(i: int) -> str:
        cont = sim.containers[i]
        if group == "namespace":
            return cont.namespace
        return cont.team or UNLABELLED

    is_shared = np.array([c.namespace in shared for c in sim.containers], dtype=bool)
    tenants = sorted({key(i) for i in range(len(sim.containers)) if not is_shared[i]})
    index = {t: k for k, t in enumerate(tenants)}
    own = np.zeros((n_steps, len(tenants)))
    shared_own = cost[:, is_shared].sum(axis=1)
    for i in range(len(sim.containers)):
        if not is_shared[i]:
            own[:, index[key(i)]] += cost[:, i]

    total_by_tenant = own.copy()
    idle_left = idle_per_step.copy()
    shared_pool = shared_own.copy()
    if idle == "proportional":
        weight_total = own.sum(axis=1) + shared_own
        frac = np.divide(
            own, weight_total[:, None], out=np.zeros_like(own), where=weight_total[:, None] > 0
        )
        frac_shared = np.divide(
            shared_own, weight_total, out=np.zeros_like(shared_own), where=weight_total > 0
        )
        total_by_tenant += frac * idle_per_step[:, None]
        shared_pool += frac_shared * idle_per_step
        idle_left = np.where(weight_total > 0, 0.0, idle_per_step)
    if shared:
        own_total = own.sum(axis=1)
        frac = np.divide(
            own, own_total[:, None], out=np.zeros_like(own), where=own_total[:, None] > 0
        )
        total_by_tenant += frac * shared_pool[:, None]
        shared_pool = np.where(own_total > 0, 0.0, shared_pool)

    return {
        "basis": basis,
        "idle": idle,
        "group_by": group,
        "shared_namespaces": list(shared),
        "tenants_usd": {t: float(total_by_tenant[:, k].sum()) for t, k in index.items()},
        "idle_usd": float(idle_left.sum()),
        "shared_usd": float(shared_pool.sum()),
    }


def ground_truth(sim: SimulatedCluster) -> dict[str, Any]:
    """All ground-truth scenarios for ``sim`` as a JSON-serialisable mapping."""
    total = 0.0
    for node in sim.nodes:
        total += node.pool.hourly_usd * (node.alive_to - node.alive_from) * sim.step_hours
    scenarios: dict[str, Any] = {}
    for basis in ("request", "max_request_usage"):
        for idle in ("keep", "proportional"):
            for group in ("team", "namespace"):
                scenarios[f"{basis}/{idle}/{group}"] = _scenario(sim, basis, idle, group, ())
    scenarios["request/proportional/team/shared-platform"] = _scenario(
        sim, "request", "proportional", "team", SHARED_NAMESPACES
    )
    return {
        "description": "Costs computed directly from the simulation (independent of the "
        "costlens ingestion and allocation code).",
        "seed": sim.seed,
        "total_node_cost_usd": total,
        "scenarios": scenarios,
    }
