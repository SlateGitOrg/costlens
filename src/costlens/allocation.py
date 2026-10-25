"""Cost allocation: from node costs to tenants, with idle and shared-cost distribution.

The pipeline (every arrow conserves nano-dollars exactly)::

    node cost (S,N) ──split by basis──▶ container cost (S,C) + node idle (S,N)
                    ──per_node idle?──▶ container idle share (S,C)
                    ──group_by───────▶ tenant cost (S,T) + shared pool (S,) + idle (S,)
                    ──idle strategy──▶ tenant idle share (S,T+1)
                    ──shared strategy▶ tenant shared share (S,T)
                    ──external costs─▶ tenant external share (S,T)

Rows that a strategy cannot place (e.g. idle at a step where no tenant runs) stay on the
explicit ``__idle__`` / ``__shared__`` rows, so ``sum(result) == node cost + external cost``
always holds with integer equality. :meth:`AllocationResult.check_invariants` asserts it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from costlens.config import (
    HOURS_PER_MONTH,
    AllocationConfig,
    CostBasis,
    IdleStrategy,
    SharedStrategy,
)
from costlens.dataset import ClusterDataset, ContainerInfo
from costlens.errors import CostlensError
from costlens.money import FloatArray, IntArray, apportion_rows, spread_evenly, to_nanos
from costlens.pricing import NodeCosts
from costlens.units import GIB

log = logging.getLogger(__name__)

IDLE = "__idle__"
SHARED = "__shared__"
UNLABELLED = "__unlabelled__"


class Component(StrEnum):
    """Ledger columns of an allocation."""

    CPU = "cpu"
    MEMORY = "memory"
    IDLE = "idle"
    SHARED = "shared"
    EXTERNAL = "external"


@dataclass(frozen=True)
class AllocationResult:
    """Per-step ledger of an allocation, one row per tenant plus ``__idle__``/``__shared__``.

    ``ledger[component]`` has shape ``(S, R)`` with ``R = len(rows)`` and holds nano-dollars.
    The ``shared`` component is a transfer: tenants receive positive amounts and the
    ``__shared__`` row carries the matching negative amount, so the shared namespaces' own
    cpu/memory/idle cost stays visible on that row while its net total is what could not be
    placed. Every other cell is non-negative.
    """

    config: AllocationConfig
    rows: list[str]
    ledger: dict[Component, IntArray]
    node_total_nanos: int
    external_total_nanos: int
    cpu_used_core_hours: FloatArray  # (R,)
    cpu_requested_core_hours: FloatArray  # (R,)
    memory_used_gib_hours: FloatArray  # (R,)
    memory_requested_gib_hours: FloatArray  # (R,)
    overcommitted_node_steps: int
    step_seconds: int
    start: int

    @property
    def grand_total_nanos(self) -> int:
        """Node cost plus external cost over the period."""
        return self.node_total_nanos + self.external_total_nanos

    def row_totals(self) -> dict[str, dict[Component, int]]:
        """Total nano-dollars per row and component over the whole period."""
        sums = {comp: arr.sum(axis=0) for comp, arr in self.ledger.items()}
        return {
            row: {comp: int(sums[comp][i]) for comp in Component} for i, row in enumerate(self.rows)
        }

    def totals_by_row(self) -> dict[str, int]:
        """Total nano-dollars per row (all components)."""
        stacked: IntArray = np.sum([arr.sum(axis=0) for arr in self.ledger.values()], axis=0)
        return {row: int(stacked[i]) for i, row in enumerate(self.rows)}

    def check_invariants(self) -> None:
        """Raise if money was created or destroyed anywhere in the pipeline."""
        allocated = int(sum(int(arr.sum()) for arr in self.ledger.values()))
        if allocated != self.grand_total_nanos:
            raise CostlensError(
                f"allocation invariant violated: allocated {allocated} != "
                f"total {self.grand_total_nanos} nano-dollars",
            )
        shared_row = self.rows.index(SHARED)
        for comp, arr in self.ledger.items():
            check = arr if comp is not Component.SHARED else np.delete(arr, shared_row, axis=1)
            if np.any(check < 0):
                raise CostlensError(f"negative {comp.value} cost in ledger")
        if any(total < 0 for total in self.totals_by_row().values()):
            raise CostlensError("a ledger row has a negative total")

    def windowed(self, window_steps: int) -> tuple[npt.NDArray[np.int64], IntArray]:
        """Aggregate the all-component ledger into windows of ``window_steps`` steps.

        Returns ``(window_start_timestamps, totals)`` with ``totals`` of shape ``(W, R)``.
        """
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        stacked: IntArray = np.sum(list(self.ledger.values()), axis=0)
        s = stacked.shape[0]
        starts = np.arange(0, s, window_steps)
        sums: IntArray = np.add.reduceat(stacked, starts, axis=0)
        return self.start + starts.astype(np.int64) * self.step_seconds, sums


def tenant_key(
    dataset: ClusterDataset, container: ContainerInfo, group_by: str, team_label: str
) -> str:
    """The tenant a container belongs to under a grouping."""
    if group_by == "namespace":
        return container.namespace
    if group_by == "team":
        return dataset.label_of(container, team_label) or UNLABELLED
    if group_by == "workload":
        return f"{container.namespace}/{container.workload}"
    if group_by == "node":
        return dataset.nodes[container.node_index].name
    if group_by == "pod":
        return f"{container.namespace}/{container.pod}"
    if group_by == "container":
        return container.key
    label = group_by.removeprefix("label:")
    return dataset.label_of(container, label) or UNLABELLED


def basis_amounts(dataset: ClusterDataset, basis: CostBasis) -> tuple[FloatArray, FloatArray]:
    """``(S, C)`` CPU cores and memory bytes each container is charged for (0 when absent)."""
    if basis is CostBasis.REQUEST:
        cpu, mem = dataset.cpu_request, dataset.memory_request
    elif basis is CostBasis.USAGE:
        cpu, mem = dataset.cpu_usage, dataset.memory_usage
    else:
        cpu = np.fmax(dataset.cpu_request, dataset.cpu_usage)
        mem = np.fmax(dataset.memory_request, dataset.memory_usage)
    return np.nan_to_num(cpu, nan=0.0), np.nan_to_num(mem, nan=0.0)


def _split_node_resource(
    totals: IntArray,
    capacity: FloatArray,
    amounts: FloatArray,
    columns: npt.NDArray[np.int64],
) -> tuple[IntArray, IntArray, int]:
    """Split one node's per-step cost for one resource across its containers and idle.

    Weights are ``[amount_1, ..., amount_k, max(capacity - sum(amount), 0)]`` so a container
    pays exactly ``amount / capacity`` of the node; when the containers' amounts exceed
    capacity the node is fully allocated pro rata and idle is zero.
    """
    sub = amounts[:, columns]
    used = sub.sum(axis=1)
    cap = np.nan_to_num(capacity, nan=0.0)
    idle_weight = np.maximum(cap - used, 0.0)
    over = int(np.count_nonzero((used > cap) & (totals > 0)))
    weights = np.concatenate([sub, idle_weight[:, None]], axis=1)
    out = apportion_rows(totals, weights, fallback_col=-1)
    return out[:, :-1], out[:, -1], over


def _distribute(
    pool: IntArray,
    weights: FloatArray,
) -> tuple[IntArray, IntArray]:
    """Apportion a per-step pool ``(S,)`` over ``(S, K)`` weights.

    Returns ``(shares (S, K), unplaced (S,))``; steps with no positive weight stay unplaced.
    """
    padded = np.concatenate([weights, np.zeros((weights.shape[0], 1))], axis=1)
    out = apportion_rows(pool, padded, fallback_col=-1)
    return out[:, :-1], out[:, -1]


def _strategy_weights(
    strategy: IdleStrategy | SharedStrategy,
    own: IntArray,
    tenants: list[str],
    weights_by_tenant: dict[str, float],
) -> FloatArray:
    """Per-step weights for distributing a pool among tenants (columns of ``own``)."""
    active = own > 0
    if strategy in (IdleStrategy.PROPORTIONAL, SharedStrategy.PROPORTIONAL):
        return own.astype(np.float64)
    if strategy in (IdleStrategy.EVEN, SharedStrategy.EVEN):
        return active.astype(np.float64)
    if strategy is SharedStrategy.WEIGHTED:
        fixed = np.array([weights_by_tenant.get(t, 0.0) for t in tenants], dtype=np.float64)
        return active * fixed[None, :]
    raise CostlensError(f"strategy {strategy} has no weights")  # pragma: no cover


def allocate(
    dataset: ClusterDataset,
    node_costs: NodeCosts,
    config: AllocationConfig,
) -> AllocationResult:
    """Run the full allocation pipeline (see module docstring)."""
    s = dataset.n_steps
    n_containers = len(dataset.containers)
    cpu_amount, mem_amount = basis_amounts(dataset, config.basis)
    node_of = dataset.container_node_indices()

    container_cpu = np.zeros((s, n_containers), dtype=np.int64)
    container_mem = np.zeros((s, n_containers), dtype=np.int64)
    container_idle = np.zeros((s, n_containers), dtype=np.int64)
    idle_left = np.zeros(s, dtype=np.int64)
    overcommitted = 0

    for j in range(len(dataset.nodes)):
        cols = np.flatnonzero(node_of == j)
        cpu_part, cpu_idle, over_c = _split_node_resource(
            node_costs.cpu_nanos[:, j], dataset.node_cpu[:, j], cpu_amount, cols
        )
        mem_part, mem_idle, over_m = _split_node_resource(
            node_costs.memory_nanos[:, j], dataset.node_memory[:, j], mem_amount, cols
        )
        overcommitted += max(over_c, over_m)
        container_cpu[:, cols] = cpu_part
        container_mem[:, cols] = mem_part
        if config.idle is IdleStrategy.PER_NODE and cols.size:
            share_c, rest_c = _distribute(cpu_idle, cpu_part.astype(np.float64))
            share_m, rest_m = _distribute(mem_idle, mem_part.astype(np.float64))
            container_idle[:, cols] = share_c + share_m
            idle_left += rest_c + rest_m
        else:
            idle_left += cpu_idle + mem_idle

    # ---- group containers into tenants; shared namespaces feed the shared pool ------------
    keys = [tenant_key(dataset, c, config.group_by, config.team_label) for c in dataset.containers]
    shared_ns = set(config.shared_namespaces)
    is_shared = np.array([c.namespace in shared_ns for c in dataset.containers], dtype=bool)
    tenants = sorted({k for k, shared in zip(keys, is_shared, strict=True) if not shared})
    rows = [*tenants, IDLE, SHARED]
    t = len(tenants)
    idle_row, shared_row = t, t + 1
    row_of = np.array(
        [
            shared_row if shared else tenants.index(k)
            for k, shared in zip(keys, is_shared, strict=True)
        ],
        dtype=np.int64,
    )
    width = len(rows)

    def group(matrix: IntArray) -> IntArray:
        out = np.zeros((s, width), dtype=np.int64)
        if n_containers:
            np.add.at(out.T, row_of, matrix.T)
        return out

    cpu_rows = group(container_cpu)
    mem_rows = group(container_mem)
    idle_rows = group(container_idle)
    idle_rows[:, idle_row] += idle_left
    shared_rows = np.zeros((s, width), dtype=np.int64)
    external_rows = np.zeros((s, width), dtype=np.int64)

    # ---- idle distribution (tenants and the shared pool participate) ---------------------
    if config.idle in (IdleStrategy.PROPORTIONAL, IdleStrategy.EVEN):
        participants = [*range(t), shared_row]
        own = (cpu_rows + mem_rows)[:, participants]
        weights = _strategy_weights(config.idle, own, [*tenants, SHARED], {})
        shares, unplaced = _distribute(idle_rows[:, idle_row], weights)
        idle_rows[:, idle_row] = unplaced
        idle_rows[:, participants] += shares

    # ---- shared pool and external costs --------------------------------------------------
    own_tenants = (cpu_rows + mem_rows)[:, :t]
    if config.shared_strategy is not SharedStrategy.KEEP and t:
        weights = _strategy_weights(
            config.shared_strategy, own_tenants, tenants, config.shared_weights
        )
        pool = cpu_rows[:, shared_row] + mem_rows[:, shared_row] + idle_rows[:, shared_row]
        shares, unplaced = _distribute(pool, weights)
        shared_rows[:, :t] = shares
        # Unplaceable shared cost stays on the shared row; move the placed part off it.
        shared_rows[:, shared_row] = unplaced - pool
    external_total = 0
    hours = dataset.n_steps * dataset.step_hours
    for cost in config.external:
        nanos = to_nanos(cost.monthly_usd * hours / HOURS_PER_MONTH)
        external_total += nanos
        per_step = spread_evenly(nanos, s)
        if config.shared_strategy is SharedStrategy.KEEP or not t:
            external_rows[:, shared_row] += per_step
            continue
        weights = _strategy_weights(
            config.shared_strategy, own_tenants, tenants, config.shared_weights
        )
        shares, unplaced = _distribute(per_step, weights)
        external_rows[:, :t] += shares
        external_rows[:, shared_row] += unplaced

    # ---- usage/efficiency metrics per row ------------------------------------------------
    hours_per_step = dataset.step_hours

    def per_row(values: FloatArray, scale: float) -> FloatArray:
        out = np.zeros(width)
        if n_containers:
            np.add.at(out, row_of, np.nan_to_num(values, nan=0.0).sum(axis=0) * scale)
        return out

    result = AllocationResult(
        config=config,
        rows=rows,
        ledger={
            Component.CPU: cpu_rows,
            Component.MEMORY: mem_rows,
            Component.IDLE: idle_rows,
            Component.SHARED: shared_rows,
            Component.EXTERNAL: external_rows,
        },
        node_total_nanos=node_costs.grand_total(),
        external_total_nanos=external_total,
        cpu_used_core_hours=per_row(dataset.cpu_usage, hours_per_step),
        cpu_requested_core_hours=per_row(dataset.cpu_request, hours_per_step),
        memory_used_gib_hours=per_row(dataset.memory_usage, hours_per_step / GIB),
        memory_requested_gib_hours=per_row(dataset.memory_request, hours_per_step / GIB),
        overcommitted_node_steps=overcommitted,
        step_seconds=dataset.step_seconds,
        start=dataset.start,
    )
    result.check_invariants()
    log.info(
        "allocated %d nano-dollars over %d steps to %d tenants (%s, idle=%s, shared=%s)",
        result.grand_total_nanos,
        s,
        t,
        config.basis,
        config.idle,
        config.shared_strategy,
    )
    return result
