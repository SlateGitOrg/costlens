"""Node pricing: hourly price per node and its split into CPU and memory cost.

A node is billed as a whole, but Kubernetes tenants consume CPU and memory separately, so the
node's hourly price ``P`` is split with a CPU-to-memory cost ratio ``r`` (the price of one
vCPU-hour divided by the price of one GiB-hour)::

    mem_rate = P / (r * cores + gib)        cpu_rate = r * mem_rate
    cpu_rate * cores + mem_rate * gib == P  (by construction)

This is the same approach OpenCost and the cloud providers' own custom-machine price sheets
take. Per step the node cost is first rounded to integer nano-dollars and then split exactly
into a CPU part and a memory part with :func:`costlens.money.apportion_rows`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from costlens.dataset import ClusterDataset, NodeInfo
from costlens.errors import PricingError
from costlens.money import NANOS_PER_DOLLAR, FloatArray, IntArray, apportion_rows
from costlens.units import GIB


@dataclass(frozen=True)
class InstancePrice:
    """Catalog entry for one instance type."""

    instance_type: str
    vcpu: float
    memory_gib: float
    on_demand_hourly: float
    spot_hourly: float | None = None


@dataclass(frozen=True)
class PricingCatalog:
    """Prices for instance types plus the fallback used for unknown types."""

    instances: dict[str, InstancePrice]
    cpu_to_memory_cost_ratio: float = 7.5
    default_spot_discount: float = 0.65
    fallback_cpu_hourly: float | None = None
    fallback_memory_gib_hourly: float | None = None
    currency: str = "USD"

    def node_hourly(self, node: NodeInfo, cores: float, memory_bytes: float) -> float:
        """Hourly price of ``node``; spot nodes use the spot price or the default discount."""
        spot = node.capacity_type.lower() == "spot"
        entry = self.instances.get(node.instance_type)
        if entry is not None:
            if spot and entry.spot_hourly is not None:
                return entry.spot_hourly
            on_demand = entry.on_demand_hourly
        elif self.fallback_cpu_hourly is not None and self.fallback_memory_gib_hourly is not None:
            on_demand = (
                self.fallback_cpu_hourly * cores
                + self.fallback_memory_gib_hourly * memory_bytes / GIB
            )
        else:
            raise PricingError(
                f"node {node.name!r}: instance type {node.instance_type!r} is not in the "
                "catalog and no [fallback] rates are configured",
            )
        return on_demand * (1.0 - self.default_spot_discount) if spot else on_demand


def load_catalog(path: Path) -> PricingCatalog:
    """Load a TOML pricing catalog (see ``examples/pricing.toml``)."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PricingError(f"{path}: pricing catalog not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PricingError(f"{path}: invalid TOML ({exc})") from exc
    return catalog_from_dict(raw, str(path))


def catalog_from_dict(raw: dict[str, Any], source: str = "<catalog>") -> PricingCatalog:
    """Build a catalog from a parsed TOML/JSON mapping, validating every number."""
    instances: dict[str, InstancePrice] = {}
    for i, item in enumerate(raw.get("instance", [])):
        where = f"{source}: instance[{i}]"
        try:
            entry = InstancePrice(
                instance_type=str(item["type"]),
                vcpu=_positive(item["vcpu"], where, "vcpu"),
                memory_gib=_positive(item["memory_gib"], where, "memory_gib"),
                on_demand_hourly=_positive(item["on_demand_hourly"], where, "on_demand_hourly"),
                spot_hourly=(
                    _positive(item["spot_hourly"], where, "spot_hourly")
                    if "spot_hourly" in item
                    else None
                ),
            )
        except (KeyError, TypeError) as exc:
            raise PricingError(f"{where}: missing or invalid field ({exc})") from exc
        if entry.instance_type in instances:
            raise PricingError(f"{where}: duplicate instance type {entry.instance_type!r}")
        instances[entry.instance_type] = entry
    ratio = _positive(raw.get("cpu_to_memory_cost_ratio", 7.5), source, "cpu_to_memory_ratio")
    discount = float(raw.get("default_spot_discount", 0.65))
    if not 0.0 <= discount < 1.0:
        raise PricingError(f"{source}: default_spot_discount must be in [0, 1)")
    fallback = raw.get("fallback", {})
    return PricingCatalog(
        instances=instances,
        cpu_to_memory_cost_ratio=ratio,
        default_spot_discount=discount,
        fallback_cpu_hourly=(
            _positive(fallback["cpu_hourly"], source, "fallback.cpu_hourly")
            if "cpu_hourly" in fallback
            else None
        ),
        fallback_memory_gib_hourly=(
            _positive(fallback["memory_gib_hourly"], source, "fallback.memory_gib_hourly")
            if "memory_gib_hourly" in fallback
            else None
        ),
        currency=str(raw.get("currency", "USD")),
    )


def _positive(value: Any, where: str, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PricingError(f"{where}: {name} must be a number") from exc
    if not number > 0 or not np.isfinite(number):
        raise PricingError(f"{where}: {name} must be positive")
    return number


@dataclass(frozen=True)
class NodeCosts:
    """Per-step node costs in nano-dollars, split exactly into CPU and memory parts."""

    hourly_price: FloatArray  # (N,) dollars per hour
    cpu_rate: FloatArray  # (N,) dollars per core-hour
    memory_rate: FloatArray  # (N,) dollars per GiB-hour
    cpu_nanos: IntArray  # (S, N)
    memory_nanos: IntArray  # (S, N)

    @property
    def total_nanos(self) -> IntArray:
        """``(S, N)`` total node cost per step."""
        out: IntArray = self.cpu_nanos + self.memory_nanos
        return out

    def grand_total(self) -> int:
        """Total cost of all nodes over the whole period, in nano-dollars."""
        return int(self.cpu_nanos.sum() + self.memory_nanos.sum())


def price_nodes(dataset: ClusterDataset, catalog: PricingCatalog) -> NodeCosts:
    """Price every node-step of ``dataset``.

    A node's capacity is taken as its median observed capacity (capacity flapping by a few
    MiB must not change the price); a node is charged only for the steps it existed.
    """
    n = len(dataset.nodes)
    present = dataset.node_present
    hourly = np.zeros(n)
    cpu_rate = np.zeros(n)
    mem_rate = np.zeros(n)
    ratio = catalog.cpu_to_memory_cost_ratio
    cores_med = _median_capacity(dataset, cpu=True)
    bytes_med = _median_capacity(dataset, cpu=False)
    for j, node in enumerate(dataset.nodes):
        if not np.any(present[:, j]):
            continue
        cores = float(cores_med[j])
        mem_bytes = float(bytes_med[j])
        gib = mem_bytes / GIB
        if cores <= 0 or gib <= 0:
            raise PricingError(f"node {node.name!r} reports non-positive capacity")
        price = catalog.node_hourly(node, cores, mem_bytes)
        hourly[j] = price
        mem_rate[j] = price / (ratio * cores + gib)
        cpu_rate[j] = ratio * mem_rate[j]
    per_step_nanos = np.rint(hourly * dataset.step_hours * NANOS_PER_DOLLAR).astype(np.int64)
    totals = np.where(present, per_step_nanos[None, :], 0).astype(np.int64)
    s = dataset.n_steps
    cpu_weight = np.broadcast_to(cpu_rate * cores_med, (s, n))
    mem_weight = np.broadcast_to(mem_rate * bytes_med / GIB, (s, n))
    split = apportion_rows(
        totals.reshape(-1),
        np.stack([cpu_weight.reshape(-1), mem_weight.reshape(-1)], axis=1),
        fallback_col=0,
    )
    return NodeCosts(
        hourly_price=hourly,
        cpu_rate=cpu_rate,
        memory_rate=mem_rate,
        cpu_nanos=split[:, 0].reshape(s, n),
        memory_nanos=split[:, 1].reshape(s, n),
    )


def _median_capacity(dataset: ClusterDataset, *, cpu: bool) -> FloatArray:
    values = dataset.node_cpu if cpu else dataset.node_memory
    out = np.zeros(values.shape[1])
    for j in range(values.shape[1]):
        col = values[:, j]
        alive = ~np.isnan(col)
        if np.any(alive):
            out[j] = float(np.median(col[alive]))
    return out
