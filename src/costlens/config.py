"""Allocation configuration (TOML) and its validation.

Example ``allocation.toml``::

    [allocation]
    basis = "max_request_usage"   # request | usage | max_request_usage
    idle = "proportional"         # keep | proportional | even | per_node
    group_by = "team"             # namespace | team | label:<name> | workload | node | pod
    team_label = "team"

    [shared]
    namespaces = ["kube-system", "monitoring"]
    strategy = "proportional"     # keep | even | proportional | weighted
    weights = { payments = 2.0, search = 1.0 }

    [[shared.external]]
    name = "eks-control-plane"
    monthly_usd = 73.0
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from costlens.errors import ConfigError

HOURS_PER_MONTH = 730.0  # 365 * 24 / 12, the convention cloud price sheets use


class CostBasis(StrEnum):
    """What a container is charged for at each step."""

    REQUEST = "request"
    USAGE = "usage"
    MAX_REQUEST_USAGE = "max_request_usage"


class IdleStrategy(StrEnum):
    """How unallocated node capacity (idle cost) is distributed."""

    KEEP = "keep"
    PROPORTIONAL = "proportional"
    EVEN = "even"
    PER_NODE = "per_node"


class SharedStrategy(StrEnum):
    """How shared costs (shared namespaces, external costs) are distributed."""

    KEEP = "keep"
    EVEN = "even"
    PROPORTIONAL = "proportional"
    WEIGHTED = "weighted"


GROUPINGS = ("namespace", "team", "workload", "node", "pod", "container")


@dataclass(frozen=True)
class ExternalCost:
    """A cost outside the nodes (control plane, load balancers, ...) billed monthly."""

    name: str
    monthly_usd: float


@dataclass(frozen=True)
class AllocationConfig:
    """Everything that changes how cost is attributed."""

    basis: CostBasis = CostBasis.MAX_REQUEST_USAGE
    idle: IdleStrategy = IdleStrategy.PROPORTIONAL
    group_by: str = "namespace"
    team_label: str = "team"
    shared_namespaces: tuple[str, ...] = ()
    shared_strategy: SharedStrategy = SharedStrategy.PROPORTIONAL
    shared_weights: dict[str, float] = field(default_factory=dict)
    external: tuple[ExternalCost, ...] = ()

    def __post_init__(self) -> None:
        validate_group_by(self.group_by)
        if self.shared_strategy is SharedStrategy.WEIGHTED and not self.shared_weights:
            raise ConfigError("shared strategy 'weighted' needs [shared.weights]")
        for key, weight in self.shared_weights.items():
            if weight < 0:
                raise ConfigError(f"shared weight for {key!r} must be non-negative")
        for cost in self.external:
            if cost.monthly_usd < 0:
                raise ConfigError(f"external cost {cost.name!r} must be non-negative")


def validate_group_by(group_by: str) -> None:
    """Raise :class:`ConfigError` unless ``group_by`` is a grouping or ``label:<name>``."""
    if group_by in GROUPINGS:
        return
    if group_by.startswith("label:") and len(group_by) > len("label:"):
        return
    raise ConfigError(
        f"invalid group_by {group_by!r}; use one of {', '.join(GROUPINGS)} or label:<name>",
    )


def _enum[E: StrEnum](enum: type[E], value: Any, name: str) -> E:
    try:
        return enum(str(value))
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum)
        raise ConfigError(f"invalid {name} {value!r}; expected one of: {allowed}") from exc


def config_from_dict(raw: dict[str, Any]) -> AllocationConfig:
    """Build an :class:`AllocationConfig` from a parsed TOML mapping."""
    alloc = raw.get("allocation", {})
    shared = raw.get("shared", {})
    if not isinstance(alloc, dict) or not isinstance(shared, dict):
        raise ConfigError("[allocation] and [shared] must be tables")
    unknown = set(alloc) - {"basis", "idle", "group_by", "team_label"}
    unknown |= {
        f"shared.{k}" for k in set(shared) - {"namespaces", "strategy", "weights", "external"}
    }
    if unknown:
        raise ConfigError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    weights_raw = shared.get("weights", {})
    if not isinstance(weights_raw, dict):
        raise ConfigError("[shared.weights] must map tenant -> weight")
    try:
        weights = {str(k): float(v) for k, v in weights_raw.items()}
        external = tuple(
            ExternalCost(name=str(item["name"]), monthly_usd=float(item["monthly_usd"]))
            for item in shared.get("external", [])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid [shared] section ({exc})") from exc
    namespaces = shared.get("namespaces", [])
    if not isinstance(namespaces, list) or not all(isinstance(n, str) for n in namespaces):
        raise ConfigError("shared.namespaces must be a list of strings")
    return AllocationConfig(
        basis=_enum(CostBasis, alloc.get("basis", CostBasis.MAX_REQUEST_USAGE), "basis"),
        idle=_enum(IdleStrategy, alloc.get("idle", IdleStrategy.PROPORTIONAL), "idle"),
        group_by=str(alloc.get("group_by", "namespace")),
        team_label=str(alloc.get("team_label", "team")),
        shared_namespaces=tuple(namespaces),
        shared_strategy=_enum(
            SharedStrategy, shared.get("strategy", SharedStrategy.PROPORTIONAL), "shared strategy"
        ),
        shared_weights=weights,
        external=external,
    )


def load_config(path: Path) -> AllocationConfig:
    """Load an allocation config file."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: config file not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML ({exc})") from exc
    return config_from_dict(raw)
