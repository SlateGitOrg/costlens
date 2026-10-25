"""Allocation engine: hand-checked scenarios and conservation properties."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from costlens.allocation import IDLE, SHARED, UNLABELLED, Component, allocate
from costlens.config import (
    AllocationConfig,
    CostBasis,
    ExternalCost,
    IdleStrategy,
    SharedStrategy,
)
from costlens.errors import ConfigError, CostlensError
from costlens.money import to_nanos
from costlens.pricing import catalog_from_dict, price_nodes
from tests.conftest import make_dataset

# 1 $/h node with 4 cores and 16 GiB; ratio 4 => 1 core costs as much as 4 GiB, so
# cores and memory each account for half the node price: 0.125 $/core-h, 0.03125 $/GiB-h.
CATALOG = catalog_from_dict(
    {
        "cpu_to_memory_cost_ratio": 4.0,
        "instance": [{"type": "m5.xlarge", "vcpu": 4, "memory_gib": 16, "on_demand_hourly": 1.0}],
    },
)
DOLLAR = to_nanos(1.0)


def two_tenants(steps: int = 1) -> object:
    containers = [
        ("a", "app", 0, {"team": "red"}),
        ("b", "app", 0, {}),
        ("kube-system", "dns", 0, {}),
    ]
    cpu = np.tile([1.0, 1.0, 0.0], (steps, 1))
    mem = np.tile([4.0, 0.0, 4.0], (steps, 1))
    return make_dataset([(4, 16)], containers, cpu, mem, namespace_labels={"b": {"team": "blue"}})


def run(ds: object, **kwargs: object) -> dict[str, int]:
    result = allocate(ds, price_nodes(ds, CATALOG), AllocationConfig(**kwargs))  # type: ignore[arg-type]
    return result.totals_by_row()


def test_request_basis_keep_idle_hand_computed() -> None:
    totals = run(two_tenants(), basis=CostBasis.REQUEST, idle=IdleStrategy.KEEP)
    # a: 1 core (0.125) + 4 GiB (0.125) ; b: 1 core ; kube-system: 4 GiB
    assert totals["a"] == 250_000_000
    assert totals["b"] == 125_000_000
    assert totals["kube-system"] == 125_000_000
    assert totals[IDLE] == 500_000_000
    assert sum(totals.values()) == DOLLAR


def test_keep_idle_leaves_idle_row() -> None:
    totals = run(two_tenants(), idle=IdleStrategy.KEEP)
    assert totals[IDLE] == 500_000_000
    assert totals["a"] == 250_000_000


def test_proportional_and_even_idle() -> None:
    prop = run(two_tenants(), idle=IdleStrategy.PROPORTIONAL)
    assert prop == {
        "a": 500_000_000,
        "b": 250_000_000,
        "kube-system": 250_000_000,
        IDLE: 0,
        SHARED: 0,
    }
    even = run(two_tenants(), idle=IdleStrategy.EVEN)
    assert even["a"] == 250_000_000 + 166_666_667
    assert even[IDLE] == 0
    assert sum(even.values()) == DOLLAR


def test_per_node_idle_follows_resource() -> None:
    # Per-node idle splits CPU idle by CPU share and memory idle by memory share.
    totals = run(two_tenants(), idle=IdleStrategy.PER_NODE)
    # cpu idle 0.25 split a:b 1:1 ; memory idle 0.25 split a:dns 1:1
    assert totals["a"] == 250_000_000 + 125_000_000 + 125_000_000
    assert totals["b"] == 125_000_000 + 125_000_000
    assert totals["kube-system"] == 125_000_000 + 125_000_000


def test_group_by_team_uses_pod_then_namespace_label() -> None:
    totals = run(two_tenants(), group_by="team", idle=IdleStrategy.KEEP)
    assert totals["red"] == 250_000_000
    assert totals["blue"] == 125_000_000
    assert totals[UNLABELLED] == 125_000_000


@pytest.mark.parametrize("group_by", ["workload", "node", "pod", "container", "label:team"])
def test_other_groupings_conserve(group_by: str) -> None:
    totals = run(two_tenants(), group_by=group_by)
    assert sum(totals.values()) == DOLLAR


def test_shared_namespace_proportional_and_even_and_weighted() -> None:
    ds = two_tenants()
    prop = run(ds, idle=IdleStrategy.KEEP, shared_namespaces=("kube-system",))
    # kube-system's 0.125 split 2:1 by own cost (a 0.25, b 0.125)
    assert prop["a"] == 250_000_000 + 83_333_333
    assert prop["b"] == 125_000_000 + 41_666_667
    assert prop[SHARED] == 0
    even = run(
        ds,
        idle=IdleStrategy.KEEP,
        shared_namespaces=("kube-system",),
        shared_strategy=SharedStrategy.EVEN,
    )
    assert even["a"] - 250_000_000 == pytest.approx(62_500_000, abs=1)
    weighted = run(
        ds,
        idle=IdleStrategy.KEEP,
        shared_namespaces=("kube-system",),
        shared_strategy=SharedStrategy.WEIGHTED,
        shared_weights={"a": 3.0},
    )
    assert weighted["a"] == 250_000_000 + 125_000_000
    keep = run(
        ds,
        idle=IdleStrategy.KEEP,
        shared_namespaces=("kube-system",),
        shared_strategy=SharedStrategy.KEEP,
    )
    assert keep[SHARED] == 125_000_000


def test_shared_row_ledger_is_a_transfer() -> None:
    ds = two_tenants()
    result = allocate(
        ds, price_nodes(ds, CATALOG), AllocationConfig(shared_namespaces=("kube-system",))
    )
    shared_col = result.rows.index(SHARED)
    detail = result.row_totals()[SHARED]
    assert detail[Component.SHARED] < 0
    assert result.totals_by_row()[SHARED] == 0
    assert result.ledger[Component.SHARED][:, shared_col].sum() == -sum(
        v[Component.SHARED] for k, v in result.row_totals().items() if k != SHARED
    )


def test_external_costs_are_spread_and_distributed() -> None:
    ds = two_tenants(steps=730)
    totals = run(ds, idle=IdleStrategy.KEEP, external=(ExternalCost("control-plane", 73.0),))
    assert sum(totals.values()) == 730 * DOLLAR + to_nanos(73.0)
    kept = run(
        ds,
        idle=IdleStrategy.KEEP,
        shared_strategy=SharedStrategy.KEEP,
        external=(ExternalCost("control-plane", 73.0),),
    )
    assert kept[SHARED] == to_nanos(73.0)


def test_usage_and_max_basis() -> None:
    containers = [("a", "app", 0, {})]
    req = np.array([[1.0], [1.0]])
    use = np.array([[2.0], [0.5]])
    ds = make_dataset(
        [(4, 16)], containers, req, np.zeros((2, 1)), cpu_usage=use, memory_usage=np.zeros((2, 1))
    )
    usage = run(ds, basis=CostBasis.USAGE, idle=IdleStrategy.KEEP)
    assert usage["a"] == 250_000_000 + 62_500_000
    maxed = run(ds, basis=CostBasis.MAX_REQUEST_USAGE, idle=IdleStrategy.KEEP)
    assert maxed["a"] == 250_000_000 + 125_000_000


def test_overcommit_allocates_whole_node_pro_rata() -> None:
    containers = [("a", "x", 0, {}), ("b", "y", 0, {})]
    use = np.array([[3.0, 3.0]])
    ds = make_dataset(
        [(4, 16)],
        containers,
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        cpu_usage=use,
        memory_usage=np.zeros((1, 2)),
    )
    result = allocate(
        ds,
        price_nodes(ds, CATALOG),
        AllocationConfig(basis=CostBasis.USAGE, idle=IdleStrategy.KEEP),
    )
    totals = result.totals_by_row()
    assert totals["a"] == totals["b"] == 250_000_000
    assert totals[IDLE] == 500_000_000  # all memory idle
    assert result.overcommitted_node_steps == 1


def test_idle_without_any_tenant_stays_idle() -> None:
    present = np.array([[False]])
    ds = make_dataset(
        [(4, 16)], [("a", "x", 0, {})], np.ones((1, 1)), np.ones((1, 1)), present=present
    )
    for idle in IdleStrategy:
        totals = run(ds, idle=idle)
        assert totals[IDLE] == DOLLAR


def test_windowed_and_invariant_failure() -> None:
    ds = two_tenants(steps=5)
    result = allocate(ds, price_nodes(ds, CATALOG), AllocationConfig())
    starts, sums = result.windowed(2)
    assert starts.tolist() == [ds.start, ds.start + 7200, ds.start + 14400]
    assert sums.sum() == 5 * DOLLAR
    with pytest.raises(ValueError):
        result.windowed(0)
    result.ledger[Component.CPU][0, 0] += 1
    with pytest.raises(CostlensError, match="invariant"):
        result.check_invariants()
    result.ledger[Component.CPU][0, 0] -= 10**12
    with pytest.raises(CostlensError):
        result.check_invariants()


def test_weighted_requires_weights() -> None:
    with pytest.raises(ConfigError):
        AllocationConfig(shared_strategy=SharedStrategy.WEIGHTED)


# ---------------------------------------------------------------------------------------------
# Property: for random clusters and every strategy combination, allocated + idle + shared ==
# total node cost + external cost, exactly, and no tenant is charged a negative amount.


@st.composite
def random_cluster(draw: st.DrawFn) -> object:
    steps = draw(st.integers(1, 6))
    n_nodes = draw(st.integers(1, 3))
    n_containers = draw(st.integers(0, 6))
    caps = [
        (draw(st.sampled_from([2.0, 4.0, 8.0])), draw(st.sampled_from([4.0, 16.0, 32.0])))
        for _ in range(n_nodes)
    ]
    namespaces = ["a", "b", "c", "kube-system"]
    containers = [
        (
            draw(st.sampled_from(namespaces)),
            f"c{i}",
            draw(st.integers(0, n_nodes - 1)),
            {"team": draw(st.sampled_from(["x", "y"]))} if draw(st.booleans()) else {},
        )
        for i in range(n_containers)
    ]
    values = st.floats(0.0, 6.0, allow_nan=False)
    shape = (steps, n_containers)

    def matrix() -> np.ndarray:
        return np.array(
            draw(st.lists(values, min_size=steps * n_containers, max_size=steps * n_containers))
        ).reshape(shape)

    present = np.array(
        draw(st.lists(st.booleans(), min_size=steps * n_containers, max_size=steps * n_containers)),
        dtype=bool,
    ).reshape(shape)
    node_present = np.array(
        draw(st.lists(st.booleans(), min_size=steps * n_nodes, max_size=steps * n_nodes)),
        dtype=bool,
    )
    node_present = node_present.reshape(steps, n_nodes)
    node_of = np.array([c[2] for c in containers], dtype=int)
    if n_containers:
        present &= node_present[:, node_of]
    return make_dataset(
        caps,
        containers,
        matrix(),
        matrix(),
        cpu_usage=matrix(),
        memory_usage=matrix(),
        present=present,
        node_present=node_present,
    )


configs = st.builds(
    AllocationConfig,
    basis=st.sampled_from(list(CostBasis)),
    idle=st.sampled_from(list(IdleStrategy)),
    group_by=st.sampled_from(["namespace", "team", "node", "container"]),
    shared_namespaces=st.sampled_from([(), ("kube-system",), ("kube-system", "c")]),
    shared_strategy=st.sampled_from(
        [SharedStrategy.KEEP, SharedStrategy.EVEN, SharedStrategy.PROPORTIONAL]
    ),
    external=st.sampled_from([(), (ExternalCost("cp", 73.0), ExternalCost("lb", 18.25))]),
)


@given(random_cluster(), configs)
@settings(max_examples=250, deadline=None)
def test_allocated_plus_idle_equals_total(ds: object, config: AllocationConfig) -> None:
    costs = price_nodes(ds, CATALOG)  # type: ignore[arg-type]
    result = allocate(ds, costs, config)  # type: ignore[arg-type]
    totals = result.totals_by_row()
    assert sum(totals.values()) == costs.grand_total() + result.external_total_nanos
    assert all(v >= 0 for v in totals.values())
    if config.idle is IdleStrategy.KEEP:
        assert all(
            result.row_totals()[row][Component.IDLE] == 0 for row in result.rows if row != IDLE
        )
    if not config.shared_namespaces and not config.external:
        assert totals[SHARED] == 0


@given(random_cluster(), st.sampled_from(list(CostBasis)))
@settings(max_examples=100, deadline=None)
def test_idle_strategy_never_changes_own_costs(ds: object, basis: CostBasis) -> None:
    costs = price_nodes(ds, CATALOG)  # type: ignore[arg-type]
    own = None
    for idle in IdleStrategy:
        result = allocate(ds, costs, AllocationConfig(basis=basis, idle=idle))  # type: ignore[arg-type]
        detail = result.row_totals()
        current = {r: d[Component.CPU] + d[Component.MEMORY] for r, d in detail.items()}
        if own is None:
            own = current
        assert current == own
