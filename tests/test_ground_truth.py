"""End-to-end: simulate -> export JSON -> ingest -> price -> allocate == ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from costlens.allocation import IDLE, SHARED, allocate
from costlens.config import AllocationConfig, CostBasis, IdleStrategy
from costlens.dataset import load_dataset
from costlens.money import to_dollars
from costlens.pricing import load_catalog, price_nodes

TRUTH_SCENARIOS = [
    "request/keep/team",
    "request/keep/namespace",
    "request/proportional/team",
    "request/proportional/namespace",
    "max_request_usage/keep/team",
    "max_request_usage/keep/namespace",
    "max_request_usage/proportional/team",
    "max_request_usage/proportional/namespace",
    "request/proportional/team/shared-platform",
]


@pytest.fixture(scope="module")
def truth(small_export: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads((small_export / "ground_truth.json").read_text())
    return data


@pytest.mark.parametrize("scenario", TRUTH_SCENARIOS)
def test_allocation_matches_ground_truth(
    small_export: Path, truth: dict[str, object], scenario: str
) -> None:
    ds = load_dataset(small_export)
    costs = price_nodes(ds, load_catalog(small_export / "pricing.toml"))
    expected = truth["scenarios"][scenario]  # type: ignore[index]
    config = AllocationConfig(
        basis=CostBasis(expected["basis"]),
        idle=IdleStrategy(expected["idle"]),
        group_by=expected["group_by"],
        shared_namespaces=tuple(expected["shared_namespaces"]),
    )
    result = allocate(ds, costs, config)
    totals = result.totals_by_row()
    assert to_dollars(costs.grand_total()) == pytest.approx(
        truth["total_node_cost_usd"],
        abs=1e-6,  # type: ignore[arg-type]
    )
    assert set(totals) - {IDLE, SHARED} == set(expected["tenants_usd"])
    for tenant, dollars in expected["tenants_usd"].items():
        # nano-dollar rounding: < 1 n$ per container-step, far below a cent
        assert to_dollars(totals[tenant]) == pytest.approx(dollars, abs=1e-4), tenant
    assert to_dollars(totals[IDLE]) == pytest.approx(expected["idle_usd"], abs=1e-4)
    assert to_dollars(totals[SHARED]) == pytest.approx(expected["shared_usd"], abs=1e-4)


def test_every_truth_scenario_is_tested(truth: dict[str, object]) -> None:
    assert set(truth["scenarios"]) == set(TRUTH_SCENARIOS)  # type: ignore[arg-type]
