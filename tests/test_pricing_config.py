"""Pricing catalog and allocation config parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from costlens.config import (
    AllocationConfig,
    CostBasis,
    IdleStrategy,
    SharedStrategy,
    config_from_dict,
    load_config,
    validate_group_by,
)
from costlens.dataset import NodeInfo
from costlens.errors import ConfigError, PricingError
from costlens.pricing import catalog_from_dict, load_catalog, price_nodes
from costlens.units import GIB
from tests.conftest import make_dataset

CATALOG = {
    "cpu_to_memory_cost_ratio": 8.0,
    "instance": [
        {"type": "m5.xlarge", "vcpu": 4, "memory_gib": 16, "on_demand_hourly": 0.192},
        {
            "type": "c5.xlarge",
            "vcpu": 4,
            "memory_gib": 8,
            "on_demand_hourly": 0.17,
            "spot_hourly": 0.06,
        },
    ],
    "fallback": {"cpu_hourly": 0.03, "memory_gib_hourly": 0.004},
}


def test_node_hourly_variants() -> None:
    cat = catalog_from_dict(CATALOG)
    on_demand = NodeInfo("a", "m5.xlarge", "on-demand", "z")
    assert cat.node_hourly(on_demand, 4, 16 * GIB) == pytest.approx(0.192)
    spot_default = NodeInfo("b", "m5.xlarge", "spot", "z")
    assert cat.node_hourly(spot_default, 4, 16 * GIB) == pytest.approx(0.192 * 0.35)
    spot_listed = NodeInfo("c", "c5.xlarge", "SPOT", "z")
    assert cat.node_hourly(spot_listed, 4, 8 * GIB) == pytest.approx(0.06)
    unknown = NodeInfo("d", "x9.huge", "on-demand", "z")
    assert cat.node_hourly(unknown, 2, 4 * GIB) == pytest.approx(0.076)
    unknown_spot = NodeInfo("e", "x9.huge", "spot", "z")
    assert cat.node_hourly(unknown_spot, 2, 4 * GIB) == pytest.approx(0.076 * 0.35)


def test_unknown_instance_without_fallback() -> None:
    cat = catalog_from_dict({"instance": CATALOG["instance"]})
    with pytest.raises(PricingError, match="not in the catalog"):
        cat.node_hourly(NodeInfo("d", "x9.huge", "on-demand", "z"), 2, GIB)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {"instance": [{"type": "a", "vcpu": 0, "memory_gib": 1, "on_demand_hourly": 1}]},
            "positive",
        ),
        (
            {"instance": [{"type": "a", "vcpu": "x", "memory_gib": 1, "on_demand_hourly": 1}]},
            "number",
        ),
        ({"instance": [{"type": "a"}]}, "missing"),
        ({"instance": [CATALOG["instance"][0], CATALOG["instance"][0]]}, "duplicate"),
        ({"default_spot_discount": 1.5}, "discount"),
        ({"cpu_to_memory_cost_ratio": -1}, "positive"),
    ],
)
def test_catalog_rejects(raw: dict[str, object], message: str) -> None:
    with pytest.raises(PricingError, match=message):
        catalog_from_dict(raw)


def test_load_catalog_errors(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="not found"):
        load_catalog(tmp_path / "none.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("x = [")
    with pytest.raises(PricingError, match="invalid TOML"):
        load_catalog(bad)


def test_price_nodes_split_is_exact_and_respects_ratio() -> None:
    present = np.array([[True, True], [True, False], [False, False]])
    ds = make_dataset(
        [(4, 16), (4, 16)],
        [],
        np.zeros((3, 0)),
        np.zeros((3, 0)),
        node_present=present,
    )
    costs = price_nodes(ds, catalog_from_dict(CATALOG))
    # 0.192 $/h for 1h steps = 192,000,000 nano-dollars per alive node-step.
    assert costs.total_nanos.tolist() == [[192_000_000, 192_000_000], [192_000_000, 0], [0, 0]]
    assert costs.grand_total() == 3 * 192_000_000
    assert costs.cpu_rate[0] / costs.memory_rate[0] == pytest.approx(8.0)
    assert costs.cpu_rate[0] * 4 + costs.memory_rate[0] * 16 == pytest.approx(0.192)
    assert np.array_equal(costs.cpu_nanos + costs.memory_nanos, costs.total_nanos)


def test_price_nodes_rejects_zero_capacity() -> None:
    ds = make_dataset([(0, 16)], [], np.zeros((1, 0)), np.zeros((1, 0)))
    with pytest.raises(PricingError, match="non-positive"):
        price_nodes(ds, catalog_from_dict(CATALOG))


def test_config_from_dict_full(tmp_path: Path) -> None:
    path = tmp_path / "alloc.toml"
    path.write_text(
        """
[allocation]
basis = "request"
idle = "per_node"
group_by = "label:app"

[shared]
namespaces = ["kube-system"]
strategy = "weighted"
weights = { a = 2, b = 1 }

[[shared.external]]
name = "control-plane"
monthly_usd = 73
""",
    )
    cfg = load_config(path)
    assert cfg.basis is CostBasis.REQUEST
    assert cfg.idle is IdleStrategy.PER_NODE
    assert cfg.shared_strategy is SharedStrategy.WEIGHTED
    assert cfg.shared_weights == {"a": 2.0, "b": 1.0}
    assert cfg.external[0].monthly_usd == 73.0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"allocation": {"basis": "vibes"}}, "invalid basis"),
        ({"allocation": {"idle": "share"}}, "invalid idle"),
        ({"allocation": {"group_by": "planet"}}, "group_by"),
        ({"allocation": {"colour": "red"}}, "unknown configuration"),
        ({"shared": {"strategy": "weighted"}}, "weights"),
        ({"shared": {"weights": [1]}}, "map"),
        ({"shared": {"weights": {"a": -1}}}, "non-negative"),
        ({"shared": {"external": [{"name": "x"}]}}, "invalid"),
        ({"shared": {"external": [{"name": "x", "monthly_usd": -5}]}}, "non-negative"),
        ({"shared": {"namespaces": "kube-system"}}, "list"),
        ({"allocation": []}, "tables"),
    ],
)
def test_config_rejects(raw: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


def test_config_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[allocation")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)


def test_group_by_validation() -> None:
    validate_group_by("label:team")
    with pytest.raises(ConfigError):
        validate_group_by("label:")
    assert AllocationConfig().group_by == "namespace"
