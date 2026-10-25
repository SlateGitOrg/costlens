"""The bundled fixture is exactly what the documented generator command produces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from costlens.dataset import load_dataset
from costlens.synth.export import write_export
from costlens.synth.generator import simulate
from costlens.synth.spec import default_spec

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "demo-cluster"


@pytest.mark.slow
def test_bundled_fixture_regenerates(tmp_path: Path) -> None:
    sim = simulate(default_spec(), start=1772409600, weeks=6, step_seconds=900, seed=7)
    write_export(sim, tmp_path)
    committed = json.loads((FIXTURE / "manifest.json").read_text())
    fresh = json.loads((tmp_path / "manifest.json").read_text())
    assert committed == fresh
    assert (FIXTURE / "pricing.toml").read_text() == (tmp_path / "pricing.toml").read_text()
    a, b = load_dataset(FIXTURE), load_dataset(tmp_path)
    assert [c.key for c in a.containers] == [c.key for c in b.containers]
    for name in (
        "cpu_usage",
        "memory_usage",
        "cpu_request",
        "memory_request",
        "cpu_limit",
        "memory_limit",
        "oom_events",
        "node_cpu",
        "node_memory",
    ):
        # equal_nan + a tolerance: libm results may differ in the last ulp across CPUs
        assert np.allclose(
            getattr(a, name), getattr(b, name), rtol=1e-9, atol=1e-6, equal_nan=True
        ), name
    truth_a = json.loads((FIXTURE / "ground_truth.json").read_text())
    truth_b = json.loads((tmp_path / "ground_truth.json").read_text())
    assert truth_a["total_node_cost_usd"] == pytest.approx(truth_b["total_node_cost_usd"])
    for key, scenario in truth_a["scenarios"].items():
        for tenant, usd in scenario["tenants_usd"].items():
            assert usd == pytest.approx(truth_b["scenarios"][key]["tenants_usd"][tenant])
