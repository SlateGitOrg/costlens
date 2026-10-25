"""Report tables and the full showback report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from costlens.allocation import allocate
from costlens.config import AllocationConfig, ExternalCost
from costlens.dataset import load_dataset, load_true_demand
from costlens.pricing import load_catalog, price_nodes
from costlens.report.showback import (
    ReportInputs,
    allocation_table,
    cluster_summary,
    config_lines,
    daily_table,
    recommendation_savings,
    write_report,
)
from costlens.report.tables import Table, bullet_list
from costlens.rightsizing.backtest import run_backtest
from costlens.rightsizing.recommender import recommend


def test_table_renderings() -> None:
    table = Table(
        ["name", "value"], [["a", 1.5], ["b", float("nan")]], csv_columns=["name", "value_usd"]
    )
    md = table.to_markdown()
    assert md.splitlines()[1] == "| :--- | ---: |"
    assert "| b | - |" in md
    assert table.to_csv() == "name,value_usd\na,1.5\nb,\n"
    assert json.loads(table.to_json()) == [
        {"name": "a", "value_usd": 1.5},
        {"name": "b", "value_usd": None},
    ]
    assert table.render("table") == md
    assert table.render("csv") == table.to_csv()
    with pytest.raises(ValueError):
        table.render("xml")
    with pytest.raises(ValueError):
        Table(["a"], [["x", "y"]])
    with pytest.raises(ValueError):
        Table(["a"], [], csv_columns=["a", "b"])
    assert Table(["z"], [[0.0]]).to_csv() == "z\n0\n"
    assert bullet_list([]) == "- none"
    assert bullet_list(["x"]) == "- x"


def test_full_report(small_export: Path, tmp_path: Path) -> None:
    ds = load_dataset(small_export)
    costs = price_nodes(ds, load_catalog(small_export / "pricing.toml"))
    config = AllocationConfig(
        group_by="team",
        shared_namespaces=("kube-system", "monitoring"),
        external=(ExternalCost("control-plane", 73.0),),
    )
    result = allocate(ds, costs, config)
    summary = cluster_summary(result)
    assert summary.node_cost + summary.external_cost == pytest.approx(
        summary.workload_cost
        + summary.shared_namespace_cost
        + summary.idle_cost
        + summary.external_cost
    )
    table = allocation_table(result)
    assert table.rows[-1][0] == "**total**"
    totals = [row[6] for row in table.rows[:-1]]
    assert sum(totals) == pytest.approx(table.rows[-1][6])  # type: ignore[arg-type]
    assert any("control-plane" in line for line in config_lines(config))
    daily = daily_table(result)
    assert sum(float(r[2]) for r in daily.rows) == pytest.approx(table.rows[-1][6])

    recs = recommend(ds)
    savings = recommendation_savings(ds, costs, recs)
    assert set(savings) == {r.key for r in recs}
    backtest = run_backtest(
        ds,
        costs,
        train_weeks=1,
        validation_weeks=0.5,
        test_weeks=0.5,
        true_demand=load_true_demand(small_export, ds),
    )
    paths = write_report(
        ReportInputs(ds, costs, {"team": result}, recs, savings, backtest), tmp_path / "rep"
    )
    names = {p.name for p in paths}
    assert names == {
        "showback.md",
        "allocation_team.csv",
        "daily_team.csv",
        "recommendations.csv",
        "backtest_test.csv",
        "backtest_validation.csv",
        "backtest_test_true_demand.csv",
    }
    text = (tmp_path / "rep" / "showback.md").read_text()
    for heading in (
        "## Cluster summary",
        "## Cost by team",
        "## Rightsizing recommendations",
        "## Rightsizing backtest",
        "uncensored demand",
        "## Data quality",
    ):
        assert heading in text
    with (tmp_path / "rep" / "allocation_team.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[-1]["tenant"] == "**total**"
    assert float(rows[-1]["total_usd"]) == pytest.approx(
        summary.node_cost + summary.external_cost, abs=1e-5
    )


def test_recommendation_savings_for_unseen_workload(small_export: Path) -> None:
    ds = load_dataset(small_export)
    costs = price_nodes(ds, load_catalog(small_export / "pricing.toml"))
    recs = recommend(ds.slice_steps(0, 24))
    later = ds.slice_steps(24 * 7, 24 * 8)
    savings = recommendation_savings(later, costs, recs)
    assert all(np.isfinite(v) for v in savings.values())
