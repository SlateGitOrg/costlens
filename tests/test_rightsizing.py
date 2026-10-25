"""Recommender and backtest behaviour."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from costlens.dataset import ContainerInfo, load_dataset, load_true_demand
from costlens.errors import ConfigError
from costlens.pricing import catalog_from_dict, load_catalog, price_nodes
from costlens.rightsizing.backtest import (
    BASELINE,
    evaluate,
    run_backtest,
    split_weeks,
)
from costlens.rightsizing.recommender import RecommenderConfig, recommend, workload_groups
from costlens.units import GIB, MIB
from tests.conftest import make_dataset

CATALOG = catalog_from_dict(
    {"instance": [{"type": "m5.xlarge", "vcpu": 4, "memory_gib": 16, "on_demand_hourly": 1.0}]}
)


def steady(
    steps: int = 24 * 14,
    cpu: float = 0.2,
    mem_gib: float = 1.0,
    req_cpu: float = 1.0,
    req_mem: float = 4.0,
) -> object:
    rng = np.random.default_rng(0)
    use_cpu = cpu * (1 + 0.1 * rng.standard_normal((steps, 1)))
    use_mem = mem_gib * (1 + 0.02 * rng.standard_normal((steps, 1)))
    return make_dataset(
        [(4, 16)],
        [("ns", "app", 0, {})],
        np.full((steps, 1), req_cpu),
        np.full((steps, 1), req_mem),
        cpu_usage=use_cpu,
        memory_usage=use_mem,
    )


def test_recommendation_tracks_usage_with_margin() -> None:
    (rec,) = recommend(steady())
    assert rec.workload == "ns/Deployment/app"
    assert rec.current_cpu_request == 1.0
    assert rec.current_memory_request == pytest.approx(4 * GIB)
    # p90 of ~N(0.2, 0.02) is ~0.226 -> *1.15 ~ 0.26, plus at most one 5% bucket
    assert 0.24 <= rec.target_cpu <= 0.29
    assert 1.1 * GIB <= rec.target_memory <= 1.3 * GIB
    assert rec.lower_cpu <= rec.target_cpu <= rec.upper_cpu
    assert rec.lower_memory <= rec.target_memory <= rec.upper_memory
    assert rec.confidence_days == pytest.approx(14.0)
    assert np.isnan(rec.current_cpu_limit)


def test_usage_gaps_do_not_lower_the_recommendation() -> None:
    base = steady()
    spread = np.random.default_rng(1).uniform(0.1, 1.0, base.cpu_usage.shape)  # type: ignore[attr-defined]
    full = replace(base, cpu_usage=spread)  # type: ignore[type-var]
    gap = np.zeros(full.present.shape, dtype=bool)  # type: ignore[attr-defined]
    gap[::2] = True  # every other scrape lost while the pod kept running
    gappy = replace(full, cpu_usage=np.where(gap, np.nan, full.cpu_usage))  # type: ignore[attr-defined,type-var]
    unsampled = replace(gappy, present=full.present & ~gap)  # type: ignore[attr-defined]
    zero_filled = replace(full, cpu_usage=np.where(gap, 0.0, full.cpu_usage))  # type: ignore[attr-defined,type-var]
    (rec,) = recommend(gappy)
    (reference,) = recommend(unsampled)
    (biased,) = recommend(zero_filled)
    # a gap contributes exactly nothing to the CPU histogram, as if it had never been sampled,
    # and does not count as a sample that would narrow the confidence bounds
    assert rec.target_cpu == reference.target_cpu
    assert rec.upper_cpu == reference.upper_cpu
    assert rec.samples == reference.samples == int((~gap).sum())
    # filling the gaps with zero (the old behaviour) would have pushed the target down
    assert biased.target_cpu < rec.target_cpu


def test_backtest_does_not_score_usage_gaps() -> None:
    ds = steady(steps=4, cpu=2.0)
    cpu = ds.cpu_usage.copy()  # type: ignore[attr-defined]
    cpu[:2] = np.nan
    gappy = replace(ds, cpu_usage=cpu)  # type: ignore[type-var]
    outcome = evaluate(gappy, (np.array([1.0]), np.array([1.0])), None, "baseline")
    # both measured steps use ~2 cores against a 1-core request; the gaps are not counted
    assert outcome.cpu_starved_pct == pytest.approx(100.0)


def test_minimums_apply_per_container() -> None:
    (rec,) = recommend(steady(cpu=0.001, mem_gib=0.01))
    assert rec.target_cpu == pytest.approx(0.025)
    assert rec.target_memory == pytest.approx(250 * MIB)


def test_short_history_widens_bounds() -> None:
    (short,) = recommend(steady(steps=12))
    (long,) = recommend(steady())
    assert short.confidence_days == pytest.approx(0.5)
    assert short.upper_cpu / short.target_cpu > long.upper_cpu / long.target_cpu


def test_oom_bump_raises_memory_target() -> None:
    steps = 24 * 7
    mem = np.full((steps, 1), 0.5)
    ds = make_dataset(
        [(4, 16)],
        [("ns", "app", 0, {})],
        np.ones((steps, 1)),
        np.full((steps, 1), 0.5),
        memory_usage=mem,
    )
    ds.memory_limit[:, 0] = 0.5 * GIB
    ds.oom_events[::12, 0] = 1.0  # killed twice a day at its 512Mi limit
    (aware,) = recommend(ds, RecommenderConfig(oom_aware=True))
    (naive,) = recommend(ds, RecommenderConfig(oom_aware=False))
    assert aware.oom_events == 14
    # bump = max(1.2 * 512Mi, 512Mi + 100Mi) = 614.4Mi, then +15% margin
    assert aware.target_memory >= 614 * MIB * 1.15
    assert naive.target_memory < 0.5 * GIB * 1.15 * 1.06


def test_cpu_weighting_by_request() -> None:
    steps = 48
    ds = make_dataset(
        [(4, 16)],
        [("ns", "app", 0, {}), ("ns", "app", 0, {})],
        np.tile([2.0, 0.0], (steps, 1)),
        np.ones((steps, 2)),
        cpu_usage=np.tile([0.1, 1.0], (steps, 1)),
    )
    (weighted,) = recommend(ds, RecommenderConfig(cpu_percentile=0.6))
    (unweighted,) = recommend(
        ds, RecommenderConfig(cpu_percentile=0.6, weight_cpu_by_request=False)
    )
    # The 2-core replica's samples carry 20x the weight of the 0-request replica's.
    assert weighted.target_cpu < 0.2
    assert unweighted.target_cpu > weighted.target_cpu


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpu_percentile": 0},
        {"upper_percentile": 1.5},
        {"safety_margin": -0.1},
        {"half_life_hours": 0},
        {"oom_bump_ratio": 0.9},
    ],
)
def test_recommender_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        RecommenderConfig(**kwargs)  # type: ignore[arg-type]


def test_workload_groups_split_by_container_name() -> None:
    ds = make_dataset(
        [(4, 16)],
        [("ns", "app", 0, {}), ("ns", "sidecar", 0, {})],
        np.ones((2, 2)),
        np.ones((2, 2)),
    )
    assert sorted(workload_groups(ds)) == [
        ("ns/Deployment/app", "app"),
        ("ns/Deployment/sidecar", "sidecar"),
    ]
    assert isinstance(ds.containers[0], ContainerInfo)


def test_evaluate_baseline_and_rightsized() -> None:
    ds = steady()
    rates = (price_nodes(ds, CATALOG).cpu_rate, price_nodes(ds, CATALOG).memory_rate)
    base = evaluate(ds, rates, None, BASELINE)
    assert base.savings_monthly == 0.0
    assert base.cpu_starved_pct == 0.0
    recs = recommend(ds)
    tuned = evaluate(ds, rates, recs, "p90")
    assert tuned.savings_pct > 50
    assert tuned.cpu_starved_pct < 10
    median = evaluate(
        ds,
        rates,
        recommend(
            ds, RecommenderConfig(cpu_percentile=0.5, memory_percentile=0.5, safety_margin=0.0)
        ),
        "p50",
    )
    assert median.cpu_starved_pct > 20
    assert median.savings_monthly > tuned.savings_monthly
    assert tuned.workloads_changed == 1
    assert tuned.oom_workloads == 0
    tiny = [r.__class__(**{**r.__dict__, "target_memory": 0.5 * GIB}) for r in recs]
    risky = evaluate(ds, rates, tiny, "tiny")
    assert risky.oom_workloads == 1 and risky.oom_risk_steps > 0


def test_split_weeks_validation() -> None:
    ds = steady(steps=24 * 14)
    train, val, train_val, test = split_weeks(ds, 1, 0.5, 0.5)
    assert (train.n_steps, val.n_steps, train_val.n_steps, test.n_steps) == (168, 84, 252, 84)
    with pytest.raises(ConfigError, match="weeks"):
        split_weeks(ds, 2, 1, 1)


def test_backtest_on_simulated_export(small_export: Path) -> None:
    ds = load_dataset(small_export)
    costs = price_nodes(ds, load_catalog(small_export / "pricing.toml"))
    demand = load_true_demand(small_export, ds)
    assert demand is not None
    report = run_backtest(
        ds, costs, train_weeks=1, validation_weeks=0.5, test_weeks=0.5, true_demand=demand
    )
    names = [o.policy for o in report.test]
    assert names[0] == BASELINE
    assert report.selected in names[1:]
    assert report.test_true_demand is not None
    assert [o.policy for o in report.test_true_demand] == names
    by_name = {o.policy: o for o in report.test}
    # More conservative percentiles never save more money.
    assert by_name["p50+15%"].savings_monthly > by_name["p95+15%"].savings_monthly
    assert by_name["p95+15%"].cpu_starved_pct <= by_name["p50+15%"].cpu_starved_pct
    with pytest.raises(ConfigError, match="unique"):
        run_backtest(
            ds,
            costs,
            train_weeks=1,
            validation_weeks=0.5,
            test_weeks=0.5,
            candidates=[RecommenderConfig(name="x"), RecommenderConfig(name="x")],
        )


def test_true_demand_absent_for_real_exports(tmp_path: Path, small_export: Path) -> None:
    import json
    import shutil

    copy = tmp_path / "real"
    shutil.copytree(small_export, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest.pop("extra")
    (copy / "manifest.json").write_text(json.dumps(manifest))
    ds = load_dataset(copy)
    assert load_true_demand(copy, ds) is None
