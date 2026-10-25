"""Backtest rightsizing policies on held-out weeks.

Protocol (no tuning on the reported split):

1. Split the timeline into ``train`` | ``validation`` | ``test`` weeks.
2. For each candidate :class:`RecommenderConfig`, fit on ``train`` and apply the targets as the
   new requests during ``validation``; pick the cheapest policy whose validation risk is
   within the budget (no OOM-risk workloads, CPU starvation <= ``max_cpu_starved_pct``).
3. Refit every policy on ``train + validation`` and report all of them on ``test``; the
   selected policy is only *marked*, never re-chosen on test.

Risk model, per held-out container-step (requests replaced by the recommendation, limits
scaled by the container's original limit/request ratio, which is what the VPA does):

* **CPU starved** — usage above the new CPU request (the container relies on spare node CPU
  and is throttled under contention).
* **CPU throttled** — usage above the new CPU limit, for containers that have a limit.
* **OOM risk** — peak memory above the new memory limit (each such container-step is a
  potential OOM kill); containers without a limit are evaluated against
  ``limit == request``, the common guidance for memory. For the current-requests baseline on
  observed data the OOM kills that actually happened are counted instead.

Observed usage is itself censored by the *current* limits (a container cannot be seen using
more CPU than its current limit), so these are lower bounds on risk; the limitation is stated
in the README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from costlens.config import HOURS_PER_MONTH
from costlens.dataset import ClusterDataset
from costlens.errors import ConfigError
from costlens.pricing import NodeCosts
from costlens.rightsizing.recommender import (
    Recommendation,
    RecommenderConfig,
    recommend,
    workload_groups,
)
from costlens.units import GIB

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BASELINE = "current requests"


def default_candidates() -> list[RecommenderConfig]:
    """The policies compared in the backtest table (VPA default is ``p90+15%``)."""
    return [
        RecommenderConfig(name="p50+15%", cpu_percentile=0.5, memory_percentile=0.5),
        RecommenderConfig(name="p90+15%"),
        RecommenderConfig(name="p90+15% no-OOM-bump", oom_aware=False),
        RecommenderConfig(name="p95+15%", cpu_percentile=0.95, memory_percentile=0.95),
        RecommenderConfig(
            name="p95+30%", cpu_percentile=0.95, memory_percentile=0.95, safety_margin=0.30
        ),
        RecommenderConfig(name="p99+15%", cpu_percentile=0.99, memory_percentile=0.99),
        RecommenderConfig(
            name="max+0%", cpu_percentile=1.0, memory_percentile=1.0, safety_margin=0.0
        ),
    ]


@dataclass(frozen=True)
class PolicyOutcome:
    """Cost and risk of one policy over one evaluation window."""

    policy: str
    request_cost_monthly: float
    savings_monthly: float
    savings_pct: float
    cpu_starved_pct: float
    cpu_throttled_pct: float
    oom_risk_steps: int
    oom_workloads: int
    workloads_changed: int


@dataclass(frozen=True)
class BacktestReport:
    """Validation and test outcomes plus the policy chosen on validation."""

    train_weeks: float
    validation_weeks: float
    test_weeks: float
    validation: list[PolicyOutcome]
    test: list[PolicyOutcome]
    selected: str
    selection_rule: str
    test_true_demand: list[PolicyOutcome] | None = None


def request_cost(
    window: ClusterDataset,
    node_costs_rates: tuple[FloatArray, FloatArray],
    cpu: FloatArray,
    memory: FloatArray,
) -> float:
    """Dollar cost of reserving ``cpu``/``memory`` ``(S, C)`` at each container's node rates."""
    cpu_rate, mem_rate = node_costs_rates
    node_of = window.container_node_indices()
    hours = window.step_hours
    per_step = (
        np.nan_to_num(cpu) * cpu_rate[node_of][None, :]
        + np.nan_to_num(memory) / GIB * mem_rate[node_of][None, :]
    )
    return float(per_step[window.present].sum() * hours)


def evaluate(
    window: ClusterDataset,
    rates: tuple[FloatArray, FloatArray],
    recommendations: list[Recommendation] | None,
    policy: str,
    demand: tuple[FloatArray, FloatArray] | None = None,
) -> PolicyOutcome:
    """Apply ``recommendations`` (``None`` = keep current requests) to ``window``.

    Risk is scored against observed usage, or against ``demand`` (uncensored CPU and memory
    demand, ``(S, C)`` each) when the export is synthetic and provides it.
    """
    present = window.present
    cpu_req = window.cpu_request.copy()
    mem_req = window.memory_request.copy()
    cpu_lim = window.cpu_limit.copy()
    mem_lim = window.memory_limit.copy()
    changed = 0
    if recommendations is not None:
        recs = {r.key: r for r in recommendations}
        for key, columns in workload_groups(window).items():
            rec = recs.get(key)
            if rec is None:
                continue  # workload first seen after training: keep its current requests
            changed += 1
            for col in columns:
                old_cpu, old_mem = cpu_req[:, col].copy(), mem_req[:, col].copy()
                cpu_req[:, col] = rec.target_cpu
                mem_req[:, col] = rec.target_memory
                with np.errstate(divide="ignore", invalid="ignore"):
                    cpu_ratio = np.where(old_cpu > 0, cpu_lim[:, col] / old_cpu, np.nan)
                    mem_ratio = np.where(old_mem > 0, mem_lim[:, col] / old_mem, np.nan)
                cpu_lim[:, col] = rec.target_cpu * cpu_ratio
                mem_lim[:, col] = rec.target_memory * mem_ratio
    baseline = request_cost(window, rates, window.cpu_request, window.memory_request)
    cost = request_cost(window, rates, cpu_req, mem_req)
    hours = window.n_steps * window.step_hours
    monthly = HOURS_PER_MONTH / hours

    cpu_seen = window.cpu_usage if demand is None else demand[0]
    mem_seen = window.memory_usage if demand is None else demand[1]
    usage = np.where(present, cpu_seen, np.nan)
    n_samples = max(int(present.sum()), 1)
    starved = np.nan_to_num(usage > cpu_req + 1e-9, nan=0.0).astype(bool) & present
    limited = present & ~np.isnan(cpu_lim)
    throttled = limited & (np.nan_to_num(usage) > np.nan_to_num(cpu_lim, nan=np.inf) + 1e-9)
    effective_mem_limit = np.where(np.isnan(mem_lim), mem_req, mem_lim)
    killed = present & (np.nan_to_num(mem_seen) > effective_mem_limit + 0.5)
    if recommendations is None and demand is None:
        # The baseline's kills are the ones that actually happened.
        killed = present & (window.oom_events > 0)
    oom_workloads = 0
    for columns in workload_groups(window).values():
        if np.any(killed[:, columns]):
            oom_workloads += 1
    return PolicyOutcome(
        policy=policy,
        request_cost_monthly=cost * monthly,
        savings_monthly=(baseline - cost) * monthly,
        savings_pct=100.0 * (baseline - cost) / baseline if baseline > 0 else 0.0,
        cpu_starved_pct=100.0 * float(starved.sum()) / n_samples,
        cpu_throttled_pct=100.0 * float(throttled.sum()) / max(int(limited.sum()), 1),
        oom_risk_steps=int(killed.sum()),
        oom_workloads=oom_workloads,
        workloads_changed=changed,
    )


def split_weeks(
    dataset: ClusterDataset, train: float, validation: float, test: float
) -> tuple[ClusterDataset, ClusterDataset, ClusterDataset, ClusterDataset]:
    """Return ``(train, validation, train+validation, test)`` windows."""
    per_week = round(7 * 86400 / dataset.step_seconds)
    a = round(train * per_week)
    b = a + round(validation * per_week)
    c = b + round(test * per_week)
    if min(a, b - a, c - b) <= 0 or c > dataset.n_steps:
        raise ConfigError(
            f"need {train}+{validation}+{test} weeks but the export has "
            f"{dataset.n_steps / per_week:.2f} weeks",
        )
    return (
        dataset.slice_steps(0, a),
        dataset.slice_steps(a, b),
        dataset.slice_steps(0, b),
        dataset.slice_steps(b, c),
    )


def run_backtest(
    dataset: ClusterDataset,
    node_costs: NodeCosts,
    *,
    train_weeks: float = 3.0,
    validation_weeks: float = 1.0,
    test_weeks: float = 2.0,
    candidates: list[RecommenderConfig] | None = None,
    max_cpu_starved_pct: float = 5.0,
    true_demand: tuple[FloatArray, FloatArray] | None = None,
) -> BacktestReport:
    """Run the full select-on-validation, report-on-test protocol.

    ``true_demand`` (synthetic exports only) adds a second test table scored against the
    uncensored demand; selection never looks at it.
    """
    candidates = candidates or default_candidates()
    names = [c.name for c in candidates]
    if len(set(names)) != len(names) or BASELINE in names:
        raise ConfigError("candidate policy names must be unique")
    rates = (node_costs.cpu_rate, node_costs.memory_rate)
    train, validation, train_val, test = split_weeks(
        dataset, train_weeks, validation_weeks, test_weeks
    )

    val_outcomes = [evaluate(validation, rates, None, BASELINE)]
    val_outcomes += [
        evaluate(validation, rates, recommend(train, cfg), cfg.name) for cfg in candidates
    ]
    eligible = [
        o
        for o in val_outcomes[1:]
        if o.oom_workloads == 0 and o.cpu_starved_pct <= max_cpu_starved_pct
    ]
    if eligible:
        selected = max(eligible, key=lambda o: o.savings_monthly).policy
    else:
        selected = min(val_outcomes[1:], key=lambda o: (o.oom_workloads, o.cpu_starved_pct)).policy
    rule = (
        f"cheapest policy on validation with 0 OOM-risk workloads and CPU starvation <= "
        f"{max_cpu_starved_pct:g}% of container-steps"
        + ("" if eligible else " (none qualified: lowest-risk policy chosen)")
    )
    fitted = [(cfg.name, recommend(train_val, cfg)) for cfg in candidates]
    test_outcomes = [evaluate(test, rates, None, BASELINE)]
    test_outcomes += [evaluate(test, rates, recs, name) for name, recs in fitted]
    true_outcomes = None
    if true_demand is not None:
        window = slice(train_val.n_steps, train_val.n_steps + test.n_steps)
        test_demand = (true_demand[0][window], true_demand[1][window])
        true_outcomes = [evaluate(test, rates, None, BASELINE, test_demand)]
        true_outcomes += [evaluate(test, rates, recs, name, test_demand) for name, recs in fitted]
    log.info("backtest selected %s on validation", selected)
    return BacktestReport(
        train_weeks=train_weeks,
        validation_weeks=validation_weeks,
        test_weeks=test_weeks,
        validation=val_outcomes,
        test=test_outcomes,
        selected=selected,
        selection_rule=rule,
        test_true_demand=true_outcomes,
    )
