"""Showback tables and the full Markdown/CSV report."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from costlens.allocation import IDLE, SHARED, AllocationResult, Component
from costlens.config import HOURS_PER_MONTH, AllocationConfig
from costlens.dataset import ClusterDataset
from costlens.money import to_dollars
from costlens.pricing import NodeCosts
from costlens.report.tables import Table, bullet_list
from costlens.rightsizing.backtest import BacktestReport, PolicyOutcome
from costlens.rightsizing.recommender import Recommendation, workload_groups
from costlens.units import GIB, format_cpu, format_memory

log = logging.getLogger(__name__)


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


@dataclass(frozen=True)
class ClusterSummary:
    """Headline numbers for an allocation (dollars)."""

    period_start: int
    period_end: int
    node_cost: float
    external_cost: float
    workload_cost: float
    shared_namespace_cost: float
    idle_cost: float
    idle_pct: float
    overcommitted_node_steps: int

    def to_table(self) -> Table:
        """Two-column key/value table."""
        rows: list[list[str | int | float]] = [
            ["Period (UTC)", f"{_iso(self.period_start)} -> {_iso(self.period_end)}"],
            ["Node cost", f"${self.node_cost:,.2f}"],
            ["External shared cost", f"${self.external_cost:,.2f}"],
            ["Charged to workloads", f"${self.workload_cost:,.2f}"],
            ["Shared namespaces (own)", f"${self.shared_namespace_cost:,.2f}"],
            ["Idle capacity", f"${self.idle_cost:,.2f} ({self.idle_pct:.1f}% of node cost)"],
            ["Over-committed node-steps", str(self.overcommitted_node_steps)],
        ]
        return Table(["Metric", "Value"], rows, align=["l", "l"])


def cluster_summary(result: AllocationResult) -> ClusterSummary:
    """Summarise where the money went before idle/shared redistribution."""
    shared_col = result.rows.index(SHARED)
    own = result.ledger[Component.CPU] + result.ledger[Component.MEMORY]
    shared_own = int(own[:, shared_col].sum())
    workload = int(own.sum()) - shared_own
    idle = int(result.ledger[Component.IDLE].sum())
    node = result.node_total_nanos
    return ClusterSummary(
        period_start=result.start,
        period_end=result.start + result.ledger[Component.CPU].shape[0] * result.step_seconds,
        node_cost=to_dollars(node),
        external_cost=to_dollars(result.external_total_nanos),
        workload_cost=to_dollars(workload),
        shared_namespace_cost=to_dollars(shared_own),
        idle_cost=to_dollars(idle),
        idle_pct=100.0 * idle / node if node else 0.0,
        overcommitted_node_steps=result.overcommitted_node_steps,
    )


def allocation_table(result: AllocationResult, include_empty: bool = False) -> Table:
    """One row per tenant: cost components, total, share and request efficiency."""
    detail = result.row_totals()
    grand = result.grand_total_nanos
    rows: list[list[str | int | float]] = []
    order = sorted(
        range(len(result.rows)),
        key=lambda i: (result.rows[i] in (IDLE, SHARED), -sum(detail[result.rows[i]].values())),
    )
    for i in order:
        name = result.rows[i]
        parts = detail[name]
        total = sum(parts.values())
        if total == 0 and not include_empty and name in (IDLE, SHARED):
            continue
        rows.append(
            [
                name,
                to_dollars(parts[Component.CPU]),
                to_dollars(parts[Component.MEMORY]),
                to_dollars(parts[Component.IDLE]),
                to_dollars(parts[Component.SHARED]),
                to_dollars(parts[Component.EXTERNAL]),
                to_dollars(total),
                100.0 * total / grand if grand else 0.0,
                100.0 * _ratio(result.cpu_used_core_hours[i], result.cpu_requested_core_hours[i]),
                100.0
                * _ratio(result.memory_used_gib_hours[i], result.memory_requested_gib_hours[i]),
            ]
        )
    rows.append(
        [
            "**total**",
            *[to_dollars(sum(d[c] for d in detail.values())) for c in Component],
            to_dollars(grand),
            100.0,
            float("nan"),
            float("nan"),
        ]
    )
    columns = [
        "tenant",
        "cpu $",
        "memory $",
        "idle $",
        "shared $",
        "external $",
        "total $",
        "share %",
        "cpu eff %",
        "mem eff %",
    ]
    return Table(
        columns,
        rows,
        align=["l"] + ["r"] * 9,
        csv_columns=[
            "tenant",
            "cpu_usd",
            "memory_usd",
            "idle_usd",
            "shared_usd",
            "external_usd",
            "total_usd",
            "share_pct",
            "cpu_efficiency_pct",
            "memory_efficiency_pct",
        ],
    )


def daily_table(result: AllocationResult) -> Table:
    """Cost per tenant per UTC day (long format: day, tenant, dollars)."""
    steps_per_day = max(1, 86400 // result.step_seconds)
    starts, sums = result.windowed(steps_per_day)
    rows: list[list[str | int | float]] = []
    for w, start in enumerate(starts):
        day = dt.datetime.fromtimestamp(int(start), dt.UTC).strftime("%Y-%m-%d")
        for i, tenant in enumerate(result.rows):
            if sums[w, i]:
                rows.append([day, tenant, to_dollars(sums[w, i])])
    return Table(
        ["day", "tenant", "cost $"],
        rows,
        align=["l", "l", "r"],
        csv_columns=["day", "tenant", "cost_usd"],
    )


def recommendation_savings(
    dataset: ClusterDataset, node_costs: NodeCosts, recs: list[Recommendation]
) -> dict[tuple[str, str], float]:
    """Monthly request-cost saving of applying each recommendation over ``dataset``'s period."""
    groups = workload_groups(dataset)
    node_of = dataset.container_node_indices()
    hours = dataset.n_steps * dataset.step_hours
    scale = HOURS_PER_MONTH / hours if hours else 0.0
    out: dict[tuple[str, str], float] = {}
    for rec in recs:
        cols = np.array(groups.get(rec.key, []), dtype=np.int64)
        if cols.size == 0:
            out[rec.key] = 0.0
            continue
        present = dataset.present[:, cols]
        cpu_rate = node_costs.cpu_rate[node_of[cols]][None, :]
        mem_rate = node_costs.memory_rate[node_of[cols]][None, :] / GIB
        current = (
            np.nan_to_num(dataset.cpu_request[:, cols]) * cpu_rate
            + np.nan_to_num(dataset.memory_request[:, cols]) * mem_rate
        )
        proposed = rec.target_cpu * cpu_rate + rec.target_memory * mem_rate
        delta = np.where(present, current - proposed, 0.0).sum() * dataset.step_hours
        out[rec.key] = float(delta * scale)
    return out


def recommendations_table(
    recs: list[Recommendation], savings: dict[tuple[str, str], float], limit: int | None = None
) -> Table:
    """Recommendations sorted by monthly saving (largest first)."""
    ordered = sorted(recs, key=lambda r: -savings.get(r.key, 0.0))
    if limit is not None:
        ordered = ordered[:limit]
    rows: list[list[str | int | float]] = [
        [
            r.workload,
            r.container,
            f"{format_cpu(r.current_cpu_request)} -> {format_cpu(r.target_cpu)}",
            f"{format_memory(r.current_memory_request)} -> {format_memory(r.target_memory)}",
            f"{format_cpu(r.lower_cpu)}..{format_cpu(r.upper_cpu)}",
            f"{format_memory(r.lower_memory)}..{format_memory(r.upper_memory)}",
            r.oom_events,
            savings.get(r.key, 0.0),
        ]
        for r in ordered
    ]
    return Table(
        [
            "workload",
            "container",
            "cpu request",
            "memory request",
            "cpu bounds",
            "memory bounds",
            "OOMs",
            "saving $/mo",
        ],
        rows,
        align=["l", "l", "l", "l", "l", "l", "r", "r"],
    )


def recommendations_csv_table(
    recs: list[Recommendation], savings: dict[tuple[str, str], float]
) -> Table:
    """Machine-readable recommendations (cores and bytes)."""
    rows: list[list[str | int | float]] = [
        [
            r.workload,
            r.container,
            r.current_cpu_request,
            r.target_cpu,
            r.lower_cpu,
            r.upper_cpu,
            r.current_memory_request,
            r.target_memory,
            r.lower_memory,
            r.upper_memory,
            r.current_cpu_limit,
            r.current_memory_limit,
            r.samples,
            r.oom_events,
            r.confidence_days,
            savings.get(r.key, 0.0),
        ]
        for r in recs
    ]
    return Table(
        [
            "workload",
            "container",
            "cpu_request",
            "cpu_target",
            "cpu_lower",
            "cpu_upper",
            "memory_request_bytes",
            "memory_target_bytes",
            "memory_lower_bytes",
            "memory_upper_bytes",
            "cpu_limit",
            "memory_limit_bytes",
            "samples",
            "oom_events",
            "confidence_days",
            "monthly_saving_usd",
        ],
        rows,
    )


def backtest_table(outcomes: list[PolicyOutcome], selected: str | None = None) -> Table:
    """Savings-vs-risk table for one evaluation window."""
    rows: list[list[str | int | float]] = [
        [
            f"{o.policy} (selected)" if o.policy == selected else o.policy,
            o.request_cost_monthly,
            o.savings_monthly,
            o.savings_pct,
            o.cpu_starved_pct,
            o.cpu_throttled_pct,
            o.oom_risk_steps,
            o.oom_workloads,
        ]
        for o in outcomes
    ]
    return Table(
        [
            "policy",
            "requests $/mo",
            "saving $/mo",
            "saving %",
            "cpu starved %",
            "cpu throttled %",
            "OOM-risk steps",
            "OOM-risk workloads",
        ],
        rows,
        align=["l"] + ["r"] * 7,
    )


def config_lines(config: AllocationConfig) -> list[str]:
    """Human-readable description of an allocation config."""
    lines = [
        f"cost basis: `{config.basis}`",
        f"idle distribution: `{config.idle}`",
        f"grouped by: `{config.group_by}`",
    ]
    if config.shared_namespaces:
        lines.append(
            f"shared namespaces: {', '.join(f'`{n}`' for n in config.shared_namespaces)} "
            f"distributed `{config.shared_strategy}`"
        )
    lines += [f"external cost `{e.name}`: ${e.monthly_usd:,.2f}/month" for e in config.external]
    return lines


@dataclass(frozen=True)
class ReportInputs:
    """Everything the full report renders."""

    dataset: ClusterDataset
    node_costs: NodeCosts
    allocations: dict[str, AllocationResult]  # title -> result
    recommendations: list[Recommendation]
    savings: dict[tuple[str, str], float]
    backtest: BacktestReport | None


def write_report(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Write ``showback.md`` and the CSV files; return the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = inputs.dataset
    written: list[Path] = []
    md: list[str] = [
        f"# Showback report: {ds.cluster}",
        "",
        f"{len(ds.nodes)} nodes, {len(ds.containers)} containers, {ds.n_steps} steps of "
        f"{ds.step_seconds // 60} min. All amounts in USD.",
        "",
    ]
    first = True
    for title, result in inputs.allocations.items():
        slug = result.config.group_by.replace(":", "-")
        if first:
            md += ["## Cluster summary", "", cluster_summary(result).to_table().to_markdown(), ""]
            first = False
        table = allocation_table(result)
        md += [
            f"## Cost by {title}",
            "",
            bullet_list(config_lines(result.config)),
            "",
            table.to_markdown(),
            "",
        ]
        written.append(table.write_csv(out_dir / f"allocation_{slug}.csv"))
        written.append(daily_table(result).write_csv(out_dir / f"daily_{slug}.csv"))
    rec_table = recommendations_table(inputs.recommendations, inputs.savings, limit=30)
    total_saving = sum(inputs.savings.values())
    md += [
        "## Rightsizing recommendations",
        "",
        f"VPA-style targets from the whole period; applying all of them changes request cost "
        f"by ${total_saving:,.2f}/month (positive = saving; negative rows are under-provisioned "
        f"workloads that need more). {len(rec_table.rows)} shown; machine-readable list in "
        "`recommendations.csv`.",
        "",
        rec_table.to_markdown(),
        "",
    ]
    written.append(
        recommendations_csv_table(inputs.recommendations, inputs.savings).write_csv(
            out_dir / "recommendations.csv"
        )
    )
    if inputs.backtest is not None:
        bt = inputs.backtest
        md += [
            "## Rightsizing backtest",
            "",
            f"Train {bt.train_weeks:g} weeks, validation {bt.validation_weeks:g} weeks, "
            f"held-out test {bt.test_weeks:g} weeks. Selection rule: {bt.selection_rule}. "
            f"Selected on validation: **{bt.selected}**.",
            "",
            "### Held-out test weeks (observed usage)",
            "",
            backtest_table(bt.test, bt.selected).to_markdown(),
            "",
        ]
        written.append(backtest_table(bt.test).write_csv(out_dir / "backtest_test.csv"))
        written.append(backtest_table(bt.validation).write_csv(out_dir / "backtest_validation.csv"))
        if bt.test_true_demand is not None:
            md += [
                "### Held-out test weeks (uncensored demand, simulation only)",
                "",
                backtest_table(bt.test_true_demand, bt.selected).to_markdown(),
                "",
            ]
            written.append(
                backtest_table(bt.test_true_demand).write_csv(
                    out_dir / "backtest_test_true_demand.csv"
                )
            )
    md += ["## Data quality", "", bullet_list(ds.warnings), ""]
    report = out_dir / "showback.md"
    report.write_text("\n".join(md), encoding="utf-8")
    written.insert(0, report)
    log.info("wrote %d report files to %s", len(written), out_dir)
    return written
