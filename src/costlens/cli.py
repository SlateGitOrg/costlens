"""Command-line interface: ``costlens <command> ...``.

Commands:
    generate   simulate a multi-team cluster and write a Prometheus export (+ ground truth)
    validate   load an export and report its shape and data-quality warnings
    allocate   allocate cost to tenants (namespace / team / label / workload / node)
    recommend  VPA-style rightsizing recommendations
    backtest   compare rightsizing policies on held-out weeks
    report     write the full showback report (Markdown + CSV)
    queries    print the PromQL costlens needs, or a chunked curl export script
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from costlens import __version__
from costlens.allocation import allocate
from costlens.catalog import METRICS, export_script, render_promql
from costlens.config import (
    AllocationConfig,
    CostBasis,
    IdleStrategy,
    SharedStrategy,
    load_config,
)
from costlens.dataset import ClusterDataset, load_dataset, load_true_demand
from costlens.errors import CostlensError
from costlens.money import to_dollars
from costlens.pricing import NodeCosts, load_catalog, price_nodes
from costlens.report.showback import (
    ReportInputs,
    allocation_table,
    backtest_table,
    cluster_summary,
    recommendation_savings,
    recommendations_csv_table,
    recommendations_table,
    write_report,
)
from costlens.rightsizing.backtest import run_backtest
from costlens.rightsizing.recommender import RecommenderConfig, recommend
from costlens.synth.export import write_export
from costlens.synth.generator import simulate
from costlens.synth.spec import default_spec
from costlens.units import parse_duration

log = logging.getLogger("costlens")

DEFAULT_START = 1772409600  # 2026-03-02T00:00:00Z (a Monday)


def _load(args: argparse.Namespace) -> tuple[ClusterDataset, NodeCosts]:
    directory = Path(args.export)
    pricing = Path(args.pricing) if args.pricing else directory / "pricing.toml"
    dataset = load_dataset(directory)
    return dataset, price_nodes(dataset, load_catalog(pricing))


def _allocation_config(args: argparse.Namespace) -> AllocationConfig:
    config = load_config(Path(args.config)) if args.config else AllocationConfig()
    overrides: dict[str, object] = {}
    if args.basis:
        overrides["basis"] = CostBasis(args.basis)
    if args.idle:
        overrides["idle"] = IdleStrategy(args.idle)
    if args.group_by:
        overrides["group_by"] = args.group_by
    if args.shared_namespaces is not None:
        overrides["shared_namespaces"] = tuple(n for n in args.shared_namespaces.split(",") if n)
    if args.shared_strategy:
        overrides["shared_strategy"] = SharedStrategy(args.shared_strategy)
    return replace(config, **overrides) if overrides else config  # type: ignore[arg-type]


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        log.info("wrote %s", out)
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


# ---- commands -------------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    step = int(parse_duration(args.step))
    sim = simulate(
        default_spec(), start=args.start, weeks=args.weeks, step_seconds=step, seed=args.seed
    )
    manifest = write_export(sim, Path(args.out), chunk_days=args.chunk_days)
    files = sum(len(v) for v in manifest.files.values())
    print(
        f"wrote {Path(args.out)}: {len(sim.nodes)} nodes, {len(sim.containers)} containers, "
        f"{sim.n_steps} steps x {step}s, {files} export files (+ pricing.toml, "
        "ground_truth.json)"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    dataset, costs = _load(args)
    days = dataset.n_steps * dataset.step_seconds / 86400
    namespaces = sorted({c.namespace for c in dataset.containers})
    print(f"cluster       {dataset.cluster}")
    print(f"period        {days:.2f} days, {dataset.n_steps} steps of {dataset.step_seconds}s")
    print(
        f"nodes         {len(dataset.nodes)} "
        f"({', '.join(sorted({n.instance_type for n in dataset.nodes}))})"
    )
    print(f"containers    {len(dataset.containers)} in {len(namespaces)} namespaces")
    print(f"node cost     ${to_dollars(costs.grand_total()):,.2f}")
    print(f"warnings      {len(dataset.warnings)}")
    for warning in dataset.warnings:
        print(f"  - {warning}")
    return 0


def cmd_allocate(args: argparse.Namespace) -> int:
    dataset, costs = _load(args)
    result = allocate(dataset, costs, _allocation_config(args))
    table = allocation_table(result)
    if args.format == "table":
        text = cluster_summary(result).to_table().to_markdown() + "\n\n" + table.to_markdown()
    else:
        text = table.render(args.format)
    _emit(text, args.out)
    return 0


def _recommender_config(args: argparse.Namespace) -> RecommenderConfig:
    return RecommenderConfig(
        name="cli",
        cpu_percentile=args.cpu_percentile,
        memory_percentile=args.memory_percentile,
        safety_margin=args.margin,
        half_life_hours=args.half_life_hours,
        oom_aware=not args.no_oom_bump,
    )


def cmd_recommend(args: argparse.Namespace) -> int:
    dataset, costs = _load(args)
    recs = recommend(dataset, _recommender_config(args))
    savings = recommendation_savings(dataset, costs, recs)
    if args.format == "table":
        text = recommendations_table(recs, savings).to_markdown()
        text += f"\n\nTotal request-cost change: ${sum(savings.values()):,.2f}/month"
    else:
        text = recommendations_csv_table(recs, savings).render(args.format)
    _emit(text, args.out)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    dataset, costs = _load(args)
    demand = load_true_demand(Path(args.export), dataset)
    report = run_backtest(
        dataset,
        costs,
        train_weeks=args.train_weeks,
        validation_weeks=args.validation_weeks,
        test_weeks=args.test_weeks,
        max_cpu_starved_pct=args.max_cpu_starved_pct,
        true_demand=demand,
    )
    if args.format != "table":
        _emit(backtest_table(report.test).render(args.format), args.out)
        return 0
    parts = [
        f"selection rule: {report.selection_rule}",
        f"selected on validation: {report.selected}",
        "",
        "validation weeks:",
        backtest_table(report.validation, report.selected).to_markdown(),
        "",
        "held-out test weeks (observed usage):",
        backtest_table(report.test, report.selected).to_markdown(),
    ]
    if report.test_true_demand is not None:
        parts += [
            "",
            "held-out test weeks (uncensored demand, simulation only):",
            backtest_table(report.test_true_demand, report.selected).to_markdown(),
        ]
    _emit("\n".join(parts), args.out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    dataset, costs = _load(args)
    base = _allocation_config(args)
    groupings = args.group_by_list.split(",") if args.group_by_list else ["team", "namespace"]
    allocations = {
        group: allocate(dataset, costs, replace(base, group_by=group)) for group in groupings
    }
    recs = recommend(dataset)
    backtest = None
    if not args.no_backtest:
        try:
            backtest = run_backtest(
                dataset, costs, true_demand=load_true_demand(Path(args.export), dataset)
            )
        except CostlensError as exc:
            log.warning("skipping backtest: %s", exc)
    written = write_report(
        ReportInputs(
            dataset,
            costs,
            allocations,
            recs,
            recommendation_savings(dataset, costs, recs),
            backtest,
        ),
        Path(args.out),
    )
    for path in written:
        print(path)
    return 0


def cmd_queries(args: argparse.Namespace) -> int:
    step = int(parse_duration(args.step))
    if args.script:
        sys.stdout.write(export_script(args.prometheus, step, int(parse_duration(args.chunk))))
        return 0
    for spec in METRICS.values():
        flag = "required" if spec.required else "optional"
        print(f"# {spec.name} ({flag}): {spec.description}")
        print(render_promql(spec, step))
        print()
    return 0


# ---- parser ---------------------------------------------------------------------------------


def _add_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("export", help="export directory containing manifest.json")
    p.add_argument("--pricing", help="pricing catalog TOML (default: <export>/pricing.toml)")


def _add_allocation_args(p: argparse.ArgumentParser, group_default: bool = True) -> None:
    p.add_argument("--config", help="allocation config TOML")
    p.add_argument("--basis", choices=[b.value for b in CostBasis])
    p.add_argument("--idle", choices=[i.value for i in IdleStrategy])
    if group_default:
        p.add_argument("--group-by", help="namespace|team|workload|node|pod|container|label:X")
    else:
        p.set_defaults(group_by=None)
    p.add_argument("--shared-namespaces", help="comma-separated namespaces treated as shared")
    p.add_argument("--shared-strategy", choices=[s.value for s in SharedStrategy])


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=["table", "csv", "json"], default="table")
    p.add_argument("--out", help="write to this file instead of stdout")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="costlens",
        description="Kubernetes cost allocation and rightsizing from Prometheus exports.",
    )
    parser.add_argument("--version", action="version", version=f"costlens {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for info logs, -vv for debug"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="simulate a cluster and write an export")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--weeks", type=float, default=6.0)
    p.add_argument("--step", default="15m", help="sample step, e.g. 15m or 300")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start", type=int, default=DEFAULT_START, help="unix start time")
    p.add_argument("--chunk-days", type=float, default=7.0, help="days per export file")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate", help="load an export and print a summary")
    _add_export_args(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("allocate", help="allocate cost to tenants")
    _add_export_args(p)
    _add_allocation_args(p)
    _add_output_args(p)
    p.set_defaults(func=cmd_allocate)

    p = sub.add_parser("recommend", help="VPA-style rightsizing recommendations")
    _add_export_args(p)
    p.add_argument("--cpu-percentile", type=float, default=0.9)
    p.add_argument("--memory-percentile", type=float, default=0.9)
    p.add_argument("--margin", type=float, default=0.15, help="safety margin fraction")
    p.add_argument("--half-life-hours", type=float, default=24.0)
    p.add_argument("--no-oom-bump", action="store_true", help="ignore OOM kills")
    _add_output_args(p)
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("backtest", help="compare rightsizing policies on held-out weeks")
    _add_export_args(p)
    p.add_argument("--train-weeks", type=float, default=3.0)
    p.add_argument("--validation-weeks", type=float, default=1.0)
    p.add_argument("--test-weeks", type=float, default=2.0)
    p.add_argument("--max-cpu-starved-pct", type=float, default=5.0)
    _add_output_args(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="write the full showback report")
    _add_export_args(p)
    _add_allocation_args(p, group_default=False)
    p.add_argument("--group-by-list", help="comma-separated groupings (default team,namespace)")
    p.add_argument("--no-backtest", action="store_true")
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("queries", help="print the PromQL costlens consumes")
    p.add_argument("--step", default="5m")
    p.add_argument("--script", action="store_true", help="print a chunked curl export script")
    p.add_argument("--prometheus", default="http://localhost:9090")
    p.add_argument("--chunk", default="7d", help="time span per request (capped at 11k points)")
    p.set_defaults(func=cmd_queries)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code (2 on a costlens error)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    try:
        code: int = args.func(args)
    except CostlensError as exc:
        print(f"costlens: error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
