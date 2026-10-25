"""Compare costlens allocations with the simulator's independent ground truth.

Usage: uv run python scripts/eval_ground_truth.py examples/demo-cluster
Prints one row per ground-truth scenario with the worst per-tenant error.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from costlens.allocation import IDLE, SHARED, allocate
from costlens.config import AllocationConfig, CostBasis, IdleStrategy
from costlens.dataset import load_dataset
from costlens.money import to_dollars
from costlens.pricing import load_catalog, price_nodes


def main(directory: Path) -> int:
    truth = json.loads((directory / "ground_truth.json").read_text())
    dataset = load_dataset(directory)
    costs = price_nodes(dataset, load_catalog(directory / "pricing.toml"))
    total = to_dollars(costs.grand_total())
    print(f"node cost: costlens ${total:,.6f} vs ground truth ${truth['total_node_cost_usd']:,.6f}")
    print(
        "| scenario | tenants | max abs error $ | max rel error | idle error $ | "
        "sum == total (exact) | ms |"
    )
    print("| :--- | ---: | ---: | ---: | ---: | :---: | ---: |")
    worst = 0.0
    for name, scenario in truth["scenarios"].items():
        config = AllocationConfig(
            basis=CostBasis(scenario["basis"]),
            idle=IdleStrategy(scenario["idle"]),
            group_by=scenario["group_by"],
            shared_namespaces=tuple(scenario["shared_namespaces"]),
        )
        t0 = time.perf_counter()
        result = allocate(dataset, costs, config)
        elapsed = (time.perf_counter() - t0) * 1000
        totals = result.totals_by_row()
        errors = [
            (abs(to_dollars(totals[tenant]) - usd), abs(to_dollars(totals[tenant]) - usd) / usd)
            for tenant, usd in scenario["tenants_usd"].items()
        ]
        max_abs = max(e for e, _ in errors)
        max_rel = max(r for _, r in errors)
        idle_err = abs(to_dollars(totals[IDLE]) - scenario["idle_usd"]) + abs(
            to_dollars(totals[SHARED]) - scenario["shared_usd"]
        )
        exact = sum(totals.values()) == result.grand_total_nanos
        worst = max(worst, max_abs, idle_err)
        print(
            f"| {name} | {len(errors)} | {max_abs:.2e} | {max_rel:.1e} | {idle_err:.1e} | "
            f"{'yes' if exact else 'NO'} | {elapsed:.0f} |"
        )
    print(f"worst absolute error across scenarios: ${worst:.2e}")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "examples/demo-cluster")))
