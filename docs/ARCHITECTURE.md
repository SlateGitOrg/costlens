# costlens architecture

This document covers how the pieces fit, the algorithms with their invariants, and what each
step costs. Notation: `S` steps, `N` nodes, `C` containers, `T` tenants, `W` workload
containers, `B` histogram buckets (176).

## Module map

```
src/costlens/
  errors.py          exception hierarchy (CostlensError and subclasses)
  units.py           Kubernetes quantities (250m, 512Mi) and Go durations (1h30m)
  money.py           int64 nano-dollars, row-wise largest-remainder apportionment
  promjson.py        /api/v1/query_range JSON reader/writer (.json, .json.gz)
  catalog.py         metric contract: PromQL, required labels, manifest.json, export script
  dataset.py         export dir -> ClusterDataset (aligned (S,N)/(S,C) arrays + metadata)
  pricing.py         price catalog (TOML) -> NodeCosts (per-step nano-$ split CPU/memory)
  config.py          AllocationConfig (TOML) and strategy enums
  allocation.py      allocation pipeline -> AllocationResult ledger (+ invariants)
  rightsizing/
    histogram.py     exponential-bucket decaying histogram
    recommender.py   per workload-container VPA-style recommendations
    backtest.py      train/validate/test policy comparison
  report/
    tables.py        Table -> Markdown / CSV / JSON
    showback.py      summary, allocation, daily, recommendation and backtest tables; report
  synth/
    spec.py          declarative cluster: pools, namespaces, workloads, demand models
    generator.py     deterministic simulation (scheduler, pods, demand, OOMs)
    export.py        simulation -> Prometheus export + pricing.toml + ground_truth.json
    truth.py         independent float reference allocation (ground truth)
  cli.py             argparse CLI
```

Dependencies point one way: `synth` -> `catalog`/`promjson` (it writes the same format real
exports use), `allocation` -> `pricing` -> `dataset` -> `catalog`/`promjson`, and `report`
depends on the rest. `synth/truth.py` imports nothing from the allocation path, so the ground
truth is independent of the code it checks.

## Data model

`ClusterDataset` holds dense `float64` arrays on a regular grid. Step `t` covers
`[t, t + step)`, and the manifest's `end` is exclusive.

| array | shape | meaning |
| --- | --- | --- |
| `node_cpu`, `node_memory` | (S, N) | capacity; `NaN` = node absent at that step |
| `present` | (S, C) | container exists (any usage/request sample) on a live node |
| `cpu_usage`, `memory_usage` | (S, C) | cores / peak working-set bytes; 0 when present but unsampled |
| `cpu_request`, `memory_request` | (S, C) | 0 for BestEffort containers |
| `cpu_limit`, `memory_limit` | (S, C) | `NaN` = unlimited |
| `oom_events` | (S, C) | OOM kills in the step |

A container is identified by `namespace/pod/container`. A pod never moves between nodes, so
each container has a single `node_index`; if the series disagree, the most frequent node wins
and a warning is recorded. Its workload comes from `kube_pod_owner`. ReplicaSet hashes and
CronJob schedule suffixes are stripped, so the recommendation targets `Deployment/api`, not a
single rollout.

Ingestion is strict where an error would corrupt results and lenient where it would only lose
detail:

- **Errors:** a malformed response, a non-matrix result, a timestamp off the grid by more than
  1% of a step, a missing required label, or a node with capacity but no labels.
- **Repairs (each recorded as a warning):** samples outside the window are dropped,
  overlapping chunks that disagree resolve to the later file, a node-step with only CPU or only
  memory capacity is dropped, containers on unknown nodes are skipped, and container-steps on
  a node without capacity are dropped.

## Money: exact apportionment

`apportion_rows(totals (R,), weights (R,K))` splits every integer total over its row of
weights using Hamilton's largest-remainder method:

1. `exact = total * w / sum(w)` and `base = floor(exact)`. Zero-weight columns are forced to
   0.
2. `remainder = total - sum(base)` goes one unit at a time to the eligible columns with the
   largest fractional parts. A stable sort breaks ties by column index, so results are
   deterministic.
3. A defensive branch handles floating-point floors that overshoot (negative remainder) by
   taking units back from the smallest fractional parts.

The guarantees, which `tests/test_money.py` checks with Hypothesis:

- `sum(out[r]) == totals[r]` exactly.
- `|out[r,k] - exact[r,k]| < 1`.
- `out >= 0`.
- Zero weight gets zero.
- A row with no positive weight sends its total to a fallback column.

Cost is `O(R·K log K)` time for one row-wise argsort, and `O(R·K)` space.

Money is held in nano-dollars. In `int64` that covers about $9.2 billion, far more than any
single report needs. The total rounding drift of a tenant is bounded by one unit per
container-step it took part in.

## Pricing

For each node, the hourly price `P` comes from the catalog. Spot nodes use `spot_hourly`, or
`on_demand × (1 − default_spot_discount)` when no spot price is listed. Unknown instance types
fall back to per-core and per-GiB rates if the catalog provides them, and otherwise raise.

The price is split with the CPU:memory cost ratio `r`:

```
mem_rate = P / (r·cores + GiB)     cpu_rate = r·mem_rate     cpu_rate·cores + mem_rate·GiB = P
```

Capacity is the node's median observed capacity, so a capacity metric that wobbles cannot
move the price. Each alive node-step costs `round(P · step_hours · 1e9)` nano-dollars, and
that integer is apportioned into a CPU part and a memory part with weights
`cpu_rate·cores : mem_rate·GiB`. Time `O(S·N)`.

## Allocation pipeline

The result is an `AllocationResult` with ledger components `cpu`, `memory`, `idle`, `shared`
and `external`. Each component is an `(S, R)` `int64` matrix, with `R = T + 2` rows: the
tenants plus `__idle__` and `__shared__`.

1. **Charge basis.** Per container-step, `amount = request | usage | max(request, usage)`, for
   CPU and memory separately.
2. **Node split.** For each node and each resource, apportion that node-step's cost over the
   weights `[amount_1 … amount_k, max(capacity − Σamount, 0)]`. A container therefore pays
   `amount/capacity` of the node, and the last column is idle. If `Σamount > capacity`
   (over-commit), idle is 0 and the node cost is split pro rata. Those steps are counted in
   `overcommitted_node_steps`. Time `O(S·C)` plus apportionment.
3. **`per_node` idle (optional).** Each node's CPU idle is apportioned over the containers on
   that node by their CPU cost; memory idle is handled the same way by memory cost. A node-step
   with idle but no containers keeps its idle.
4. **Grouping.** Each container maps to a tenant key (`namespace`, `team`, `label:x`, …).
   Containers in a shared namespace go to the `__shared__` row instead, and `np.add.at`
   scatters container columns into rows. Time `O(S·C)`.
5. **`proportional` / `even` idle.** Per step, the idle pool is apportioned over the tenants
   plus the `__shared__` row. Weights are own cost (`proportional`) or 1 for each active
   participant (`even`). If no participant is active, the idle stays on `__idle__`.
   Time `O(S·T log T)`.
6. **Shared pool.** Per step, the `__shared__` row's cpu + memory + idle is apportioned over
   the tenants with `proportional`, `even` or `weighted` weights (fixed weights only for
   active tenants). The ledger records this as a transfer: tenants receive positive `shared`
   amounts and `__shared__` carries the matching negative amount. That way the shared
   namespaces' own cost stays visible on their row, while the row's net total is whatever
   could not be placed.
7. **External costs.** `monthly_usd × period_hours / 730` is converted to nano-dollars, spread
   evenly over the steps (`spread_evenly`, exact), then distributed with the shared strategy.

**Invariants.** `check_invariants()` runs at the end of every allocation. It requires
`Σ ledger == node cost + external cost` exactly, every cell non-negative except the
`__shared__` transfer column, and every row total non-negative. The property tests check the
same facts, plus that the idle strategy never changes a tenant's own CPU and memory cost.

**Complexity.** Total time is `O(S·C + S·T log T)` and memory is `O(S·(C + T))`. The
benchmark shows roughly linear scaling: 0.7–1.0 s for 2,000 containers × 2,016 steps,
depending on machine load.

## Rightsizing

### Decaying histogram

Bucket `i` starts at `first·(ratio^i − 1)/(ratio − 1)`. With the VPA's options this is CPU
from 0.01 cores and memory from 10 MB, with ratio 1.05 up to 1000 cores or 1 TB, which gives
176 buckets. A sample at time `t` adds `w · 2^((t − ref)/half_life)` to its bucket.

The Go VPA periodically shifts its reference time to avoid overflow. costlens instead anchors
the reference at the newest sample (every factor is ≤ 1) and rescales the existing weights
when a newer sample arrives. Percentiles are scale-invariant, so the results are the same.
`percentile(p)` returns the *upper edge* of the bucket where the cumulative weight reaches
`p`, which errs high by at most one bucket (5%).

Adding `n` samples is `O(n)` (vectorised bucket index plus `bincount`). A query is `O(B)`.

### Recommender

Samples are grouped by `(namespace/workload, container name)`:

- **CPU.** Every present step's usage is a sample, weighted by `max(request, 0.1)` as the VPA
  does.
- **Memory.** One sample per container per 24-hour interval, holding that interval's peak.
  When the interval contains an OOM kill, the peak becomes at least
  `max(1.2·M, M + 100 MiB)`, where `M = max(usage, limit)` at the kill.
- **Targets.** `percentile(p_target) · (1 + margin)`, floored at the VPA's per-pod minimums
  (25 millicores, 250 MiB) divided by the number of containers in the pod.
- **Bounds.** The lower bound is the p50 estimate × `(1 + 0.001/conf)^−2`; the upper bound is
  the p95 estimate × `(1 + 1/conf)`. Here `conf = min(lifespan days, sample-days)`.

Time is `O(S·C)` for the samples plus `O(W·B)` for the queries.

### Backtest

`split_weeks` cuts the timeline into train, validation, train+validation and test windows.

1. Each candidate policy is fit on train and evaluated on validation.
2. The selected policy is the cheapest one with zero OOM-risk workloads and CPU starvation at
   or below the threshold. If no candidate qualifies, the lowest-risk one is taken.
3. Every policy is refit on train+validation and evaluated on test. The selection is only
   *marked* there.

`evaluate` replaces requests with the targets. Limits are scaled by the original
limit/request ratio, which is what the VPA does.

| metric | definition |
| --- | --- |
| request cost | `Σ request · node rate · step_hours`, scaled to 730 h |
| cpu starved % | share of present container-steps with usage > new CPU request |
| cpu throttled % | share of CPU-limited container-steps with usage > new CPU limit |
| OOM-risk steps | present container-steps with memory > new memory limit; `limit = request` where there is no limit; for the current baseline on observed data, the real OOM kills |

Synthetic exports also list uncensored demand files under `manifest.extra.truth_files`. When
those are present, the test table is computed a second time against that demand.

## Synthetic generator

- **Nodes.** Node pools (on-demand general and memory pools, a spot batch pool, with one node
  joining on day 9.5) become `SimNode`s, each with a lifetime.
- **Pods.**
  - Deployments are rolled out at random intervals around `rollout_days`, and each rollout
    gets new ReplicaSet and pod names.
  - StatefulSets run for the whole period.
  - CronJobs spawn Jobs named `<cronjob>-<minute>`.
  - DaemonSets run one pod per node for the node's lifetime.
- **Scheduler.** First-fit by requests over the pod's lifetime, against 92% of capacity.
  DaemonSets are placed first, then other workloads in decreasing order of CPU request.
- **Demand.**
  - CPU: `base · (1 + amp·diurnal) · weekend · growth · lognormal noise`, plus spikes.
  - Memory: base plus a diurnal term plus noise, and for leaky containers a linear leak
    since the last restart.
  - Observed usage is capped at the limits; a memory value at or above the limit records an
    OOM kill and restarts the leak.
- **Rounding.** Values are rounded (1e-6 cores, whole bytes) *before* export and before the
  ground truth is computed, so both paths see identical inputs.
- **Determinism.** One `numpy.random.Generator(PCG64(seed))` drives the whole run, and
  gzip output uses `mtime=0`. `tests/test_fixture.py` regenerates the bundled fixture and
  compares it with tolerances, because libm can differ in the last ulp across CPUs.

## Error handling and logging

Every deliberate failure raises a `CostlensError` subclass carrying a message that names the
file or field involved. The CLI turns these into `costlens: error: …` and exit code 2, while
argparse errors exit with code 2 through `SystemExit`.

Modules log through `logging.getLogger(__name__)`. Data-quality repairs are logged at WARNING
and also kept in `ClusterDataset.warnings` for the report; stage summaries are logged at INFO.
`-v` and `-vv` raise the CLI log level.

## Extension points

- **New groupings:** extend `tenant_key` and `GROUPINGS`.
- **New idle or shared strategies:** add an enum member and a branch in `_strategy_weights`.
  The apportionment and the invariants carry over unchanged.
- **New metrics:** add a `MetricSpec` to `catalog.METRICS`, and the loader picks it up from
  the manifest.
- **Real prices:** generate a `pricing.toml` from a billing export. The catalog format is a
  plain list of instance types.
