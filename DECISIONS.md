# Design decisions

Key design decisions in costlens, each with the reasoning and what it would cost if it turned out wrong.

- Python 3.12 with numpy as the only runtime dependency; argparse for the CLI and
  tomllib for configuration — fewer dependencies keep a fresh-clone `make check` fast and
  reliable, and numpy covers every vectorised kernel needed — cost if wrong: a richer CLI
  (typer, rich tables) would mean rewriting cli.py, about 300 lines, without touching the core.
- money is `int64` nano-dollars, and every split uses largest-remainder apportionment
  — "allocated + idle == total" becomes an exact identity that tests assert with `==`; nano
  rather than micro keeps the per-tenant rounding drift under $1e-4 over six weeks (measured
  worst $3.4e-05 by `make eval`) — cost if wrong: the ceiling is about $9.2B per ledger sum,
  so a far larger bill would need a wider integer type.
- the input is Prometheus `/api/v1/query_range` JSON plus a `manifest.json`, with
  several chunk files per metric allowed — this is what an operator can export with curl, and
  Prometheus caps range queries at 11,000 points per series, so long fine-step exports must
  be chunked — cost if wrong: users with only an OpenCost or Kubecost API would need an
  adapter.
- grid semantics — sample `t` covers `[t, t + step)`, the manifest `end` is
  exclusive, timestamps may deviate from the grid by at most 1% of a step, and anything
  further off is an error — misaligned data silently mis-assigns cost, so it must fail
  loudly — cost if wrong: exports with unusual alignment have to be re-exported with the
  manifest's start and step.
- memory usage is `max_over_time(working_set[step])` — peaks matter for OOM risk and
  for VPA memory histograms, and averages hide them — cost if wrong: under the `usage` basis,
  memory charges come out slightly high compared with average-based tools.
- the default cost basis is `max_request_usage` — a tenant pays for its reservation
  plus any burst beyond it, which is the OpenCost convention and penalises both
  over-requesting and unrequested bursting — cost if wrong: it is one flag
  (`--basis request`).
- when the charged amounts on a node exceed its capacity, the node cost is split pro
  rata and idle is zero, and the occurrence is counted — no tenant can be charged more than
  the node costs — cost if wrong: bursting tenants pay slightly less than their raw usage
  share during contention.
- the node price is split into CPU and memory with a fixed vCPU:GiB cost ratio
  (default 7.5) — this is the standard approach (OpenCost, custom-machine price sheets) when
  a provider does not price the resources separately — cost if wrong: CPU-heavy and
  memory-heavy tenants are each mis-weighted by the ratio error. The ratio is configurable
  per catalog.
- proportional and even idle are distributed per step, not over the whole period —
  idle at 03:00 should land on the tenants running at 03:00 — cost if wrong: the results
  differ from period-level tools such as Kubecost's aggregate share.
- shared namespaces take part in idle distribution as one participant, and their
  pool (own cost plus idle share) is then redistributed — this matches the "share idle,
  then share overhead" order of common showback tools — cost if wrong: under `even` idle the
  platform pool counts as one tenant, which slightly changes everyone's share.
- the shared component is a ledger transfer, negative on the `__shared__` row — the
  platform namespaces' own cost stays auditable while totals still conserve — cost if wrong:
  consumers that sum only positive cells would double-count. The report exposes row totals
  rather than raw cells.
- external costs use 730 hours per month and are spread evenly over the steps — this
  is the cloud price-sheet convention — cost if wrong: months of 28 or 31 days are off by up
  to 4% in per-day showback.
- team resolution is the pod label `team`, then the namespace label `team`, then
  `__unlabelled__` — pod overrides cover shared namespaces that host several teams'
  workloads — cost if wrong: organisations using another label name set `team_label`.
- recommendations aggregate by (workload, container name), with the workload derived
  from the owner reference minus the ReplicaSet hash or CronJob suffix — this is what a VPA
  object targets, and it survives rollouts — cost if wrong: bare pods or exotic controllers
  are recommended per pod or per owner name.
- the recommender defaults are the VPA's: p90 target, p50/p95 bounds, 15% margin,
  24 h half-life, 24 h memory-peak interval, OOM bump of max(1.2×, +100 MiB), pod minimums
  of 25m / 250 MiB, and CPU samples weighted by request — the behaviour is familiar to
  Kubernetes users and well documented — cost if wrong: the pod minimums push tiny sidecars
  up to 250 MiB (visible in the demo), which teams may prefer to override.
- the histogram reference is anchored at the newest sample rather than shifted
  periodically — this is equivalent for percentiles, simpler, and cannot overflow — cost if
  wrong: none for percentiles; absolute weights are not comparable with Go VPA checkpoints.
- the backtest selects the policy on validation weeks (cheapest with zero OOM-risk
  workloads and at most 5% CPU starvation), refits on train+validation, and reports every
  policy on the held-out test weeks — this avoids tuning on the reported split — cost if
  wrong: a single validation week can select a policy that is marginally worse on test, and
  the table shows when that happens.
- backtest limits scale with the original limit/request ratio, and an unlimited
  memory container is evaluated as `limit == request` — this matches the VPA's behaviour and
  common memory guidance — cost if wrong: OOM risk for unlimited containers is overstated
  compared with node-pressure reality.
- the simulator also exports uncensored demand as simulation-only side files under
  `manifest.extra.truth_files` — observed usage is capped by limits, so only the simulator
  can show what that censoring hides — cost if wrong: none for real exports, which never have
  these files, and the code path is skipped there.
- the ground truth is an independent float implementation (`synth/truth.py`) that
  works from the simulator's own objects — agreement between two different code paths is
  stronger evidence than re-running the engine — cost if wrong: both paths share the
  pricing-split formula by definition, so a conceptual error in that formula would not be
  caught. It is unit-tested separately (`test_price_nodes_split_is_exact_and_respects_ratio`).
- the bundled fixture is a 6-week, 15-minute, seed-7 export of about 5.5 MB of gzip
  JSON — it is the smallest period that fits 3 train + 1 validation + 2 test weeks while
  staying well under the 20 MB committed-data limit — cost if wrong: 15-minute peaks smooth
  CPU spikes that a 1-minute VPA feed would see. `costlens generate --step 5m` is available
  for finer data.
- the fixture test compares regenerated data with tolerances, not bytes — numpy's
  PCG64 stream is portable, but libm `exp`/`cos` can differ in the last ulp across CPUs —
  cost if wrong: a real generator regression smaller than 1e-9 relative would go unnoticed.
- prices are illustrative approximations of public on-demand list prices
  (m5.2xlarge $0.384/h, r5.xlarge $0.252/h, c5.2xlarge $0.34/h), and spot is a synthetic 65%
  discount ($0.119/h) — real billing integration is out of scope — cost if wrong: absolute
  dollar figures are not quotes. Relative shares and invariants do not depend on them.
- the coverage gate is 90% line+branch over the whole package (measured 98%) — this
  exceeds the 80% core-logic bar while tolerating defensive branches such as the
  negative-remainder path in apportionment — cost if wrong: an uncovered defensive branch
  could hide a bug that only pathological float input reaches.
- a usage gap (container present through a request sample but with no
  usage sample at that step) stays `NaN` and is reported as a warning. The recommender's
  histograms skip it and it does not count toward `samples` or confidence. The backtest leaves
  it out of the starvation and throttling denominators. Cost allocation (`usage` and
  `max_request_usage` bases, plus the used-hours efficiency columns) fills it from the
  container's most recent earlier usage sample, or the first later one for a leading gap.
  Request gaps take the neighbouring request because pod requests are immutable, and a
  container that never reported a request stays at 0 (BestEffort) — the old zero-fill fed fake
  idle samples into the histograms. A probe that dropped every other payments-api CPU
  sample moved the p90+15% target from 0.411 to 0.351 cores with no warning — cost if wrong:
  for a long gap, the forward-filled allocation charge can be stale. The warning makes that
  visible, and the bundled fixture has no gaps, so no published number changes.
