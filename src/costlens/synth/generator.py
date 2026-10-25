"""Simulate a cluster described by a :class:`ClusterSpec` on a regular time grid.

The simulation is deterministic for a given seed (numpy ``PCG64``). It produces the *true*
state of the cluster — node lifetimes, pod placements, true demand, the usage Prometheus would
observe (demand censored by limits), OOM kills — which :mod:`costlens.synth.export` turns into
Prometheus range-query exports and :mod:`costlens.synth.truth` turns into ground-truth costs.

Values are quantised (CPU to 1e-6 cores, memory to whole bytes) *before* they are stored so
that the exported text and the ground-truth computation see bit-identical numbers.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from costlens.errors import ConfigError
from costlens.synth.spec import ClusterSpec, ContainerSpec, NodePoolSpec, WorkloadSpec
from costlens.units import GIB, MIB

log = logging.getLogger(__name__)

FloatMatrix = npt.NDArray[np.float64]
_SUFFIX_ALPHABET = "bcdfghjklmnpqrstvwxz2456789"  # the alphabet Kubernetes uses for names


@dataclass(frozen=True)
class SimNode:
    """A simulated node and the steps ``[alive_from, alive_to)`` it existed."""

    name: str
    pool: NodePoolSpec
    zone: str
    alive_from: int
    alive_to: int

    @property
    def cores(self) -> float:
        """CPU capacity."""
        return self.pool.cores

    @property
    def memory_bytes(self) -> float:
        """Memory capacity in bytes."""
        return self.pool.memory_gib * GIB


@dataclass(frozen=True)
class SimContainer:
    """One container of one pod, alive over steps ``[start, stop)`` on ``node``."""

    namespace: str
    pod: str
    container: str
    workload: WorkloadSpec
    spec: ContainerSpec
    owner_kind: str
    owner_name: str
    node: int
    start: int
    stop: int
    team: str | None
    pod_labels: dict[str, str] = field(default_factory=dict)

    @property
    def cpu_request(self) -> float:
        """CPU request in cores."""
        return self.spec.cpu_request

    @property
    def memory_request(self) -> float:
        """Memory request in bytes."""
        return self.spec.mem_request_mib * MIB


@dataclass
class SimulatedCluster:
    """Output of :func:`simulate`: arrays are ``(S, C)`` with NaN where a container is absent."""

    spec: ClusterSpec
    start: int
    step_seconds: int
    n_steps: int
    seed: int
    nodes: list[SimNode]
    containers: list[SimContainer]
    cpu_demand: FloatMatrix
    cpu_usage: FloatMatrix
    memory_usage: FloatMatrix
    oom_events: FloatMatrix
    memory_demand: FloatMatrix

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        """Start timestamp of every step."""
        return self.start + self.step_seconds * np.arange(self.n_steps, dtype=np.int64)

    @property
    def step_hours(self) -> float:
        """Step length in hours."""
        return self.step_seconds / 3600.0


class _Scheduler:
    """First-fit placement of pods onto nodes by requests over their lifetime."""

    def __init__(self, nodes: list[SimNode], n_steps: int, fraction: float) -> None:
        self.nodes = nodes
        self.cpu = np.zeros((n_steps, len(nodes)))
        self.mem = np.zeros((n_steps, len(nodes)))
        self.fraction = fraction

    def place(
        self,
        pool: str | None,
        start: int,
        stop: int,
        cpu: float,
        mem: float,
        node: int | None = None,
    ) -> int:
        candidates = (
            [node]
            if node is not None
            else [i for i, n in enumerate(self.nodes) if pool is None or n.pool.name == pool]
        )
        for i in candidates:
            n = self.nodes[i]
            if n.alive_from > start or n.alive_to < stop:
                continue
            free_cpu = n.cores * self.fraction - self.cpu[start:stop, i].max(initial=0.0)
            free_mem = n.memory_bytes * self.fraction - self.mem[start:stop, i].max(initial=0.0)
            if node is not None or (free_cpu >= cpu and free_mem >= mem):
                self.cpu[start:stop, i] += cpu
                self.mem[start:stop, i] += mem
                return i
        raise ConfigError(
            f"cannot schedule pod needing {cpu} cores / {mem / GIB:.1f}GiB in pool {pool!r} "
            f"for steps [{start}, {stop}); enlarge the node pool",
        )


def _suffix(rng: np.random.Generator, length: int) -> str:
    picks = rng.integers(0, len(_SUFFIX_ALPHABET), size=length)
    return "".join(_SUFFIX_ALPHABET[int(i)] for i in picks)


def _build_nodes(spec: ClusterSpec, n_steps: int, step_seconds: int) -> list[SimNode]:
    nodes: list[SimNode] = []
    for pool in spec.pools:
        for i in range(pool.count + pool.late_nodes):
            late = i >= pool.count
            join = int(pool.late_join_day * 86400 // step_seconds) if late else 0
            nodes.append(
                SimNode(
                    name=f"ip-10-0-{len(nodes) + 1}-{(i * 37 + 11) % 250}.{pool.name}.internal",
                    pool=pool,
                    zone=pool.zones[i % len(pool.zones)],
                    alive_from=min(join, n_steps),
                    alive_to=n_steps,
                ),
            )
    return nodes


def _pod_intervals(
    workload: WorkloadSpec, n_steps: int, step_seconds: int, rng: np.random.Generator
) -> list[tuple[int, int, str, str, str]]:
    """``(start, stop, pod_name, owner_kind, owner_name)`` for every pod of a workload."""
    per_hour = 3600 / step_seconds
    pods: list[tuple[int, int, str, str, str]] = []
    if workload.kind == "Deployment":
        cuts = [0]
        if workload.rollout_days > 0:
            t = 0.0
            while True:
                t += rng.uniform(0.5, 1.5) * workload.rollout_days * 86400 / step_seconds
                if t >= n_steps:
                    break
                cuts.append(int(t))
        cuts.append(n_steps)
        for a, b in itertools.pairwise(cuts):
            rs = f"{workload.name}-{_suffix(rng, 10)}"
            for _ in range(workload.replicas):
                pods.append((a, b, f"{rs}-{_suffix(rng, 5)}", "ReplicaSet", rs))
    elif workload.kind == "StatefulSet":
        for i in range(workload.replicas):
            pods.append((0, n_steps, f"{workload.name}-{i}", "StatefulSet", workload.name))
    elif workload.kind == "CronJob":
        every = workload.cron_every_hours * per_hour
        duration = max(1, round(workload.cron_duration_hours * per_hour))
        t = workload.cron_offset_hours * per_hour
        while t < n_steps:
            a = int(t)
            b = min(n_steps, a + duration)
            minute = 28_000_000 + a * step_seconds // 60
            job = f"{workload.name}-{minute}"
            pods.append((a, b, f"{job}-{_suffix(rng, 5)}", "Job", job))
            t += every
    elif workload.kind != "DaemonSet":
        raise ConfigError(f"unsupported workload kind {workload.kind!r}")
    return pods


def _season(workload: WorkloadSpec, timestamps: npt.NDArray[np.int64]) -> FloatMatrix:
    """Diurnal shape in [0, 1] peaking at 14:00 UTC, and a weekday/weekend multiplier."""
    hour = (timestamps % 86400) / 3600.0
    diurnal = 0.5 * (1.0 - np.cos(2 * np.pi * (hour - 2.0) / 24.0))
    weekday = ((timestamps // 86400) + 3) % 7  # 1970-01-01 was a Thursday -> 0 = Monday
    weekend = np.where(weekday >= 5, workload.weekend_factor, 1.0)
    return np.stack([diurnal, weekend])


def _demand(
    c: ContainerSpec,
    workload: WorkloadSpec,
    timestamps: npt.NDArray[np.int64],
    origin: int,
    rng: np.random.Generator,
) -> tuple[FloatMatrix, FloatMatrix]:
    """True CPU (cores) and memory (bytes, before leak) demand over ``timestamps``."""
    diurnal, weekend = _season(workload, timestamps)
    weeks = (timestamps - origin) / (7 * 86400.0)
    growth = (1.0 + workload.weekly_growth) ** weeks
    noise = rng.lognormal(mean=0.0, sigma=c.cpu_noise, size=timestamps.size)
    cpu = c.cpu_base * (1.0 + c.cpu_diurnal * diurnal) * weekend * growth * noise
    if c.cpu_spike_prob > 0:
        spikes = rng.random(timestamps.size) < c.cpu_spike_prob
        cpu = np.where(spikes, cpu * c.cpu_spike_mult, cpu)
    mem_mib = (
        c.mem_base_mib
        + c.mem_diurnal_mib * diurnal * weekend
        + np.abs(rng.normal(0.0, c.mem_noise_mib, size=timestamps.size))
    ) * np.sqrt(growth)
    return cpu, mem_mib * MIB


def _apply_leak_and_oom(
    mem: FloatMatrix, c: ContainerSpec, step_hours: float
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    """Add the memory leak and record OOM kills against the limit.

    Returns ``(observed, oom, uncensored)``: ``observed`` is what Prometheus sees (capped at the
    limit, leak reset by each restart); ``uncensored`` is the memory the container would use if
    it were never killed (leak accumulated since pod start) — the counterfactual demand the
    rightsizing backtest scores policies against.
    """
    limit = c.mem_limit_mib * MIB if c.mem_limit_mib is not None else np.inf
    leak_per_step = c.mem_leak_per_hour_mib * MIB * step_hours
    uncensored = mem + leak_per_step * np.arange(mem.size)
    if leak_per_step == 0:
        hit = mem >= limit
        return np.where(hit, limit, mem), hit.astype(np.float64), uncensored
    observed = mem.copy()
    oom = np.zeros_like(mem)
    since_restart = 0
    for k in range(mem.size):
        value = mem[k] + leak_per_step * since_restart
        if value >= limit:
            observed[k] = limit
            oom[k] = 1.0
            since_restart = 0
        else:
            observed[k] = value
            since_restart += 1
    return observed, oom, uncensored


def simulate(
    spec: ClusterSpec,
    *,
    start: int,
    weeks: float,
    step_seconds: int,
    seed: int,
) -> SimulatedCluster:
    """Simulate ``weeks`` weeks of the cluster at ``step_seconds`` resolution."""
    if step_seconds <= 0 or 86400 % step_seconds:
        raise ConfigError("step_seconds must divide a day evenly")
    n_steps = round(weeks * 7 * 86400 / step_seconds)
    if n_steps <= 0:
        raise ConfigError("the simulated period must contain at least one step")
    rng = np.random.default_rng(seed)
    nodes = _build_nodes(spec, n_steps, step_seconds)
    scheduler = _Scheduler(nodes, n_steps, spec.schedulable_fraction)
    timestamps = start + step_seconds * np.arange(n_steps, dtype=np.int64)
    step_hours = step_seconds / 3600.0

    containers: list[SimContainer] = []
    columns: list[tuple[FloatMatrix, ...]] = []

    def add_pod(
        workload: WorkloadSpec, a: int, b: int, pod: str, kind: str, owner: str, node: int | None
    ) -> None:
        cpu_req = sum(c.cpu_request for c in workload.containers)
        mem_req = sum(c.mem_request_mib for c in workload.containers) * MIB
        placed = scheduler.place(workload.pool, a, b, cpu_req, mem_req, node=node)
        ns = spec.namespace(workload.namespace)
        team = workload.team_label_on_pod or ns.team
        labels = {"app": workload.app or workload.name}
        if workload.team_label_on_pod:
            labels["team"] = workload.team_label_on_pod
        if ns.cost_center:
            labels["cost_center"] = ns.cost_center
        for c in workload.containers:
            ts = timestamps[a:b]
            cpu, mem = _demand(c, workload, ts, start, rng)
            mem, oom, mem_demand = _apply_leak_and_oom(mem, c, step_hours)
            usage = np.minimum(cpu, c.cpu_limit) if c.cpu_limit is not None else cpu
            columns.append(
                (
                    _place(np.round(cpu, 6), a, b, n_steps),
                    _place(np.round(usage, 6), a, b, n_steps),
                    _place(np.round(mem), a, b, n_steps),
                    _place(oom, a, b, n_steps),
                    _place(np.round(mem_demand), a, b, n_steps),
                ),
            )
            containers.append(
                SimContainer(
                    namespace=workload.namespace,
                    pod=pod,
                    container=c.name,
                    workload=workload,
                    spec=c,
                    owner_kind=kind,
                    owner_name=owner,
                    node=placed,
                    start=a,
                    stop=b,
                    team=team,
                    pod_labels=labels,
                ),
            )

    # DaemonSets first (one pod per node for the node's lifetime), then the rest by size.
    for workload in spec.workloads:
        if workload.kind != "DaemonSet":
            continue
        for i, node in enumerate(nodes):
            if node.alive_from >= node.alive_to:
                continue
            pod = f"{workload.name}-{_suffix(rng, 5)}"
            add_pod(workload, node.alive_from, node.alive_to, pod, "DaemonSet", workload.name, i)
    others = [w for w in spec.workloads if w.kind != "DaemonSet"]
    others.sort(key=lambda w: -sum(c.cpu_request for c in w.containers))
    for workload in others:
        for a, b, pod, kind, owner in _pod_intervals(workload, n_steps, step_seconds, rng):
            add_pod(workload, a, b, pod, kind, owner, None)

    def stack(i: int) -> FloatMatrix:
        if not columns:
            return np.zeros((n_steps, 0))
        return np.stack([col[i] for col in columns], axis=1)

    log.info("simulated %d nodes, %d containers, %d steps", len(nodes), len(containers), n_steps)
    return SimulatedCluster(
        spec=spec,
        start=start,
        step_seconds=step_seconds,
        n_steps=n_steps,
        seed=seed,
        nodes=nodes,
        containers=containers,
        cpu_demand=stack(0),
        cpu_usage=stack(1),
        memory_usage=stack(2),
        oom_events=stack(3),
        memory_demand=stack(4),
    )


def _place(values: FloatMatrix, a: int, b: int, n_steps: int) -> FloatMatrix:
    out = np.full(n_steps, np.nan)
    out[a:b] = values
    return out
