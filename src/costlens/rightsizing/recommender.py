"""VPA-style request recommendations from historical usage.

For every *workload container* (all pods of one Deployment/StatefulSet/CronJob/DaemonSet,
grouped by container name, as a VerticalPodAutoscaler target would be) the recommender keeps:

* a CPU histogram of per-step usage, each sample weighted by the container's CPU request
  (``max(request, 0.1)``, as the VPA does, so busy-but-small replicas do not dominate), and
* a memory histogram of per-interval *peaks* (one sample per container per 24h by default),
  where an OOM kill inside an interval raises that interval's peak to
  ``max(1.2 * M, M + 100MiB)`` with ``M`` the memory at the kill (the limit it hit).

Targets are histogram percentiles plus a safety margin, floored at per-container minimums.
Lower/upper bounds use other percentiles and widen with low confidence (short history), using
the VPA's ``(1 + 1/confidence)`` upper and ``(1 + 0.001/confidence)**-2`` lower multipliers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from costlens.dataset import ClusterDataset
from costlens.errors import ConfigError
from costlens.rightsizing.histogram import CPU_OPTIONS, MEMORY_OPTIONS, DecayingHistogram
from costlens.units import MIB

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
_MIN_CPU_SAMPLE_WEIGHT = 0.1


@dataclass(frozen=True)
class RecommenderConfig:
    """Tunable recommender parameters; the defaults are the VPA recommender's defaults."""

    name: str = "p90+15%"
    cpu_percentile: float = 0.90
    memory_percentile: float = 0.90
    lower_percentile: float = 0.50
    upper_percentile: float = 0.95
    safety_margin: float = 0.15
    half_life_hours: float = 24.0
    memory_peak_interval_hours: float = 24.0
    oom_aware: bool = True
    oom_bump_ratio: float = 1.2
    oom_min_bump_mib: float = 100.0
    weight_cpu_by_request: bool = True
    pod_min_cpu_cores: float = 0.025
    pod_min_memory_mib: float = 250.0

    def __post_init__(self) -> None:
        for name in ("cpu_percentile", "memory_percentile", "lower_percentile", "upper_percentile"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ConfigError(f"{name} must be in (0, 1], got {value}")
        if self.safety_margin < 0:
            raise ConfigError("safety_margin must be non-negative")
        if self.half_life_hours <= 0 or self.memory_peak_interval_hours <= 0:
            raise ConfigError("half-life and peak interval must be positive")
        if self.oom_bump_ratio < 1.0 or self.oom_min_bump_mib < 0:
            raise ConfigError("OOM bump must not shrink memory")


@dataclass(frozen=True)
class Recommendation:
    """Recommended requests for one workload container (CPU in cores, memory in bytes)."""

    workload: str
    container: str
    target_cpu: float
    target_memory: float
    lower_cpu: float
    lower_memory: float
    upper_cpu: float
    upper_memory: float
    current_cpu_request: float
    current_memory_request: float
    current_cpu_limit: float  # NaN = no limit
    current_memory_limit: float  # NaN = no limit
    samples: int
    oom_events: int
    confidence_days: float

    @property
    def key(self) -> tuple[str, str]:
        """``(workload, container)``."""
        return (self.workload, self.container)


def workload_groups(dataset: ClusterDataset) -> dict[tuple[str, str], list[int]]:
    """Map ``(namespace/workload, container name)`` to dataset column indices."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, c in enumerate(dataset.containers):
        groups[(f"{c.namespace}/{c.workload}", c.container)].append(i)
    return dict(groups)


def _containers_per_pod(dataset: ClusterDataset, columns: list[int]) -> int:
    pods: dict[str, set[str]] = defaultdict(set)
    for i in columns:
        c = dataset.containers[i]
        pods[c.pod].add(c.container)
    return max(len(v) for v in pods.values())


def _latest(values: FloatArray, present: npt.NDArray[np.bool_]) -> float:
    """Most recent present value across columns (NaN if none)."""
    rows = np.flatnonzero(present.any(axis=1))
    if rows.size == 0:
        return float("nan")
    last = rows[-1]
    vals = values[last][present[last]]
    return float(vals[0]) if vals.size else float("nan")


def _memory_peaks(
    dataset: ClusterDataset, col: int, config: RecommenderConfig
) -> tuple[FloatArray, FloatArray]:
    """Per-interval memory peaks (with OOM bumps) and their interval-end timestamps."""
    interval = max(1, round(config.memory_peak_interval_hours * 3600 / dataset.step_seconds))
    present = dataset.present[:, col]
    usage = np.where(present, dataset.memory_usage[:, col], np.nan)
    if config.oom_aware:
        ooms = (dataset.oom_events[:, col] > 0) & present
        if np.any(ooms):
            limit = np.nan_to_num(dataset.memory_limit[:, col], nan=0.0)
            at_kill = np.fmax(usage, limit)
            bumped = np.maximum(
                at_kill * config.oom_bump_ratio, at_kill + config.oom_min_bump_mib * MIB
            )
            usage = np.where(ooms, np.fmax(usage, bumped), usage)
    starts = np.arange(0, dataset.n_steps, interval)
    with np.errstate(invalid="ignore"):
        peaks = np.fmax.reduceat(np.nan_to_num(usage, nan=-1.0), starts)
    ends = np.minimum(starts + interval, dataset.n_steps) - 1
    keep = peaks >= 0
    timestamps = dataset.start + ends.astype(np.float64) * dataset.step_seconds
    return peaks[keep], timestamps[keep]


def recommend(
    dataset: ClusterDataset, config: RecommenderConfig | None = None
) -> list[Recommendation]:
    """Compute a recommendation for every workload container seen in ``dataset``."""
    config = config or RecommenderConfig()
    half_life = config.half_life_hours * 3600.0
    margin = 1.0 + config.safety_margin

    def bound(hist: DecayingHistogram, p: float, floor: float, mult: float = 1.0) -> float:
        return max(hist.percentile(p) * margin * mult, floor)

    ts = dataset.timestamps.astype(np.float64)
    out: list[Recommendation] = []
    for (workload, container), columns in sorted(workload_groups(dataset).items()):
        cols = np.array(columns)
        present = dataset.present[:, cols]
        if not np.any(present):
            continue
        cpu_hist = DecayingHistogram(CPU_OPTIONS, half_life)
        mem_hist = DecayingHistogram(MEMORY_OPTIONS, half_life)
        for col in columns:
            mask = dataset.present[:, col]
            weights = (
                np.maximum(dataset.cpu_request[mask, col], _MIN_CPU_SAMPLE_WEIGHT)
                if config.weight_cpu_by_request
                else None
            )
            cpu_hist.add_samples(dataset.cpu_usage[mask, col], ts[mask], weights)
            peaks, peak_ts = _memory_peaks(dataset, col, config)
            mem_hist.add_samples(peaks, peak_ts)

        per_pod = _containers_per_pod(dataset, columns)
        min_cpu = config.pod_min_cpu_cores / per_pod
        min_mem = config.pod_min_memory_mib * MIB / per_pod
        rows_present = np.flatnonzero(present.any(axis=1))
        lifespan_days = (rows_present[-1] - rows_present[0] + 1) * dataset.step_seconds / 86400
        sample_days = int(present.sum()) * dataset.step_seconds / 86400
        confidence = max(min(lifespan_days, sample_days), 1e-6)
        upper_mult = 1.0 + 1.0 / confidence
        lower_mult = (1.0 + 0.001 / confidence) ** -2

        oom_count = int((dataset.oom_events[:, cols] * present).sum())
        out.append(
            Recommendation(
                workload=workload,
                container=container,
                target_cpu=bound(cpu_hist, config.cpu_percentile, min_cpu),
                target_memory=bound(mem_hist, config.memory_percentile, min_mem),
                lower_cpu=bound(cpu_hist, config.lower_percentile, min_cpu, lower_mult),
                lower_memory=bound(mem_hist, config.lower_percentile, min_mem, lower_mult),
                upper_cpu=bound(cpu_hist, config.upper_percentile, min_cpu, upper_mult),
                upper_memory=bound(mem_hist, config.upper_percentile, min_mem, upper_mult),
                current_cpu_request=_latest(dataset.cpu_request[:, cols], present),
                current_memory_request=_latest(dataset.memory_request[:, cols], present),
                current_cpu_limit=_latest(dataset.cpu_limit[:, cols], present),
                current_memory_limit=_latest(dataset.memory_limit[:, cols], present),
                samples=int(present.sum()),
                oom_events=oom_count,
                confidence_days=confidence,
            ),
        )
    log.info("computed %d recommendations with %s", len(out), config.name)
    return out
