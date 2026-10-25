"""Declarative description of a synthetic multi-team cluster.

The default spec (:func:`default_spec`) models a small e-commerce company: four product teams,
a data team, an unlabelled sandbox namespace, and platform namespaces (``kube-system``,
``monitoring``) that are natural candidates for shared-cost treatment. Workloads are chosen to
exercise every code path: over- and under-provisioned requests, diurnal and weekly seasonality,
spiky CPU, sidecars, CronJobs that come and go, DaemonSets on every node, a memory leak that
OOM-kills against its limit, spot nodes and a node added part-way through the period.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NodePoolSpec:
    """A group of identical nodes."""

    name: str
    instance_type: str
    capacity_type: str
    cores: float
    memory_gib: float
    hourly_usd: float
    count: int
    zones: tuple[str, ...] = ("us-east-1a", "us-east-1b")
    late_nodes: int = 0  # extra nodes that join after ``late_join_day``
    late_join_day: float = 10.0


@dataclass(frozen=True)
class ContainerSpec:
    """Demand model and resource settings for one container of a workload.

    CPU demand (cores) at time ``t`` is
    ``cpu_base * season(t) * growth(t) * lognormal(cpu_noise)`` plus occasional spikes of
    ``cpu_spike_mult``; memory demand (MiB) is ``mem_base + mem_diurnal * diurnal(t)`` plus noise
    plus ``mem_leak_per_hour`` accumulated since the container (re)started.
    """

    name: str
    cpu_base: float
    mem_base_mib: float
    cpu_request: float
    mem_request_mib: float
    cpu_diurnal: float = 0.0
    cpu_noise: float = 0.15
    cpu_spike_prob: float = 0.0
    cpu_spike_mult: float = 3.0
    mem_diurnal_mib: float = 0.0
    mem_noise_mib: float = 8.0
    mem_leak_per_hour_mib: float = 0.0
    cpu_limit: float | None = None
    mem_limit_mib: float | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    """A controller (Deployment, StatefulSet, CronJob or DaemonSet) and its pods."""

    name: str
    namespace: str
    kind: str
    containers: tuple[ContainerSpec, ...]
    pool: str = "general"
    replicas: int = 1
    team_label_on_pod: str | None = None  # overrides the namespace team label
    rollout_days: float = 0.0  # Deployments: mean days between rollouts (0 = never)
    cron_every_hours: float = 24.0
    cron_duration_hours: float = 1.0
    cron_offset_hours: float = 0.0
    weekend_factor: float = 1.0
    weekly_growth: float = 0.0
    app: str = ""


@dataclass(frozen=True)
class NamespaceSpec:
    """A namespace and the labels kube-state-metrics would export for it."""

    name: str
    team: str | None
    cost_center: str | None = None


@dataclass(frozen=True)
class ClusterSpec:
    """Everything the generator needs to simulate a cluster."""

    name: str
    pools: tuple[NodePoolSpec, ...]
    namespaces: tuple[NamespaceSpec, ...]
    workloads: tuple[WorkloadSpec, ...]
    cpu_to_memory_cost_ratio: float = 7.5
    schedulable_fraction: float = 0.92
    extra: dict[str, str] = field(default_factory=dict)

    def namespace(self, name: str) -> NamespaceSpec:
        """Look up a namespace by name."""
        for ns in self.namespaces:
            if ns.name == name:
                return ns
        raise KeyError(name)


def _c(name: str, **kwargs: float | None) -> ContainerSpec:
    return ContainerSpec(name=name, **kwargs)  # type: ignore[arg-type]


def default_spec() -> ClusterSpec:
    """The demo cluster used by ``costlens generate``, the bundled fixture and the tests."""
    pools = (
        NodePoolSpec("general", "m5.2xlarge", "on-demand", 8, 32, 0.384, 3),
        NodePoolSpec("memory", "r5.xlarge", "on-demand", 4, 32, 0.252, 2),
        NodePoolSpec(
            "batch",
            "c5.2xlarge",
            "spot",
            8,
            16,
            0.34,
            1,
            ("us-east-1c",),
            late_nodes=1,
            late_join_day=9.5,
        ),
    )
    namespaces = (
        NamespaceSpec("payments", "payments", "cc-100"),
        NamespaceSpec("search", "search", "cc-200"),
        NamespaceSpec("storefront-web", "storefront", "cc-300"),
        NamespaceSpec("storefront-api", "storefront", "cc-300"),
        NamespaceSpec("data-pipelines", "data", "cc-400"),
        NamespaceSpec("ml-training", "data", "cc-400"),
        NamespaceSpec("sandbox", None),
        NamespaceSpec("kube-system", "platform", "cc-900"),
        NamespaceSpec("monitoring", "platform", "cc-900"),
    )
    envoy = _c(
        "envoy",
        cpu_base=0.05,
        mem_base_mib=60,
        cpu_request=0.1,
        mem_request_mib=128,
        cpu_diurnal=0.8,
        mem_noise_mib=3,
    )
    workloads = (
        # payments: an over-provisioned API, a memory-heavy ledger and a spiky scorer
        WorkloadSpec(
            "payments-api",
            "payments",
            "Deployment",
            (
                _c(
                    "api",
                    cpu_base=0.18,
                    mem_base_mib=380,
                    cpu_request=1.0,
                    mem_request_mib=1024,
                    cpu_diurnal=1.2,
                    mem_diurnal_mib=60,
                    cpu_limit=2.0,
                    mem_limit_mib=2048,
                ),
            ),
            replicas=3,
            rollout_days=6,
            weekend_factor=0.6,
            weekly_growth=0.02,
            app="payments-api",
        ),
        WorkloadSpec(
            "ledger",
            "payments",
            "StatefulSet",
            (
                _c(
                    "postgres",
                    cpu_base=0.35,
                    mem_base_mib=5200,
                    cpu_request=1.0,
                    mem_request_mib=8192,
                    cpu_diurnal=0.6,
                    mem_diurnal_mib=600,
                    mem_noise_mib=40,
                ),
            ),
            pool="memory",
            replicas=2,
            app="ledger",
        ),
        WorkloadSpec(
            "fraud-scorer",
            "payments",
            "Deployment",
            (
                _c(
                    "scorer",
                    cpu_base=0.25,
                    mem_base_mib=700,
                    cpu_request=0.5,
                    mem_request_mib=1536,
                    cpu_diurnal=0.5,
                    cpu_spike_prob=0.04,
                    cpu_spike_mult=4.0,
                    cpu_limit=1.5,
                    mem_limit_mib=2048,
                ),
            ),
            replicas=2,
            rollout_days=9,
            app="fraud-scorer",
        ),
        # search: sidecar-carrying query pods, a batch indexer on spot, a cache
        WorkloadSpec(
            "search-query",
            "search",
            "Deployment",
            (
                _c(
                    "query",
                    cpu_base=0.3,
                    mem_base_mib=900,
                    cpu_request=0.75,
                    mem_request_mib=2048,
                    cpu_diurnal=1.0,
                    mem_diurnal_mib=150,
                    cpu_limit=1.5,
                    mem_limit_mib=3072,
                ),
                envoy,
            ),
            replicas=3,
            rollout_days=4,
            weekend_factor=0.7,
            weekly_growth=0.03,
            app="search-query",
        ),
        WorkloadSpec(
            "search-indexer",
            "search",
            "CronJob",
            (
                _c(
                    "indexer",
                    cpu_base=2.2,
                    mem_base_mib=2500,
                    cpu_request=3.0,
                    mem_request_mib=4096,
                    cpu_noise=0.1,
                    mem_noise_mib=50,
                    cpu_limit=4.0,
                    mem_limit_mib=6144,
                ),
            ),
            pool="batch",
            cron_every_hours=6,
            cron_duration_hours=1.5,
            app="search-indexer",
        ),
        WorkloadSpec(
            "search-cache",
            "search",
            "StatefulSet",
            (
                _c(
                    "redis",
                    cpu_base=0.08,
                    mem_base_mib=3000,
                    cpu_request=0.25,
                    mem_request_mib=6144,
                    mem_diurnal_mib=400,
                    mem_noise_mib=20,
                ),
            ),
            pool="memory",
            replicas=2,
            app="search-cache",
        ),
        # storefront: diurnal web, an API with a team override label, a leaky resizer
        WorkloadSpec(
            "web",
            "storefront-web",
            "Deployment",
            (
                _c(
                    "nginx",
                    cpu_base=0.12,
                    mem_base_mib=150,
                    cpu_request=0.5,
                    mem_request_mib=512,
                    cpu_diurnal=2.0,
                    mem_diurnal_mib=30,
                    cpu_limit=1.0,
                    mem_limit_mib=512,
                ),
            ),
            replicas=3,
            rollout_days=3,
            weekend_factor=0.5,
            weekly_growth=0.01,
            app="web",
        ),
        WorkloadSpec(
            "storefront-api",
            "storefront-api",
            "Deployment",
            (
                _c(
                    "api",
                    cpu_base=0.4,
                    mem_base_mib=600,
                    cpu_request=0.4,
                    mem_request_mib=768,
                    cpu_diurnal=1.0,
                    mem_diurnal_mib=120,
                    cpu_limit=1.0,
                    mem_limit_mib=1536,
                ),
                envoy,
            ),
            replicas=2,
            rollout_days=5,
            weekend_factor=0.6,
            app="storefront-api",
        ),
        WorkloadSpec(
            "image-resizer",
            "storefront-api",
            "Deployment",
            (
                _c(
                    "resizer",
                    cpu_base=0.15,
                    mem_base_mib=300,
                    cpu_request=0.25,
                    mem_request_mib=512,
                    cpu_diurnal=1.0,
                    mem_leak_per_hour_mib=9.0,
                    mem_limit_mib=640,
                    mem_noise_mib=10,
                ),
            ),
            replicas=2,
            team_label_on_pod="media",
            rollout_days=12,
            app="image-resizer",
        ),
        # data: nightly ETL and a training job on spot, an always-on feature store
        WorkloadSpec(
            "nightly-etl",
            "data-pipelines",
            "CronJob",
            (
                _c(
                    "etl",
                    cpu_base=1.5,
                    mem_base_mib=3000,
                    cpu_request=2.0,
                    mem_request_mib=6144,
                    cpu_noise=0.2,
                    mem_noise_mib=100,
                    mem_limit_mib=8192,
                ),
            ),
            pool="batch",
            cron_every_hours=24,
            cron_duration_hours=3,
            cron_offset_hours=1,
            app="nightly-etl",
        ),
        WorkloadSpec(
            "trainer",
            "ml-training",
            "CronJob",
            (
                _c(
                    "trainer",
                    cpu_base=3.0,
                    mem_base_mib=5000,
                    cpu_request=4.0,
                    mem_request_mib=8192,
                    cpu_noise=0.08,
                    mem_noise_mib=80,
                ),
            ),
            pool="batch",
            cron_every_hours=12,
            cron_duration_hours=4,
            cron_offset_hours=7,
            app="trainer",
        ),
        WorkloadSpec(
            "feature-store",
            "data-pipelines",
            "Deployment",
            (
                _c(
                    "server",
                    cpu_base=0.1,
                    mem_base_mib=1200,
                    cpu_request=0.5,
                    mem_request_mib=2048,
                    cpu_diurnal=0.4,
                    mem_noise_mib=15,
                ),
            ),
            replicas=2,
            rollout_days=14,
            app="feature-store",
        ),
        # sandbox: an unlabelled namespace (shows up as __unlabelled__ when grouping by team)
        WorkloadSpec(
            "experiments",
            "sandbox",
            "Deployment",
            (
                _c(
                    "notebook",
                    cpu_base=0.05,
                    mem_base_mib=800,
                    cpu_request=1.0,
                    mem_request_mib=2048,
                    cpu_spike_prob=0.02,
                    cpu_spike_mult=10.0,
                ),
            ),
            replicas=1,
            app="experiments",
        ),
        # platform: shared namespaces
        WorkloadSpec(
            "coredns",
            "kube-system",
            "Deployment",
            (
                _c(
                    "coredns",
                    cpu_base=0.03,
                    mem_base_mib=40,
                    cpu_request=0.1,
                    mem_request_mib=70,
                    cpu_diurnal=1.0,
                    mem_noise_mib=2,
                    mem_limit_mib=170,
                ),
            ),
            replicas=2,
            app="coredns",
        ),
        WorkloadSpec(
            "kube-proxy",
            "kube-system",
            "DaemonSet",
            (
                _c(
                    "kube-proxy",
                    cpu_base=0.02,
                    mem_base_mib=30,
                    cpu_request=0.1,
                    mem_request_mib=64,
                    mem_noise_mib=1,
                ),
            ),
            app="kube-proxy",
        ),
        WorkloadSpec(
            "node-exporter",
            "monitoring",
            "DaemonSet",
            (
                _c(
                    "node-exporter",
                    cpu_base=0.01,
                    mem_base_mib=25,
                    cpu_request=0.05,
                    mem_request_mib=64,
                    mem_noise_mib=1,
                    mem_limit_mib=128,
                ),
            ),
            app="node-exporter",
        ),
        WorkloadSpec(
            "prometheus",
            "monitoring",
            "StatefulSet",
            (
                _c(
                    "prometheus",
                    cpu_base=0.4,
                    mem_base_mib=3500,
                    cpu_request=1.0,
                    mem_request_mib=6144,
                    mem_diurnal_mib=500,
                    mem_noise_mib=60,
                ),
            ),
            pool="memory",
            replicas=1,
            app="prometheus",
        ),
    )
    return ClusterSpec(name="demo-shop", pools=pools, namespaces=namespaces, workloads=workloads)
