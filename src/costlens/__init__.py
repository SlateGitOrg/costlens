"""costlens: Kubernetes cost allocation and VPA-style rightsizing from Prometheus exports.

Typical use::

    from pathlib import Path
    from costlens import AllocationConfig, allocate, load_catalog, load_dataset, price_nodes

    dataset = load_dataset(Path("export"))
    costs = price_nodes(dataset, load_catalog(Path("export/pricing.toml")))
    result = allocate(dataset, costs, AllocationConfig(group_by="team"))
    result.check_invariants()  # allocated + idle + shared == total, exactly
"""

from costlens.allocation import AllocationResult, allocate
from costlens.config import (
    AllocationConfig,
    CostBasis,
    ExternalCost,
    IdleStrategy,
    SharedStrategy,
    load_config,
)
from costlens.dataset import ClusterDataset, load_dataset
from costlens.errors import CostlensError
from costlens.pricing import NodeCosts, PricingCatalog, load_catalog, price_nodes
from costlens.rightsizing.backtest import run_backtest
from costlens.rightsizing.recommender import Recommendation, RecommenderConfig, recommend

__version__ = "0.1.0"

__all__ = [
    "AllocationConfig",
    "AllocationResult",
    "ClusterDataset",
    "CostBasis",
    "CostlensError",
    "ExternalCost",
    "IdleStrategy",
    "NodeCosts",
    "PricingCatalog",
    "Recommendation",
    "RecommenderConfig",
    "SharedStrategy",
    "__version__",
    "allocate",
    "load_catalog",
    "load_config",
    "load_dataset",
    "price_nodes",
    "recommend",
    "run_backtest",
]
