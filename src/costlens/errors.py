"""Exception hierarchy for costlens.

Every error raised deliberately by the library derives from :class:`CostlensError`, so
callers (and the CLI) can catch one type and report a clean message instead of a traceback.
"""

from __future__ import annotations


class CostlensError(Exception):
    """Base class for all costlens errors."""


class ExportFormatError(CostlensError):
    """A Prometheus export file is malformed or not a range-query (matrix) response."""


class ManifestError(CostlensError):
    """The export manifest is missing, malformed or inconsistent with the export files."""


class DataQualityError(CostlensError):
    """The ingested series are internally inconsistent (e.g. a pod on an unknown node)."""


class PricingError(CostlensError):
    """A node cannot be priced, or the pricing catalog is malformed."""


class ConfigError(CostlensError):
    """An allocation / rightsizing configuration value is invalid."""


class UnitParseError(CostlensError):
    """A Kubernetes quantity or duration string could not be parsed."""
