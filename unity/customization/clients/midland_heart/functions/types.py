"""Shared types for Midland Heart repairs metrics.

Provides strictly-typed Pydantic models and enums used across all metric
queries.  Using Pydantic ensures validation at runtime and provides clear
contracts for data shapes.

Types Defined
-------------
- GroupBy: Dimension for breaking down metrics
- TimePeriod: Time granularity for aggregations
- MetricResult: Standard result shape for all metrics
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# ENUMERATIONS
# =============================================================================


class GroupBy(str, Enum):
    """Breakdown/grouping dimension for metrics.

    Determines how metric results are segmented.  Each value corresponds
    to a column in the repairs/telematics data that can be used for grouping.

    When ``None`` is passed instead of a GroupBy value, metrics aggregate
    across all records without grouping (no plots generated).
    """

    OPERATIVE = "operative"
    TRADE = "trade"
    PATCH = "patch"
    REGION = "region"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class TimePeriod(str, Enum):
    """Time period granularity for aggregations.

    Used when grouping metrics by time to specify the bucket size.
    """

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class MetricResult(BaseModel):
    """Standard result shape for all metric queries.

    This is the canonical output format for every metric function in the
    repairs module.  Using a consistent shape enables uniform handling in
    CLI scripts, logging, and downstream consumers.

    Attributes
    ----------
    metric_name : str
        The name of the metric (e.g., ``"first_time_fix_rate"``).
    group_by : str or None
        Grouping dimension used (e.g., ``"operative"``, ``"patch"``).
    time_period : str or None
        Time period granularity if time-based grouping was used.
    start_date : str or None
        Start date of the analysis period (ISO format).
    end_date : str or None
        End date of the analysis period (ISO format).
    results : list[dict]
        List of result rows, each a dict with group key and values.
    total : float or None
        Aggregate total across all groups.
    metadata : dict or None
        Additional metadata about the calculation.
    plots : list[dict]
        Generated plot results (empty if ``include_plots=False``).
    """

    metric_name: str
    group_by: Optional[str] = None
    time_period: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    plots: List[Dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "GroupBy",
    "TimePeriod",
    "MetricResult",
]
