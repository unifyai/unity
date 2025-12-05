"""
Shared types for repairs queries.

Defines common enums and type aliases used across all metric queries.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class GroupBy(str, Enum):
    """Breakdown/grouping dimension for metrics."""

    OPERATIVE = "operative"
    TRADE = "trade"
    PATCH = "patch"
    REGION = "region"
    TIME_PERIOD = "time_period"
    TOTAL = "total"  # No grouping, aggregate total


class TimePeriod(str, Enum):
    """Time period granularity for aggregations."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class MetricResult(TypedDict, total=False):
    """Standard result shape for metric queries."""

    metric_name: str
    group_by: str
    time_period: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    results: List[Dict[str, Any]]
    total: Optional[float]
    metadata: Optional[Dict[str, Any]]


# Type alias for tools dict passed to queries
ToolsDict = Dict[str, Any]
