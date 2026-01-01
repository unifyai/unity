"""
Shared types for repairs queries.

This module provides strictly-typed Pydantic models and enums used across
all metric queries in the repairs module. Using Pydantic ensures validation
at runtime and provides clear contracts for data shapes.

Types Defined:
    - GroupBy: Dimension for breaking down metrics
    - TimePeriod: Time granularity for aggregations
    - PlotType: Supported visualization chart types (re-exported from viz_utils)
    - PlotConfig: Configuration for a single plot (re-exported from viz_utils)
    - PlotResult: Result of a plot generation attempt (re-exported from viz_utils)
    - MetricResult: Standard result shape for all metrics
    - ToolsDict: Type alias for file manager tools dictionary
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Re-export plot types from viz_utils (canonical source)
from unity.file_manager.managers.utils.viz_utils import (
    PlotConfig,
    PlotResult,
    PlotType,
)

# Type alias for the visualize tool function signature
VisualizeTool = Callable[..., Union[PlotResult, List[PlotResult]]]


# =============================================================================
# ENUMERATIONS
# =============================================================================


class GroupBy(str, Enum):
    """
    Breakdown/grouping dimension for metrics.

    Determines how metric results are segmented. Each value corresponds
    to a column in the repairs/telematics data that can be used for grouping.

    Values:
        OPERATIVE: Group by individual operative (worker)
        TRADE: Group by trade/skill type (e.g., plumber, electrician)
        PATCH: Group by geographic patch/area
        REGION: Group by broader geographic region
        TIME_PERIOD: Group by time bucket (day, week, month, etc.)
        TOTAL: No grouping - aggregate total across all records
    """

    OPERATIVE = "operative"
    TRADE = "trade"
    PATCH = "patch"
    REGION = "region"
    TIME_PERIOD = "time_period"
    TOTAL = "total"


class TimePeriod(str, Enum):
    """
    Time period granularity for aggregations.

    Used when grouping metrics by time to specify the bucket size.

    Values:
        DAY: Daily aggregation
        WEEK: Weekly aggregation
        MONTH: Monthly aggregation
        QUARTER: Quarterly aggregation
        YEAR: Yearly aggregation
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
    """
    Standard result shape for all metric queries.

    This is the canonical output format for every metric function in the
    repairs module. Using a consistent shape enables uniform handling in
    CLI scripts, logging, and downstream consumers.

    Attributes:
        metric_name: The name of the metric (e.g., "first_time_fix_rate")
        group_by: The grouping dimension used (e.g., "operative", "total")
        time_period: Time period granularity if time-based grouping was used
        start_date: Start date of the analysis period (ISO format)
        end_date: End date of the analysis period (ISO format)
        results: List of result rows, each a dict with group key and values
        total: Aggregate total across all groups (if applicable)
        metadata: Optional additional metadata about the calculation
        plots: List of generated plot results (empty if include_plots=False)

    Example:
        >>> result = MetricResult(
        ...     metric_name="first_time_fix_rate",
        ...     group_by="operative",
        ...     results=[
        ...         {"group": "John Smith", "rate": 0.85, "count": 120},
        ...         {"group": "Jane Doe", "rate": 0.92, "count": 95},
        ...     ],
        ...     total=0.88,
        ...     plots=[]
        ... )
    """

    metric_name: str
    group_by: str
    time_period: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    plots: List[PlotResult] = Field(default_factory=list)

    class Config:
        """Pydantic configuration for MetricResult."""

        # Allow arbitrary types for flexibility in results dicts
        arbitrary_types_allowed = True


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Type alias for the tools dictionary passed to metric functions
# Contains file manager tools like _reduce, _filter_files, _list_columns, _visualize
ToolsDict = Dict[str, Any]


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "GroupBy",
    "TimePeriod",
    # Plot types (re-exported from viz_utils)
    "PlotType",
    "PlotConfig",
    "PlotResult",
    # Models
    "MetricResult",
    # Type aliases
    "ToolsDict",
    "VisualizeTool",
]
