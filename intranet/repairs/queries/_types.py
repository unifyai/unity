"""
Shared types for repairs queries.

This module provides strictly-typed Pydantic models and enums used across
all metric queries in the repairs module. Using Pydantic ensures validation
at runtime and provides clear contracts for data shapes.

Types Defined:
    - GroupBy: Dimension for breaking down metrics
    - TimePeriod: Time granularity for aggregations
    - PlotType: Supported visualization chart types
    - PlotConfig: Configuration for a single plot
    - PlotResult: Result of a plot generation attempt
    - MetricResult: Standard result shape for all metrics
    - ToolsDict: Type alias for file manager tools dictionary
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class PlotType(str, Enum):
    """
    Supported plot types from the Plot API.

    These map directly to the 'type' field in the Plot API's plot_config.

    Values:
        SCATTER: Scatter plot for correlations between two numeric variables
        BAR: Bar chart for comparing values across categories
        HISTOGRAM: Histogram for distribution of a single variable
        LINE: Line chart for trends over time/sequences
    """

    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"
    LINE = "line"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class PlotConfig(BaseModel):
    """
    Configuration for a single plot visualization.

    This model defines the parameters needed to generate a plot via the
    Plot API. It maps to the 'plot_config' object in the API request body.

    Attributes:
        type: The chart type to generate (bar, line, histogram, scatter)
        x_axis: Column name for the x-axis
        y_axis: Column name for the y-axis (optional for histograms)
        group_by: Optional column for grouping/coloring data points
        aggregate: Aggregation function (sum, mean, count, min, max)
        scale_x: Scale type for x-axis (linear or log)
        scale_y: Scale type for y-axis (linear or log)
        metric: Metric for aggregation (alias for aggregate in some contexts)
        bin_count: Number of bins for histogram plots
        show_regression: Whether to show regression line (scatter plots)
        title: Human-readable title for the plot

    Example:
        >>> config = PlotConfig(
        ...     type=PlotType.BAR,
        ...     x_axis="Full Name",
        ...     y_axis="count",
        ...     aggregate="sum",
        ...     title="Jobs Completed per Operative"
        ... )
    """

    type: PlotType
    x_axis: str
    y_axis: Optional[str] = None
    group_by: Optional[str] = None
    aggregate: Optional[str] = None
    scale_x: Optional[str] = None
    scale_y: Optional[str] = None
    metric: Optional[str] = None
    bin_count: Optional[int] = None
    show_regression: Optional[bool] = None
    title: Optional[str] = None

    class Config:
        """Pydantic configuration for PlotConfig."""

        use_enum_values = True  # Serialize enums as their string values


class PlotResult(BaseModel):
    """
    Result of a plot generation attempt.

    Contains either a successful plot URL or error information if generation
    failed. The plot_config and project_config fields preserve the exact
    parameters used for the API call, enabling debugging and retry logic.

    Attributes:
        url: The generated plot URL (None if generation failed)
        token: Access token for the plot (from API response)
        expires_in_hours: Hours until the plot URL expires
        plot_config: The plot configuration that was used
        project_config: The project/context configuration that was used
        title: Human-readable title for the plot
        error: Error message if generation failed
        traceback: Full traceback if generation failed

    Example:
        >>> # Successful result
        >>> result = PlotResult(
        ...     url="https://console.unify.ai/plot/view/abc123",
        ...     token="abc123",
        ...     expires_in_hours=24,
        ...     plot_config={"type": "bar", "x_axis": "operative"},
        ...     project_config={"project_name": "RepairsAgent5M"},
        ...     title="Jobs by Operative"
        ... )
        >>> # Failed result
        >>> result = PlotResult(
        ...     plot_config={"type": "bar"},
        ...     project_config={"project_name": "RepairsAgent5M"},
        ...     error="Connection timeout",
        ...     traceback="..."
        ... )
    """

    url: Optional[str] = None
    token: Optional[str] = None
    expires_in_hours: Optional[int] = None
    plot_config: Dict[str, Any] = Field(default_factory=dict)
    project_config: Dict[str, Any] = Field(default_factory=dict)
    title: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """Return True if plot generation succeeded (has URL, no error)."""
        return self.url is not None and self.error is None


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
# Contains file manager tools like _reduce, _filter_files, _list_columns
ToolsDict = Dict[str, Any]
