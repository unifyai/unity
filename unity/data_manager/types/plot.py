"""
Plot-related type definitions for DataManager.

This module defines Pydantic models for plot configuration and results.
These types are used by DataManager's visualization methods.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PlotType(str, Enum):
    """
    Supported plot types for visualization.

    Values
    ------
    SCATTER
        Scatter plot for correlations between two numeric variables.
    BAR
        Bar chart for comparing values across categories.
    HISTOGRAM
        Histogram for distribution of a single variable.
    LINE
        Line chart for trends over time/sequences.
    """

    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"
    LINE = "line"


class PlotConfig(BaseModel):
    """
    Configuration for a plot visualization.

    This model defines all parameters needed to generate a plot via
    DataManager's ``plot()`` method. Maps to the Plot API request body.

    Attributes
    ----------
    plot_type : str
        Chart type: "scatter", "bar", "histogram", "line".
    x_axis : str
        Column name for the X-axis.
    y_axis : str | None
        Column name for the Y-axis. Required for scatter, bar, line.
    group_by : str | None
        Column to group/color data points by.
    metric : str | None
        Metric for aggregation (deprecated, use ``aggregate``).
    aggregate : str | None
        Aggregation function for bar charts: "sum", "count", "avg", "min", "max".
    scale_x : str | None
        X-axis scale: "linear" or "log".
    scale_y : str | None
        Y-axis scale: "linear" or "log".
    bin_count : int | None
        Number of bins for histogram plots.
    show_regression : bool | None
        Whether to show regression line on scatter plots.
    title : str | None
        Chart title displayed above the plot.

    Usage Examples
    --------------
    >>> config = PlotConfig(
    ...     plot_type="bar",
    ...     x_axis="category",
    ...     y_axis="revenue",
    ...     aggregate="sum",
    ...     title="Revenue by Category"
    ... )
    """

    model_config = ConfigDict(use_enum_values=True)

    plot_type: str
    x_axis: str
    y_axis: Optional[str] = None
    group_by: Optional[str] = None
    metric: Optional[str] = None
    aggregate: Optional[str] = None
    scale_x: Optional[str] = None
    scale_y: Optional[str] = None
    bin_count: Optional[int] = None
    show_regression: Optional[bool] = None
    title: Optional[str] = None


class PlotResult(BaseModel):
    """
    Result of a plot generation attempt.

    Contains either a successful plot URL or error information if generation failed.

    Attributes
    ----------
    url : str | None
        URL to the generated plot image. None if generation failed.
    token : str | None
        Authentication token for accessing the plot (if applicable).
    expires_in_hours : int | None
        Hours until the plot URL expires.
    title : str | None
        Title of the generated plot.
    context : str | None
        Source context path for the plot data.
    error : str | None
        Error message if plot generation failed.
    traceback_str : str | None
        Full traceback string for debugging failures.

    Properties
    ----------
    succeeded : bool
        True if the plot was generated successfully.

    Usage Examples
    --------------
    >>> result = dm.plot("Data/MH/arrears", plot_type="bar", x="region", y="amount")
    >>> if result.succeeded:
    ...     print(f"Plot URL: {result.url}")
    ... else:
    ...     print(f"Error: {result.error}")
    """

    url: Optional[str] = None
    token: Optional[str] = None
    expires_in_hours: Optional[int] = None
    title: Optional[str] = None
    context: Optional[str] = None
    error: Optional[str] = None
    traceback_str: Optional[str] = Field(default=None, alias="traceback")

    @property
    def succeeded(self) -> bool:
        """True if the plot was generated successfully."""
        return self.url is not None and self.error is None
