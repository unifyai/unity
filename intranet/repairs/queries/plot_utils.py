"""
Plot generation utilities for repairs metrics.

This module provides a thin wrapper for generating plot visualizations via
FileManager's _visualize tool. It handles the mapping from metric×group_by
combinations to plot configurations defined in _plots.py.

Architecture Note:
    This module delegates all plot generation to FileManager._visualize,
    which handles context resolution, API communication, retry logic, and
    authentication. The plot configurations themselves are defined in _plots.py.

Usage:
    >>> from .plot_utils import generate_plots
    >>> plots = generate_plots(
    ...     visualize_tool=tools["visualize"],
    ...     metric_name="first_time_fix_rate",
    ...     group_by=GroupBy.OPERATIVE,
    ...     tables="/path/to/file.xlsx.Tables.TableName",
    ...     filter_expr='`Status` == "Completed"',
    ...     include_plots=True
    ... )
    >>> for plot in plots:
    ...     if plot.succeeded:
    ...         print(f"{plot.title}: {plot.url}")
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from ._plots import get_plot_configs
from ._types import GroupBy, PlotConfig, PlotResult, VisualizeTool

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API
# =============================================================================


def generate_plots(
    *,
    visualize_tool: VisualizeTool,
    metric_name: str,
    group_by: GroupBy,
    tables: Union[str, List[str]],
    filter_expr: Optional[str] = None,
    include_plots: bool = False,
) -> List[PlotResult]:
    """
    Generate plot visualizations for a metric result.

    This function looks up the plot configurations for the given metric
    and group_by combination, then generates each plot via the FileManager's
    _visualize tool. If any plot generation fails, the error is captured
    in the PlotResult rather than raising an exception.

    Parameters
    ----------
    visualize_tool : VisualizeTool
        The FileManager._visualize tool (obtained via tools["visualize"]).
    metric_name : str
        Name of the metric (e.g., "first_time_fix_rate").
        Must match a key in METRIC_PLOT_CONFIGS.
    group_by : GroupBy
        Grouping dimension used in the query. Determines which
        plot configurations are selected.
    tables : str | list[str]
        Table reference(s) in FileManager format. Accepts either:
        - A single string for metrics with one data source
        - A list of strings for metrics spanning multiple tables
          (e.g., telematics data with monthly tables)
        When multiple tables are provided, plots are generated for EACH
        table, with the table name appended to the plot title.
    filter_expr : str | None
        Optional filter expression applied to the data. Same syntax as
        used in _reduce calls.
    include_plots : bool
        If False, returns empty list immediately without making any API
        calls. This enables the caller to skip plot generation when not
        needed (default behavior).

    Returns
    -------
    list[PlotResult]
        List of PlotResult objects. Each PlotResult contains either:
        - A successful result with url, token, and expires_in_hours
        - A failed result with error message

        Returns empty list if:
        - include_plots is False
        - No plot configurations are defined for the metric×group_by combination
        - tables is empty

    Example (single table):
        >>> plots = generate_plots(
        ...     visualize_tool=tools["visualize"],
        ...     metric_name="first_time_fix_rate",
        ...     group_by=GroupBy.OPERATIVE,
        ...     tables="/path/to/data.xlsx.Tables.Repairs",
        ...     filter_expr='`Status` == "Completed"',
        ...     include_plots=True
        ... )

    Example (multiple tables - telematics):
        >>> plots = generate_plots(
        ...     visualize_tool=tools["visualize"],
        ...     metric_name="distance_travelled_per_day",
        ...     group_by=GroupBy.OPERATIVE,
        ...     tables=[
        ...         "/path/to/data.xlsx.Tables.July_2025",
        ...         "/path/to/data.xlsx.Tables.August_2025",
        ...     ],
        ...     include_plots=True
        ... )
    """
    # Early exit if plot generation is disabled
    if not include_plots:
        return []

    # Normalize tables input: convert string to list
    table_list: List[str] = []
    if isinstance(tables, str):
        if tables:
            table_list = [tables]
    else:
        table_list = [t for t in tables if t]  # Filter out empty strings

    if not table_list:
        logger.warning(f"[{metric_name}] No tables provided for plot generation")
        return []

    # Look up plot configurations for this metric×group_by
    configs = get_plot_configs(metric_name, group_by)
    if not configs:
        logger.debug(
            f"[{metric_name}] No plot configs defined for group_by={group_by.value}",
        )
        return []

    total_plots = len(configs) * len(table_list)
    logger.debug(
        f"[{metric_name}] Generating {total_plots} plot(s) "
        f"({len(configs)} configs × {len(table_list)} tables) for group_by={group_by.value}",
    )

    # Generate plots for each config
    results: List[PlotResult] = []
    for config in configs:
        config_results = _generate_plots_for_config(
            visualize_tool=visualize_tool,
            config=config,
            tables=table_list,
            filter_expr=filter_expr,
            metric_name=metric_name,
        )
        results.extend(config_results)

    return results


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


def _generate_plots_for_config(
    *,
    visualize_tool: VisualizeTool,
    config: PlotConfig,
    tables: List[str],
    filter_expr: Optional[str],
    metric_name: str,
) -> List[PlotResult]:
    """
    Generate plots for a single config across one or more tables.

    Parameters
    ----------
    visualize_tool : VisualizeTool
        The FileManager._visualize tool.
    config : PlotConfig
        Plot configuration from _plots.py.
    tables : list[str]
        List of table references.
    filter_expr : str | None
        Optional filter expression.
    metric_name : str
        Metric name for logging.

    Returns
    -------
    list[PlotResult]
        One PlotResult per table.
    """
    try:
        # Call the visualize tool with config parameters
        # The visualize tool handles batch processing internally when multiple tables
        raw_result = visualize_tool(
            tables=tables,
            plot_type=config.plot_type,
            x_axis=config.x_axis,
            y_axis=config.y_axis,
            group_by=config.group_by,
            filter=filter_expr,
            title=config.title,
            aggregate=config.aggregate,
            scale_x=config.scale_x,
            scale_y=config.scale_y,
            bin_count=config.bin_count,
            show_regression=config.show_regression,
        )

        # Normalize to list
        if isinstance(raw_result, list):
            results = raw_result
        else:
            results = [raw_result]

        # Log results
        for result in results:
            if result.succeeded:
                logger.info(
                    f"[{metric_name}] Generated plot: {result.title} -> {result.url}",
                )
            else:
                logger.warning(
                    f"[{metric_name}] Plot generation failed: {result.title} -> {result.error}",
                )

        return results

    except Exception as e:
        # If visualize tool raises, wrap in PlotResult error
        logger.exception(f"[{metric_name}] Unexpected error calling visualize tool")
        error_result = PlotResult(
            title=config.title or f"{config.plot_type} chart",
            error=f"Visualize tool error: {type(e).__name__}: {e}",
        )
        # Return one error result per table
        return [error_result] * len(tables)
