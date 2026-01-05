"""
Metrics module - Single Source of Truth for repairs/telematics metrics.

This module contains all metric function definitions, helpers, types,
and plot configurations. Both the static BespokeRepairsAgent and dynamic
DynamicRepairsAgent use these definitions.

Components:
    - definitions: All 16+ metric function implementations (to be migrated)
    - helpers: Reusable compositional helper functions (to be migrated)
    - types: GroupBy, TimePeriod, MetricResult, PlotResult
    - constants: Table names, filters, column mappings (to be migrated)
    - plots: Plot configurations per metric and group_by

Usage:
    from intranet.repairs_agent.metrics import GroupBy, TimePeriod, MetricResult
    from intranet.repairs_agent.metrics.plots import get_plot_configs
"""

# Re-export commonly used types
from .types import (
    GroupBy,
    MetricResult,
    PlotConfig,
    PlotResult,
    PlotType,
    TimePeriod,
    ToolsDict,
    VisualizeTool,
)

# Re-export plot functions
from .plots import (
    get_all_metrics_with_plots,
    get_plot_configs,
    get_supported_group_bys,
)

__all__ = [
    # Types
    "GroupBy",
    "TimePeriod",
    "MetricResult",
    "PlotResult",
    "PlotConfig",
    "PlotType",
    "ToolsDict",
    "VisualizeTool",
    # Plot functions
    "get_plot_configs",
    "get_all_metrics_with_plots",
    "get_supported_group_bys",
]
