"""
Metrics module - Single Source of Truth for repairs/telematics metrics.

This module contains all metric function definitions, helpers, types,
and plot configurations. Both the static BespokeRepairsAgent and dynamic
DynamicRepairsAgent use these definitions.

Components:
    - core: All 16 metric function implementations
    - helpers: Reusable compositional helper functions
    - types: GroupBy, TimePeriod, MetricResult, PlotResult
    - plots: Plot configurations and generation per metric and group_by

Usage:
    from intranet.repairs_agent.metrics import GroupBy, TimePeriod, MetricResult
    from intranet.repairs_agent.metrics import first_time_fix_rate
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
    VisualizeTool,
    FileTools,
)

# Re-export plot query functions (plot generation is now inline in each metric)
from .plots import (
    get_all_metrics_with_plots,
    get_plot_configs,
    get_supported_group_bys,
)

# Re-export all metric functions from core
from .core import (
    ALL_METRICS,
    appointment_adherence_rate,
    avg_duration_at_merchant,
    avg_repairs_per_property,
    avg_time_travelling,
    complaints_rate,
    distance_travelled_per_day,
    first_time_fix_rate,
    follow_on_materials_rate,
    follow_on_required_rate,
    job_completed_on_time_rate,
    jobs_completed_per_day,
    jobs_issued_per_day,
    jobs_requiring_materials_rate,
    merchant_stops_per_day,
    no_access_rate,
    repairs_completed_per_day,
)

__all__ = [
    # Types
    "GroupBy",
    "TimePeriod",
    "MetricResult",
    "PlotResult",
    "PlotConfig",
    "PlotType",
    "FileTools",
    "VisualizeTool",
    # Plot query functions (plot generation is inline in each metric)
    "get_plot_configs",
    "get_all_metrics_with_plots",
    "get_supported_group_bys",
    # Metric functions
    "jobs_completed_per_day",
    "no_access_rate",
    "first_time_fix_rate",
    "follow_on_required_rate",
    "follow_on_materials_rate",
    "job_completed_on_time_rate",
    "merchant_stops_per_day",
    "avg_duration_at_merchant",
    "distance_travelled_per_day",
    "avg_time_travelling",
    "repairs_completed_per_day",
    "jobs_issued_per_day",
    "jobs_requiring_materials_rate",
    "avg_repairs_per_property",
    "complaints_rate",
    "appointment_adherence_rate",
    # List of all metrics
    "ALL_METRICS",
]
