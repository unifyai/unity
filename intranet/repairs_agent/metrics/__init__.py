"""
Metrics module - Single Source of Truth for repairs/telematics metrics.

This module contains all metric function definitions, helpers, types,
and plot configurations. Both the static BespokeRepairsAgent and dynamic
DynamicRepairsAgent use these definitions.

Components:
    - definitions: All 16+ metric function implementations
    - helpers: Reusable compositional helper functions
    - types: GroupBy, TimePeriod, MetricResult, PlotResult
    - constants: Table names, filters, column mappings
    - plots: Plot configurations per metric and group_by

Usage:
    from intranet.repairs_agent.metrics import GroupBy, TimePeriod, MetricResult
    from intranet.repairs_agent.metrics.definitions import first_time_fix_rate
"""

# Re-export commonly used types
# Note: These will be available after migration from intranet/repairs/queries/
# from .types import GroupBy, TimePeriod, MetricResult, PlotResult
# from .constants import REPAIRS_TABLE, TELEMATICS_TABLES

__all__ = [
    # Types (will be re-exported after migration)
    # "GroupBy",
    # "TimePeriod",
    # "MetricResult",
    # "PlotResult",
]
