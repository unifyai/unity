"""
Plot configurations for repairs metrics.

This module defines the default visualization configurations for each metric
and group_by combination. The configurations are based on the recommendations
from the "Performance Metrics for Midland Heart Repairs Operations" analysis,
which specifies the most effective chart types for communicating each metric
to executives and stakeholders.

Design Principles:
    1. Each metric has sensible default visualizations per group_by dimension
    2. Multiple plots per combination are supported where valuable
    3. Rationale for each configuration is documented inline
    4. Configurations map directly to the Plot API's plot_config schema

Column Naming:
    Column names in x_axis/y_axis are specified directly as they appear in
    the data tables (e.g., "OperativeWhoCompletedJob", "RepairsPatch").

Repairs Table Columns:
    - JobTicketReference: Unique job identifier (use for counting)
    - OperativeWhoCompletedJob: Operative who completed the job
    - OperativeName: Operative assigned to job
    - RepairsRegion: Region (North/South)
    - RepairsPatch: Patch area
    - WorksOrderReportedCompletedDate: Completion date
    - WorksOrderRaisedDate: Job raised date
    - WorksOrderIssuedDate: Job issued date
    - FirstTimeFix: Yes/No indicator
    - NoAccess: No access reason (empty if accessed)
    - FollowOn: Yes/No indicator
    - FollowOnDescription: Reason for follow-on
    - PropertyReference: Property identifier

Telematics Table Columns:
    - Trip: Trip identifier
    - Driver: Driver name
    - Vehicle: Vehicle identifier
    - Total distance: Total distance traveled
    - Business distance: Business miles
    - Trip travel time: Travel duration
    - StartLocation, EndLocation: Location names

Usage:
    >>> from intranet.repairs_agent.metrics.plots import get_plot_configs
    >>> configs = get_plot_configs("first_time_fix_rate", GroupBy.OPERATIVE)
    >>> for config in configs:
    ...     print(config.type, config.title)

Plot API Reference:
    - type: scatter | bar | histogram | line
    - x_axis: Column name for x-axis (required)
    - y_axis: Column name for y-axis (required for scatter/bar/line)
    - metric: count | sum | mean | min | max (aggregation for y values)
    - group_by: Optional grouping column for color coding
    - aggregate: Used when grouping - applies metric to each group
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .types import GroupBy, PlotConfig

logger = logging.getLogger(__name__)

# Note: PlotConfig is re-exported from unity.file_manager.managers.utils.viz_utils
# via types.py for type consistency with FileManager.visualize.
# PlotConfig uses plot_type (str) instead of type (enum), and aggregate instead of metric.


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Registry type: metric_name -> {group_by -> [plot_configs]}
PlotConfigRegistry = Dict[str, Dict[GroupBy, List[PlotConfig]]]


# =============================================================================
# PLOT CONFIGURATION REGISTRY
# =============================================================================

METRIC_PLOT_CONFIGS: PlotConfigRegistry = {
    # =========================================================================
    # 1. JOBS COMPLETED (Productivity)
    # =========================================================================
    # PDF: "Ranked bar chart showing each operative's average jobs per day"
    # PDF: "Time series line chart could show trends over months"
    "jobs_completed": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="Jobs Completed per Operative",
                # Rationale: Bar chart showing count of completed jobs per operative
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="Jobs Completed per Patch",
                # Rationale: Bar chart showing jobs completed in each patch area
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="Jobs Completed per Region",
                # Rationale: Compare North vs South region productivity
            ),
        ],
    },
    # =========================================================================
    # 2. JOBS ISSUED (Incoming Demand)
    # =========================================================================
    # PDF: "Time series line chart for jobs issued per day"
    "jobs_issued": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeName",
                y_axis="JobTicketReference",
                group_by="OperativeName",
                metric="count",
                title="Jobs Issued per Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="Jobs Issued per Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="Jobs Issued per Region",
            ),
        ],
    },
    # =========================================================================
    # 4. FIRST-TIME FIX RATE (%)
    # =========================================================================
    # PDF: "Bar chart by trade or patch showing each group's FTFR"
    # Note: Rate is calculated by the metric function. Plot shows raw counts
    # of FirstTimeFix='Yes' jobs per group for visual comparison.
    "first_time_fix_rate": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="First-Time Fix Count by Operative",
                # Rationale: Shows volume of FTF jobs per operative
                # (filter for FirstTimeFix='Yes' applied by metric function)
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="First-Time Fix Count by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="First-Time Fix Count by Region",
            ),
        ],
    },
    # =========================================================================
    # 5. FOLLOW-ON REQUIRED RATE (%)
    # =========================================================================
    # PDF: "Bar chart by region or operative of follow-on %"
    "follow_on_required_rate": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="Follow-On Jobs by Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="Follow-On Jobs by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="Follow-On Jobs by Region",
            ),
        ],
    },
    # =========================================================================
    # 6. FOLLOW-ON DUE TO MATERIALS (%)
    # =========================================================================
    # PDF: "Bar chart by trade for this specific metric"
    "follow_on_materials_rate": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="Materials Follow-On by Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="Materials Follow-On by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="Materials Follow-On by Region",
            ),
        ],
    },
    # =========================================================================
    # 7. NO-ACCESS RATE (%)
    # =========================================================================
    # PDF: "Bar chart by patch or region showing each area's no-access %"
    "no_access_rate": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="No-Access Jobs by Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="No-Access Jobs by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="No-Access Jobs by Region",
            ),
        ],
    },
    # =========================================================================
    # 8. ON-TIME COMPLETION RATE (SLA Compliance)
    # =========================================================================
    # PDF: "Bar chart by priority category"
    "job_completed_on_time_rate": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="On-Time Completions by Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="On-Time Completions by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="On-Time Completions by Region",
            ),
        ],
    },
    # =========================================================================
    # 9. MERCHANT STOPS PER DAY (Telematics)
    # =========================================================================
    # PDF: "Bar chart of average merchant stops per day by operative"
    # Note: Uses telematics data - columns prefixed with table label at runtime
    "merchant_stops": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="Driver",
                y_axis="Trip",
                group_by="Driver",
                metric="count",
                title="Trips per Driver",
                # Rationale: Count of trips (stops) per driver
            ),
        ],
    },
    # =========================================================================
    # 10. AVERAGE DURATION AT MERCHANT (Telematics)
    # =========================================================================
    # PDF: "Bar chart by patch or by trade"
    "merchant_dwell_time": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="Driver",
                y_axis="Trip travel time",
                group_by="Driver",
                metric="mean",
                title="Average Trip Duration per Driver",
            ),
        ],
    },
    # =========================================================================
    # 11. DISTANCE TRAVELLED PER DAY (Telematics)
    # =========================================================================
    # PDF: "Comparative bar chart by region showing average daily miles"
    "total_distance_travelled": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="Driver",
                y_axis="Total distance",
                group_by="Driver",
                metric="sum",
                title="Total Distance per Driver",
            ),
        ],
    },
    # =========================================================================
    # 12. AVERAGE TIME TRAVELLING (Telematics)
    # =========================================================================
    # PDF: "Bar chart showing average travel hours per day for each region"
    "travel_time": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="Driver",
                y_axis="Trip travel time",
                group_by="Driver",
                metric="sum",
                title="Total Travel Time per Driver",
            ),
        ],
    },
    # =========================================================================
    # 13. JOBS REQUIRING MATERIALS RATE (%)
    # =========================================================================
    # PDF: "Comparative chart by trade"
    "jobs_requiring_materials_rate": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="Jobs with Materials by Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="Jobs with Materials by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="Jobs with Materials by Region",
            ),
        ],
    },
    # =========================================================================
    # 14. AVERAGE REPAIRS PER PROPERTY
    # =========================================================================
    # PDF: "Single average number with breakdown by property type"
    "avg_repairs_per_property": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="Repairs by Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="Repairs by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="Repairs by Region",
            ),
        ],
    },
    # =========================================================================
    # 15. COMPLAINTS RATE (%)
    # =========================================================================
    # Note: Complaints data not available in current dataset
    "complaints_rate": {},
    # =========================================================================
    # 16. APPOINTMENT ADHERENCE RATE (%)
    # =========================================================================
    # Shows whether operatives arrive within scheduled appointment windows
    "appointment_adherence_rate": {
        GroupBy.OPERATIVE: [
            PlotConfig(
                plot_type="bar",
                x_axis="OperativeWhoCompletedJob",
                y_axis="JobTicketReference",
                group_by="OperativeWhoCompletedJob",
                metric="count",
                title="Scheduled Jobs by Operative",
            ),
        ],
        GroupBy.PATCH: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsPatch",
                y_axis="JobTicketReference",
                group_by="RepairsPatch",
                metric="count",
                title="Scheduled Jobs by Patch",
            ),
        ],
        GroupBy.REGION: [
            PlotConfig(
                plot_type="bar",
                x_axis="RepairsRegion",
                y_axis="JobTicketReference",
                group_by="RepairsRegion",
                metric="count",
                title="Scheduled Jobs by Region",
            ),
        ],
    },
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_plot_configs(metric_name: str, group_by: GroupBy) -> List[PlotConfig]:
    """
    Get plot configurations for a specific metric and group_by combination.

    Args:
        metric_name: Name of the metric (e.g., "first_time_fix_rate")
        group_by: Grouping dimension (e.g., GroupBy.OPERATIVE)

    Returns:
        List of PlotConfig objects for the combination, or empty list if
        no configurations are defined.

    Example:
        >>> configs = get_plot_configs("first_time_fix_rate", GroupBy.OPERATIVE)
        >>> len(configs)
        1
        >>> configs[0].type
        'bar'
    """
    metric_configs = METRIC_PLOT_CONFIGS.get(metric_name, {})
    return metric_configs.get(group_by, [])


def get_all_metrics_with_plots() -> List[str]:
    """
    Get list of all metrics that have plot configurations defined.

    Returns:
        List of metric names that have at least one plot configuration.

    Example:
        >>> metrics = get_all_metrics_with_plots()
        >>> "first_time_fix_rate" in metrics
        True
    """
    return list(METRIC_PLOT_CONFIGS.keys())


def get_supported_group_bys(metric_name: str) -> List[GroupBy]:
    """
    Get list of group_by dimensions that have plots defined for a metric.

    Args:
        metric_name: Name of the metric

    Returns:
        List of GroupBy values that have configurations for this metric.

    Example:
        >>> group_bys = get_supported_group_bys("first_time_fix_rate")
        >>> GroupBy.OPERATIVE in group_bys
        True
    """
    metric_configs = METRIC_PLOT_CONFIGS.get(metric_name, {})
    return list(metric_configs.keys())


# =============================================================================
# NOTE: Plot generation is now INLINE in each metric function (core.py)
# =============================================================================
# The generate_plots function has been removed. Plot configurations are now
# defined inline within each metric function to make them visible to CodeActActor
# when the function source is retrieved via FunctionManager.
#
# See core.py for inline plot generation patterns.
