"""
Performance metrics queries for repairs.

Each metric is a registered async function with standard parameters for
grouping and time filtering. Metrics can be broken down by:
- operative
- trade
- patch
- region
- time_period

Use execute_tool/execute_tools for tool calls and write any custom
Python aggregation logic you need.

Metric Reference (from requirements):
--------------------------------------
1.  jobs_completed_per_day - Jobs completed per man per day
2.  no_access_rate - No Access % / Absolute number
3.  first_time_fix_rate - First Time Fix % / Absolute Number
4.  follow_on_required_rate - Follow on Required % / Absolute Number
5.  follow_on_materials_rate - Follow on Required for Materials %
6.  job_completed_on_time_rate - Job completed on time % / Absolute Number
7.  merchant_stops_per_day - No of merchant stops per day
8.  avg_duration_at_merchant - Average duration at a Merchant per day
9.  distance_travelled_per_day - Distance Travelled per day
10. avg_time_travelling - Average time travelling per day
11. repairs_completed_per_day - Repairs completed per day
12. jobs_issued_per_day - Jobs issued per day
13. jobs_requiring_materials_rate - % of jobs completed that require materials
14. avg_repairs_per_property - Average no of repairs per property completed
15. complaints_rate - Complaints as % of total jobs completed
"""

from __future__ import annotations

from typing import Optional

from intranet.core.bespoke_repairs_agent import register

from ._types import GroupBy, MetricResult, TimePeriod, ToolsDict


# =============================================================================
# 1. Jobs Completed Per Day
# =============================================================================


@register(
    "jobs_completed_per_day",
    "Jobs completed per man per day (by operative/trade/patch/region/time)",
)
async def jobs_completed_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
) -> MetricResult:
    """
    Get jobs completed per man per day.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by (operative, trade, patch, region)
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation

    Returns
    -------
    MetricResult
        Aggregated results with grouping metadata
    """
    # TODO: Implement query logic
    raise NotImplementedError("jobs_completed_per_day not yet implemented")


# =============================================================================
# 2. No Access Rate
# =============================================================================


@register(
    "no_access_rate",
    "No Access % / Absolute number (by operative/trade/patch/region/time)",
)
async def no_access_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
) -> MetricResult:
    """
    Get No Access rate as percentage or absolute number.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation
    return_absolute : bool
        If True, return absolute count; if False, return percentage

    Returns
    -------
    MetricResult
        Rate or count of no-access jobs
    """
    # TODO: Implement query logic
    raise NotImplementedError("no_access_rate not yet implemented")


# =============================================================================
# 3. First Time Fix Rate
# =============================================================================


@register(
    "first_time_fix_rate",
    "First Time Fix % / Absolute Number (by operative/trade/patch/region/time)",
)
async def first_time_fix_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
) -> MetricResult:
    """
    Get First Time Fix rate as percentage or absolute number.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation
    return_absolute : bool
        If True, return absolute count; if False, return percentage

    Returns
    -------
    MetricResult
        Rate or count of first-time-fix jobs
    """
    # TODO: Implement query logic
    raise NotImplementedError("first_time_fix_rate not yet implemented")


# =============================================================================
# 4. Follow On Required Rate
# =============================================================================


@register(
    "follow_on_required_rate",
    "Follow on Required % / Absolute Number (by operative/trade/patch/region/time)",
)
async def follow_on_required_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
) -> MetricResult:
    """
    Get Follow On Required rate as percentage or absolute number.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation
    return_absolute : bool
        If True, return absolute count; if False, return percentage

    Returns
    -------
    MetricResult
        Rate or count of jobs requiring follow-on
    """
    # TODO: Implement query logic
    raise NotImplementedError("follow_on_required_rate not yet implemented")


# =============================================================================
# 5. Follow On Required for Materials Rate
# =============================================================================


@register(
    "follow_on_materials_rate",
    "Follow on Required for Materials % (by operative/trade/patch/region/time)",
)
async def follow_on_materials_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
) -> MetricResult:
    """
    Get Follow On Required specifically for Materials as percentage or absolute.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation
    return_absolute : bool
        If True, return absolute count; if False, return percentage

    Returns
    -------
    MetricResult
        Rate or count of jobs requiring follow-on due to materials
    """
    # TODO: Implement query logic
    raise NotImplementedError("follow_on_materials_rate not yet implemented")


# =============================================================================
# 6. Job Completed On Time Rate
# =============================================================================


@register(
    "job_completed_on_time_rate",
    "Job completed on time % / Absolute Number (by operative/trade/patch/region/time)",
)
async def job_completed_on_time_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
) -> MetricResult:
    """
    Get Job Completed On Time rate as percentage or absolute number.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation
    return_absolute : bool
        If True, return absolute count; if False, return percentage

    Returns
    -------
    MetricResult
        Rate or count of jobs completed on time
    """
    # TODO: Implement query logic
    raise NotImplementedError("job_completed_on_time_rate not yet implemented")


# =============================================================================
# 7. Merchant Stops Per Day
# =============================================================================


@register(
    "merchant_stops_per_day",
    "Number of merchant stops per day (by operative/trade/patch/region/time)",
)
async def merchant_stops_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
) -> MetricResult:
    """
    Get number of merchant stops per day.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation

    Returns
    -------
    MetricResult
        Count of merchant stops grouped by dimension
    """
    # TODO: Implement query logic (requires telematics data)
    raise NotImplementedError("merchant_stops_per_day not yet implemented")


# =============================================================================
# 8. Average Duration at Merchant
# =============================================================================


@register(
    "avg_duration_at_merchant",
    "Average duration at a Merchant per day (by operative/trade/patch/region/time)",
)
async def avg_duration_at_merchant(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
) -> MetricResult:
    """
    Get average duration at a merchant per day.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation

    Returns
    -------
    MetricResult
        Average duration (in minutes) at merchant stops
    """
    # TODO: Implement query logic (requires telematics data)
    raise NotImplementedError("avg_duration_at_merchant not yet implemented")


# =============================================================================
# 9. Distance Travelled Per Day
# =============================================================================


@register(
    "distance_travelled_per_day",
    "Distance travelled per day (by operative/trade/patch/region/time)",
)
async def distance_travelled_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
) -> MetricResult:
    """
    Get distance travelled per day.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation

    Returns
    -------
    MetricResult
        Distance travelled (in miles/km) grouped by dimension
    """
    # TODO: Implement query logic (requires telematics data)
    raise NotImplementedError("distance_travelled_per_day not yet implemented")


# =============================================================================
# 10. Average Time Travelling Per Day
# =============================================================================


@register(
    "avg_time_travelling",
    "Average time travelling per day (by operative/trade/patch/region/time)",
)
async def avg_time_travelling(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
) -> MetricResult:
    """
    Get average time spent travelling per day.

    Note: This metric may depend on telematics data availability.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation

    Returns
    -------
    MetricResult
        Average travel time (in minutes) grouped by dimension
    """
    # TODO: Implement query logic (requires telematics data, feasibility TBD)
    raise NotImplementedError("avg_time_travelling not yet implemented")


# =============================================================================
# 11. Repairs Completed Per Day
# =============================================================================


@register(
    "repairs_completed_per_day",
    "Repairs completed per day (total/by trade/patch/region/time)",
)
async def repairs_completed_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.TOTAL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
) -> MetricResult:
    """
    Get total repairs completed per day.

    Unlike jobs_completed_per_day which is per-operative, this gives
    aggregate totals across all patches.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by (default: TOTAL for all patches)
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation

    Returns
    -------
    MetricResult
        Total repair count grouped by dimension
    """
    # TODO: Implement query logic
    raise NotImplementedError("repairs_completed_per_day not yet implemented")


# =============================================================================
# 12. Jobs Issued Per Day
# =============================================================================


@register(
    "jobs_issued_per_day",
    "Jobs issued per day (total/by trade/patch/region/time)",
)
async def jobs_issued_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.TOTAL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
) -> MetricResult:
    """
    Get jobs issued per day.

    Note: Need to identify how orders/jobs work (multiple jobs per order?).

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by (default: TOTAL for all patches)
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation

    Returns
    -------
    MetricResult
        Count of jobs issued grouped by dimension
    """
    # TODO: Implement query logic
    # NOTE: Need to clarify order vs job relationship
    raise NotImplementedError("jobs_issued_per_day not yet implemented")


# =============================================================================
# 13. Jobs Requiring Materials Rate
# =============================================================================


@register(
    "jobs_requiring_materials_rate",
    "% of jobs completed that require materials (by operative/trade/patch/region/time)",
)
async def jobs_requiring_materials_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
) -> MetricResult:
    """
    Get percentage of completed jobs that required materials.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation
    return_absolute : bool
        If True, return absolute count; if False, return percentage

    Returns
    -------
    MetricResult
        Rate or count of jobs requiring materials
    """
    # TODO: Implement query logic
    raise NotImplementedError("jobs_requiring_materials_rate not yet implemented")


# =============================================================================
# 14. Average Repairs Per Property
# =============================================================================


@register(
    "avg_repairs_per_property",
    "Average number of repairs per property completed (per operative/month/quarter/year)",
)
async def avg_repairs_per_property(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    time_period: TimePeriod = TimePeriod.MONTH,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> MetricResult:
    """
    Get average number of repairs per property.

    This helps identify properties with multiple repairs (potential issues).

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by (per operative)
    time_period : TimePeriod
        Time granularity (month, quarter, year)
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)

    Returns
    -------
    MetricResult
        Average repairs per property, with list of properties having multiple repairs
    """
    # TODO: Implement query logic
    # NOTE: Want to identify properties with multiple repairs
    raise NotImplementedError("avg_repairs_per_property not yet implemented")


# =============================================================================
# 15. Complaints Rate
# =============================================================================


@register(
    "complaints_rate",
    "Complaints as % of total jobs completed (by operative/trade/patch/region/time)",
)
async def complaints_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
) -> MetricResult:
    """
    Get complaints as percentage of total jobs completed.

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager
    group_by : GroupBy
        Dimension to group results by
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    time_period : TimePeriod
        Time granularity for aggregation
    return_absolute : bool
        If True, return absolute count; if False, return percentage

    Returns
    -------
    MetricResult
        Rate or count of complaints
    """
    # TODO: Implement query logic
    raise NotImplementedError("complaints_rate not yet implemented")


# =============================================================================
# Utility: List all registered metrics
# =============================================================================

ALL_METRICS = [
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
]
