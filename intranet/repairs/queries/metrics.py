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

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from intranet.core.bespoke_repairs_agent import register

from ._types import GroupBy, MetricResult, PlotResult, TimePeriod, ToolsDict
from .plot_utils import generate_plots, get_active_project

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def metric_timer(metric_name: str):
    """Decorator to add timing and logging to metric functions."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            logger.info(f"[{metric_name}] Starting metric calculation...")
            logger.debug(f"[{metric_name}] Parameters: {kwargs}")

            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time

                # Log summary of results
                if isinstance(result, dict):
                    num_results = len(result.get("results", []))
                    total = result.get("total", "N/A")
                    logger.info(
                        f"[{metric_name}] ✓ Completed in {_format_duration(elapsed)} | "
                        f"Results: {num_results} groups | Total: {total}",
                    )
                else:
                    logger.info(
                        f"[{metric_name}] ✓ Completed in {_format_duration(elapsed)}",
                    )

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"[{metric_name}] ✗ Failed after {_format_duration(elapsed)}: {type(e).__name__}: {e}",
                )
                raise

        return wrapper  # type: ignore

    return decorator


def _timed_tool(
    metric_name: str,
    tool_name: str,
    tool: Callable,
    label: str = "",
    **kwargs: Any,
) -> Any:
    """
    Generic timing wrapper for any file manager tool call.

    Parameters
    ----------
    metric_name : str
        Name of the metric (for log prefix)
    tool_name : str
        Name of the tool being called (e.g., "reduce", "filter_files")
    tool : Callable
        The tool function to call
    label : str, optional
        Additional context label for the log message
    **kwargs : Any
        Arguments to pass to the tool

    Returns
    -------
    Any
        The result from the tool call

    Example
    -------
    >>> result = _timed_tool(
    ...     "jobs_completed_per_day", "reduce", reduce_tool,
    ...     label="count jobs",
    ...     table=REPAIRS_TABLE, metric="count", keys="JobTicketReference",
    ...     filter=filter_expr, group_by=group_by_field,
    ... )
    """
    start_time = time.perf_counter()
    result = tool(**kwargs)
    elapsed = time.perf_counter() - start_time

    # Generic result summary based on type
    if result is None:
        summary = "None"
    elif isinstance(result, dict):
        summary = f"{len(result)} items"
    elif isinstance(result, list):
        summary = f"{len(result)} rows"
    elif isinstance(result, (int, float)):
        summary = str(result)
    else:
        summary = f"{type(result).__name__}"

    label_str = f" ({label})" if label else ""
    logger.debug(
        f"[{metric_name}] {tool_name}{label_str} took {_format_duration(elapsed)}: {summary}",
    )
    return result


def _extract_count(value: Any) -> int:
    """
    Extract count from reduce result value.

    The reduce tool can return values in different formats:
    - Direct int: 123
    - Dict with 'count' key: {'shared_value': None, 'count': 123}
    - Dict with 'count' as None: {'shared_value': 'xxx', 'count': None}

    Returns 0 if count cannot be extracted.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        count = value.get("count")
        if count is not None:
            return int(count)
        # Fallback: try shared_value if it's numeric
        shared = value.get("shared_value")
        if isinstance(shared, (int, float)):
            return int(shared)
        return 0
    logger.warning(
        f"Unexpected value type in reduce result: {type(value).__name__} = {value}",
    )
    return 0


def _extract_sum(value: Any) -> float:
    """
    Extract sum from reduce result value.

    Similar to _extract_count but for sum operations, returns float.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        # Try 'sum' key first, then 'count'
        for key in ("sum", "count", "value"):
            val = value.get(key)
            if val is not None and isinstance(val, (int, float)):
                return float(val)
        return 0.0
    logger.warning(
        f"Unexpected value type in reduce sum result: {type(value).__name__} = {value}",
    )
    return 0.0


def _normalize_grouped_result(
    result: Dict[str, Any],
    extract_fn: Callable[[Any], Any] = _extract_count,
) -> Dict[str, Any]:
    """
    Normalize grouped reduce results to simple {group: value} format.

    Handles the nested {'shared_value': ..., 'count': ...} format.
    """
    if not isinstance(result, dict):
        return result

    normalized = {}
    for key, value in result.items():
        normalized[key] = extract_fn(value)
    return normalized


# =============================================================================
# TABLE CONSTANTS
# =============================================================================

REPAIRS_FILE = "/home/hmahmood24/unity/intranet/repairs/MDH Repairs Data July - Nov 25 - DL V1.xlsx"
TELEMATICS_FILE = "/home/hmahmood24/unity/intranet/repairs/MDH Telematics Data July - Nov 25 - DL V1.xlsx"

REPAIRS_TABLE = f"{REPAIRS_FILE}.Tables.Raised_01-07-2025_to_30-11-2025"

TELEMATICS_TABLES = {
    "july": f"{TELEMATICS_FILE}.Tables.July_2025",
    "august": f"{TELEMATICS_FILE}.Tables.August_2025",
    "september": f"{TELEMATICS_FILE}.Tables.September_2025",
    "october": f"{TELEMATICS_FILE}.Tables.October_2025",
    "november": f"{TELEMATICS_FILE}.Tables.November_2025",
}

ALL_TELEMATICS_TABLES = list(TELEMATICS_TABLES.values())

# =============================================================================
# FILTER CONSTANTS
# =============================================================================

COMPLETED_FILTER = "`WorksOrderStatusDescription` in ['Complete', 'Closed']"
NO_ACCESS_FILTER = "`NoAccess` != 'None' and `NoAccess` != ''"
FIRST_TIME_FIX_FILTER = "`FirstTimeFix` == 'Yes'"
SECOND_TIME_FIX_FILTER = "`SecondTimeFix` == 'Yes'"
THIRD_TIME_FIX_FILTER = "`ThirdTimePlusFix` == 'Yes'"
FOLLOW_ON_FILTER = "`FollowOn` == 'Yes'"
ISSUED_FILTER = "`WorksOrderStatusDescription` == 'Issued'"

# Known merchant names for telematics location matching
MERCHANT_NAMES = [
    "Travis Perkins",
    "Screwfix",
    "Toolstation",
    "Plumb Center",
    "City Plumbing",
    "Jewson",
    "Selco",
    "Wickes",
]

# =============================================================================
# GROUP BY FIELD MAPPINGS
# =============================================================================

GROUP_BY_FIELDS = {
    GroupBy.OPERATIVE: "OperativeWhoCompletedJob",
    GroupBy.PATCH: "RepairsPatch",
    GroupBy.REGION: "RepairsRegion",
    GroupBy.TOTAL: None,  # No grouping
}

TELEMATICS_GROUP_BY_FIELDS = {
    GroupBy.OPERATIVE: "Vehicle",  # Vehicle contains operative name
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _resolve_group_by(group_by: GroupBy | str) -> Optional[str]:
    """Resolve GroupBy enum or string to actual column name."""
    # Handle string inputs (from CLI JSON params)
    if isinstance(group_by, str):
        try:
            group_by = GroupBy(group_by)
        except ValueError:
            logger.warning(f"Unknown group_by value: {group_by}, returning None")
            return None
    return GROUP_BY_FIELDS.get(group_by)


def _resolve_telematics_group_by(group_by: GroupBy | str) -> Optional[str]:
    """Resolve GroupBy enum or string to telematics column name."""
    # Handle string inputs (from CLI JSON params)
    if isinstance(group_by, str):
        try:
            group_by = GroupBy(group_by)
        except ValueError:
            return None
    # Only operative grouping is supported for telematics
    if group_by == GroupBy.OPERATIVE:
        return TELEMATICS_GROUP_BY_FIELDS.get(group_by)
    return None


def _build_date_filter(
    base_filter: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    date_column: str = "`VisitDate`",
) -> Optional[str]:
    """Build combined filter with date range."""
    filters = []
    if base_filter:
        filters.append(f"({base_filter})")
    if start_date:
        filters.append(f"{date_column} >= '{start_date}'")
    if end_date:
        filters.append(f"{date_column} <= '{end_date}'")
    return " and ".join(filters) if filters else None


def _compute_percentage(numerator: int, denominator: int) -> float:
    """Compute percentage safely."""
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _build_metric_result(
    metric_name: str,
    group_by: GroupBy | str,
    time_period: TimePeriod | str,
    start_date: Optional[str],
    end_date: Optional[str],
    results: List[Dict[str, Any]],
    total: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    plots: Optional[List[PlotResult]] = None,
) -> MetricResult:
    """
    Build standardized MetricResult with optional plot visualizations.

    This function creates a consistent output format for all metrics,
    handling enum-to-string conversions and optional fields.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g., "first_time_fix_rate")
    group_by : GroupBy | str
        Grouping dimension used (enum or string value)
    time_period : TimePeriod | str
        Time granularity used (enum or string value)
    start_date : str | None
        Start date of analysis period
    end_date : str | None
        End date of analysis period
    results : list
        List of result dicts with metric values
    total : float | None
        Aggregate total across all groups
    metadata : dict | None
        Additional metadata about the calculation
    plots : list[PlotResult] | None
        Generated plot visualizations (empty list if plots were not requested)

    Returns
    -------
    MetricResult
        Pydantic model with all metric data and optional plots
    """
    # Handle both enum and string inputs (strings come from CLI JSON params)
    group_by_value = group_by.value if isinstance(group_by, GroupBy) else group_by
    time_period_value = (
        time_period.value if isinstance(time_period, TimePeriod) else time_period
    )

    return MetricResult(
        metric_name=metric_name,
        group_by=group_by_value,
        time_period=time_period_value,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=total,
        metadata=metadata,
        plots=plots or [],
    )


# =============================================================================
# 1. Jobs Completed Per Day
# =============================================================================


@register(
    "jobs_completed_per_day",
    "Jobs completed per man per day (by operative/trade/patch/region/time)",
)
@metric_timer("jobs_completed_per_day")
async def jobs_completed_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Aggregated results with grouping metadata and optional plots
    """
    metric_name = "jobs_completed_per_day"

    try:
        reduce_tool = tools.get("reduce")
        if not reduce_tool:
            logger.error(f"[{metric_name}] 'reduce' tool not found in tools dict")
            raise ValueError("Required 'reduce' tool not available")

        group_by_field = _resolve_group_by(group_by)
        filter_expr = _build_date_filter(COMPLETED_FILTER, start_date, end_date)

        logger.debug(f"[{metric_name}] Filter: {filter_expr}")
        logger.debug(f"[{metric_name}] Group by: {group_by_field}")

        # Count completed jobs grouped by dimension
        raw_result = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="completed jobs",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=filter_expr,
            group_by=group_by_field,
        )

        # Format results - normalize nested reduce output
        if isinstance(raw_result, dict):
            result = _normalize_grouped_result(raw_result, _extract_count)
            results = [{"group": k, "count": v} for k, v in result.items()]
            total = sum(result.values())
            logger.debug(f"[{metric_name}] Found {len(result)} groups, total={total}")
        else:
            count_val = _extract_count(raw_result)
            results = [{"group": "total", "count": count_val}]
            total = count_val
            logger.debug(f"[{metric_name}] Single total result: {total}")

        # Generate plots if requested
        group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
        plots = generate_plots(
            metric_name=metric_name,
            group_by=group_by_enum,
            project_name=get_active_project(),
            tables=REPAIRS_TABLE,
            filter_expr=filter_expr,
            include_plots=include_plots,
        )

        return _build_metric_result(
            metric_name=metric_name,
            group_by=group_by,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            results=results,
            total=float(total),
            plots=plots,
        )

    except Exception as e:
        logger.exception(f"[{metric_name}] Unexpected error during calculation")
        raise RuntimeError(f"Failed to calculate {metric_name}: {e}") from e


# =============================================================================
# 2. No Access Rate
# =============================================================================


@register(
    "no_access_rate",
    "No Access % / Absolute number (by operative/trade/patch/region/time)",
)
@metric_timer("no_access_rate")
async def no_access_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Rate or count of no-access jobs with optional plots
    """
    metric_name = "no_access_rate"

    try:
        reduce_tool = tools.get("reduce")
        if not reduce_tool:
            logger.error(f"[{metric_name}] 'reduce' tool not found")
            raise ValueError("Required 'reduce' tool not available")

        group_by_field = _resolve_group_by(group_by)
        logger.debug(
            f"[{metric_name}] return_absolute={return_absolute}, group_by={group_by_field}",
        )

        # Count no-access jobs
        no_access_filter = _build_date_filter(NO_ACCESS_FILTER, start_date, end_date)
        raw_no_access = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="no-access jobs",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=no_access_filter,
            group_by=group_by_field,
        )

        # Normalize results
        if isinstance(raw_no_access, dict):
            no_access_result = _normalize_grouped_result(raw_no_access, _extract_count)
            logger.debug(
                f"[{metric_name}] Normalized no-access: {len(no_access_result)} groups",
            )
        else:
            no_access_result = _extract_count(raw_no_access)
            logger.debug(
                f"[{metric_name}] Extracted no-access count: {no_access_result}",
            )

        if return_absolute:
            # Return absolute counts
            if isinstance(no_access_result, dict):
                results = [
                    {"group": k, "count": v} for k, v in no_access_result.items()
                ]
                total = sum(no_access_result.values())
            else:
                results = [{"group": "total", "count": no_access_result}]
                total = no_access_result
        else:
            # Calculate percentages - need total completed jobs
            logger.debug(
                f"[{metric_name}] Fetching total completed jobs for percentage calculation",
            )
            completed_filter = _build_date_filter(
                COMPLETED_FILTER,
                start_date,
                end_date,
            )
            raw_total = _timed_tool(
                metric_name,
                "reduce",
                reduce_tool,
                label="total completed",
                table=REPAIRS_TABLE,
                metric="count",
                keys="JobTicketReference",
                filter=completed_filter,
                group_by=group_by_field,
            )

            # Normalize total results
            if isinstance(raw_total, dict):
                total_result = _normalize_grouped_result(raw_total, _extract_count)
            else:
                total_result = _extract_count(raw_total)

            if isinstance(no_access_result, dict) and isinstance(total_result, dict):
                results = []
                for k in total_result:
                    na_count = no_access_result.get(k, 0)
                    tot_count = total_result.get(k, 0)
                    pct = _compute_percentage(na_count, tot_count)
                    results.append(
                        {
                            "group": k,
                            "percentage": pct,
                            "count": na_count,
                            "total": tot_count,
                        },
                    )
                total = _compute_percentage(
                    sum(no_access_result.values()),
                    sum(total_result.values()),
                )
            else:
                na_count = no_access_result if isinstance(no_access_result, int) else 0
                tot_count = total_result if isinstance(total_result, int) else 0
                pct = _compute_percentage(na_count, tot_count)
                results = [
                    {
                        "group": "total",
                        "percentage": pct,
                        "count": na_count,
                        "total": tot_count,
                    },
                ]
                total = pct

        # Generate plots if requested
        group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
        plots = generate_plots(
            metric_name=metric_name,
            group_by=group_by_enum,
            project_name=get_active_project(),
            tables=REPAIRS_TABLE,
            filter_expr=no_access_filter,
            include_plots=include_plots,
        )

        return _build_metric_result(
            metric_name=metric_name,
            group_by=group_by,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            results=results,
            total=float(total),
            metadata={"return_absolute": return_absolute},
            plots=plots,
        )

    except Exception as e:
        logger.exception(f"[{metric_name}] Unexpected error during calculation")
        raise RuntimeError(f"Failed to calculate {metric_name}: {e}") from e


# =============================================================================
# 3. First Time Fix Rate
# =============================================================================


@register(
    "first_time_fix_rate",
    "First Time Fix % / Absolute Number (by operative/trade/patch/region/time)",
)
@metric_timer("first_time_fix_rate")
async def first_time_fix_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Rate or count of first-time-fix jobs with optional plots
    """
    metric_name = "first_time_fix_rate"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_group_by(group_by)
    logger.debug(
        f"[{metric_name}] group_by_field={group_by_field}, return_absolute={return_absolute}",
    )

    # Count first-time-fix jobs
    ftf_filter = _build_date_filter(FIRST_TIME_FIX_FILTER, start_date, end_date)
    raw_ftf = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="first-time-fix jobs",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=ftf_filter,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_ftf, dict):
        ftf_result = _normalize_grouped_result(raw_ftf, _extract_count)
        logger.debug(f"[{metric_name}] Normalized FTF: {len(ftf_result)} groups")
    else:
        ftf_result = _extract_count(raw_ftf)
        logger.debug(f"[{metric_name}] Extracted FTF count: {ftf_result}")

    if return_absolute:
        if isinstance(ftf_result, dict):
            results = [{"group": k, "count": v} for k, v in ftf_result.items()]
            total = sum(ftf_result.values())
        else:
            results = [{"group": "total", "count": ftf_result}]
            total = ftf_result
        logger.debug(
            f"[{metric_name}] Returning absolute: total={total}, groups={len(results)}",
        )
    else:
        # Calculate percentages
        completed_filter = _build_date_filter(COMPLETED_FILTER, start_date, end_date)
        raw_total = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="total completed",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        # Normalize total results
        if isinstance(raw_total, dict):
            total_result = _normalize_grouped_result(raw_total, _extract_count)
        else:
            total_result = _extract_count(raw_total)

        if isinstance(ftf_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                ftf_count = ftf_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = _compute_percentage(ftf_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": ftf_count,
                        "total": tot_count,
                    },
                )
            total = _compute_percentage(
                sum(ftf_result.values()),
                sum(total_result.values()),
            )
        else:
            ftf_count = ftf_result if isinstance(ftf_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = _compute_percentage(ftf_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": ftf_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        filter_expr=ftf_filter,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name="first_time_fix_rate",
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        metadata={"return_absolute": return_absolute},
        plots=plots,
    )


# =============================================================================
# 4. Follow On Required Rate
# =============================================================================


@register(
    "follow_on_required_rate",
    "Follow on Required % / Absolute Number (by operative/trade/patch/region/time)",
)
@metric_timer("follow_on_required_rate")
async def follow_on_required_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Rate or count of jobs requiring follow-on with optional plots
    """
    metric_name = "follow_on_required_rate"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_group_by(group_by)

    # Count follow-on jobs
    fo_filter = _build_date_filter(FOLLOW_ON_FILTER, start_date, end_date)
    raw_fo = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="follow-on count",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=fo_filter,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_fo, dict):
        fo_result = _normalize_grouped_result(raw_fo, _extract_count)
    else:
        fo_result = _extract_count(raw_fo)

    if return_absolute:
        if isinstance(fo_result, dict):
            results = [{"group": k, "count": v} for k, v in fo_result.items()]
            total = sum(fo_result.values())
        else:
            results = [{"group": "total", "count": fo_result}]
            total = fo_result
    else:
        # Calculate percentages
        completed_filter = _build_date_filter(COMPLETED_FILTER, start_date, end_date)
        raw_total = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="total completed",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        # Normalize total results
        if isinstance(raw_total, dict):
            total_result = _normalize_grouped_result(raw_total, _extract_count)
        else:
            total_result = _extract_count(raw_total)

        if isinstance(fo_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                fo_count = fo_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = _compute_percentage(fo_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": fo_count,
                        "total": tot_count,
                    },
                )
            total = _compute_percentage(
                sum(fo_result.values()),
                sum(total_result.values()),
            )
        else:
            fo_count = fo_result if isinstance(fo_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = _compute_percentage(fo_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": fo_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        filter_expr=fo_filter,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name="follow_on_required_rate",
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        metadata={"return_absolute": return_absolute},
        plots=plots,
    )


# =============================================================================
# 5. Follow On Required for Materials Rate
# =============================================================================


@register(
    "follow_on_materials_rate",
    "Follow on Required for Materials % (by operative/trade/patch/region/time)",
)
@metric_timer("follow_on_materials_rate")
async def follow_on_materials_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Rate or count of jobs requiring follow-on due to materials with optional plots
    """
    # Note: This searches FollowOnDescription/FollowOnNotes for material-related keywords
    # This is a heuristic approach since there's no dedicated "materials" flag
    metric_name = "follow_on_materials_rate"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_group_by(group_by)

    # Get all follow-on jobs and search for material-related keywords in descriptions
    fo_filter = _build_date_filter(FOLLOW_ON_FILTER, start_date, end_date)

    # Count total follow-on jobs
    raw_total_fo = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="total follow-on",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=fo_filter,
        group_by=group_by_field,
    )

    # Normalize
    if isinstance(raw_total_fo, dict):
        total_fo = _normalize_grouped_result(raw_total_fo, _extract_count)
    else:
        total_fo = _extract_count(raw_total_fo)

    # For materials, we'd need to search FollowOnDescription - using filter for now
    # Material-related keywords: "material", "part", "order", "stock"
    materials_filter = _build_date_filter(
        f"({FOLLOW_ON_FILTER}) and (FollowOnDescription != 'None')",
        start_date,
        end_date,
    )
    raw_materials_fo = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="materials follow-on",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=materials_filter,
        group_by=group_by_field,
    )

    # Normalize
    if isinstance(raw_materials_fo, dict):
        materials_fo = _normalize_grouped_result(raw_materials_fo, _extract_count)
    else:
        materials_fo = _extract_count(raw_materials_fo)

    if return_absolute:
        if isinstance(materials_fo, dict):
            results = [{"group": k, "count": v} for k, v in materials_fo.items()]
            total = sum(materials_fo.values())
        else:
            results = [{"group": "total", "count": materials_fo}]
            total = materials_fo
    else:
        if isinstance(materials_fo, dict) and isinstance(total_fo, dict):
            results = []
            for k in total_fo:
                mat_count = materials_fo.get(k, 0)
                tot_count = total_fo.get(k, 0)
                pct = _compute_percentage(mat_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": mat_count,
                        "total_follow_on": tot_count,
                    },
                )
            total = _compute_percentage(
                sum(materials_fo.values()),
                sum(total_fo.values()),
            )
        else:
            mat_count = materials_fo if isinstance(materials_fo, int) else 0
            tot_count = total_fo if isinstance(total_fo, int) else 0
            pct = _compute_percentage(mat_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": mat_count,
                    "total_follow_on": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        filter_expr=materials_filter,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name="follow_on_materials_rate",
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        metadata={
            "return_absolute": return_absolute,
            "note": "Approximation based on FollowOnDescription presence",
        },
        plots=plots,
    )


# =============================================================================
# 6. Job Completed On Time Rate
# =============================================================================


@register(
    "job_completed_on_time_rate",
    "Job completed on time % / Absolute Number (by operative/trade/patch/region/time)",
)
@metric_timer("job_completed_on_time_rate")
async def job_completed_on_time_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Rate or count of jobs completed on time with optional plots
    """
    # Compare WorksOrderReportedCompletedDate vs WorksOrderTargetDate
    metric_name = "job_completed_on_time_rate"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_group_by(group_by)

    # Jobs completed on time: CompletedDate <= TargetDate
    on_time_filter = _build_date_filter(
        f"({COMPLETED_FILTER}) and `WorksOrderReportedCompletedDate` != 'None' and `WorksOrderTargetDate` != 'None' and `WorksOrderReportedCompletedDate` <= `WorksOrderTargetDate`",
        start_date,
        end_date,
    )
    raw_on_time = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="on-time count",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=on_time_filter,
        group_by=group_by_field,
    )

    # Normalize
    if isinstance(raw_on_time, dict):
        on_time_result = _normalize_grouped_result(raw_on_time, _extract_count)
    else:
        on_time_result = _extract_count(raw_on_time)

    if return_absolute:
        if isinstance(on_time_result, dict):
            results = [{"group": k, "count": v} for k, v in on_time_result.items()]
            total = sum(on_time_result.values())
        else:
            results = [{"group": "total", "count": on_time_result}]
            total = on_time_result
    else:
        # Calculate percentage of all completed jobs
        completed_filter = _build_date_filter(COMPLETED_FILTER, start_date, end_date)
        raw_total = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="total completed",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        # Normalize
        if isinstance(raw_total, dict):
            total_result = _normalize_grouped_result(raw_total, _extract_count)
        else:
            total_result = _extract_count(raw_total)

        if isinstance(on_time_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                ot_count = on_time_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = _compute_percentage(ot_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "on_time": ot_count,
                        "total": tot_count,
                    },
                )
            total = _compute_percentage(
                sum(on_time_result.values()),
                sum(total_result.values()),
            )
        else:
            ot_count = on_time_result if isinstance(on_time_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = _compute_percentage(ot_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "on_time": ot_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        filter_expr=on_time_filter,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name="job_completed_on_time_rate",
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        metadata={"return_absolute": return_absolute},
        plots=plots,
    )


# =============================================================================
# 7. Merchant Stops Per Day
# =============================================================================


@register(
    "merchant_stops_per_day",
    "Number of merchant stops per day (by operative/trade/patch/region/time)",
)
@metric_timer("merchant_stops_per_day")
async def merchant_stops_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Count of merchant stops grouped by dimension with optional plots
    """
    # Search telematics EndLocation for known merchant names
    metric_name = "merchant_stops_per_day"
    filter_files_tool = tools["filter_files"]

    # Build filter to match merchant locations
    merchant_filter_parts = [
        f"`EndLocation`.contains('{m}', case=False)" for m in MERCHANT_NAMES
    ]
    # Simplified: count rows where EndLocation mentions any merchant
    # Note: This is approximate as filter syntax may not support complex string matching

    total_stops = 0
    results_by_group: Dict[str, int] = {}

    for table in ALL_TELEMATICS_TABLES:
        for merchant in MERCHANT_NAMES:
            try:
                # Try to filter by each merchant name
                rows = _timed_tool(
                    metric_name,
                    "filter_files",
                    filter_files_tool,
                    label=f"{merchant}@{table.split('.')[-1]}",
                    filter=f"'{merchant}' in `EndLocation`",
                    tables=[table],
                    limit=1000,
                )
                if rows:
                    total_stops += len(rows)
                    for row in rows:
                        vehicle = row.get("Vehicle", "Unknown")
                        results_by_group[vehicle] = results_by_group.get(vehicle, 0) + 1
            except Exception:
                # Filter syntax may not support this; continue
                pass

    if results_by_group:
        results = [{"group": k, "stops": v} for k, v in results_by_group.items()]
    else:
        results = [
            {
                "group": "total",
                "stops": total_stops,
                "note": "Merchant detection requires location string matching",
            },
        ]

    # Generate plots if requested - for each month's telematics table
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=ALL_TELEMATICS_TABLES,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name="merchant_stops_per_day",
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total_stops),
        metadata={
            "merchants_searched": MERCHANT_NAMES,
            "note": "Approximate - based on EndLocation text matching",
        },
        plots=plots,
    )


# =============================================================================
# 8. Average Duration at Merchant
# =============================================================================


@register(
    "avg_duration_at_merchant",
    "Average duration at a Merchant per day (by operative/trade/patch/region/time)",
)
@metric_timer("avg_duration_at_merchant")
async def avg_duration_at_merchant(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Average duration (in minutes) at merchant stops with optional plots
    """
    metric_name = "avg_duration_at_merchant"

    # Note: Duration at merchant would require:
    # 1. Identifying merchant stops via EndLocation
    # 2. Calculating dwell time from arrival/departure timestamps
    # This is complex due to telematics data structure

    # Generate plots if requested - for each month's telematics table
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=ALL_TELEMATICS_TABLES,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name=metric_name,
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=[
            {
                "note": "Duration calculation requires timestamp parsing from telematics Arrival/Departure fields",
                "merchants_list": MERCHANT_NAMES,
            },
        ],
        total=0.0,
        metadata={
            "status": "requires_advanced_implementation",
            "reason": "Needs timestamp parsing and location matching logic",
        },
        plots=plots,
    )


# =============================================================================
# 9. Distance Travelled Per Day
# =============================================================================


@register(
    "distance_travelled_per_day",
    "Distance travelled per day (by operative/trade/patch/region/time)",
)
@metric_timer("distance_travelled_per_day")
async def distance_travelled_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Distance travelled (in miles/km) grouped by dimension with optional plots
    """
    metric_name = "distance_travelled_per_day"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # For telematics, group by Vehicle which contains operative name
    group_by_field = _resolve_telematics_group_by(group_by)

    # Filter for valid business distance values
    base_filter = "`Business distance` != 'None'"

    # Aggregate across all telematics tables (monthly)
    total_distance: Dict[str, float] = {}
    grand_total = 0.0

    for table in ALL_TELEMATICS_TABLES:
        raw_result = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label=f"distance sum ({table.split('.')[-1]})",
            table=table,
            metric="sum",
            keys="Business distance",
            filter=base_filter,
            group_by=group_by_field,
        )

        if isinstance(raw_result, dict):
            result = _normalize_grouped_result(raw_result, _extract_sum)
            for k, v in result.items():
                total_distance[k] = total_distance.get(k, 0.0) + v
        else:
            grand_total += _extract_sum(raw_result)

    if total_distance:
        results = [
            {"group": k, "distance_miles": round(v, 2)}
            for k, v in total_distance.items()
        ]
        grand_total = sum(total_distance.values())
    else:
        results = [{"group": "total", "distance_miles": round(grand_total, 2)}]

    # Generate plots if requested - for each month's telematics table
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=ALL_TELEMATICS_TABLES,
        filter_expr=base_filter,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name=metric_name,
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=round(grand_total, 2),
        metadata={"unit": "miles"},
        plots=plots,
    )


# =============================================================================
# 10. Average Time Travelling Per Day
# =============================================================================


@register(
    "avg_time_travelling",
    "Average time travelling per day (by operative/trade/patch/region/time)",
)
@metric_timer("avg_time_travelling")
async def avg_time_travelling(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Average travel time (in minutes) grouped by dimension with optional plots
    """
    # Telematics has "Trip travel time" in HH:MM:SS format
    # Summing string times requires parsing; return event count as proxy
    metric_name = "avg_time_travelling"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_telematics_group_by(group_by)

    total_trips: Dict[str, int] = {}
    grand_total = 0

    for table in ALL_TELEMATICS_TABLES:
        raw_result = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label=f"trip count ({table.split('.')[-1]})",
            table=table,
            metric="count",
            keys="Trip travel time",
            filter="`Trip travel time` != 'None'",
            group_by=group_by_field,
        )

        if isinstance(raw_result, dict):
            result = _normalize_grouped_result(raw_result, _extract_count)
            for k, v in result.items():
                total_trips[k] = total_trips.get(k, 0) + v
        else:
            grand_total += _extract_count(raw_result)

    if total_trips:
        results = [{"group": k, "trip_count": v} for k, v in total_trips.items()]
        grand_total = sum(total_trips.values())
    else:
        results = [{"group": "total", "trip_count": grand_total}]

    # Generate plots if requested - for each month's telematics table
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=ALL_TELEMATICS_TABLES,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name=metric_name,
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(grand_total),
        metadata={
            "note": "Returns trip count - actual time averaging requires HH:MM:SS parsing",
            "unit": "trip_events",
        },
        plots=plots,
    )


# =============================================================================
# 11. Repairs Completed Per Day
# =============================================================================


@register(
    "repairs_completed_per_day",
    "Repairs completed per day (total/by trade/patch/region/time)",
)
@metric_timer("repairs_completed_per_day")
async def repairs_completed_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.TOTAL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Total repair count grouped by dimension with optional plots
    """
    metric_name = "repairs_completed_per_day"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_group_by(group_by)
    filter_expr = _build_date_filter(COMPLETED_FILTER, start_date, end_date)

    raw_result = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="completed repairs",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=filter_expr,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_result, dict):
        result = _normalize_grouped_result(raw_result, _extract_count)
        results = [{"group": k, "count": v} for k, v in result.items()]
        total = sum(result.values())
    else:
        count_val = _extract_count(raw_result)
        results = [{"group": "total", "count": count_val}]
        total = count_val

    # Generate plots if requested
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        filter_expr=filter_expr,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name=metric_name,
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        plots=plots,
    )


# =============================================================================
# 12. Jobs Issued Per Day
# =============================================================================


@register(
    "jobs_issued_per_day",
    "Jobs issued per day (total/by trade/patch/region/time)",
)
@metric_timer("jobs_issued_per_day")
async def jobs_issued_per_day(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.TOTAL,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Count of jobs issued grouped by dimension with optional plots
    """
    metric_name = "jobs_issued_per_day"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_group_by(group_by)

    # Filter by WorksOrderIssuedDate for issued jobs
    filter_expr = _build_date_filter(
        ISSUED_FILTER,
        start_date,
        end_date,
        date_column="`WorksOrderIssuedDate`",
    )

    raw_result = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="issued jobs",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=filter_expr,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_result, dict):
        result = _normalize_grouped_result(raw_result, _extract_count)
        results = [{"group": k, "count": v} for k, v in result.items()]
        total = sum(result.values())
    else:
        count_val = _extract_count(raw_result)
        results = [{"group": "total", "count": count_val}]
        total = count_val

    # Generate plots if requested
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        filter_expr=filter_expr,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name=metric_name,
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        plots=plots,
    )


# =============================================================================
# 13. Jobs Requiring Materials Rate
# =============================================================================


@register(
    "jobs_requiring_materials_rate",
    "% of jobs completed that require materials (by operative/trade/patch/region/time)",
)
@metric_timer("jobs_requiring_materials_rate")
async def jobs_requiring_materials_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Rate or count of jobs requiring materials with optional plots
    """
    # Note: No dedicated "materials required" column exists
    # Proxy: jobs with FollowOnDescription mentioning materials/parts
    metric_name = "jobs_requiring_materials_rate"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    group_by_field = _resolve_group_by(group_by)

    # Count completed jobs with non-empty FollowOnDescription (proxy for materials)
    materials_filter = _build_date_filter(
        f"({COMPLETED_FILTER}) and `FollowOnDescription` != 'None' and `FollowOnDescription` != ''",
        start_date,
        end_date,
    )
    raw_materials = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="materials jobs",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=materials_filter,
        group_by=group_by_field,
    )

    # Normalize
    if isinstance(raw_materials, dict):
        materials_result = _normalize_grouped_result(raw_materials, _extract_count)
    else:
        materials_result = _extract_count(raw_materials)

    if return_absolute:
        if isinstance(materials_result, dict):
            results = [{"group": k, "count": v} for k, v in materials_result.items()]
            total = sum(materials_result.values())
        else:
            results = [{"group": "total", "count": materials_result}]
            total = materials_result
    else:
        completed_filter = _build_date_filter(COMPLETED_FILTER, start_date, end_date)
        raw_total = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="total completed",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        # Normalize
        if isinstance(raw_total, dict):
            total_result = _normalize_grouped_result(raw_total, _extract_count)
        else:
            total_result = _extract_count(raw_total)

        if isinstance(materials_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                mat_count = materials_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = _compute_percentage(mat_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": mat_count,
                        "total": tot_count,
                    },
                )
            total = _compute_percentage(
                sum(materials_result.values()),
                sum(total_result.values()),
            )
        else:
            mat_count = materials_result if isinstance(materials_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = _compute_percentage(mat_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": mat_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        filter_expr=materials_filter,
        include_plots=include_plots,
    )

    return _build_metric_result(
        metric_name="jobs_requiring_materials_rate",
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        metadata={
            "return_absolute": return_absolute,
            "note": "Proxy: jobs with FollowOnDescription (no dedicated materials column)",
        },
        plots=plots,
    )


# =============================================================================
# 14. Average Repairs Per Property
# =============================================================================


@register(
    "avg_repairs_per_property",
    "Average number of repairs per property completed (per operative/month/quarter/year)",
)
@metric_timer("avg_repairs_per_property")
async def avg_repairs_per_property(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    time_period: TimePeriod = TimePeriod.MONTH,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Average repairs per property, with list of properties having multiple repairs
        and optional plots
    """
    metric_name = "avg_repairs_per_property"
    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    filter_expr = _build_date_filter(COMPLETED_FILTER, start_date, end_date)

    # Group by FullAddress to count repairs per property
    raw_property_counts = _timed_tool(
        metric_name,
        "reduce",
        reduce_tool,
        label="property counts",
        table=REPAIRS_TABLE,
        metric="count",
        keys="JobTicketReference",
        filter=filter_expr,
        group_by="FullAddress",
    )

    if isinstance(raw_property_counts, dict):
        # Normalize the results
        property_counts = _normalize_grouped_result(raw_property_counts, _extract_count)

        # Calculate average repairs per property
        total_repairs = sum(property_counts.values())
        num_properties = len(property_counts)
        avg_repairs = (
            round(total_repairs / num_properties, 2) if num_properties > 0 else 0.0
        )

        # Find properties with multiple repairs (threshold: 2+)
        multi_repair_properties = {k: v for k, v in property_counts.items() if v >= 2}

        results = [
            {
                "total_repairs": total_repairs,
                "unique_properties": num_properties,
                "average_repairs_per_property": avg_repairs,
                "properties_with_multiple_repairs": len(multi_repair_properties),
            },
        ]

        # Generate plots if requested
        group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
        plots = generate_plots(
            metric_name=metric_name,
            group_by=group_by_enum,
            project_name=get_active_project(),
            tables=REPAIRS_TABLE,
            filter_expr=filter_expr,
            include_plots=include_plots,
        )

        return _build_metric_result(
            metric_name="avg_repairs_per_property",
            group_by=group_by,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            results=results,
            total=avg_repairs,
            metadata={
                "top_repeat_properties": sorted(
                    multi_repair_properties.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10],
            },
            plots=plots,
        )
    else:
        return _build_metric_result(
            metric_name="avg_repairs_per_property",
            group_by=group_by,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            results=[{"error": "Could not aggregate by property"}],
            total=0.0,
            plots=[],
        )


# =============================================================================
# 15. Complaints Rate
# =============================================================================


@register(
    "complaints_rate",
    "Complaints as % of total jobs completed (by operative/trade/patch/region/time)",
)
@metric_timer("complaints_rate")
async def complaints_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
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
    include_plots : bool
        If True, generate visualization URLs for the results

    Returns
    -------
    MetricResult
        Rate or count of complaints with optional plots
    """
    metric_name = "complaints_rate"

    # Generate plots if requested (will be empty for this unavailable metric)
    group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    plots = generate_plots(
        metric_name=metric_name,
        group_by=group_by_enum,
        project_name=get_active_project(),
        tables=REPAIRS_TABLE,
        include_plots=include_plots,
    )

    # Note: No complaints column exists in the repairs data
    # This metric cannot be calculated from the current dataset
    return _build_metric_result(
        metric_name=metric_name,
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=[
            {
                "error": "Complaints data not available in current dataset",
                "note": "Repairs data does not include a complaints column",
            },
        ],
        total=0.0,
        metadata={
            "status": "data_not_available",
            "required_column": "Complaints or similar",
        },
        plots=plots,
    )


# =============================================================================
# 16. Appointment Adherence Rate
# =============================================================================


@register(
    "appointment_adherence_rate",
    "Percentage of appointments where operative arrived within scheduled window "
    "(by operative/patch/region/time)",
)
@metric_timer("appointment_adherence_rate")
async def appointment_adherence_rate(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get appointment adherence rate as percentage or absolute number.

    This metric measures punctuality and scheduling reliability by calculating
    the percentage of repair appointments where the operative arrived within
    the scheduled appointment window.

    Business Context (from PDF Analysis):
        - Target: 95%+ on-time arrival
        - High adherence indicates reliable scheduling and respects tenants' time
        - Low adherence causes tenant frustration and can lead to no-access incidents
        - "Even if the job is ultimately completed on time, being late to an agreed
          appointment can frustrate tenants"

    Calculation Logic:
        - Compares ArrivedOnSite timestamp against ScheduledAppointmentStart/End window
        - A visit is "on-time" if arrival occurred between start and end of window
        - Visits where no arrival time is recorded are excluded from calculation

    Parameters
    ----------
    tools : ToolsDict
        Tools from FileManager containing reduce, filter_files, list_columns
    group_by : GroupBy
        Dimension to group results by (operative, patch, region, total)
    start_date : str, optional
        Start date filter (YYYY-MM-DD format)
    end_date : str, optional
        End date filter (YYYY-MM-DD format)
    time_period : TimePeriod
        Time granularity for aggregation (day, week, month, quarter, year)
    return_absolute : bool
        If True, return absolute counts of on-time/late arrivals
        If False, return percentage adherence rate

    Returns
    -------
    MetricResult
        Appointment adherence rate or counts per group, with:
        - percentage: Adherence rate (if return_absolute=False)
        - on_time_count: Number of on-time arrivals
        - late_count: Number of late arrivals
        - total_scheduled: Total scheduled appointments

    Example Usage
    -------------
    >>> result = await appointment_adherence_rate(
    ...     group_by=GroupBy.OPERATIVE,
    ...     return_absolute=False,
    ... )
    >>> # Result: {"results": [{"group": "John Smith", "percentage": 92.5, ...}], ...}
    """
    metric_name = "appointment_adherence_rate"

    try:
        reduce_tool = tools.get("reduce")
        filter_files_tool = tools.get("filter_files")

        if not reduce_tool:
            logger.error(f"[{metric_name}] 'reduce' tool not found")
            raise ValueError("Required 'reduce' tool not available")

        group_by_field = _resolve_group_by(group_by)
        logger.debug(
            f"[{metric_name}] return_absolute={return_absolute}, group_by={group_by_field}",
        )

        # Note: This metric requires ScheduledAppointmentStart, ScheduledAppointmentEnd,
        # and ArrivedOnSite columns. If these don't exist in the repairs data,
        # we need to check and return appropriate message.

        # First, check if the required columns exist
        list_columns_tool = tools.get("list_columns")
        if list_columns_tool:
            try:
                columns = list_columns_tool(table=REPAIRS_TABLE)
                required_cols = [
                    "ScheduledAppointmentStart",
                    "ScheduledAppointmentEnd",
                    "ArrivedOnSite",
                ]
                missing_cols = [
                    col for col in required_cols if col not in (columns or {})
                ]

                if missing_cols:
                    logger.warning(
                        f"[{metric_name}] Missing required columns: {missing_cols}. "
                        "Appointment adherence cannot be calculated.",
                    )
                    return _build_metric_result(
                        metric_name=metric_name,
                        group_by=group_by,
                        time_period=time_period,
                        start_date=start_date,
                        end_date=end_date,
                        results=[
                            {
                                "error": "Required columns not available",
                                "missing_columns": missing_cols,
                                "note": (
                                    "Appointment adherence requires "
                                    "ScheduledAppointmentStart, ScheduledAppointmentEnd, "
                                    "and ArrivedOnSite columns"
                                ),
                            },
                        ],
                        total=0.0,
                        metadata={"status": "data_not_available"},
                    )
            except Exception as e:
                logger.warning(f"[{metric_name}] Could not check columns: {e}")

        # Build filter for scheduled appointments with arrival times
        # Appointments must have both scheduled window and arrival recorded
        scheduled_filter = (
            "`ScheduledAppointmentStart` != '' and "
            "`ScheduledAppointmentStart` is not None and "
            "`ArrivedOnSite` != '' and "
            "`ArrivedOnSite` is not None"
        )
        base_filter = _build_date_filter(scheduled_filter, start_date, end_date)

        # Count total scheduled appointments
        raw_total = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="total scheduled",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=base_filter,
            group_by=group_by_field,
        )

        # Count on-time arrivals
        # An arrival is on-time if ArrivedOnSite <= ScheduledAppointmentEnd
        # (i.e., arrived before or at the end of the window)
        on_time_filter = (
            f"({base_filter}) and " "`ArrivedOnSite` <= `ScheduledAppointmentEnd`"
        )
        raw_on_time = _timed_tool(
            metric_name,
            "reduce",
            reduce_tool,
            label="on-time arrivals",
            table=REPAIRS_TABLE,
            metric="count",
            keys="JobTicketReference",
            filter=on_time_filter,
            group_by=group_by_field,
        )

        # Normalize results
        if isinstance(raw_total, dict):
            total_result = _normalize_grouped_result(raw_total, _extract_count)
        else:
            total_result = _extract_count(raw_total)

        if isinstance(raw_on_time, dict):
            on_time_result = _normalize_grouped_result(raw_on_time, _extract_count)
        else:
            on_time_result = _extract_count(raw_on_time)

        # Build results based on return_absolute flag
        if isinstance(total_result, dict) and isinstance(on_time_result, dict):
            results = []
            total_on_time = 0
            total_scheduled = 0

            for group_key in total_result:
                scheduled_count = total_result.get(group_key, 0)
                on_time_count = on_time_result.get(group_key, 0)
                late_count = scheduled_count - on_time_count

                total_on_time += on_time_count
                total_scheduled += scheduled_count

                if return_absolute:
                    results.append(
                        {
                            "group": group_key,
                            "on_time_count": on_time_count,
                            "late_count": late_count,
                            "total_scheduled": scheduled_count,
                        },
                    )
                else:
                    pct = _compute_percentage(on_time_count, scheduled_count)
                    results.append(
                        {
                            "group": group_key,
                            "percentage": pct,
                            "on_time_count": on_time_count,
                            "late_count": late_count,
                            "total_scheduled": scheduled_count,
                        },
                    )

            # Calculate overall total
            if return_absolute:
                total = float(total_on_time)
            else:
                total = _compute_percentage(total_on_time, total_scheduled)

        else:
            # Non-grouped result
            scheduled_count = total_result if isinstance(total_result, int) else 0
            on_time_count = on_time_result if isinstance(on_time_result, int) else 0
            late_count = scheduled_count - on_time_count

            if return_absolute:
                results = [
                    {
                        "group": "total",
                        "on_time_count": on_time_count,
                        "late_count": late_count,
                        "total_scheduled": scheduled_count,
                    },
                ]
                total = float(on_time_count)
            else:
                pct = _compute_percentage(on_time_count, scheduled_count)
                results = [
                    {
                        "group": "total",
                        "percentage": pct,
                        "on_time_count": on_time_count,
                        "late_count": late_count,
                        "total_scheduled": scheduled_count,
                    },
                ]
                total = pct

        # Generate plots if requested
        group_by_enum = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
        plots = generate_plots(
            metric_name=metric_name,
            group_by=group_by_enum,
            project_name=get_active_project(),
            tables=REPAIRS_TABLE,
            filter_expr=base_filter,
            include_plots=include_plots,
        )

        return _build_metric_result(
            metric_name=metric_name,
            group_by=group_by,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            results=results,
            total=float(total),
            metadata={"return_absolute": return_absolute},
            plots=plots,
        )

    except Exception as e:
        logger.exception(f"[{metric_name}] Unexpected error during calculation")
        raise RuntimeError(f"Failed to calculate {metric_name}: {e}") from e


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
    "appointment_adherence_rate",
]
