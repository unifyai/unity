"""
Performance metrics queries for repairs.

Each metric is a registered async function with standard parameters for
grouping and time filtering. Metrics can be broken down by:
- operative
- trade
- patch
- region
- day (temporal grouping)
- time_period

Metric Reference (from requirements):
--------------------------------------
1.  jobs_completed - Jobs completed (groupable by operative/patch/region/day)
2.  no_access_rate - No Access % / Absolute number
3.  first_time_fix_rate - First Time Fix % / Absolute Number
4.  follow_on_required_rate - Follow on Required % / Absolute Number
5.  follow_on_materials_rate - Follow on Required for Materials %
6.  job_completed_on_time_rate - Job completed on time % / Absolute Number
7.  merchant_stops - No of merchant stops (SKIPPED - no merchant list)
8.  merchant_dwell_time - Average duration at merchant (SKIPPED - no merchant list)
9.  total_distance_travelled - Total distance travelled
10. travel_time - Travel time (SKIPPED - HH:MM:SS parsing not supported)
11. jobs_issued - Jobs issued per day
12. jobs_requiring_materials_rate - % of jobs completed that require materials
13. avg_repairs_per_property - Average no of repairs per property completed
14. complaints_rate - Complaints % (SKIPPED - no data column)
15. appointment_adherence_rate - Appointment adherence rate
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, Dict, List, Optional

from intranet.repairs_agent.static.registry import register


# =============================================================================
# @skip Decorator for Placeholder Metrics
# =============================================================================

# Registry for skipped metrics (for documentation/introspection)
SKIPPED_METRICS: dict[str, str] = {}


def skip(metric_id: str, reason: str):
    """
    Mark a metric as skipped (not registered) with a documented reason.

    The function is still defined for reference but won't appear in
    the query registry or be executable via scripts.

    Usage:
        @skip("merchant_dwell_time", "Blocked: no exhaustive merchant list available")
        async def merchant_dwell_time(...): ...
    """

    def decorator(fn: Callable) -> Callable:
        SKIPPED_METRICS[metric_id] = reason

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            raise NotImplementedError(f"Metric '{metric_id}' is skipped: {reason}")

        wrapper._skipped = True
        wrapper._skip_reason = reason
        return wrapper

    return decorator


# Import types from local module (canonical source)
from .types import GroupBy, MetricResult, PlotResult, TimePeriod, FileTools

# Import ONLY helper functions - NO global constants
# Note: Plot configs are INLINE in each metric for CodeActActor visibility
# All filter expressions and column mappings are INLINE in each metric
from .helpers import (
    build_filter,
    build_metric_result,
    compute_percentage,
    discover_repairs_table,
    discover_telematics_tables,
    extract_count,
    extract_plot_succeeded,
    extract_plot_url,
    extract_sum,
    normalize_grouped_result,
    resolve_group_by,
)


# =============================================================================
# 1. Jobs Completed
# =============================================================================


@register(
    "jobs_completed",
    "Total jobs completed (groupable by operative/patch/region/day)",
)
async def jobs_completed(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get total jobs completed.

    This metric measures operative productivity by counting the total number
    of jobs completed, grouped by the specified dimension.
    Used for performance benchmarking and workload analysis.

    Discovery Pattern
    -----------------
    table_info = discover_repairs_table(tools)
    # Returns: {"table": "<path>", "description": "...", "columns": [...]}
    repairs_table = table_info["table"]

    Tool Chain
    ----------
    1. reduce(table=repairs_table, metric="count", keys="JobTicketReference",
              filter="`WorksOrderStatusDescription` in ['Complete', 'Closed']",
              group_by="OperativeWhoCompletedJob")
       → Returns: {"John Smith": 150, "Jane Doe": 120, ...}

    Filter Expressions (INLINE - no globals)
    ----------------------------------------
    - Completed jobs: `WorksOrderStatusDescription` in ['Complete', 'Closed']

    Column Mappings (INLINE - no globals)
    -------------------------------------
    - "operative" → "OperativeWhoCompletedJob"
    - "patch" → "RepairsPatch"
    - "region" → "RepairsRegion"
    - "day" → "WorksOrderReportedCompletedDateDay"

    Parameters
    ----------
    tools : FileTools
        Tools from FileManager (reduce, filter_files, visualize, tables_overview, schema_explain)
    group_by : GroupBy | str
        "operative", "patch", "region", "day", or None for total
    start_date : str, optional
        Start date filter (YYYY-MM-DD format)
    end_date : str, optional
        End date filter (YYYY-MM-DD format)
    time_period : TimePeriod
        Time granularity for aggregation
    include_plots : bool
        If True, generate visualization URLs

    Returns
    -------
    MetricResult
        Aggregated results with grouping metadata and optional plots
    """
    metric_name = "jobs_completed"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discover repairs table
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by to column name
    group_by_str = group_by.value if hasattr(group_by, "value") else str(group_by)
    group_by_field = resolve_group_by(group_by_str)

    # Filter for completed jobs
    completed_filter = "`WorksOrderStatusDescription` in ['Complete', 'Closed']"
    filter_expr = build_filter([completed_filter], start_date, end_date)

    # Query: count completed jobs
    raw_result = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=filter_expr,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_result, dict):
        counts = normalize_grouped_result(raw_result)
        results = [{"group": k, "count": v} for k, v in counts.items()]
        total = sum(counts.values())
    else:
        count_val = extract_count(raw_result)
        results = [{"group": "total", "count": count_val}]
        total = count_val

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=filter_expr,
                    title=f"Jobs Completed by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"Jobs Completed by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"Jobs Completed by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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
# 2. No Access Rate
# =============================================================================


@register(
    "no_access_rate",
    "No Access % / Absolute number (by operative/trade/patch/region/time)",
)
async def no_access_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get No Access rate as percentage or absolute number.

    No Access measures repair visits where the operative could not gain access
    to the property (tenant not home, no answer, etc.).

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    columns = tools["list_columns"](table=repairs_table)
    # Required: NoAccess, JobTicketReference

    Tool Chain
    ----------
    1. reduce(table=repairs_table, metric="count", keys="JobTicketReference",
              filter="`NoAccess` != 'None' and `NoAccess` != ''", group_by=...)
    2. reduce(table=repairs_table, metric="count", keys="JobTicketReference",
              filter=COMPLETED_FILTER, group_by=...) for percentage
    3. Python: percentage = (no_access / total) * 100

    Filter Expressions
    ------------------
    - No Access: `NoAccess` != 'None' and `NoAccess` != ''
    - Completed: `WorksOrderStatusDescription` in ['Complete', 'Closed']

    Parameters
    ----------
    tools : FileTools
        FileManager tools
    group_by : GroupBy | str
        "operative", "patch", "region", or "total"
    return_absolute : bool
        If True return counts, if False return percentages
    """
    metric_name = "no_access_rate"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str)

    # Query: count no-access jobs
    no_access_filter = build_filter(
        ["`NoAccess` != 'None' and `NoAccess` != ''"],
        start_date,
        end_date,
    )
    raw_no_access = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=no_access_filter,
        group_by=group_by_field,
    )

    # Normalize no-access results
    if isinstance(raw_no_access, dict):
        no_access_result = normalize_grouped_result(raw_no_access)
    else:
        no_access_result = extract_count(raw_no_access)

    if return_absolute:
        # Return absolute counts
        if isinstance(no_access_result, dict):
            results = [{"group": k, "count": v} for k, v in no_access_result.items()]
            total = sum(no_access_result.values())
        else:
            results = [{"group": "total", "count": no_access_result}]
            total = no_access_result
    else:
        # Calculate percentages - need total completed jobs
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = reduce_tool(
            table=repairs_table,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        if isinstance(raw_total, dict):
            total_result = normalize_grouped_result(raw_total)
        else:
            total_result = extract_count(raw_total)

        if isinstance(no_access_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                na_count = no_access_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = compute_percentage(na_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": na_count,
                        "total": tot_count,
                    },
                )
            total = compute_percentage(
                sum(no_access_result.values()),
                sum(total_result.values()),
            )
        else:
            na_count = no_access_result if isinstance(no_access_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = compute_percentage(na_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": na_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=no_access_filter,
                    title=f"No-Access Rate by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"No-Access Rate by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"No-Access Rate by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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


# =============================================================================
# 3. First Time Fix Rate
# =============================================================================


@register(
    "first_time_fix_rate",
    "First Time Fix % / Absolute Number (by operative/trade/patch/region/time)",
)
async def first_time_fix_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get First Time Fix rate as percentage or absolute number.

    First Time Fix (FTF) measures the percentage of repair jobs completed
    successfully on the first visit without requiring a follow-up appointment.

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    columns = tools["list_columns"](table=repairs_table)
    # Required: FirstTimeFix, JobTicketReference, WorksOrderStatusDescription

    Tool Chain
    ----------
    1. reduce(table=repairs_table, metric="count", keys="JobTicketReference",
              filter="`FirstTimeFix` == 'Yes'", group_by=...)
    2. reduce(table=repairs_table, metric="count", keys="JobTicketReference",
              filter=COMPLETED_FILTER, group_by=...) for percentage
    3. Python: percentage = (ftf / total) * 100

    Filter Expressions
    ------------------
    - First Time Fix: `FirstTimeFix` == 'Yes'
    - Completed: `WorksOrderStatusDescription` in ['Complete', 'Closed']

    Parameters
    ----------
    tools : FileTools
        FileManager tools
    group_by : GroupBy | str
        "operative", "patch", "region", or "total"
    return_absolute : bool
        If True return counts, if False return percentages
    """
    metric_name = "first_time_fix_rate"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str)

    # Query: count first-time-fix jobs
    ftf_filter = build_filter(["`FirstTimeFix` == 'Yes'"], start_date, end_date)
    raw_ftf = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=ftf_filter,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_ftf, dict):
        ftf_result = normalize_grouped_result(raw_ftf)
    else:
        ftf_result = extract_count(raw_ftf)

    if return_absolute:
        if isinstance(ftf_result, dict):
            results = [{"group": k, "count": v} for k, v in ftf_result.items()]
            total = sum(ftf_result.values())
        else:
            results = [{"group": "total", "count": ftf_result}]
            total = ftf_result
    else:
        # Calculate percentages
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = reduce_tool(
            table=repairs_table,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        if isinstance(raw_total, dict):
            total_result = normalize_grouped_result(raw_total)
        else:
            total_result = extract_count(raw_total)

        if isinstance(ftf_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                ftf_count = ftf_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = compute_percentage(ftf_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": ftf_count,
                        "total": tot_count,
                    },
                )
            total = compute_percentage(
                sum(ftf_result.values()),
                sum(total_result.values()),
            )
        else:
            ftf_count = ftf_result if isinstance(ftf_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = compute_percentage(ftf_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": ftf_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=ftf_filter,
                    title=f"First-Time Fix by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"First-Time Fix by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"First-Time Fix by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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


# =============================================================================
# 4. Follow On Required Rate
# =============================================================================


@register(
    "follow_on_required_rate",
    "Follow on Required % / Absolute Number (by operative/trade/patch/region/time)",
)
async def follow_on_required_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get Follow On Required rate as percentage or absolute number.

    Follow-on required measures repairs where additional work was needed after
    the initial visit.

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    columns = tools["list_columns"](table=repairs_table)
    # Required: FollowOn, JobTicketReference, WorksOrderStatusDescription

    Tool Chain
    ----------
    1. reduce(table=repairs_table, metric="count", keys="JobTicketReference",
              filter="`FollowOn` == 'Yes'", group_by=...)
    2. reduce(..., filter=COMPLETED_FILTER, ...) for percentage
    3. Python: percentage = (follow_on / total) * 100

    Filter Expressions
    ------------------
    - Follow-on: `FollowOn` == 'Yes'
    - Completed: `WorksOrderStatusDescription` in ['Complete', 'Closed']
    """
    metric_name = "follow_on_required_rate"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str)

    # Count follow-on jobs (among completed jobs only - must match denominator universe)
    fo_filter = build_filter(
        [
            "`FollowOn` == 'Yes'",
            "`WorksOrderStatusDescription` in ['Complete', 'Closed']",
        ],
        start_date,
        end_date,
    )
    raw_fo = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=fo_filter,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_fo, dict):
        fo_result = normalize_grouped_result(raw_fo)
    else:
        fo_result = extract_count(raw_fo)

    if return_absolute:
        if isinstance(fo_result, dict):
            results = [{"group": k, "count": v} for k, v in fo_result.items()]
            total = sum(fo_result.values())
        else:
            results = [{"group": "total", "count": fo_result}]
            total = fo_result
    else:
        # Calculate percentages
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = reduce_tool(
            table=repairs_table,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        if isinstance(raw_total, dict):
            total_result = normalize_grouped_result(raw_total)
        else:
            total_result = extract_count(raw_total)

        if isinstance(fo_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                fo_count = fo_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = compute_percentage(fo_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": fo_count,
                        "total": tot_count,
                    },
                )
            total = compute_percentage(
                sum(fo_result.values()),
                sum(total_result.values()),
            )
        else:
            fo_count = fo_result if isinstance(fo_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = compute_percentage(fo_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": fo_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=fo_filter,
                    title=f"Follow-On Required by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"Follow-On Required by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"Follow-On Required by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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
async def follow_on_materials_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get Follow On Required specifically for Materials as percentage or absolute.

    Measures follow-on jobs caused by material/parts issues.

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    # Required: FollowOn, FollowOnDescription, JobTicketReference

    Tool Chain
    ----------
    1. reduce(..., filter="`FollowOn` == 'Yes' and `FollowOnDescription` != 'None'")
    2. reduce(..., filter="`FollowOn` == 'Yes'") for total
    3. Python: percentage = (materials_fo / total_fo) * 100

    Filter Expressions
    ------------------
    - Materials follow-on: `FollowOn` == 'Yes' and `FollowOnDescription` != 'None'
    - All follow-on: `FollowOn` == 'Yes'
    """
    metric_name = "follow_on_materials_rate"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str)

    # Count total follow-on jobs
    fo_filter = build_filter(["`FollowOn` == 'Yes'"], start_date, end_date)
    raw_total_fo = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=fo_filter,
        group_by=group_by_field,
    )

    if isinstance(raw_total_fo, dict):
        total_fo = normalize_grouped_result(raw_total_fo)
    else:
        total_fo = extract_count(raw_total_fo)

    # Count materials-related follow-on jobs
    # Filter: follow-on jobs where description contains 'MATERIALS REQUIRED'
    materials_filter = build_filter(
        ["`FollowOn` == 'Yes' and 'MATERIALS REQUIRED' in `FollowOnDescription`"],
        start_date,
        end_date,
    )
    raw_materials_fo = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=materials_filter,
        group_by=group_by_field,
    )

    if isinstance(raw_materials_fo, dict):
        materials_fo = normalize_grouped_result(raw_materials_fo)
    else:
        materials_fo = extract_count(raw_materials_fo)

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
                pct = compute_percentage(mat_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": mat_count,
                        "total_follow_on": tot_count,
                    },
                )
            total = compute_percentage(
                sum(materials_fo.values()),
                sum(total_fo.values()),
            )
        else:
            mat_count = materials_fo if isinstance(materials_fo, int) else 0
            tot_count = total_fo if isinstance(total_fo, int) else 0
            pct = compute_percentage(mat_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": mat_count,
                    "total_follow_on": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=materials_filter,
                    title=f"Follow-On Materials by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"Follow-On Materials by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"Follow-On Materials by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
        metric_name="follow_on_materials_rate",
        group_by=group_by,
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        results=results,
        total=float(total),
        metadata={
            "return_absolute": return_absolute,
            "note": "Filters for 'MATERIALS REQUIRED' in FollowOnDescription",
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
async def job_completed_on_time_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get Job Completed On Time rate as percentage or absolute number.

    SLA compliance: CompletedDate <= TargetDate.

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    # Required: WorksOrderReportedCompletedDate, WorksOrderTargetDate, JobTicketReference

    Tool Chain
    ----------
    1. reduce(..., filter="COMPLETED and CompletedDate <= TargetDate")
    2. reduce(..., filter=COMPLETED_FILTER) for percentage
    3. Python: percentage = (on_time / total) * 100
    """
    metric_name = "job_completed_on_time_rate"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str)

    # Jobs completed on time: CompletedDate <= TargetDate
    # Filter: completed jobs where completion date is on or before target
    on_time_condition = (
        "(`WorksOrderStatusDescription` in ['Complete', 'Closed']) "
        "and `WorksOrderReportedCompletedDate` != 'None' "
        "and `WorksOrderTargetDate` != 'None' "
        "and `WorksOrderReportedCompletedDate` <= `WorksOrderTargetDate`"
    )
    on_time_filter = build_filter([on_time_condition], start_date, end_date)
    raw_on_time = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=on_time_filter,
        group_by=group_by_field,
    )

    if isinstance(raw_on_time, dict):
        on_time_result = normalize_grouped_result(raw_on_time)
    else:
        on_time_result = extract_count(raw_on_time)

    if return_absolute:
        if isinstance(on_time_result, dict):
            results = [{"group": k, "count": v} for k, v in on_time_result.items()]
            total = sum(on_time_result.values())
        else:
            results = [{"group": "total", "count": on_time_result}]
            total = on_time_result
    else:
        # Calculate percentage
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = reduce_tool(
            table=repairs_table,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        if isinstance(raw_total, dict):
            total_result = normalize_grouped_result(raw_total)
        else:
            total_result = extract_count(raw_total)

        if isinstance(on_time_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                ot_count = on_time_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = compute_percentage(ot_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "on_time": ot_count,
                        "total": tot_count,
                    },
                )
            total = compute_percentage(
                sum(on_time_result.values()),
                sum(total_result.values()),
            )
        else:
            ot_count = on_time_result if isinstance(on_time_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = compute_percentage(ot_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "on_time": ot_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=on_time_filter,
                    title=f"On-Time Completions by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"On-Time Completions by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"On-Time Completions by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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
# 7. Merchant Stops (SKIPPED)
# =============================================================================


@skip("merchant_stops", "Blocked: requires exhaustive merchant name/address list")
async def merchant_stops(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get number of merchant stops.

    STATUS: SKIPPED - requires exhaustive merchant name/address list.

    Uses telematics data to count trips ending at merchant locations.

    Discovery Pattern
    -----------------
    telematics_tables = discover_telematics_tables(tools)
    # Required: EndLocation, Vehicle

    Tool Chain
    ----------
    1. filter_files(..., filter="'MerchantName' in `EndLocation`")
    2. Python: Aggregate counts by Vehicle
    """
    # This metric is skipped - decorator will raise NotImplementedError


# =============================================================================
# 8. Merchant Dwell Time (SKIPPED)
# =============================================================================


@skip("merchant_dwell_time", "Blocked: requires exhaustive merchant name/address list")
async def merchant_dwell_time(
    tools: FileTools,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get average duration at a merchant.

    STATUS: SKIPPED - requires exhaustive merchant name/address list.
    Timestamp subtraction (Departure - Arrival) IS supported once merchant
    locations are identified.

    Calculates time spent at merchant locations from telematics data.
    Requires parsing arrival/departure timestamps for dwell time calculation.
    """
    # This metric is skipped - decorator will raise NotImplementedError


# =============================================================================
# 9. Total Distance Travelled
# =============================================================================


@register(
    "total_distance_travelled",
    "Total distance travelled (groupable by vehicle/day)",
)
async def total_distance_travelled(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get total distance travelled.

    Aggregates business miles from telematics data.

    Discovery Pattern
    -----------------
    telematics_tables = discover_telematics_tables(tools)
    # Required: Business distance, Vehicle

    Tool Chain
    ----------
    1. reduce(table=telematics_table, metric="sum", keys="Business distance", group_by="Vehicle")
    2. Python: Aggregate across monthly tables

    Column Mappings
    ---------------
    - "operative" → "Vehicle"
    - "day" → "ArrivalDay"
    """
    metric_name = "total_distance_travelled"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    telematics_tables = [t["table"] for t in discover_telematics_tables(tools)]

    # For telematics, group by Vehicle column
    group_by_str = group_by.value if hasattr(group_by, "value") else str(group_by)
    group_by_field = resolve_group_by(group_by_str, telematics=True)

    base_filter = "`Business distance` != 'None'"

    # Aggregate across all telematics tables
    total_distance: Dict[str, float] = {}
    grand_total = 0.0

    for table in telematics_tables:
        raw_result = reduce_tool(
            table=table,
            metric="sum",
            keys="Business distance",
            filter=base_filter,
            group_by=group_by_field,
        )

        if isinstance(raw_result, dict):
            result = normalize_grouped_result(raw_result, extract_sum)
            for k, v in result.items():
                total_distance[k] = total_distance.get(k, 0.0) + v
        else:
            grand_total += extract_sum(raw_result)

    if total_distance:
        results = [
            {"group": k, "distance_miles": round(v, 2)}
            for k, v in total_distance.items()
        ]
        grand_total = sum(total_distance.values())
    else:
        results = [{"group": "total", "distance_miles": round(grand_total, 2)}]

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool:
            try:
                # Inline plot config - visible to CodeActActor
                # Telematics: x_axis="Driver", y_axis="Total distance", metric="sum"
                result = visualize_tool(
                    tables=telematics_tables,
                    plot_type="bar",
                    x_axis="Driver",
                    y_axis="Total distance",
                    group_by="Driver",
                    metric="sum",
                    filter=base_filter,
                    title=f"Distance Travelled by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"Distance Travelled by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"Distance Travelled by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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
# 10. Travel Time (SKIPPED)
# =============================================================================


@skip("travel_time", "Blocked: HH:MM:SS string parsing not supported by backend")
async def travel_time(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get travel time.

    STATUS: SKIPPED - Trip travel time column is HH:MM:SS string format
    which cannot be aggregated by the backend.

    Would calculate average time spent travelling from telematics data.
    """
    # This metric is skipped - decorator will raise NotImplementedError


# =============================================================================
# 11. (DELETED - repairs_completed_per_day was duplicate of jobs_completed)
# =============================================================================


# =============================================================================
# 12. Jobs Issued
# =============================================================================


@register(
    "jobs_issued",
    "Jobs issued (groupable by operative/patch/region/day)",
)
async def jobs_issued(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get jobs issued.

    Incoming demand metric for workload analysis.

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    # Required: JobTicketReference, WorksOrderStatusDescription

    Tool Chain
    ----------
    1. reduce(table=repairs_table, metric="count", keys="JobTicketReference",
              filter=ISSUED_FILTER, group_by=...)

    Filter Expressions
    ------------------
    - Issued: `WorksOrderStatusDescription` == 'Issued'

    Column Mappings
    ---------------
    - "day" → "WorksOrderIssuedDateDay" (via date_context="issued")
    """
    metric_name = "jobs_issued"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by (use date_context="issued" for day grouping)
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str, date_context="issued")

    # Filter by WorksOrderIssuedDate
    filter_expr = build_filter(
        ["`WorksOrderStatusDescription` == 'Issued'"],
        start_date,
        end_date,
        date_column="WorksOrderIssuedDate",
    )

    raw_result = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=filter_expr,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_result, dict):
        counts = normalize_grouped_result(raw_result)
        results = [{"group": k, "count": v} for k, v in counts.items()]
        total = sum(counts.values())
    else:
        count_val = extract_count(raw_result)
        results = [{"group": "total", "count": count_val}]
        total = count_val

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=filter_expr,
                    title=f"Jobs Issued by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"Jobs Issued by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"Jobs Issued by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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
async def jobs_requiring_materials_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get percentage of completed jobs that required materials.

    Uses FollowOnDescription as proxy (no dedicated materials column).

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    # Required: FollowOnDescription, JobTicketReference, WorksOrderStatusDescription

    Tool Chain
    ----------
    1. reduce(..., filter=COMPLETED and FollowOnDescription present)
    2. reduce(..., filter=COMPLETED) for percentage
    3. Python: percentage = (materials / total) * 100

    Filter Expressions
    ------------------
    - Materials proxy: FollowOnDescription != 'None' and != ''
    - Completed: `WorksOrderStatusDescription` in ['Complete', 'Closed']
    """
    metric_name = "jobs_requiring_materials_rate"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str)

    # Count jobs with FollowOnDescription (proxy for materials)
    # Filter: completed jobs with follow-on description (proxy for materials)
    materials_condition = (
        "(`WorksOrderStatusDescription` in ['Complete', 'Closed']) and "
        "`FollowOnDescription` != 'None' and `FollowOnDescription` != ''"
    )
    materials_filter = build_filter([materials_condition], start_date, end_date)
    raw_materials = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=materials_filter,
        group_by=group_by_field,
    )

    if isinstance(raw_materials, dict):
        materials_result = normalize_grouped_result(raw_materials)
    else:
        materials_result = extract_count(raw_materials)

    if return_absolute:
        if isinstance(materials_result, dict):
            results = [{"group": k, "count": v} for k, v in materials_result.items()]
            total = sum(materials_result.values())
        else:
            results = [{"group": "total", "count": materials_result}]
            total = materials_result
    else:
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = reduce_tool(
            table=repairs_table,
            metric="count",
            keys="JobTicketReference",
            filter=completed_filter,
            group_by=group_by_field,
        )

        if isinstance(raw_total, dict):
            total_result = normalize_grouped_result(raw_total)
        else:
            total_result = extract_count(raw_total)

        if isinstance(materials_result, dict) and isinstance(total_result, dict):
            results = []
            for k in total_result:
                mat_count = materials_result.get(k, 0)
                tot_count = total_result.get(k, 0)
                pct = compute_percentage(mat_count, tot_count)
                results.append(
                    {
                        "group": k,
                        "percentage": pct,
                        "count": mat_count,
                        "total": tot_count,
                    },
                )
            total = compute_percentage(
                sum(materials_result.values()),
                sum(total_result.values()),
            )
        else:
            mat_count = materials_result if isinstance(materials_result, int) else 0
            tot_count = total_result if isinstance(total_result, int) else 0
            pct = compute_percentage(mat_count, tot_count)
            results = [
                {
                    "group": "total",
                    "percentage": pct,
                    "count": mat_count,
                    "total": tot_count,
                },
            ]
            total = pct

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=materials_filter,
                    title=f"Jobs Requiring Materials by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"Jobs Requiring Materials by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"Jobs Requiring Materials by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
        metric_name=metric_name,
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
async def avg_repairs_per_property(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    time_period: TimePeriod = TimePeriod.MONTH,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get average number of repairs per property.

    Identifies properties with multiple repairs.

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    # Required: FullAddress, JobTicketReference, WorksOrderStatusDescription

    Tool Chain
    ----------
    1. reduce(..., filter=COMPLETED, group_by="FullAddress")
       → {property: count}
    2. Python: avg = total_repairs / unique_properties
    """
    metric_name = "avg_repairs_per_property"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    filter_expr = build_filter(
        ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
        start_date,
        end_date,
    )

    # Group by FullAddress to count repairs per property
    raw_property_counts = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=filter_expr,
        group_by="FullAddress",
    )

    if isinstance(raw_property_counts, dict):
        property_counts = normalize_grouped_result(raw_property_counts)

        # Calculate average
        total_repairs = sum(property_counts.values())
        num_properties = len(property_counts)
        avg_repairs = (
            round(total_repairs / num_properties, 2) if num_properties > 0 else 0.0
        )

        # Find properties with multiple repairs
        multi_repair_properties = {k: v for k, v in property_counts.items() if v >= 2}

        results = [
            {
                "total_repairs": total_repairs,
                "unique_properties": num_properties,
                "average_repairs_per_property": avg_repairs,
                "properties_with_multiple_repairs": len(multi_repair_properties),
            },
        ]

        # Note: This metric groups by FullAddress internally, not by user's group_by
        # Plots are not supported for this metric
        plots: List[PlotResult] = []

        return build_metric_result(
            metric_name=metric_name,
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
        return build_metric_result(
            metric_name=metric_name,
            group_by=group_by,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            results=[{"error": "Could not aggregate by property"}],
            total=0.0,
            plots=[],
        )


# =============================================================================
# 15. Complaints Rate (SKIPPED)
# =============================================================================


@skip("complaints_rate", "Blocked: no complaints column in available data")
async def complaints_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get complaints as percentage of total jobs completed.

    STATUS: SKIPPED - Complaints data not currently available in repairs dataset.

    Would calculate complaints as percentage of completed jobs if data available.
    """
    # This metric is skipped - decorator will raise NotImplementedError


# =============================================================================
# 16. Appointment Adherence Rate
# =============================================================================


@register(
    "appointment_adherence_rate",
    "Percentage of appointments where operative arrived within scheduled window "
    "(by operative/patch/region/time)",
)
async def appointment_adherence_rate(
    tools: FileTools,
    group_by: Optional[GroupBy | str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
    """
    Get appointment adherence rate as percentage or absolute number.

    Measures punctuality: arrived within scheduled window.

    Discovery Pattern
    -----------------
    repairs_table = discover_repairs_table(tools)
    columns = tools["list_columns"](table=repairs_table)
    # Required: ArrivedOnSite, ScheduledAppointmentStart, ScheduledAppointmentEnd

    Tool Chain
    ----------
    1. reduce(..., filter="scheduled and arrived") → total scheduled
    2. reduce(..., filter="arrived <= end_window") → on-time
    3. Python: adherence = (on_time / total) * 100

    Filter Expressions
    ------------------
    - Scheduled: ScheduledAppointmentStart non-empty
    - Arrived: ArrivedOnSite non-empty
    - On-time: ArrivedOnSite <= ScheduledAppointmentEnd
    """
    metric_name = "appointment_adherence_rate"

    reduce_tool = tools.get("reduce")
    if not reduce_tool:
        raise ValueError("Required 'reduce' tool not available")

    # Discovery with fallback
    repairs_table = discover_repairs_table(tools)["table"]

    # Resolve group_by (use date_context="scheduled" for day grouping since
    # this metric is about scheduled appointments, not completion dates)
    group_by_str = (
        group_by.value
        if hasattr(group_by, "value")
        else str(group_by) if group_by else None
    )
    group_by_field = resolve_group_by(group_by_str, date_context="scheduled")

    # Check if required columns exist
    list_columns_tool = tools.get("list_columns")
    if list_columns_tool:
        try:
            columns = list_columns_tool(table=repairs_table)
            required_cols = [
                "ScheduledAppointmentStart",
                "ScheduledAppointmentEnd",
                "ArrivedOnSite",
            ]
            missing_cols = [col for col in required_cols if col not in (columns or {})]
            if missing_cols:
                return build_metric_result(
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
        except Exception:
            pass  # Proceed anyway if column check fails

    # Filter for scheduled appointments with arrival times
    # Also exclude garbage dates like 1900-01-02 from source data
    scheduled_condition = (
        "`ScheduledAppointmentStart` != '' and "
        "`ScheduledAppointmentStart` is not None and "
        "`ScheduledAppointmentStartDay` not in ['1900-01-02', 'None', ''] and "
        "`ArrivedOnSite` != '' and "
        "`ArrivedOnSite` is not None"
    )
    base_filter = build_filter([scheduled_condition], start_date, end_date)

    # Count total scheduled appointments
    raw_total = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=base_filter,
        group_by=group_by_field,
    )

    # Count on-time arrivals
    on_time_filter = f"({base_filter}) and `ArrivedOnSite` <= `ScheduledAppointmentEnd`"
    raw_on_time = reduce_tool(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=on_time_filter,
        group_by=group_by_field,
    )

    # Normalize results
    if isinstance(raw_total, dict):
        total_result = normalize_grouped_result(raw_total)
    else:
        total_result = extract_count(raw_total)

    if isinstance(raw_on_time, dict):
        on_time_result = normalize_grouped_result(raw_on_time)
    else:
        on_time_result = extract_count(raw_on_time)

    # Build results
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
                pct = compute_percentage(on_time_count, scheduled_count)
                results.append(
                    {
                        "group": group_key,
                        "percentage": pct,
                        "on_time_count": on_time_count,
                        "late_count": late_count,
                        "total_scheduled": scheduled_count,
                    },
                )

        if return_absolute:
            total = float(total_on_time)
        else:
            total = compute_percentage(total_on_time, total_scheduled)
    else:
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
            pct = compute_percentage(on_time_count, scheduled_count)
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

    # Generate plots if requested (skip if no grouping)
    plots: List[PlotResult] = []
    if include_plots and group_by is not None:
        visualize_tool = tools.get("visualize")
        if visualize_tool and group_by_field:
            try:
                # Inline plot config - visible to CodeActActor
                result = visualize_tool(
                    tables=repairs_table,
                    plot_type="bar",
                    x_axis=group_by_field,
                    y_axis="JobTicketReference",
                    group_by=group_by_field,
                    metric="count",
                    filter=base_filter,
                    title=f"Scheduled Jobs by {group_by_str.title()}",
                )
                if result:
                    plots.append(
                        PlotResult(
                            url=extract_plot_url(result),
                            title=f"Scheduled Jobs by {group_by_str.title()}",
                            succeeded=extract_plot_succeeded(result),
                        ),
                    )
            except Exception as e:
                plots.append(
                    PlotResult(
                        title=f"Scheduled Jobs by {group_by_str.title()}",
                        error=str(e),
                        succeeded=False,
                    ),
                )

    return build_metric_result(
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


# =============================================================================
# Utility: List all registered metrics
# =============================================================================

# Fully implemented metrics (registered via @register)
ALL_METRICS = [
    "jobs_completed",
    "no_access_rate",
    "first_time_fix_rate",
    "follow_on_required_rate",
    "follow_on_materials_rate",
    "job_completed_on_time_rate",
    "total_distance_travelled",
    "jobs_issued",
    "jobs_requiring_materials_rate",
    "avg_repairs_per_property",
    "appointment_adherence_rate",
]

# Skipped metrics are tracked in SKIPPED_METRICS dict (populated by @skip decorator):
# - merchant_stops: No exhaustive merchant name/address list
# - merchant_dwell_time: No exhaustive merchant name/address list
# - travel_time: HH:MM:SS string parsing not supported by backend
# - complaints_rate: No complaints column in available data
