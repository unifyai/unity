"""Performance metrics queries for Midland Heart repairs.

Each metric is a registered async function with standard parameters for
grouping and time filtering.  Metrics can be broken down by:
- operative
- trade
- patch
- region
- day (temporal grouping)
- time_period

Metric Reference (from requirements)
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

All metrics use ``data_primitives`` (the DataManager async primitives
surface) and follow a consistent interface::

    async def repairs_kpi_*(
        data_primitives,
        group_by=None,
        start_date=None,
        end_date=None,
        time_period="day",
        include_plots=False,
    ) -> dict

Rate metrics additionally accept ``return_absolute=False``.
"""

from __future__ import annotations

from unity.function_manager.custom import custom_function

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

# Registry for skipped metrics (for documentation/introspection)
SKIPPED_METRICS: dict[str, str] = {
    "merchant_stops": "Blocked: requires exhaustive merchant name/address list",
    "merchant_dwell_time": "Blocked: requires exhaustive merchant name/address list",
    "travel_time": "Blocked: HH:MM:SS string parsing not supported by backend",
    "complaints_rate": "Blocked: no complaints column in available data",
}


# =============================================================================
# 1. Jobs Completed
# =============================================================================


@custom_function()
async def repairs_kpi_jobs_completed(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Count completed repair jobs.

    Domain context
    --------------
    Completed jobs are the primary productivity measure for repairs operations.
    They reflect operative throughput and can be segmented by the business dimensions
    used in reporting (operative/patch/region/day).

    This memoizable KPI expects ``data_primitives`` to be the **async** DataManager
    primitives surface (e.g. ``primitives.data``).  All DataManager calls are awaited.

    Filter expressions
    ------------------
    - Completed jobs: ``WorksOrderStatusDescription`` in ['Complete', 'Closed']

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).  Uses: ``describe_table``, ``reduce``,
        and optionally ``plot``.
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds applied via ``build_filter(...)``.
    time_period : str
        Included for output metadata.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)`` with keys:
        metric_name, group_by, time_period, start_date, end_date, results,
        total, metadata, plots.

    Example
    -------
    result = await repairs_kpi_jobs_completed(
        data_primitives=primitives.data,
        group_by="operative",
        start_date="2025-07-01",
        end_date="2025-11-30",
    )
    """
    metric_name = "jobs_completed"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    completed_filter = "`WorksOrderStatusDescription` in ['Complete', 'Closed']"
    filter_expr = build_filter([completed_filter], start_date, end_date)

    raw_result = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=filter_expr,
        group_by=group_by_field,
    )

    if isinstance(raw_result, dict):
        counts = normalize_grouped_result(raw_result)
        results = [{"group": k, "count": v} for k, v in counts.items()]
        total = sum(counts.values())
    else:
        count_val = extract_count(raw_result)
        results = [{"group": "total", "count": count_val}]
        total = count_val

    plots = []
    if include_plots and group_by_field:
        title = f"Jobs Completed by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=filter_expr,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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


@custom_function()
async def repairs_kpi_no_access_rate(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    return_absolute=False,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute the No Access rate (percentage) or No Access counts.

    Domain context
    --------------
    No Access measures repair visits where the operative could not gain access
    to the property (tenant not home, no answer, etc.).  It's a key driver of
    wasted time and schedule inefficiency.

    Filter expressions
    ------------------
    - No Access numerator: ``NoAccess`` != 'None' and ``NoAccess`` != ''
    - Completed denominator: ``WorksOrderStatusDescription`` in ['Complete', 'Closed']

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    time_period : str
        Included for output metadata.
    return_absolute : bool
        True: return No Access counts.  False: return percentage vs completed jobs.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "no_access_rate"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    no_access_filter = build_filter(
        ["`NoAccess` != 'None' and `NoAccess` != ''"],
        start_date,
        end_date,
    )
    raw_no_access = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=no_access_filter,
        group_by=group_by_field,
    )

    if isinstance(raw_no_access, dict):
        no_access_result = normalize_grouped_result(raw_no_access)
    else:
        no_access_result = extract_count(raw_no_access)

    if return_absolute:
        if isinstance(no_access_result, dict):
            results = [{"group": k, "count": v} for k, v in no_access_result.items()]
            total = sum(no_access_result.values())
        else:
            results = [{"group": "total", "count": no_access_result}]
            total = no_access_result
    else:
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = await data_primitives.reduce(
            context,
            metric="count",
            columns="JobTicketReference",
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

    plots = []
    if include_plots and group_by_field:
        title = f"No-Access Rate by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=no_access_filter,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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


@custom_function()
async def repairs_kpi_first_time_fix_rate(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    return_absolute=False,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute the First Time Fix (FTF) rate (percentage) or FTF counts.

    Domain context
    --------------
    First Time Fix measures the percentage of repair jobs completed
    successfully on the first visit without requiring a follow-up.
    Higher FTF correlates with better customer experience and lower cost-to-serve.

    Filter expressions
    ------------------
    - FTF numerator: ``FirstTimeFix`` == 'Yes'
    - Completed denominator: ``WorksOrderStatusDescription`` in ['Complete', 'Closed']

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    time_period : str
        Included for output metadata.
    return_absolute : bool
        True: return FTF counts.  False: return FTF percentage vs completed jobs.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "first_time_fix_rate"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    ftf_filter = build_filter(["`FirstTimeFix` == 'Yes'"], start_date, end_date)
    raw_ftf = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=ftf_filter,
        group_by=group_by_field,
    )

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
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = await data_primitives.reduce(
            context,
            metric="count",
            columns="JobTicketReference",
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

    plots = []
    if include_plots and group_by_field:
        title = f"First-Time Fix by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=ftf_filter,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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


@custom_function()
async def repairs_kpi_follow_on_required_rate(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    return_absolute=False,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute the Follow On Required rate (percentage) or Follow On counts.

    Domain context
    --------------
    Follow-on required measures repairs where additional work was needed after
    the initial visit.  It can indicate diagnosis quality, parts availability,
    or complexity of the work.

    Filter expressions
    ------------------
    - Follow-on numerator (completed only): ``FollowOn`` == 'Yes' and
      ``WorksOrderStatusDescription`` in ['Complete', 'Closed']
    - Completed denominator: ``WorksOrderStatusDescription`` in ['Complete', 'Closed']

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    time_period : str
        Included for output metadata.
    return_absolute : bool
        True: return Follow On counts.  False: percentage vs completed jobs.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "follow_on_required_rate"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    fo_filter = build_filter(
        [
            "`FollowOn` == 'Yes'",
            "`WorksOrderStatusDescription` in ['Complete', 'Closed']",
        ],
        start_date,
        end_date,
    )
    raw_fo = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=fo_filter,
        group_by=group_by_field,
    )

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
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = await data_primitives.reduce(
            context,
            metric="count",
            columns="JobTicketReference",
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

    plots = []
    if include_plots and group_by_field:
        title = f"Follow-On Required by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=fo_filter,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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
# 5. Follow On Required for Materials Rate
# =============================================================================


@custom_function()
async def repairs_kpi_follow_on_materials_rate(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    return_absolute=False,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute the materials-related follow-on rate (percentage) or counts.

    Domain context
    --------------
    Materials-related follow-ons highlight cases where extra visits were driven
    by parts/materials constraints.  Useful for supply-chain and van stock
    optimization.

    Filter expressions
    ------------------
    - Materials numerator: ``FollowOn`` == 'Yes' and
      'MATERIALS REQUIRED' in ``FollowOnDescription``
    - Follow-on denominator: ``FollowOn`` == 'Yes'

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    time_period : str
        Included for output metadata.
    return_absolute : bool
        True: return materials-follow-on counts.  False: percentage vs all follow-on jobs.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "follow_on_materials_rate"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    fo_filter = build_filter(["`FollowOn` == 'Yes'"], start_date, end_date)
    raw_total_fo = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=fo_filter,
        group_by=group_by_field,
    )

    if isinstance(raw_total_fo, dict):
        total_fo = normalize_grouped_result(raw_total_fo)
    else:
        total_fo = extract_count(raw_total_fo)

    materials_filter = build_filter(
        ["`FollowOn` == 'Yes' and 'MATERIALS REQUIRED' in `FollowOnDescription`"],
        start_date,
        end_date,
    )
    raw_materials_fo = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
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

    plots = []
    if include_plots and group_by_field:
        title = f"Follow-On Materials by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=materials_filter,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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
            "note": "Filters for 'MATERIALS REQUIRED' in FollowOnDescription",
        },
        plots=plots,
    )


# =============================================================================
# 6. Job Completed On Time Rate
# =============================================================================


@custom_function()
async def repairs_kpi_job_completed_on_time_rate(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    return_absolute=False,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute the "completed on time" rate (percentage) or counts.

    Domain context
    --------------
    This KPI is a proxy for SLA compliance: completing jobs by their target dates.

    Filter expressions
    ------------------
    - On-time numerator:
      (``WorksOrderStatusDescription`` in ['Complete', 'Closed']) and
      ``WorksOrderReportedCompletedDate`` != 'None' and
      ``WorksOrderReportedCompletedDate`` is not None and
      ``WorksOrderTargetDate`` != 'None' and
      ``WorksOrderTargetDate`` is not None and
      ``WorksOrderReportedCompletedDate`` <= ``WorksOrderTargetDate``
    - Completed denominator: ``WorksOrderStatusDescription`` in ['Complete', 'Closed']

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    time_period : str
        Included for output metadata.
    return_absolute : bool
        True: return on-time counts.  False: return on-time percentage.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "job_completed_on_time_rate"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    on_time_condition = (
        "(`WorksOrderStatusDescription` in ['Complete', 'Closed']) "
        "and `WorksOrderReportedCompletedDate` != 'None' "
        "and `WorksOrderReportedCompletedDate` is not None "
        "and `WorksOrderTargetDate` != 'None' "
        "and `WorksOrderTargetDate` is not None "
        "and `WorksOrderReportedCompletedDate` <= `WorksOrderTargetDate`"
    )
    on_time_filter = build_filter([on_time_condition], start_date, end_date)
    raw_on_time = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
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
        completed_filter = build_filter(
            ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
            start_date,
            end_date,
        )
        raw_total = await data_primitives.reduce(
            context,
            metric="count",
            columns="JobTicketReference",
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

    plots = []
    if include_plots and group_by_field:
        title = f"On-Time Completions by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=on_time_filter,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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
# 9. Total Distance Travelled
# =============================================================================


@custom_function()
async def repairs_kpi_total_distance_travelled(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Sum business miles from telematics data (optionally grouped).

    Domain context
    --------------
    Total distance travelled is a productivity/cost proxy for field operations.
    This KPI aggregates telematics mileage to understand travel overhead and
    routing efficiency.

    Filter expressions
    ------------------
    - Base filter: ``Business distance`` != 'None'

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        For telematics grouping, ``resolve_group_by(..., telematics=True)`` is used.
        Supported: "operative" (Vehicle), "day" (Arrival_Date), or None.
    start_date, end_date : str | None
        Currently unused for telematics tables (kept for API consistency).
    time_period : str
        Included for output metadata.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "total_distance_travelled"

    telematics_info = await discover_telematics_tables(data_primitives)
    telematics_tables = [
        t.get("table")
        for t in (telematics_info or [])
        if isinstance(t, dict) and t.get("table")
    ]
    if not telematics_tables:
        raise ValueError("Failed to discover telematics tables")

    group_by_field = resolve_group_by(group_by, telematics=True)

    base_filter = "`Business distance` != 'None'"

    total_distance: dict = {}
    grand_total = 0.0

    for table in telematics_tables:
        raw_result = await data_primitives.reduce(
            table,
            metric="sum",
            columns="Business distance",
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

    plots = []
    if include_plots and group_by_field:
        title = f"Distance Travelled by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                telematics_tables[0],
                plot_type="bar",
                x="Driver",
                aggregate="sum",
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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
# 12. Jobs Issued
# =============================================================================


@custom_function()
async def repairs_kpi_jobs_issued(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Count issued jobs (incoming demand).

    Domain context
    --------------
    Jobs issued is an upstream demand measure for workload analysis.
    It helps track inflow versus completion throughput.

    Filter expressions
    ------------------
    - Issued jobs: ``WorksOrderStatusDescription`` == 'Issued'

    Date filtering
    --------------
    Uses ``build_filter(..., date_column="WorksOrderIssuedDate")`` for
    start/end date bounds.

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
        Day grouping uses ``resolve_group_by(..., date_context="issued")``.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds applied to ``WorksOrderIssuedDate``.
    time_period : str
        Included for output metadata.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "jobs_issued"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by, date_context="issued")

    filter_expr = build_filter(
        ["`WorksOrderStatusDescription` == 'Issued'"],
        start_date,
        end_date,
        date_column="WorksOrderIssuedDate",
    )

    raw_result = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=filter_expr,
        group_by=group_by_field,
    )

    if isinstance(raw_result, dict):
        counts = normalize_grouped_result(raw_result)
        results = [{"group": k, "count": v} for k, v in counts.items()]
        total = sum(counts.values())
    else:
        count_val = extract_count(raw_result)
        results = [{"group": "total", "count": count_val}]
        total = count_val

    plots = []
    if include_plots and group_by_field:
        title = f"Jobs Issued by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=filter_expr,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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


@custom_function()
async def repairs_kpi_jobs_requiring_materials_rate(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    return_absolute=False,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute the % of completed jobs that required materials (proxy-based).

    Domain context
    --------------
    This KPI approximates materials impact when a dedicated "materials required"
    column is not present, using ``FollowOnDescription`` as a proxy signal.

    Filter expressions
    ------------------
    - Materials proxy numerator:
      (``WorksOrderStatusDescription`` in ['Complete', 'Closed']) and
      ``FollowOnDescription`` != 'None' and ``FollowOnDescription`` != ''
    - Completed denominator: ``WorksOrderStatusDescription`` in ['Complete', 'Closed']

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    time_period : str
        Included for output metadata.
    return_absolute : bool
        True: return numerator counts.  False: return percentage.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "jobs_requiring_materials_rate"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    materials_condition = (
        "(`WorksOrderStatusDescription` in ['Complete', 'Closed']) and "
        "`FollowOnDescription` != 'None' and `FollowOnDescription` != ''"
    )
    materials_filter = build_filter([materials_condition], start_date, end_date)
    raw_materials = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
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
        raw_total = await data_primitives.reduce(
            context,
            metric="count",
            columns="JobTicketReference",
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

    plots = []
    if include_plots and group_by_field:
        title = f"Jobs Requiring Materials by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=materials_filter,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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
            "note": "Proxy: jobs with FollowOnDescription",
        },
        plots=plots,
    )


# =============================================================================
# 14. Average Repairs Per Property
# =============================================================================


@custom_function()
async def repairs_kpi_avg_repairs_per_property(
    data_primitives,
    group_by=None,
    time_period="month",
    start_date=None,
    end_date=None,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute average repairs per property (completed jobs only).

    Domain context
    --------------
    Identifies properties with repeat repairs and provides a coarse signal for
    underlying asset issues, workmanship quality, or recurring defects.

    Filter expressions
    ------------------
    - Completed jobs: ``WorksOrderStatusDescription`` in ['Complete', 'Closed']
    - Groups by ``FullAddress`` to get repairs-per-property counts
    - Computes average: total_repairs / unique_properties
    - Metadata includes ``repeat_properties`` (properties with 2+ repairs)

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        Included for output metadata (this KPI always groups internally
        by FullAddress).
    time_period : str
        Included for output metadata.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    include_plots : bool
        Currently ignored (no plots for this KPI).

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "avg_repairs_per_property"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by)

    filter_expr = build_filter(
        ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
        start_date,
        end_date,
    )

    raw_property_counts = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=filter_expr,
        group_by="FullAddress",
    )

    if isinstance(raw_property_counts, dict):
        property_counts = normalize_grouped_result(raw_property_counts)

        total_repairs = sum(property_counts.values())
        num_properties = len(property_counts)
        avg_repairs = (
            round(total_repairs / num_properties, 2) if num_properties > 0 else 0.0
        )

        multi_repair_properties = {k: v for k, v in property_counts.items() if v >= 2}

        results = [
            {
                "total_repairs": total_repairs,
                "unique_properties": num_properties,
                "average_repairs_per_property": avg_repairs,
                "properties_with_multiple_repairs": len(multi_repair_properties),
            },
        ]

        return build_metric_result(
            metric_name=metric_name,
            group_by=group_by,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            results=results,
            total=avg_repairs,
            metadata={
                "repeat_properties": [
                    {"property": k, "count": v}
                    for k, v in sorted(
                        multi_repair_properties.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ],
            },
            plots=[],
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
# 16. Appointment Adherence Rate
# =============================================================================


@custom_function()
async def repairs_kpi_appointment_adherence_rate(
    data_primitives,
    group_by=None,
    start_date=None,
    end_date=None,
    time_period="day",
    return_absolute=False,
    include_plots=False,
):
    """SKILL_KIND: KPI | DOMAIN: repairs

    Compute appointment adherence (arrived within scheduled window).

    Domain context
    --------------
    Appointment adherence measures punctuality and scheduling reliability:
    arriving within the scheduled appointment window.

    Filter expressions
    ------------------
    Universe ("scheduled appointments with arrival times"):
    - ScheduledAppointmentStart != '' and != 'None' and is not None
    - ScheduledAppointmentStart_Date not in ['1900-01-02', 'None', '']
    - ArrivedOnSite != '' and != 'None' and is not None

    On-time condition:
    - ``ArrivedOnSite`` <= ``ScheduledAppointmentEnd``

    Parameters
    ----------
    data_primitives
        DataManager primitives (async).
    group_by : str | None
        One of: "operative", "patch", "region", "trade", "day", or None.
        Day grouping uses ``resolve_group_by(..., date_context="scheduled")``.
    start_date, end_date : str | None
        Optional YYYY-MM-DD bounds.
    time_period : str
        Included for output metadata.
    return_absolute : bool
        True: return on-time/scheduled counts.  False: return adherence percentage.
    include_plots : bool
        If True and ``group_by`` is provided, attempts a bar plot.

    Returns
    -------
    dict
        Standard metric dict from ``build_metric_result(...)``.
    """
    metric_name = "appointment_adherence_rate"

    table_info = await discover_repairs_table(data_primitives)
    context = table_info.get("table") if isinstance(table_info, dict) else None
    if not context:
        raise ValueError("Failed to discover repairs table")

    group_by_field = resolve_group_by(group_by, date_context="scheduled")

    # Check if required columns exist
    try:
        desc = await data_primitives.describe_table(context)
        if desc:
            col_names = {
                getattr(c, "name", "") for c in getattr(desc, "columns", []) or []
            }
            required_cols = [
                "ScheduledAppointmentStart",
                "ScheduledAppointmentEnd",
                "ArrivedOnSite",
            ]
            missing_cols = [c for c in required_cols if c not in col_names]
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
                            "note": "Appointment adherence requires ScheduledAppointmentStart, ScheduledAppointmentEnd, and ArrivedOnSite columns",
                        },
                    ],
                    total=0.0,
                    metadata={"status": "data_not_available"},
                )
    except Exception:
        pass

    scheduled_condition = (
        "`ScheduledAppointmentStart` != '' and "
        "`ScheduledAppointmentStart` != 'None' and "
        "`ScheduledAppointmentStart` is not None and "
        "`ScheduledAppointmentStart_Date` not in ['1900-01-02', 'None', ''] and "
        "`ArrivedOnSite` != '' and "
        "`ArrivedOnSite` != 'None' and "
        "`ArrivedOnSite` is not None"
    )
    base_filter = build_filter([scheduled_condition], start_date, end_date)

    raw_total = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=base_filter,
        group_by=group_by_field,
    )

    on_time_filter = f"({base_filter}) and `ArrivedOnSite` <= `ScheduledAppointmentEnd`"
    raw_on_time = await data_primitives.reduce(
        context,
        metric="count",
        columns="JobTicketReference",
        filter=on_time_filter,
        group_by=group_by_field,
    )

    if isinstance(raw_total, dict):
        total_result = normalize_grouped_result(raw_total)
    else:
        total_result = extract_count(raw_total)

    if isinstance(raw_on_time, dict):
        on_time_result = normalize_grouped_result(raw_on_time)
    else:
        on_time_result = extract_count(raw_on_time)

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
        total = (
            float(total_on_time)
            if return_absolute
            else compute_percentage(total_on_time, total_scheduled)
        )
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

    plots = []
    if include_plots and group_by_field:
        title = f"Scheduled Jobs by {(group_by or 'Group').title()}"
        try:
            plot_result = await data_primitives.plot(
                context,
                plot_type="bar",
                x=group_by_field,
                aggregate="count",
                filter=base_filter,
                title=title,
            )
            plots.append(
                {
                    "url": extract_plot_url(plot_result),
                    "title": title,
                    "succeeded": extract_plot_succeeded(plot_result),
                },
            )
        except Exception as e:
            plots.append(
                {"url": None, "title": title, "succeeded": False, "error": str(e)},
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
# ALL METRICS registry
# =============================================================================

ALL_METRICS = [
    "repairs_kpi_jobs_completed",
    "repairs_kpi_no_access_rate",
    "repairs_kpi_first_time_fix_rate",
    "repairs_kpi_follow_on_required_rate",
    "repairs_kpi_follow_on_materials_rate",
    "repairs_kpi_job_completed_on_time_rate",
    "repairs_kpi_total_distance_travelled",
    "repairs_kpi_jobs_issued",
    "repairs_kpi_jobs_requiring_materials_rate",
    "repairs_kpi_avg_repairs_per_property",
    "repairs_kpi_appointment_adherence_rate",
]
