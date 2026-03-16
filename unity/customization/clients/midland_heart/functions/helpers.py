"""Shared helper functions for Midland Heart repairs KPI metrics.

All helpers are registered via ``@custom_function()`` so the dependency
graph is resolved automatically -- metric functions reference these by
bare name without explicit imports.

DESIGN PRINCIPLES:
1. Discovery-first: Use describe_table() to discover schema and table contexts
2. No magic globals: All filter expressions and column mappings are inline
3. Rich return values: Discovery functions return table + description + columns
4. Pure functions: No logging or side effects

API Surface (data_primitives):
- describe_table(context) -> TableDescription with .columns
- list_tables(prefix=...) -> list of table context paths
- reduce(context, metric=..., columns=..., filter=..., group_by=...) -> value
- plot(context, plot_type=..., x=..., aggregate=..., ...) -> PlotResult
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from unity.function_manager.custom import custom_function

# =============================================================================
# DISCOVERY FUNCTIONS -- Return rich schema info
# =============================================================================


@custom_function()
async def discover_repairs_table(data_primitives) -> Optional[Dict[str, Any]]:
    """Discover the Midland Heart repairs table (context path + schema).

    Uses the well-known context ``MidlandHeart/Repairs2025`` seeded by the
    MH seed script.  Returns a rich dict so the calling metric can confirm
    the table exists and inspect its columns.

    Returns
    -------
    dict or None
        ``{"table": <context path>, "description": str,
          "columns": [{"name": str, "description": str}, ...]}``
        Returns ``None`` when the context does not exist.

    Example
    -------
    >>> info = await discover_repairs_table(data_primitives)
    >>> info["table"]
    'MidlandHeart/Repairs2025'
    """
    context = "MidlandHeart/Repairs2025"
    try:
        desc = await data_primitives.describe_table(context)
    except Exception:
        return None

    columns: List[Dict[str, str]] = []
    if desc:
        for col in getattr(desc, "columns", []) or []:
            col_name = getattr(col, "name", "") or ""
            col_desc = getattr(col, "description", None) or ""
            if col_name:
                columns.append({"name": col_name, "description": col_desc})

    return {
        "table": context,
        "description": getattr(desc, "description", "") or "",
        "columns": columns,
    }


@custom_function()
async def discover_telematics_tables(data_primitives) -> List[Dict[str, Any]]:
    """Discover all monthly telematics tables (context paths + schemas).

    Lists tables under ``MidlandHeart/Telematics2025/`` and enriches each
    with column metadata via ``describe_table()``.

    Returns
    -------
    list[dict]
        Each entry: ``{"table": <context path>, "table_name": str,
        "description": str, "columns": [{"name": str, "description": str}, ...]}``
    """
    prefix = "MidlandHeart/Telematics2025"
    try:
        raw_tables = await data_primitives.list_tables(prefix=prefix)
    except Exception:
        return []

    if not isinstance(raw_tables, list):
        return []

    result: List[Dict[str, Any]] = []
    for entry in raw_tables:
        ctx = entry if isinstance(entry, str) else getattr(entry, "context", str(entry))
        if not ctx:
            continue

        columns: List[Dict[str, str]] = []
        description = ""
        try:
            desc = await data_primitives.describe_table(ctx)
            if desc:
                description = getattr(desc, "description", "") or ""
                for col in getattr(desc, "columns", []) or []:
                    col_name = getattr(col, "name", "") or ""
                    col_desc = getattr(col, "description", None) or ""
                    if col_name:
                        columns.append({"name": col_name, "description": col_desc})
        except Exception:
            pass

        table_name = ctx.rsplit("/", 1)[-1] if "/" in ctx else ctx
        result.append(
            {
                "table": ctx,
                "table_name": table_name,
                "description": description,
                "columns": columns,
            },
        )

    return result


# =============================================================================
# INLINE HELPERS -- No magic globals, everything visible
# =============================================================================


@custom_function()
def resolve_group_by(
    group_by: Optional[str] = None,
    telematics: bool = False,
    date_context: str = "completion",
) -> Optional[str]:
    """Resolve a human-friendly group_by string to the actual column name.

    The mapping is INLINE so the CodeActActor can see exactly which
    columns are used for each grouping dimension.

    Parameters
    ----------
    group_by : str or None
        Group dimension: "operative", "patch", "region", "trade", "day".
        Pass None for no grouping (aggregate total).
    telematics : bool
        If True, use telematics column mappings (Vehicle instead of Operative).
    date_context : str
        Context for temporal grouping to select appropriate date column:
        - "completion": WorksOrderReportedCompletedDateDay (default)
        - "issued": WorksOrderIssuedDateDay
        - "arrival": ArrivedOnSiteDay
        - "scheduled": ScheduledAppointmentStartDay

    Returns
    -------
    str or None
        Column name for grouping, or None if group_by is None.

    Example
    -------
    >>> resolve_group_by("operative")
    'OperativeWhoCompletedJob'
    >>> resolve_group_by("operative", telematics=True)
    'Vehicle'
    >>> resolve_group_by("day", date_context="completion")
    'WorksOrderReportedCompletedDateDay'
    >>> resolve_group_by(None)
    None
    """
    if group_by is None:
        return None

    key = group_by.lower()

    date_columns = {
        "completion": "WorksOrderReportedCompletedDateDay",
        "issued": "WorksOrderIssuedDateDay",
        "arrival": "ArrivedOnSiteDay",
        "scheduled": "ScheduledAppointmentStartDay",
    }

    telematics_mapping = {
        "operative": "Vehicle",
        "day": "ArrivalDay",
    }

    repairs_mapping = {
        "operative": "OperativeWhoCompletedJob",
        "patch": "RepairsPatch",
        "region": "RepairsRegion",
        "trade": "Trade",
        "day": date_columns.get(date_context, "WorksOrderReportedCompletedDateDay"),
    }

    mapping = telematics_mapping if telematics else repairs_mapping
    return mapping.get(key)


# =============================================================================
# FILTER BUILDERS
# =============================================================================


@custom_function()
def build_filter(
    base_conditions: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_column: str = "VisitDate",
) -> str:
    """Build a filter expression from conditions and optional date range.

    Parameters
    ----------
    base_conditions : list[str]
        List of filter conditions (will be AND-ed together).
        Pass the ACTUAL FILTER STRINGS, not global constant names.
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.
    date_column : str
        Column name for date filtering (default: VisitDate).

    Returns
    -------
    str
        Combined filter expression (empty string when nothing to filter).

    Example
    -------
    >>> build_filter(
    ...     ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
    ...     start_date="2025-07-01",
    ... )
    "(`WorksOrderStatusDescription` in ['Complete', 'Closed']) and `VisitDate` >= '2025-07-01'"
    """
    parts = [f"({c})" for c in base_conditions if c]
    if start_date:
        parts.append(f"`{date_column}` >= '{start_date}'")
    if end_date:
        parts.append(f"`{date_column}` <= '{end_date}'")
    return " and ".join(parts) if parts else ""


# =============================================================================
# RESULT EXTRACTORS
# =============================================================================


@custom_function()
def extract_count(value: Any) -> int:
    """Extract count from a reduce result.

    Handles various return formats:
    - Direct int/float: ``123`` -> ``123``
    - Dict with count: ``{"count": 123}`` -> ``123``
    - Dict with shared_value: ``{"shared_value": None, "count": 123}`` -> ``123``
    - None -> ``0``

    Parameters
    ----------
    value : Any
        Result from reduce tool.

    Returns
    -------
    int
        Extracted count value.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        count = value.get("count")
        if count is not None:
            return int(count)
        shared = value.get("shared_value")
        if isinstance(shared, (int, float)):
            return int(shared)
    return 0


@custom_function()
def extract_sum(value: Any) -> float:
    """Extract sum from a reduce result.

    Handles various return formats:
    - Direct int/float: ``123.5`` -> ``123.5``
    - Dict with sum: ``{"sum": 123.5}`` -> ``123.5``
    - Dict with count fallback: ``{"count": 10}`` -> ``10.0``
    - None -> ``0.0``

    Parameters
    ----------
    value : Any
        Result from reduce tool.

    Returns
    -------
    float
        Extracted sum value.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("sum", "count", "value"):
            val = value.get(key)
            if val is not None and isinstance(val, (int, float)):
                return float(val)
    return 0.0


@custom_function()
def normalize_grouped_result(
    result: Dict[str, Any],
    extract_fn: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """Normalize grouped reduce results to ``{group: value}`` format.

    Parameters
    ----------
    result : dict
        Raw grouped result from reduce tool.
    extract_fn : callable, optional
        Function to extract values from each group's raw result.
        Defaults to ``extract_count`` when not supplied.

    Returns
    -------
    dict[str, Any]
        Normalized ``{group_name: value}`` mapping.
    """
    if not isinstance(result, dict):
        return {}
    fn = extract_fn if extract_fn is not None else extract_count
    return {k: fn(v) for k, v in result.items()}


# =============================================================================
# NUMERIC HELPERS
# =============================================================================


@custom_function()
def compute_percentage(numerator: int, denominator: int, decimals: int = 2) -> float:
    """Compute percentage safely.

    Returns 0.0 if denominator is 0 (avoids division by zero).

    Parameters
    ----------
    numerator : int
        Top value.
    denominator : int
        Bottom value.
    decimals : int
        Decimal places to round to (default: 2).

    Returns
    -------
    float
        Percentage value (0-100 scale).
    """
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, decimals)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


@custom_function()
def build_metric_result(
    metric_name: str,
    group_by: Any,
    time_period: Any,
    start_date: Optional[str],
    end_date: Optional[str],
    results: List[Dict[str, Any]],
    total: float,
    metadata: Optional[Dict[str, Any]] = None,
    plots: Optional[List[Any]] = None,
) -> dict:
    """Build standardized metric result.

    Creates a consistent output format for all metrics, handling
    enum-to-string conversions.  This is the canonical shape consumed
    by the Actor, tests, and downstream reporting.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g., ``"first_time_fix_rate"``).
    group_by : str or Enum
        Grouping dimension used.
    time_period : str or Enum
        Time granularity used.
    start_date : str or None
        Start date of analysis period.
    end_date : str or None
        End date of analysis period.
    results : list
        List of result dicts with metric values.
    total : float
        Aggregate total across all groups.
    metadata : dict or None
        Additional metadata about the calculation.
    plots : list or None
        Generated plot visualizations.

    Returns
    -------
    dict
        JSON-like standardized metric result (safe for FunctionManager
        memoization) with keys: metric_name, group_by, time_period,
        start_date, end_date, results, total, metadata, plots.
    """
    group_by_value = group_by.value if hasattr(group_by, "value") else group_by
    time_period_value = (
        time_period.value if hasattr(time_period, "value") else time_period
    )
    return {
        "metric_name": metric_name,
        "group_by": group_by_value,
        "time_period": time_period_value,
        "start_date": start_date,
        "end_date": end_date,
        "results": results,
        "total": total,
        "metadata": metadata,
        "plots": plots or [],
    }


# =============================================================================
# PLOT RESULT HELPERS
# =============================================================================


@custom_function()
def extract_plot_url(result: Any) -> Optional[str]:
    """Extract URL from a plot result (handles both PlotResult and dict)."""
    if result is None:
        return None
    if hasattr(result, "url"):
        return result.url
    if isinstance(result, dict):
        return result.get("url") or result.get("image_url")
    return None


@custom_function()
def extract_plot_succeeded(result: Any) -> bool:
    """Extract succeeded status from a plot result."""
    if result is None:
        return False
    if hasattr(result, "error") and result.error:
        return False
    if isinstance(result, dict):
        if "error" in result and result["error"]:
            return False
        return bool(result.get("url") or result.get("image_url"))
    url = getattr(result, "url", None) or getattr(result, "image_url", None)
    return bool(url)
