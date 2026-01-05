"""
Facade helpers for repairs metrics - synced to FunctionManager.

These functions encapsulate common patterns used across all metrics,
providing a stable interface for both static and dynamic agents.
All helpers are pure functions with no logging or side effects.
"""

from typing import Any, Dict, List, Optional

from intranet.repairs_agent.metrics.types import MetricResult

# =============================================================================
# TABLE CONSTANTS
# =============================================================================

REPAIRS_FILE = (
    "/home/hmahmood24/unity/intranet/repairs/"
    "MDH Repairs Data July - Nov 25 - DL V1.xlsx"
)
TELEMATICS_FILE = (
    "/home/hmahmood24/unity/intranet/repairs/"
    "MDH Telematics Data July - Nov 25 - DL V1.xlsx"
)

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
    "operative": "OperativeWhoCompletedJob",
    "patch": "RepairsPatch",
    "region": "RepairsRegion",
    "trade": "Trade",
    "total": None,
}

TELEMATICS_GROUP_BY_FIELDS = {
    "operative": "Vehicle",  # Vehicle contains operative name in telematics
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def discover_repairs_table(tools: Dict[str, Any]) -> Optional[str]:
    """
    Discover the repairs data table dynamically.

    Uses tables_overview() to find a table containing "Repairs" in its name
    or context. Returns the table path/context for use in queries.

    Parameters
    ----------
    tools : dict
        Tools dict containing tables_overview callable

    Returns
    -------
    str or None
        Table path if found, None otherwise

    Example
    -------
    >>> repairs_table = discover_repairs_table(tools)
    >>> if repairs_table:
    ...     result = tools["reduce"](table=repairs_table, ...)
    """
    tables_overview = tools.get("tables_overview")
    if not tables_overview:
        return None

    all_tables = tables_overview()
    for name, info in all_tables.items():
        context = info.get("context", "")
        if "Repairs" in name or "repairs" in context.lower():
            return context or name
    return None


def discover_telematics_tables(tools: Dict[str, Any]) -> List[str]:
    """
    Discover telematics data tables dynamically.

    Telematics data is split by month (July-November 2025).
    Returns all matching table paths for aggregation.

    Parameters
    ----------
    tools : dict
        Tools dict containing tables_overview callable

    Returns
    -------
    list[str]
        List of telematics table paths
    """
    tables_overview = tools.get("tables_overview")
    if not tables_overview:
        return []

    all_tables = tables_overview()
    telematics = []
    for name, info in all_tables.items():
        context = info.get("context", "")
        if "Telematics" in name or "telematics" in context.lower():
            telematics.append(context or name)
    return telematics


def resolve_group_by(group_by: str) -> Optional[str]:
    """
    Resolve group_by string to actual column name.

    Mapping:
    - "operative" → "OperativeWhoCompletedJob"
    - "patch" → "RepairsPatch"
    - "region" → "RepairsRegion"
    - "total" → None (no grouping)

    Parameters
    ----------
    group_by : str
        Group dimension: "operative", "patch", "region", or "total"

    Returns
    -------
    str or None
        Column name for grouping, or None for total aggregation
    """
    mapping = {
        "operative": "OperativeWhoCompletedJob",
        "patch": "RepairsPatch",
        "region": "RepairsRegion",
        "total": None,
    }
    return mapping.get(group_by.lower() if isinstance(group_by, str) else "total")


def build_filter(
    base_conditions: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_column: str = "VisitDate",
) -> str:
    """
    Build a filter expression from conditions and optional date range.

    Parameters
    ----------
    base_conditions : list[str]
        List of filter conditions (will be AND-ed together)
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    date_column : str
        Column name for date filtering (default: VisitDate)

    Returns
    -------
    str
        Combined filter expression

    Example
    -------
    >>> build_filter(
    ...     ["`FirstTimeFix` == 'Yes'", "`WorksOrderStatusDescription` == 'Complete'"],
    ...     start_date="2025-07-01"
    ... )
    "(`FirstTimeFix` == 'Yes') and (`WorksOrderStatusDescription` == 'Complete') and `VisitDate` >= '2025-07-01'"
    """
    parts = [f"({c})" for c in base_conditions if c]
    if start_date:
        parts.append(f"`{date_column}` >= '{start_date}'")
    if end_date:
        parts.append(f"`{date_column}` <= '{end_date}'")
    return " and ".join(parts) if parts else ""


def extract_count(value: Any) -> int:
    """
    Extract count from a reduce result.

    Handles various return formats:
    - Direct int/float: 123 → 123
    - Dict with count: {"count": 123} → 123
    - Dict with shared_value: {"shared_value": None, "count": 123} → 123
    - None → 0

    Parameters
    ----------
    value : Any
        Result from reduce tool

    Returns
    -------
    int
        Extracted count value
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


def extract_sum(value: Any) -> float:
    """
    Extract sum from a reduce result.

    Handles various return formats:
    - Direct int/float: 123.5 → 123.5
    - Dict with sum: {"sum": 123.5} → 123.5
    - Dict with count fallback: {"count": 10} → 10.0
    - None → 0.0

    Parameters
    ----------
    value : Any
        Result from reduce tool

    Returns
    -------
    float
        Extracted sum value
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


def normalize_grouped_result(
    result: Dict[str, Any],
    extract_fn: Any = None,
) -> Dict[str, Any]:
    """
    Normalize grouped reduce results to {group: value} format.

    Parameters
    ----------
    result : dict
        Raw grouped result from reduce tool
    extract_fn : callable, optional
        Function to extract values (default: extract_count)

    Returns
    -------
    dict[str, Any]
        Normalized {group_name: value} mapping
    """
    if not isinstance(result, dict):
        return {}
    fn = extract_fn if extract_fn is not None else extract_count
    return {k: fn(v) for k, v in result.items()}


def compute_percentage(numerator: int, denominator: int, decimals: int = 2) -> float:
    """
    Compute percentage safely.

    Returns 0.0 if denominator is 0 (avoids division by zero).

    Parameters
    ----------
    numerator : int
        Top value
    denominator : int
        Bottom value
    decimals : int
        Decimal places to round to (default: 2)

    Returns
    -------
    float
        Percentage value (0-100 scale)
    """
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, decimals)


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
) -> MetricResult:
    """
    Build standardized metric result dictionary.

    Creates a consistent output format for all metrics, handling
    enum-to-string conversions.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g., "first_time_fix_rate")
    group_by : str or Enum
        Grouping dimension used
    time_period : str or Enum
        Time granularity used
    start_date : str or None
        Start date of analysis period
    end_date : str or None
        End date of analysis period
    results : list
        List of result dicts with metric values
    total : float
        Aggregate total across all groups
    metadata : dict or None
        Additional metadata about the calculation
    plots : list or None
        Generated plot visualizations

    Returns
    -------
    dict
        Standardized metric result with all fields
    """
    # Handle enum to string conversion
    group_by_value = group_by.value if hasattr(group_by, "value") else group_by
    time_period_value = (
        time_period.value if hasattr(time_period, "value") else time_period
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


# List of all helper functions for sync to FunctionManager
HELPER_FUNCTIONS = [
    "discover_repairs_table",
    "discover_telematics_tables",
    "resolve_group_by",
    "build_filter",
    "extract_count",
    "extract_sum",
    "normalize_grouped_result",
    "compute_percentage",
    "build_metric_result",
]
