"""
Facade helpers for repairs metrics - synced to FunctionManager.

These functions encapsulate common patterns used across all metrics,
providing a stable interface for both static and dynamic agents.
All helpers are pure functions with no logging or side effects.
"""

from typing import Any, Dict, List, Optional


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


def normalize_grouped_result(result: Dict[str, Any]) -> Dict[str, int]:
    """
    Normalize grouped reduce results to {group: count} format.

    Parameters
    ----------
    result : dict
        Raw grouped result from reduce tool

    Returns
    -------
    dict[str, int]
        Normalized {group_name: count} mapping
    """
    if not isinstance(result, dict):
        return {}
    return {k: extract_count(v) for k, v in result.items()}


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


# List of all helper functions for sync to FunctionManager
HELPER_FUNCTIONS = [
    "discover_repairs_table",
    "discover_telematics_tables",
    "resolve_group_by",
    "build_filter",
    "extract_count",
    "normalize_grouped_result",
    "compute_percentage",
]
