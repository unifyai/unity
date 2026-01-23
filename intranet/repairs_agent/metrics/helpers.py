"""
Facade helpers for repairs metrics - synced to FunctionManager.

These functions encapsulate common patterns used across all metrics,
providing a stable interface for both static and dynamic agents.

DESIGN PRINCIPLES:
1. Discovery-first: Use tables_overview and schema_explain to discover schema
2. No magic globals: All filter expressions and column mappings are inline
3. Rich return values: Discovery functions return table + description + columns
4. Pure functions: No logging or side effects
"""

from typing import Any, Callable, Dict, List, Optional, TypedDict

from intranet.repairs_agent.metrics.types import MetricResult

# =============================================================================
# FILE PATH CONSTANTS (Data sources - these identify WHERE the data is)
# =============================================================================

REPAIRS_FILE = (
    "/home/hmahmood24/unity/intranet/repairs/"
    "MDH Repairs Data July - Nov 25 - DL V1.xlsx"
)
TELEMATICS_FILE = (
    "/home/hmahmood24/unity/intranet/repairs/"
    "MDH Telematics Data July - Nov 25 - DL V1.xlsx"
)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class ColumnInfo(TypedDict, total=False):
    """Column information from schema discovery."""

    name: str
    description: str


class TableInfo(TypedDict, total=False):
    """Table information from discovery."""

    table: str  # The table path/context for queries
    description: str  # Table description
    columns: List[ColumnInfo]  # Column names and descriptions


# =============================================================================
# DISCOVERY FUNCTIONS - Return rich schema info
# =============================================================================


def discover_repairs_table(tools: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Discover the repairs data table with full schema information.

    Uses tables_overview(file=REPAIRS_FILE) to find the repairs table,
    then schema_explain to get column details.

    Parameters
    ----------
    tools : Dict[str, Any]
        Tools dict containing tables_overview, schema_explain callables

    Returns
    -------
    TableInfo or None
        Dict with 'table' (path), 'description', and 'columns' list,
        or None if discovery fails

    Example
    -------
    >>> info = discover_repairs_table(tools)
    >>> if info:
    ...     print(f"Table: {info['table']}")
    ...     print(f"Description: {info['description']}")
    ...     for col in info['columns']:
    ...         print(f"  - {col['name']}: {col.get('description', '')}")
    ...     # Now query using the table path
    ...     result = tools["reduce"](table=info['table'], ...)
    """
    tables_overview = tools.get("tables_overview")
    schema_explain = tools.get("schema_explain")

    if not tables_overview:
        return None

    # Scope to the repairs file only
    file_tables = tables_overview(file=REPAIRS_FILE)

    # Navigate the nested structure to find the repairs table
    # Structure: {<safe_file_path>: {"Tables": {<label>: {"context": ..., "description": ...}}}}
    table_path: Optional[str] = None
    table_desc: str = ""

    for file_key, file_info in file_tables.items():
        if file_key == "FileRecords":
            continue
        if isinstance(file_info, dict) and "Tables" in file_info:
            tables = file_info["Tables"]
            for label, tinfo in tables.items():
                if isinstance(tinfo, dict):
                    ctx = tinfo.get("context", "")
                    if "Repairs" in label or "Raised" in label:
                        table_path = ctx
                        table_desc = tinfo.get("description", "")
                        break

    if not table_path:
        return None

    # Get column information via schema_explain
    columns: List[ColumnInfo] = []
    if schema_explain:
        try:
            schema_text = schema_explain(table=table_path)
            # Parse the schema_explain output to extract columns
            # Format: "Table: ...\n\nColumns:\n  - ColName: Description\n  - ColName\n\nRow count: N"
            in_columns = False
            for line in schema_text.split("\n"):
                line = line.strip()
                if line == "Columns:":
                    in_columns = True
                elif in_columns and line.startswith("- "):
                    col_part = line[2:]  # Remove "- "
                    if ": " in col_part:
                        name, desc = col_part.split(": ", 1)
                        columns.append({"name": name, "description": desc})
                    else:
                        columns.append({"name": col_part, "description": ""})
                elif in_columns and not line.startswith("-"):
                    in_columns = False
        except Exception:
            pass

    return {
        "table": table_path,
        "description": table_desc,
        "columns": columns,
    }


def discover_telematics_tables(tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Discover telematics data tables with full schema information.

    Uses tables_overview(file=TELEMATICS_FILE) to find monthly tables,
    then schema_explain for each to get column details.

    Parameters
    ----------
    tools : Dict[str, Any]
        Tools dict containing tables_overview, schema_explain callables

    Returns
    -------
    list[TableInfo]
        List of dicts, each with 'table', 'description', 'columns'

    Example
    -------
    >>> tables = discover_telematics_tables(tools)
    >>> for tinfo in tables:
    ...     print(f"Table: {tinfo['table']}")
    ...     # Aggregate across all monthly tables
    ...     result = tools["reduce"](table=tinfo['table'], ...)
    """
    tables_overview = tools.get("tables_overview")
    schema_explain = tools.get("schema_explain")

    if not tables_overview:
        return []

    # Scope to the telematics file only
    file_tables = tables_overview(file=TELEMATICS_FILE)

    # Collect all telematics tables
    result: List[TableInfo] = []

    for file_key, file_info in file_tables.items():
        if file_key == "FileRecords":
            continue
        if isinstance(file_info, dict) and "Tables" in file_info:
            tables = file_info["Tables"]
            for label, tinfo in tables.items():
                if isinstance(tinfo, dict):
                    ctx = tinfo.get("context", "")
                    desc = tinfo.get("description", "")

                    # Get column information
                    columns: List[ColumnInfo] = []
                    if schema_explain:
                        try:
                            schema_text = schema_explain(table=ctx)
                            in_columns = False
                            for line in schema_text.split("\n"):
                                line = line.strip()
                                if line == "Columns:":
                                    in_columns = True
                                elif in_columns and line.startswith("- "):
                                    col_part = line[2:]
                                    if ": " in col_part:
                                        name, col_desc = col_part.split(": ", 1)
                                        columns.append(
                                            {"name": name, "description": col_desc},
                                        )
                                    else:
                                        columns.append(
                                            {"name": col_part, "description": ""},
                                        )
                                elif in_columns and not line.startswith("-"):
                                    in_columns = False
                        except Exception:
                            pass

                    result.append(
                        {
                            "table": ctx,
                            "description": desc,
                            "columns": columns,
                        },
                    )

    return result


# =============================================================================
# INLINE HELPERS - No magic globals, everything visible
# =============================================================================


def resolve_group_by(
    group_by: Optional[str],
    telematics: bool = False,
    date_context: str = "completion",
) -> Optional[str]:
    """
    Resolve group_by string to actual column name.

    The mapping is INLINE so the CodeActActor can see exactly which
    columns are used for each grouping dimension.

    Parameters
    ----------
    group_by : str or None
        Group dimension: "operative", "patch", "region", "trade", "day".
        Pass None for no grouping (aggregate total).
    telematics : bool
        If True, use telematics column mappings (Vehicle instead of Operative)
    date_context : str
        Context for temporal grouping to select appropriate date column:
        - "completion": WorksOrderReportedCompletedDateDay (default)
        - "issued": WorksOrderIssuedDateDay
        - "arrival": ArrivedOnSiteDay
        - "scheduled": ScheduledAppointmentStartDay

    Returns
    -------
    str or None
        Column name for grouping, or None if group_by is None

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
    # Return None early if no grouping requested
    if group_by is None:
        return None

    key = group_by.lower()

    # Temporal column selection based on context
    date_columns = {
        "completion": "WorksOrderReportedCompletedDateDay",
        "issued": "WorksOrderIssuedDateDay",
        "arrival": "ArrivedOnSiteDay",
        "scheduled": "ScheduledAppointmentStartDay",
    }

    # Telematics data column mappings:
    telematics_mapping = {
        "operative": "Vehicle",  # Vehicle contains operative name in telematics
        "day": "ArrivalDay",
    }

    # INLINE mapping - no hidden globals
    # Repairs data column mappings:
    repairs_mapping = {
        "operative": "OperativeWhoCompletedJob",
        "patch": "RepairsPatch",
        "region": "RepairsRegion",
        "trade": "Trade",
        # Temporal mappings - context-dependent
        "day": date_columns.get(date_context, "WorksOrderReportedCompletedDateDay"),
    }

    mapping = telematics_mapping if telematics else repairs_mapping
    return mapping.get(key)


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
        List of filter conditions (will be AND-ed together).
        Pass the ACTUAL FILTER STRINGS, not global constant names.

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
    >>> # GOOD: Inline filter expression - CodeActActor can see exactly what it means
    >>> build_filter(
    ...     ["`WorksOrderStatusDescription` in ['Complete', 'Closed']"],
    ...     start_date="2025-07-01"
    ... )
    "(`WorksOrderStatusDescription` in ['Complete', 'Closed']) and `VisitDate` >= '2025-07-01'"

    >>> # BAD: Using a magic constant - CodeActActor can't see what COMPLETED_FILTER means
    >>> build_filter([COMPLETED_FILTER], start_date="2025-07-01")  # Don't do this!
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
    extract_fn: Optional[Callable[[Any], Any]] = None,
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
    Build standardized metric result.

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
    MetricResult
        Standardized metric result
    """
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


# =============================================================================
# PLOT RESULT HELPER
# =============================================================================


def extract_plot_url(result: Any) -> Optional[str]:
    """Extract URL from visualize result (handles both PlotResult and dict)."""
    if result is None:
        return None
    if hasattr(result, "url"):
        return result.url
    if isinstance(result, dict):
        return result.get("url")
    return None


def extract_plot_succeeded(result: Any) -> bool:
    """Extract succeeded status from visualize result."""
    if result is None:
        return False
    if hasattr(result, "succeeded"):
        return result.succeeded
    if isinstance(result, dict):
        return result.get("succeeded", True)
    return True


# =============================================================================
# EXPORTS - List of helpers for FunctionManager sync
# =============================================================================

HELPER_FUNCTIONS = [
    "discover_repairs_table",
    "discover_telematics_tables",
    "resolve_group_by",
    "build_filter",
    "extract_count",
    "extract_sum",
    "normalize_grouped_result",
    "compute_percentage",
    # Note: build_metric_result is NOT synced - it requires MetricResult import
    # Metrics use it internally; LLM should compose results directly
]
