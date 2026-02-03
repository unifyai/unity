"""
Facade helpers for repairs metrics - synced to FunctionManager.

These functions encapsulate common patterns used across all metrics,
providing a stable interface for both static and dynamic agents.

DESIGN PRINCIPLES:
1. Discovery-first: Use describe() to discover schema and table contexts
2. No magic globals: All filter expressions and column mappings are inline
3. Rich return values: Discovery functions return table + description + columns
4. Pure functions: No logging or side effects

API Reconciliation (2025-01):
- tables_overview() → describe(file_path=...) returning FileStorageMap
- schema_explain() → list_columns(context=...) returning column info
- reduce(table=..., keys=...) → reduce(context=..., columns=...)
- list_columns(table=...) → list_columns(context=...)
"""

from typing import Any, Callable, Dict, List, Optional, TypedDict, TYPE_CHECKING

from intranet.repairs_agent.metrics.types import MetricResult

if TYPE_CHECKING:
    from unity.file_manager.types.describe import FileStorageMap

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

    Uses describe(file_path=REPAIRS_FILE) to find the repairs table,
    then extracts table description and column details from the FileStorageMap.

    Parameters
    ----------
    tools : Dict[str, Any]
        Tools dict containing describe, list_columns callables

    Returns
    -------
    TableInfo or None
        Dict with 'table' (context path), 'description', and 'columns' list,
        or None if discovery fails. Columns include 'name' and 'description'.

    Example
    -------
    >>> info = discover_repairs_table(tools)
    >>> if info:
    ...     print(f"Table: {info['table']}")
    ...     print(f"Description: {info['description']}")
    ...     for col in info['columns']:
    ...         print(f"  - {col['name']}: {col['description']}")
    ...     # Now query using the table context path
    ...     result = tools["reduce"](context=info['table'], ...)
    """
    describe = tools.get("describe")
    list_columns = tools.get("list_columns")

    if not describe:
        return None

    try:
        # Get FileStorageMap for the repairs file
        storage: "FileStorageMap" = describe(file_path=REPAIRS_FILE)
    except Exception:
        return None

    # Check if file is indexed and has tables
    if not getattr(storage, "indexed_exists", False):
        return None
    if not getattr(storage, "has_tables", False):
        return None

    # Find the repairs table (look for "Repairs" or "Raised" in table name)
    table_path: Optional[str] = None
    table_description: str = ""
    matched_table = None
    tables = getattr(storage, "tables", [])

    for table_info in tables:
        name = getattr(table_info, "name", "")
        if "Repairs" in name or "Raised" in name:
            table_path = getattr(table_info, "context_path", None)
            table_description = getattr(table_info, "description", None) or ""
            matched_table = table_info
            break

    # Fallback: use first table if no specific match
    if not table_path and tables:
        matched_table = tables[0]
        table_path = getattr(matched_table, "context_path", None)
        table_description = getattr(matched_table, "description", None) or ""

    if not table_path:
        return None

    # Get column information from TableInfo.column_schema if available
    columns: List[ColumnInfo] = []
    if matched_table:
        column_schema = getattr(matched_table, "column_schema", None)
        if column_schema:
            schema_columns = getattr(column_schema, "columns", [])
            for col in schema_columns:
                col_name = getattr(col, "name", "")
                col_desc = getattr(col, "description", None) or ""
                if col_name:
                    columns.append({"name": col_name, "description": col_desc})

    # Fallback to list_columns if schema didn't have columns
    if not columns and list_columns:
        try:
            cols_result = list_columns(context=table_path)
            # list_columns returns Dict[str, Any] (name -> info) when include_types=True
            if isinstance(cols_result, dict):
                for name, info in cols_result.items():
                    desc = ""
                    if isinstance(info, dict):
                        desc = info.get("description", "") or ""
                    columns.append({"name": name, "description": desc})
            elif isinstance(cols_result, list):
                columns = [{"name": name, "description": ""} for name in cols_result]
        except Exception:
            pass

    return {
        "table": table_path,
        "description": table_description,
        "columns": columns,
    }


def discover_telematics_tables(tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Discover telematics data tables with full schema information.

    Uses describe(file_path=TELEMATICS_FILE) to find monthly tables,
    then list_columns for each to get column details.

    Parameters
    ----------
    tools : Dict[str, Any]
        Tools dict containing describe, list_columns callables

    Returns
    -------
    list[TableInfo]
        List of dicts, each with 'table' (context path), 'description', 'columns'

    Example
    -------
    >>> tables = discover_telematics_tables(tools)
    >>> for tinfo in tables:
    ...     print(f"Table: {tinfo['table']}")
    ...     # Aggregate across all monthly tables
    ...     result = tools["reduce"](context=tinfo['table'], ...)
    """
    describe = tools.get("describe")
    list_columns = tools.get("list_columns")

    if not describe:
        return []

    try:
        # Get FileStorageMap for the telematics file
        storage: "FileStorageMap" = describe(file_path=TELEMATICS_FILE)
    except Exception:
        return []

    # Check if file is indexed and has tables
    if not getattr(storage, "indexed_exists", False):
        return []
    if not getattr(storage, "has_tables", False):
        return []

    # Collect all telematics tables
    result: List[TableInfo] = []
    tables = getattr(storage, "tables", [])

    for table_info in tables:
        ctx = getattr(table_info, "context_path", "")
        if not ctx:
            continue

        # Extract table description from TableInfo
        table_description = getattr(table_info, "description", None) or ""

        # Get column information from TableInfo.column_schema if available
        columns: List[ColumnInfo] = []
        column_schema = getattr(table_info, "column_schema", None)
        if column_schema:
            schema_columns = getattr(column_schema, "columns", [])
            for col in schema_columns:
                col_name = getattr(col, "name", "")
                col_desc = getattr(col, "description", None) or ""
                if col_name:
                    columns.append({"name": col_name, "description": col_desc})

        # Fallback to list_columns if schema didn't have columns
        if not columns and list_columns:
            try:
                cols_result = list_columns(context=ctx)
                # list_columns returns Dict[str, Any] (name -> info) when include_types=True
                if isinstance(cols_result, dict):
                    for name, info in cols_result.items():
                        desc = ""
                        if isinstance(info, dict):
                            desc = info.get("description", "") or ""
                        columns.append({"name": name, "description": desc})
                elif isinstance(cols_result, list):
                    columns = [
                        {"name": name, "description": ""} for name in cols_result
                    ]
            except Exception:
                pass

        result.append(
            {
                "table": ctx,
                "description": table_description,
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
