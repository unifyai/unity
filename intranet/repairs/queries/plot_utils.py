"""
Plot API client for generating visualizations.

This module provides utilities for generating plot visualizations via the
Console Plot API. It handles the API communication, error handling, and
result formatting.

Architecture Note:
    TODO: Currently uses Console API (POST {CONSOLE_BASE_URL}/api/plot/create).
          This is a temporary integration path. Once the Plot API is migrated
          to Orchestra, this module should be updated to use the Orchestra
          endpoint instead (via the unify client). This is the ONLY place in
          the codebase that calls console -> orchestra directly rather than
          going through the standard unify -> orchestra path.

Context Resolution:
    The Plot API requires fully-qualified Unify context paths, not the local
    table patterns like "/path/to/file.xlsx.Tables.Sheet1" used by FileManager.
    This module resolves these patterns to fully-qualified contexts like:
    "DefaultUser/Assistant/Files/Local/path_to_file_xlsx/Tables/Sheet1"

    The resolution uses the same logic as FileManager's storage utilities,
    with hardcoded base context values for the default user setup.

Usage:
    >>> from .plot_utils import generate_plots
    >>> plots = generate_plots(
    ...     metric_name="first_time_fix_rate",
    ...     group_by=GroupBy.OPERATIVE,
    ...     project_name="RepairsAgent5M",
    ...     table="/path/to/file.xlsx.Tables.TableName",
    ...     filter_expr='`Status` == "Completed"',
    ...     include_plots=True
    ... )
    >>> for plot in plots:
    ...     if plot.succeeded:
    ...         print(f"{plot.title}: {plot.url}")
"""

from __future__ import annotations

import logging
import os
import re
import traceback
from typing import Any, List, Optional, Union

import httpx

from ._plots import get_plot_configs
from ._types import GroupBy, PlotConfig, PlotResult

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Switch to ORCHESTRA_URL once Plot API is migrated to Orchestra.
# Currently using Console API as a temporary integration path.
CONSOLE_BASE_URL = os.environ.get("CONSOLE_BASE_URL", "https://console.unify.ai")
PLOT_API_ENDPOINT = f"{CONSOLE_BASE_URL}/api/plot/create"

# Timeout for plot generation API calls (seconds)
PLOT_API_TIMEOUT = 30.0

# -----------------------------------------------------------------------------
# Context Resolution Configuration
# -----------------------------------------------------------------------------
# These values mirror FileManager's internal context structure for the default
# user setup. They are used to resolve table references to fully-qualified
# Unify contexts without needing access to a FileManager instance.

_BASE_CTX = "DefaultUser/Assistant"
_FS_ALIAS = "Local"
_PER_FILE_ROOT = f"{_BASE_CTX}/Files/{_FS_ALIAS}"


# =============================================================================
# CONTEXT RESOLUTION UTILITIES
# =============================================================================


def _safe(value: Any) -> str:
    """
    Uniform sanitizer for a single context path component.

    This mirrors FileManager.safe() - sanitizes values for safe inclusion
    in a context path component.

    Parameters
    ----------
    value : Any
        Value to sanitize for safe inclusion in a context path component.

    Returns
    -------
    str
        A lowercase-safe string containing only [a-zA-Z0-9_-], with other
        characters replaced by '_'. The result is truncated to 64
        characters; returns 'item' when empty.
    """
    try:
        s = str(value)

        # Detect OS-invariant path separators; split into head and tail
        last_slash = max(s.rfind("/"), s.rfind("\\"))
        if last_slash >= 0:
            head_raw, tail_raw = s[:last_slash], s[last_slash + 1 :]
        else:
            head_raw, tail_raw = "", s

        def _sanitize(part: str) -> str:
            # Replace non [a-zA-Z0-9_-] (including dots and path punctuation) with underscores
            return re.sub(r"[^a-zA-Z0-9_-]", "_", part)

        tail = _sanitize(tail_raw) or "item"
        head = _sanitize(head_raw)

        if not head:
            # No head: return sanitized tail as-is
            return tail

        def _compress_center(text: str, target_len: int) -> str:
            if len(text) <= target_len:
                return text
            # Use multiple underscores as an ellipsis in the middle
            marker = "____"
            if target_len <= len(marker):
                return marker[:target_len]
            left = (target_len - len(marker)) // 2
            right = target_len - len(marker) - left
            return text[:left] + marker + text[-right:]

        head_limit = 32
        head_comp = _compress_center(head, head_limit)
        return f"{head_comp}_{tail}"
    except Exception:
        return "item"


def _ctx_for_file(file_path: str) -> str:
    """
    Return the fully-qualified per-file Content context.

    Shape: <base>/Files/<alias>/<safe(file_path)>/Content

    This mirrors storage.ctx_for_file() from FileManager.
    """
    return f"{_PER_FILE_ROOT}/{_safe(file_path)}/Content"


def _ctx_for_file_table(file_path: str, table: str) -> str:
    """
    Return the fully-qualified per-file table context.

    Shape: <base>/Files/<alias>/<safe(file_path)>/Tables/<safe(table)>

    This mirrors storage.ctx_for_file_table() from FileManager.
    """
    return f"{_PER_FILE_ROOT}/{_safe(file_path)}/Tables/{_safe(table)}"


def _resolve_table_ref(table: str) -> str:
    """
    Resolve a table reference to a fully-qualified Unify context.

    Accepted forms:
    - "<file_path>" → per-file Content context
    - "<file_path>.Tables.<label>" → per-file table context

    This mirrors the logic in search.ctx_for_table() from FileManager.

    Parameters
    ----------
    table : str
        Table reference in FileManager format.

    Returns
    -------
    str
        Fully-qualified Unify context path.

    Raises
    ------
    ValueError
        If table is empty.
    """
    t = (table or "").strip()
    if not t:
        raise ValueError("table must be non-empty")

    # Per-file Content (no .Tables. in the path)
    if ".tables." not in t.lower():
        return _ctx_for_file(file_path=t)

    # Per-file table: split on .Tables.
    root, label = t.split(".Tables.", 1)
    return _ctx_for_file_table(file_path=root, table=label)


# =============================================================================
# PUBLIC API
# =============================================================================


def generate_plots(
    *,
    metric_name: str,
    group_by: GroupBy,
    project_name: str,
    tables: Union[str, List[str]],
    filter_expr: Optional[str] = None,
    include_plots: bool = False,
) -> List[PlotResult]:
    """
    Generate plot visualizations for a metric result.

    This function looks up the plot configurations for the given metric
    and group_by combination, then generates each plot via the Plot API.
    If any plot generation fails, the error is captured in the PlotResult
    rather than raising an exception.

    Args:
        metric_name: Name of the metric (e.g., "first_time_fix_rate").
            Must match a key in METRIC_PLOT_CONFIGS.
        group_by: Grouping dimension used in the query. Determines which
            plot configurations are selected.
        project_name: Unify project name (e.g., "RepairsAgent5M"). This is
            passed to the Plot API's project_config.project_name field.
        tables: Table reference(s) in FileManager format. Accepts either:
            - A single string for metrics with one data source
            - A list of strings for metrics spanning multiple tables
              (e.g., telematics data with monthly tables)
            When multiple tables are provided, plots are generated for EACH
            table, with the table name appended to the plot title
            (e.g., "Distance by Operative (July_2025)").
        filter_expr: Optional filter expression applied to the data. Same
            syntax as used in _reduce calls. Passed to project_config.filter_expr.
        include_plots: If False, returns empty list immediately without
            making any API calls. This enables the caller to skip plot
            generation when not needed (default behavior).

    Returns:
        List of PlotResult objects. Each PlotResult contains either:
        - A successful result with url, token, and expires_in_hours
        - A failed result with error message and traceback

        Returns empty list if:
        - include_plots is False
        - No plot configurations are defined for the metric×group_by combination
        - tables is empty
        - Context resolution fails (error captured in PlotResult)

    Example (single table):
        >>> plots = generate_plots(
        ...     metric_name="first_time_fix_rate",
        ...     group_by=GroupBy.OPERATIVE,
        ...     project_name="RepairsAgent5M",
        ...     tables="/path/to/data.xlsx.Tables.Repairs",
        ...     filter_expr='`Status` == "Completed"',
        ...     include_plots=True
        ... )

    Example (multiple tables - telematics):
        >>> plots = generate_plots(
        ...     metric_name="distance_travelled_per_day",
        ...     group_by=GroupBy.OPERATIVE,
        ...     project_name="RepairsAgent5M",
        ...     tables=[
        ...         "/path/to/data.xlsx.Tables.July_2025",
        ...         "/path/to/data.xlsx.Tables.August_2025",
        ...         ...
        ...     ],
        ...     include_plots=True
        ... )
        # Generates plots for each month: "Distance (July_2025)", "Distance (August_2025)", etc.
    """
    # Early exit if plot generation is disabled
    if not include_plots:
        return []

    # Normalize tables input: convert string to list
    table_list: List[str] = []
    if isinstance(tables, str):
        if tables:
            table_list = [tables]
    else:
        table_list = [t for t in tables if t]  # Filter out empty strings

    if not table_list:
        logger.warning(f"[{metric_name}] No tables provided for plot generation")
        return []

    # Look up plot configurations for this metric×group_by
    configs = get_plot_configs(metric_name, group_by)
    if not configs:
        logger.debug(
            f"[{metric_name}] No plot configs defined for group_by={group_by.value}",
        )
        return []

    total_plots = len(configs) * len(table_list)
    logger.debug(
        f"[{metric_name}] Generating {total_plots} plot(s) "
        f"({len(configs)} configs × {len(table_list)} tables) for group_by={group_by.value}",
    )

    # Generate plots for each table
    results: List[PlotResult] = []
    for tbl in table_list:
        # Extract table label for multi-table scenarios (e.g., "July_2025" from the path)
        # Used for title suffix when multiple tables
        table_label = _extract_table_label(tbl) if len(table_list) > 1 else None

        # Resolve the table reference to a fully-qualified Unify context
        try:
            context = _resolve_table_ref(tbl)
            logger.debug(
                f"[{metric_name}] Resolved table '{tbl}' -> context '{context}'",
            )
        except Exception as e:
            logger.warning(
                f"[{metric_name}] Failed to resolve table '{tbl}' to context: {e}",
            )
            # Context resolution failed - return error for all configs for this table
            for config in configs:
                title = _format_title_with_label(config.title, table_label)
                results.append(
                    PlotResult(
                        plot_config={},
                        project_config={"project_name": project_name, "table": tbl},
                        title=title,
                        error=f"Failed to resolve context for table: {tbl} - {e}",
                    ),
                )
            continue

        # Generate each configured plot for this table
        for config in configs:
            # Append table label to title for multi-table scenarios
            title = _format_title_with_label(config.title, table_label)

            result = _generate_single_plot(
                config=config,
                project_name=project_name,
                context=context,
                filter_expr=filter_expr,
                title_override=title,
            )
            results.append(result)

            # Log result
            if result.succeeded:
                logger.info(
                    f"[{metric_name}] Generated plot: {result.title} -> {result.url}",
                )
            else:
                logger.warning(
                    f"[{metric_name}] Plot generation failed: {result.title} -> {result.error}",
                )

    return results


def _extract_table_label(table: str) -> str:
    """
    Extract a human-readable label from a table path.

    For paths like "/path/to/file.xlsx.Tables.July_2025", extracts "July_2025".
    For paths without .Tables., returns the filename stem.
    """
    if ".Tables." in table:
        return table.split(".Tables.")[-1]
    # Fallback: extract filename without extension
    parts = table.replace("\\", "/").split("/")
    filename = parts[-1] if parts else table
    # Remove common extensions
    for ext in [".xlsx", ".xls", ".csv", ".json"]:
        if filename.lower().endswith(ext):
            filename = filename[: -len(ext)]
            break
    return filename


def _format_title_with_label(base_title: str, label: Optional[str]) -> str:
    """
    Format a plot title with an optional table label suffix.

    Args:
        base_title: Original plot title (e.g., "Distance by Operative")
        label: Optional table label (e.g., "July_2025")

    Returns:
        Formatted title. If label is provided, appends it in parentheses:
        "Distance by Operative (July_2025)"
    """
    if not label:
        return base_title
    return f"{base_title} ({label})"


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


def _generate_single_plot(
    *,
    config: PlotConfig,
    project_name: str,
    context: str,
    filter_expr: Optional[str] = None,
    title_override: Optional[str] = None,
) -> PlotResult:
    """
    Generate a single plot via the Console Plot API.

    This function makes the actual HTTP request to the Plot API and
    handles response parsing and error capture.

    Args:
        config: PlotConfig defining the visualization parameters
        project_name: Unify project name
        context: Table context path
        filter_expr: Optional filter expression
        title_override: Optional title to use instead of config.title.
            Used when generating plots for multiple tables to include
            the table label in the title.

    Returns:
        PlotResult with either successful URL or error information.
        Never raises exceptions - errors are captured in the result.
    """
    # Use title_override if provided, otherwise fall back to config.title
    title = title_override or config.title or f"{config.type.value} chart"

    # Build plot_config dict for API request
    plot_config_dict = _build_plot_config_dict(config)

    # Build project_config dict for API request
    project_config_dict = _build_project_config_dict(
        project_name=project_name,
        context=context,
        filter_expr=filter_expr,
    )

    # Build full request body
    request_body = {
        "plot_config": plot_config_dict,
        "project_config": project_config_dict,
        "title": title,
    }

    try:
        # Make API request
        # TODO: Replace with orchestra client once Plot API is migrated
        response = httpx.post(
            PLOT_API_ENDPOINT,
            json=request_body,
            headers=_get_auth_headers(),
            timeout=PLOT_API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        # Build successful result
        return PlotResult(
            url=data.get("url"),
            token=data.get("token"),
            expires_in_hours=data.get("expires_in_hours"),
            plot_config=plot_config_dict,
            project_config=project_config_dict,
            title=config.title,
        )

    except httpx.HTTPStatusError as e:
        # HTTP error (4xx, 5xx)
        error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        logger.warning(f"Plot API HTTP error: {error_msg}")
        return PlotResult(
            plot_config=plot_config_dict,
            project_config=project_config_dict,
            title=config.title,
            error=error_msg,
            traceback=traceback.format_exc(),
        )

    except httpx.RequestError as e:
        # Network/connection error
        error_msg = f"Request error: {type(e).__name__}: {e}"
        logger.warning(f"Plot API request error: {error_msg}")
        return PlotResult(
            plot_config=plot_config_dict,
            project_config=project_config_dict,
            title=config.title,
            error=error_msg,
            traceback=traceback.format_exc(),
        )

    except Exception as e:
        # Unexpected error
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"Plot generation failed unexpectedly: {error_msg}")
        return PlotResult(
            plot_config=plot_config_dict,
            project_config=project_config_dict,
            title=config.title,
            error=error_msg,
            traceback=traceback.format_exc(),
        )


def _build_plot_config_dict(config: PlotConfig) -> dict:
    """
    Build the plot_config dictionary for the API request.

    Only includes non-None fields to keep the request minimal.

    Args:
        config: PlotConfig with visualization parameters

    Returns:
        Dictionary suitable for plot_config field in API request
    """
    result = {
        "type": config.type.value if hasattr(config.type, "value") else config.type,
        "x_axis": config.x_axis,
    }

    # Add optional fields if present
    if config.y_axis is not None:
        result["y_axis"] = config.y_axis
    if config.group_by is not None:
        result["group_by"] = config.group_by
    if config.aggregate is not None:
        result["aggregate"] = config.aggregate
    if config.scale_x is not None:
        result["scale_x"] = config.scale_x
    if config.scale_y is not None:
        result["scale_y"] = config.scale_y
    if config.metric is not None:
        result["metric"] = config.metric
    if config.bin_count is not None:
        result["bin_count"] = config.bin_count
    if config.show_regression is not None:
        result["show_regression"] = config.show_regression

    return result


def _build_project_config_dict(
    *,
    project_name: str,
    context: str,
    filter_expr: Optional[str] = None,
) -> dict:
    """
    Build the project_config dictionary for the API request.

    Args:
        project_name: Unify project name
        context: Table context path (same as FileManager table refs)
        filter_expr: Optional filter expression

    Returns:
        Dictionary suitable for project_config field in API request
    """
    result = {
        "project_name": project_name,
        "context": context,
        "randomize": False,  # Enable randomization for better sampling
    }

    if filter_expr is not None:
        result["filter_expr"] = filter_expr

    return result


def _get_auth_headers() -> dict:
    """
    Get authentication headers for the Plot API request.

    Uses the UNIFY_KEY environment variable for authentication.

    Returns:
        Dictionary with Authorization header
    """
    unify_key = os.environ.get("UNIFY_KEY", "")
    if not unify_key:
        logger.warning("UNIFY_KEY not set - Plot API requests may fail")

    return {
        "Authorization": f"Bearer {unify_key}",
        "Content-Type": "application/json",
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_active_project() -> str:
    """
    Get the currently active Unify project name.

    This function attempts to get the active project from the unify client.
    Falls back to empty string if not available.

    Returns:
        Project name string, or empty string if not available
    """
    try:
        import unify

        project = unify.active_project()
        return project if project else ""
    except Exception:
        return ""
