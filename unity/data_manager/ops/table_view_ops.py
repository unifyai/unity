"""
Table view operations for DataManager.

Implementation functions for table_view, table_view_batch.
These are called by DataManager methods and should not be used directly.

Architecture Note:
    Uses Console API (POST {ORCHESTRA_URL}/logs/table).
    Mirrors the plot_ops.py pattern for consistency.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx

from unity.data_manager.types.table_view import TableViewConfig, TableViewResult
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

logger = logging.getLogger(__name__)


# =============================================================================
# AUTHENTICATION
# =============================================================================


def _get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for the Table View API request."""
    unify_key = SESSION_DETAILS.unify_key
    if not unify_key:
        logger.warning(
            "UNIFY_KEY not set in SESSION_DETAILS - Table View API requests may fail",
        )

    return {
        "Authorization": f"Bearer {unify_key}",
        "Content-Type": "application/json",
    }


# =============================================================================
# REQUEST BUILDERS
# =============================================================================


def _build_table_config_dict(config: TableViewConfig) -> Dict[str, Any]:
    """
    Build the table_config dictionary for the API request.

    Only includes non-None fields to keep the request minimal.
    """
    result: Dict[str, Any] = {}
    columns: Dict[str, Any] = {}

    if config.columns_visible is not None:
        columns["visible"] = config.columns_visible
    if config.columns_hidden is not None:
        columns["hidden"] = config.columns_hidden
    if config.columns_order is not None:
        columns["order"] = config.columns_order

    if columns:
        result["columns"] = columns
    if config.row_limit is not None:
        result["row_limit"] = config.row_limit
    if config.sort_by is not None:
        result["sort_by"] = config.sort_by
    if config.sort_order is not None:
        result["sort_order"] = config.sort_order

    return result


def _build_project_config_dict(
    *,
    project_name: str,
    context: str,
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the project_config dictionary for the API request."""
    result: Dict[str, Any] = {
        "project_name": project_name,
        "context": context,
    }

    if filter_expr is not None:
        result["filter_expr"] = filter_expr

    return result


# =============================================================================
# HTTP CLIENT WITH RETRY
# =============================================================================


def _make_table_view_request(
    endpoint: str,
    request_body: Dict[str, Any],
    headers: Dict[str, str],
) -> httpx.Response:
    """Make HTTP POST request to Table View API with retry and exponential backoff."""
    max_retries = getattr(SETTINGS.file, "TABLE_VIEW_API_MAX_RETRIES", 3)
    backoff = getattr(SETTINGS.file, "TABLE_VIEW_API_RETRY_BACKOFF", 1.0)
    timeout = getattr(SETTINGS.file, "TABLE_VIEW_API_TIMEOUT", 30.0)

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                endpoint,
                json=request_body,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = backoff * (2**attempt)
                logger.warning(
                    f"Table View API request failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time:.1f}s: {e}",
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Table View API request failed after {max_retries} attempts: {e}",
                )

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected state: no exception captured after retries")


# =============================================================================
# TABLE VIEW GENERATION
# =============================================================================


def _get_active_project() -> str:
    """Get the currently active Unify project name."""
    try:
        import unify

        project = unify.active_project()
        return project if project else ""
    except Exception:
        return ""


def generate_table_view(
    *,
    config: TableViewConfig,
    context: str,
    filter_expr: Optional[str] = None,
    project_name: Optional[str] = None,
    title: Optional[str] = None,
    title_suffix: Optional[str] = None,
) -> TableViewResult:
    """
    Generate a single table view via the Console Table View API.

    Parameters
    ----------
    config : TableViewConfig
        Configuration defining columns, sorting, and row limits.
    context : str
        Fully-qualified table context path.
    filter_expr : str | None
        Optional filter expression.
    project_name : str | None
        Unify project name. If None, uses the active project.
    title : str | None
        Title for the table view.
    title_suffix : str | None
        Optional suffix to append to the title (e.g., table label).

    Returns
    -------
    TableViewResult
        Result with either successful URL or error information.
    """
    if project_name is None:
        project_name = _get_active_project()
        if not project_name:
            return TableViewResult(
                error="No active Unify project. Set project with unify.activate().",
                title=title,
                context=context,
            )

    final_title = title or "Table View"
    if title_suffix:
        final_title = f"{final_title} ({title_suffix})"

    table_config_dict = _build_table_config_dict(config)
    project_config_dict = _build_project_config_dict(
        project_name=project_name,
        context=context,
        filter_expr=filter_expr,
    )

    request_body: Dict[str, Any] = {
        "project_config": project_config_dict,
        "title": final_title,
    }
    if table_config_dict:
        request_body["table_config"] = table_config_dict

    try:
        orchestra_base_url = SETTINGS.ORCHESTRA_URL
        table_view_endpoint = SETTINGS.file.TABLE_VIEW_API_ENDPOINT
        endpoint = f"{orchestra_base_url}{table_view_endpoint}"

        response = _make_table_view_request(
            endpoint=endpoint,
            request_body=request_body,
            headers=_get_auth_headers(),
        )
        data = response.json()

        return TableViewResult(
            url=data.get("url"),
            token=data.get("token"),
            title=final_title,
            context=context,
        )

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        logger.warning(
            f"Table View API HTTP error: {error_msg}. Request Body: {request_body}",
        )
        return TableViewResult(
            title=final_title,
            context=context,
            error=error_msg,
            traceback_str=traceback.format_exc(),
        )

    except httpx.RequestError as e:
        error_msg = f"Request error: {type(e).__name__}: {e}"
        logger.warning(
            f"Table View API request error: {error_msg}. Request Body: {request_body}",
        )
        return TableViewResult(
            title=final_title,
            context=context,
            error=error_msg,
            traceback_str=traceback.format_exc(),
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(
            f"Table view creation failed unexpectedly: {error_msg}. Request Body: {request_body}",
        )
        return TableViewResult(
            title=final_title,
            context=context,
            error=error_msg,
            traceback_str=traceback.format_exc(),
        )


def generate_table_views_batch(
    *,
    contexts: List[str],
    config: TableViewConfig,
    filter_expr: Optional[str] = None,
    project_name: Optional[str] = None,
    title: Optional[str] = None,
) -> List[TableViewResult]:
    """
    Generate table views for multiple tables with the same configuration.

    Parameters
    ----------
    contexts : list[str]
        List of fully-qualified context paths.
    config : TableViewConfig
        Configuration defining columns, sorting, and row limits.
    filter_expr : str | None
        Optional filter expression.
    project_name : str | None
        Unify project name. If None, uses the active project.
    title : str | None
        Base title for the table views.

    Returns
    -------
    list[TableViewResult]
        List of results, one per context.
    """
    if project_name is None:
        project_name = _get_active_project()
        if not project_name:
            return [
                TableViewResult(
                    error="No active Unify project. Set project with unify.activate().",
                    title=title,
                    context=ctx,
                )
                for ctx in contexts
            ]

    results: List[TableViewResult] = []

    for context in contexts:
        table_label = context.rsplit("/", 1)[-1] if "/" in context else context

        result = generate_table_view(
            config=config,
            context=context,
            filter_expr=filter_expr,
            project_name=project_name,
            title=title,
            title_suffix=table_label if len(contexts) > 1 else None,
        )

        results.append(result)

        if result.succeeded:
            logger.info(f"Created table view: {result.title} -> {result.url}")
        else:
            logger.warning(
                f"Table view creation failed: {result.title} -> {result.error}",
            )

    return results
