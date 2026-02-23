"""
Plot operations for DataManager.

Implementation functions for plot, plot_batch.
These are called by DataManager methods and should not be used directly.

Architecture Note:
    Currently uses Console API (POST {ORCHESTRA_URL}/logs/plot).
    This is the integration path until an equivalent path exists through unify.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx

from unity.data_manager.types.plot import PlotConfig, PlotResult
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

logger = logging.getLogger(__name__)


# =============================================================================
# AUTHENTICATION
# =============================================================================


def _get_auth_headers() -> Dict[str, str]:
    """
    Get authentication headers for the Plot API request.

    Uses SESSION_DETAILS.unify_key for authentication.
    """
    unify_key = SESSION_DETAILS.unify_key
    if not unify_key:
        logger.warning(
            "UNIFY_KEY not set in SESSION_DETAILS - Plot API requests may fail",
        )

    return {
        "Authorization": f"Bearer {unify_key}",
        "Content-Type": "application/json",
    }


# =============================================================================
# REQUEST BUILDERS
# =============================================================================


def _build_plot_config_dict(config: PlotConfig) -> Dict[str, Any]:
    """
    Build the plot_config dictionary for the API request.

    Only includes non-None fields to keep the request minimal.
    """
    result: Dict[str, Any] = {
        "type": config.plot_type,
        "x_axis": config.x_axis,
    }

    if config.y_axis is not None:
        result["y_axis"] = config.y_axis
    if config.group_by is not None:
        result["group_by"] = config.group_by
    if config.metric is not None:
        result["metric"] = config.metric
    if config.aggregate is not None:
        result["aggregate"] = config.aggregate
    if config.scale_x is not None:
        result["scale_x"] = config.scale_x
    if config.scale_y is not None:
        result["scale_y"] = config.scale_y
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
    randomize: bool = False,
    exclude_fields: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the project_config dictionary for the API request.
    """
    result: Dict[str, Any] = {
        "project_name": project_name,
        "context": context,
        "randomize": randomize,
    }

    if filter_expr is not None:
        result["filter_expr"] = filter_expr
    if exclude_fields is not None:
        result["exclude_fields"] = exclude_fields
    if group_by is not None:
        result["group_by"] = group_by

    return result


# =============================================================================
# HTTP CLIENT WITH RETRY
# =============================================================================


def _make_plot_request(
    endpoint: str,
    request_body: Dict[str, Any],
    headers: Dict[str, str],
) -> httpx.Response:
    """
    Make HTTP POST request to Plot API with retry and exponential backoff.
    """
    max_retries = getattr(SETTINGS.file, "PLOT_API_MAX_RETRIES", 3)
    backoff = getattr(SETTINGS.file, "PLOT_API_RETRY_BACKOFF", 1.0)
    timeout = getattr(SETTINGS.file, "PLOT_API_TIMEOUT", 30.0)

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
                    f"Plot API request failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time:.1f}s: {e}",
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Plot API request failed after {max_retries} attempts: {e}",
                )

    # Re-raise the last exception
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected state: no exception captured after retries")


# =============================================================================
# PLOT GENERATION
# =============================================================================


def _get_active_project() -> str:
    """
    Get the currently active Unify project name.
    """
    try:
        import unify

        project = unify.active_project()
        return project if project else ""
    except Exception:
        return ""


def generate_plot(
    *,
    config: PlotConfig,
    context: str,
    filter_expr: Optional[str] = None,
    project_name: Optional[str] = None,
    randomize: bool = False,
    exclude_fields: Optional[List[str]] = None,
    project_group_by: Optional[str] = None,
    title_suffix: Optional[str] = None,
) -> PlotResult:
    """
    Generate a single plot via the Console Plot API.

    This function makes the HTTP request to the Plot API and handles
    response parsing and error capture.

    Parameters
    ----------
    config : PlotConfig
        PlotConfig defining the visualization parameters.
    context : str
        Fully-qualified table context path.
    filter_expr : str | None
        Optional filter expression.
    project_name : str | None
        Unify project name. If None, uses the active project.
    randomize : bool
        Whether to randomize the data order.
    exclude_fields : list[str] | None
        Fields to exclude from the response.
    project_group_by : str | None
        Column to group data by in the project config.
    title_suffix : str | None
        Optional suffix to append to the title (e.g., table label).

    Returns
    -------
    PlotResult
        Result with either successful URL or error information.
        Never raises exceptions - errors are captured in the result.
    """
    # Get project name if not provided
    if project_name is None:
        project_name = _get_active_project()
        if not project_name:
            return PlotResult(
                error="No active Unify project. Set project with unify.activate().",
                title=config.title,
                context=context,
            )

    # Compute final title
    base_title = config.title or f"{config.plot_type} chart"
    title = f"{base_title} ({title_suffix})" if title_suffix else base_title

    # Build request
    plot_config_dict = _build_plot_config_dict(config)
    project_config_dict = _build_project_config_dict(
        project_name=project_name,
        context=context,
        filter_expr=filter_expr,
        randomize=randomize,
        exclude_fields=exclude_fields,
        group_by=project_group_by,
    )

    request_body = {
        "plot_config": plot_config_dict,
        "project_config": project_config_dict,
        "title": title,
    }

    try:
        orchestra_base_url = SETTINGS.ORCHESTRA_URL
        plot_endpoint = SETTINGS.file.PLOT_API_ENDPOINT
        endpoint = f"{orchestra_base_url}{plot_endpoint}"

        response = _make_plot_request(
            endpoint=endpoint,
            request_body=request_body,
            headers=_get_auth_headers(),
        )
        data = response.json()

        return PlotResult(
            url=data.get("url"),
            token=data.get("token"),
            expires_in_hours=data.get("expires_in_hours"),
            title=title,
            context=context,
        )

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        logger.warning(
            f"Plot API HTTP error: {error_msg}. Request Body: {request_body}",
        )
        return PlotResult(
            title=title,
            context=context,
            error=error_msg,
            traceback_str=traceback.format_exc(),
        )

    except httpx.RequestError as e:
        error_msg = f"Request error: {type(e).__name__}: {e}"
        logger.warning(
            f"Plot API request error: {error_msg}. Request Body: {request_body}",
        )
        return PlotResult(
            title=title,
            context=context,
            error=error_msg,
            traceback_str=traceback.format_exc(),
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(
            f"Plot generation failed unexpectedly: {error_msg}. Request Body: {request_body}",
        )
        return PlotResult(
            title=title,
            context=context,
            error=error_msg,
            traceback_str=traceback.format_exc(),
        )


def generate_plots_batch(
    *,
    contexts: List[str],
    config: PlotConfig,
    filter_expr: Optional[str] = None,
    project_name: Optional[str] = None,
    randomize: bool = False,
    exclude_fields: Optional[List[str]] = None,
    project_group_by: Optional[str] = None,
) -> List[PlotResult]:
    """
    Generate plots for multiple tables with the same configuration.

    This function iterates over the provided contexts and generates a plot
    for each using the same configuration. Useful for tables with identical
    schemas (e.g., monthly data tables).

    Parameters
    ----------
    contexts : list[str]
        List of fully-qualified context paths.
    config : PlotConfig
        PlotConfig defining the visualization parameters.
    filter_expr : str | None
        Optional filter expression.
    project_name : str | None
        Unify project name. If None, uses the active project.
    randomize : bool
        Whether to randomize the data order.
    exclude_fields : list[str] | None
        Fields to exclude from the response.
    project_group_by : str | None
        Column to group data by in the project config.

    Returns
    -------
    list[PlotResult]
        List of PlotResult objects, one per context.
    """
    # Get project name if not provided
    if project_name is None:
        project_name = _get_active_project()
        if not project_name:
            # Return error results for all contexts
            base_title = config.title or f"{config.plot_type} chart"
            return [
                PlotResult(
                    error="No active Unify project. Set project with unify.activate().",
                    title=base_title,
                    context=ctx,
                )
                for ctx in contexts
            ]

    results: List[PlotResult] = []

    for context in contexts:
        # Extract table label from the resolved context (last part after final "/")
        table_label = context.rsplit("/", 1)[-1] if "/" in context else context

        # Generate plot for this context
        result = generate_plot(
            config=config,
            context=context,
            filter_expr=filter_expr,
            project_name=project_name,
            randomize=randomize,
            exclude_fields=exclude_fields,
            project_group_by=project_group_by,
            title_suffix=table_label if len(contexts) > 1 else None,
        )

        results.append(result)

        # Log result
        if result.succeeded:
            logger.info(f"Generated plot: {result.title} -> {result.url}")
        else:
            logger.warning(f"Plot generation failed: {result.title} -> {result.error}")

    return results
