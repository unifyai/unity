"""
Query operations for DataManager.

Implementation functions for filter, search, reduce.
These are called by DataManager methods and should not be used directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import unify

from unity.common.filter_utils import normalize_filter_expr

logger = logging.getLogger(__name__)


def filter_impl(
    context: str,
    *,
    filter: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: Optional[str] = None,
    descending: bool = False,
) -> List[Dict[str, Any]]:
    """
    Implementation of filter operation.

    Filters rows from a table by expression.
    """
    logger.debug(
        "Filtering context=%s filter=%s limit=%d offset=%d",
        context,
        filter,
        limit,
        offset,
    )

    # Normalize filter expression
    filter_expr = normalize_filter_expr(filter) if filter else None

    try:
        logs = unify.get_logs(
            context=context,
            filter=filter_expr,
            from_fields=columns,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.warning("Filter query failed: %s", e)
        return []

    # Extract entries from Log objects
    results = []
    for log in logs or []:
        if hasattr(log, "entries") and isinstance(log.entries, dict):
            results.append(log.entries)
        elif isinstance(log, dict):
            results.append(log)

    # Apply ordering if specified (post-query since Unify may not support it directly)
    if order_by and results:
        try:
            results = sorted(
                results,
                key=lambda r: r.get(order_by, ""),
                reverse=descending,
            )
        except Exception as e:
            logger.debug("Could not sort results: %s", e)

    return results


def search_impl(
    context: str,
    *,
    query: str,
    k: int = 10,
    filter: Optional[str] = None,
    vector_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Implementation of search operation.

    Performs semantic search over embedded column.
    """
    logger.debug(
        "Searching context=%s query=%s k=%d vector_column=%s",
        context,
        query[:50] if query else None,
        k,
        vector_column,
    )

    filter_expr = normalize_filter_expr(filter) if filter else None

    # Build references dict for semantic search
    # The vector_column determines which embedding to search
    ref_column = vector_column or "text"
    references = {ref_column: query}

    try:
        logs = unify.get_logs(
            context=context,
            references=references,
            k=k,
            filter=filter_expr,
            from_fields=columns,
        )
    except Exception as e:
        logger.warning("Search query failed: %s", e)
        return []

    # Extract entries from Log objects
    results = []
    for log in logs or []:
        if hasattr(log, "entries") and isinstance(log.entries, dict):
            entry = dict(log.entries)
            # Include similarity score if available
            if hasattr(log, "similarity"):
                entry["_similarity"] = log.similarity
            results.append(entry)
        elif isinstance(log, dict):
            results.append(log)

    return results


def reduce_impl(
    context: str,
    *,
    metric: str,
    column: Optional[str] = None,
    filter: Optional[str] = None,
    group_by: Optional[Union[str, List[str]]] = None,
) -> Any:
    """
    Implementation of reduce operation.

    Computes aggregate metrics over rows.
    """
    logger.debug(
        "Reducing context=%s metric=%s column=%s group_by=%s",
        context,
        metric,
        column,
        group_by,
    )

    filter_expr = normalize_filter_expr(filter) if filter else None

    # Normalize group_by to list
    group_by_list = None
    if group_by:
        group_by_list = [group_by] if isinstance(group_by, str) else list(group_by)

    # For count without column, use "id" as the default key
    key = column or "id"

    try:
        result = unify.get_logs_metric(
            metric=metric,
            key=key,
            context=context,
            filter=filter_expr,
            group_by=group_by_list,
        )
        return result
    except Exception as e:
        logger.warning("Reduce query failed: %s", e)
        # Return sensible default based on metric
        if metric == "count":
            return 0
        elif metric in ("sum", "avg", "mean"):
            return 0.0
        return None
