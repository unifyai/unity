"""
Join operations for DataManager.

Implementation functions for filter_join, search_join, filter_multi_join, search_multi_join.
These are called by DataManager methods and should not be used directly.

NOTE: Join operations may need to be implemented client-side if the Unify API
does not support them directly. The implementations below assume API support
or provide fallback client-side join logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import unify

from unity.common.filter_utils import normalize_filter_expr

logger = logging.getLogger(__name__)


def filter_join_impl(
    *,
    left_context: str,
    right_context: str,
    join_column: str,
    filter: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Implementation of filter_join operation.

    Joins two tables and filters the result.
    """
    logger.debug(
        "Filter join: %s JOIN %s ON %s",
        left_context,
        right_context,
        join_column,
    )

    filter_expr = normalize_filter_expr(filter) if filter else None

    # Try native join if available
    try:
        if hasattr(unify, "filter_join"):
            return unify.filter_join(
                left_context=left_context,
                right_context=right_context,
                join_column=join_column,
                filter_expr=filter_expr,
                columns=columns,
                limit=limit,
            )
    except Exception as e:
        logger.debug("Native filter_join not available: %s", e)

    # Fallback: client-side join
    return _client_side_filter_join(
        left_context=left_context,
        right_context=right_context,
        join_column=join_column,
        filter_expr=filter_expr,
        columns=columns,
        limit=limit,
    )


def _client_side_filter_join(
    *,
    left_context: str,
    right_context: str,
    join_column: str,
    filter_expr: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Client-side implementation of filter join."""
    # Fetch both tables
    left_rows = unify.get_logs(context=left_context, limit=10000)
    right_rows = unify.get_logs(context=right_context, limit=10000)

    # Extract entries
    left_data = []
    for log in left_rows or []:
        if hasattr(log, "entries"):
            left_data.append(log.entries)
        elif isinstance(log, dict):
            left_data.append(log)

    right_data = []
    for log in right_rows or []:
        if hasattr(log, "entries"):
            right_data.append(log.entries)
        elif isinstance(log, dict):
            right_data.append(log)

    # Build index on right table
    right_index: Dict[Any, List[Dict[str, Any]]] = {}
    for row in right_data:
        key = row.get(join_column)
        if key is not None:
            if key not in right_index:
                right_index[key] = []
            right_index[key].append(row)

    # Perform join
    results = []
    for left_row in left_data:
        key = left_row.get(join_column)
        if key in right_index:
            for right_row in right_index[key]:
                # Merge rows (left takes precedence for conflicts)
                merged = {**right_row, **left_row}
                results.append(merged)

    # Apply filter if provided
    if filter_expr:
        filtered = []
        for row in results:
            try:
                # Evaluate filter expression with row values in scope
                if eval(filter_expr, {"__builtins__": {}}, row):
                    filtered.append(row)
            except Exception:
                pass
        results = filtered

    # Select columns if specified
    if columns:
        results = [{k: row.get(k) for k in columns} for row in results]

    # Apply limit
    if limit:
        results = results[:limit]

    return results


def search_join_impl(
    *,
    left_context: str,
    right_context: str,
    join_column: str,
    query: str,
    k: int = 10,
    filter: Optional[str] = None,
    vector_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Implementation of search_join operation.

    Joins two tables and performs semantic search.
    """
    logger.debug(
        "Search join: %s JOIN %s ON %s, query=%s",
        left_context,
        right_context,
        join_column,
        query[:50] if query else None,
    )

    filter_expr = normalize_filter_expr(filter) if filter else None

    # Try native search_join if available
    try:
        if hasattr(unify, "search_join"):
            return unify.search_join(
                left_context=left_context,
                right_context=right_context,
                join_column=join_column,
                references={"query": query},
                k=k,
                filter_expr=filter_expr,
                vector_column=vector_column,
            )
    except Exception as e:
        logger.debug("Native search_join not available: %s", e)

    # Fallback: client-side join then search
    joined = _client_side_filter_join(
        left_context=left_context,
        right_context=right_context,
        join_column=join_column,
        filter_expr=filter_expr,
    )

    # TODO: Implement client-side semantic ranking
    # For now, return first k results
    return joined[:k]


def filter_multi_join_impl(
    *,
    contexts: List[str],
    join_columns: List[str],
    filter: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Implementation of filter_multi_join operation.

    Chains multiple joins across several tables.
    """
    if len(contexts) < 2:
        raise ValueError("filter_multi_join requires at least 2 contexts")
    if len(join_columns) != len(contexts) - 1:
        raise ValueError(
            f"Expected {len(contexts) - 1} join columns, got {len(join_columns)}",
        )

    logger.debug(
        "Multi-join: %s on columns %s",
        " -> ".join(contexts),
        join_columns,
    )

    filter_expr = normalize_filter_expr(filter) if filter else None

    # Try native multi-join if available
    try:
        if hasattr(unify, "filter_multi_join"):
            return unify.filter_multi_join(
                contexts=contexts,
                join_columns=join_columns,
                filter_expr=filter_expr,
                columns=columns,
                limit=limit,
            )
    except Exception as e:
        logger.debug("Native filter_multi_join not available: %s", e)

    # Fallback: chain binary joins
    result = _client_side_filter_join(
        left_context=contexts[0],
        right_context=contexts[1],
        join_column=join_columns[0],
    )

    # Continue joining with remaining tables
    for i, (ctx, jcol) in enumerate(zip(contexts[2:], join_columns[1:]), start=2):
        # Fetch next table
        next_rows = unify.get_logs(context=ctx, limit=10000)
        next_data = []
        for log in next_rows or []:
            if hasattr(log, "entries"):
                next_data.append(log.entries)
            elif isinstance(log, dict):
                next_data.append(log)

        # Build index
        next_index: Dict[Any, List[Dict[str, Any]]] = {}
        for row in next_data:
            key = row.get(jcol)
            if key is not None:
                if key not in next_index:
                    next_index[key] = []
                next_index[key].append(row)

        # Join
        new_result = []
        for row in result:
            key = row.get(jcol)
            if key in next_index:
                for next_row in next_index[key]:
                    merged = {**next_row, **row}
                    new_result.append(merged)
        result = new_result

    # Apply filter
    if filter_expr:
        filtered = []
        for row in result:
            try:
                if eval(filter_expr, {"__builtins__": {}}, row):
                    filtered.append(row)
            except Exception:
                pass
        result = filtered

    # Select columns
    if columns:
        result = [{k: row.get(k) for k in columns} for row in result]

    # Apply limit
    if limit:
        result = result[:limit]

    return result


def search_multi_join_impl(
    *,
    contexts: List[str],
    join_columns: List[str],
    query: str,
    k: int = 10,
    filter: Optional[str] = None,
    vector_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Implementation of search_multi_join operation.

    Chains multiple joins and performs semantic search.
    """
    if len(contexts) < 2:
        raise ValueError("search_multi_join requires at least 2 contexts")
    if len(join_columns) != len(contexts) - 1:
        raise ValueError(
            f"Expected {len(contexts) - 1} join columns, got {len(join_columns)}",
        )

    logger.debug(
        "Multi-join search: %s on columns %s, query=%s",
        " -> ".join(contexts),
        join_columns,
        query[:50] if query else None,
    )

    filter_expr = normalize_filter_expr(filter) if filter else None

    # Try native multi-join search if available
    try:
        if hasattr(unify, "search_multi_join"):
            return unify.search_multi_join(
                contexts=contexts,
                join_columns=join_columns,
                references={"query": query},
                k=k,
                filter_expr=filter_expr,
                vector_column=vector_column,
            )
    except Exception as e:
        logger.debug("Native search_multi_join not available: %s", e)

    # Fallback: multi-join then return first k
    joined = filter_multi_join_impl(
        contexts=contexts,
        join_columns=join_columns,
        filter=filter,
    )

    # TODO: Implement client-side semantic ranking
    return joined[:k]
