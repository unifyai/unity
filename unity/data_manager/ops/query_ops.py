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
from unity.common.search_utils import table_search_top_k
from unity.common.metrics_utils import reduce_logs
from unity.common.embed_utils import list_private_fields

logger = logging.getLogger(__name__)


def filter_impl(
    context: str,
    *,
    filter: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Implementation of filter operation.

    Filters rows from a context by expression.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    filter : str | None
        Python boolean expression evaluated with column names in scope.
    columns : list[str] | None
        Specific columns to return. When None, returns all non-private columns.
    limit : int, default 100
        Maximum rows to return (must be <= 1000).
    offset : int, default 0
        Pagination offset.

    Returns
    -------
    list[dict[str, Any]]
        List of row dictionaries.

    Raises
    ------
    ValueError
        If limit > 1000.
    """
    if limit > 1000:
        raise ValueError("Limit must be <= 1000")

    logger.debug(
        "Filtering context=%s filter=%s limit=%d offset=%d",
        context,
        filter,
        limit,
        offset,
    )

    filter_expr = normalize_filter_expr(filter)

    # Determine fields to exclude (private fields) or include (specific columns)
    exclude_fields = None
    from_fields = None
    if columns is not None:
        from_fields = columns
    else:
        exclude_fields = list_private_fields(context)

    logs = unify.get_logs(
        context=context,
        filter=filter_expr,
        from_fields=from_fields,
        exclude_fields=exclude_fields,
        limit=limit,
        offset=offset,
    )

    # Extract entries from Log objects
    results = []
    for log in logs or []:
        if hasattr(log, "entries") and isinstance(log.entries, dict):
            results.append(log.entries)
        elif isinstance(log, dict):
            results.append(log)

    return results


def search_impl(
    context: str,
    *,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Implementation of search operation.

    Performs semantic search over embedded columns using common search utilities.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    references : dict[str, str] | None
        Mapping of source_column/expression → reference_text for semantic matching.
        Keys specify which columns to search (must have embeddings).
        Values are the reference text to match against.

        Examples:
        - ``{"text": "budget allocation"}`` — search the ``text`` column
        - ``{"content": "Q4 priorities", "summary": "budget"}`` — multi-column search

        When ``None`` or empty, returns rows without semantic ranking.
    k : int, default 10
        Number of rows to return (1..1000).
    filter : str | None
        Row-level predicate to filter before/during search.
    columns : list[str] | None
        Specific columns to return. When None, returns all non-private columns.

    Returns
    -------
    list[dict[str, Any]]
        Up to k rows ranked by semantic similarity (best match first).

    Raises
    ------
    ValueError
        If k < 1 or k > 1000.
    """
    if k < 1 or k > 1000:
        raise ValueError("k must be between 1 and 1000")

    logger.debug(
        "Searching context=%s references=%s k=%d",
        context,
        {
            k: v[:30] + "..." if len(v) > 30 else v
            for k, v in (references or {}).items()
        },
        k,
    )

    filter_expr = normalize_filter_expr(filter)

    # Use the common semantic search utility which handles:
    # - Embedding column creation/lookup
    # - Cosine similarity ranking
    # - Backfilling if similarity results are insufficient
    rows = table_search_top_k(
        context=context,
        references=references,
        k=k,
        row_filter=filter_expr,
        allowed_fields=columns,
    )

    return rows


def reduce_impl(
    context: str,
    *,
    metric: str,
    columns: Union[str, List[str]],
    filter: Optional[str] = None,
    group_by: Optional[Union[str, List[str]]] = None,
) -> Any:
    """
    Implementation of reduce operation.

    Computes aggregate metrics over rows using common metrics utilities.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    metric : str
        Reduction metric: "count", "sum", "mean", "var", "std",
        "min", "max", "median", "mode", "count_distinct".
    columns : str | list[str]
        Column(s) to compute the metric on. **Required parameter**.

        - Single column (str): Returns scalar or grouped list
        - Multiple columns (list[str]): Returns dict mapping column → value,
          or grouped list with all column values per group
    filter : str | None
        Row-level filter expression.
    group_by : str | list[str] | None
        Column(s) to group by. Results become list of dicts keyed by group values.

    Returns
    -------
    Any
        Metric value depends on columns and group_by:

        - Single column, no grouping → scalar (int for count, float for avg/sum)
        - Multiple columns, no grouping → dict {column_name: value}
        - Single column, with grouping → list of dicts [{group_col: val, metric: result}]
        - Multiple columns, with grouping → list of dicts with all metrics

    Raises
    ------
    ValueError
        If metric is not supported.
    """
    logger.debug(
        "Reducing context=%s metric=%s columns=%s group_by=%s",
        context,
        metric,
        columns,
        group_by,
    )

    # Use the common reduce_logs utility which handles:
    # - Metric validation
    # - Filter normalization
    # - Grouped vs ungrouped results
    # - Multiple column aggregation
    return reduce_logs(
        context=context,
        metric=metric,
        keys=columns,  # reduce_logs accepts str or list[str] for keys
        filter=filter,
        group_by=group_by,
    )
