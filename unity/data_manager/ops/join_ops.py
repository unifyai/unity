"""
Join operations for DataManager.

Implementation functions for filter_join, search_join, filter_multi_join, search_multi_join.
These are called by DataManager methods and should not be used directly.

This module follows the exact same pattern as KnowledgeManager's search.py join implementation.
It uses join_utils from common/ for the actual join operations.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

import unify

from unity.common.filter_utils import normalize_filter_expr
from unity.common.search_utils import table_search_top_k
from unity.common.embed_utils import list_private_fields
from unity.data_manager.ops.query_ops import reduce_impl

logger = logging.getLogger(__name__)


def join_tables_impl(
    *,
    left_table: str,
    right_table: str,
    join_expr: str,
    dest_table: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
) -> str:
    """
    Implementation of join_tables operation.

    Creates a joined table from two source tables. This is the low-level
    primitive used by higher-level join operations (filter_join, search_join, etc.).

    Parameters
    ----------
    left_table : str
        Fully-qualified context path of the left table.
    right_table : str
        Fully-qualified context path of the right table.
    join_expr : str
        Join condition expression using table paths as prefixes.
        Example: "Data/orders.customer_id == Data/customers.id"
    dest_table : str
        Fully-qualified context path for the destination (result) table.
    select : dict[str, str]
        Mapping of source columns to output column names.
        Keys use table paths as prefixes; values are output aliases.
        Example: {"Data/orders.amount": "order_amount", "Data/customers.name": "customer_name"}
    mode : str, default "inner"
        Join mode: "inner", "left", "right", "outer".
    left_where : str | None
        Optional filter expression for left table (applied before joining).
    right_where : str | None
        Optional filter expression for right table (applied before joining).

    Returns
    -------
    str
        The destination table path.

    Notes
    -----
    - The destination table is created automatically.
    - Caller is responsible for cleaning up the destination table if temporary.
    - Column references in join_expr and select keys use full table paths as prefixes.
    """
    logger.debug(
        "join_tables: %s JOIN %s -> %s",
        left_table,
        right_table,
        dest_table,
    )

    unify.join_logs(
        pair_of_args=(
            {
                "context": left_table,
                **({} if left_where is None else {"filter_expr": left_where}),
            },
            {
                "context": right_table,
                **({} if right_where is None else {"filter_expr": right_where}),
            },
        ),
        join_expr=join_expr,
        mode=mode,
        new_context=dest_table,
        columns=select,
    )
    return dest_table


def _create_join(
    *,
    dest_context: str,
    tables: List[str],
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
) -> str:
    """
    Internal helper: create a joined context from a list of two tables.

    Delegates to join_tables_impl after unpacking the tables list.
    Used by filter_join_impl, search_join_impl, etc.
    """
    if len(tables) != 2:
        raise ValueError("Exactly TWO tables are required.")

    return join_tables_impl(
        left_table=tables[0],
        right_table=tables[1],
        join_expr=join_expr,
        dest_table=dest_context,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )


def filter_join_impl(
    *,
    tables: Union[str, List[str]],
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
    tmp_context_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Join two contexts and filter the result.

    Follows the exact same pattern as KnowledgeManager.filter_join.

    Parameters
    ----------
    tables : str | list[str]
        Exactly TWO fully-qualified context paths to join.
    join_expr : str
        Join condition expression using context paths as prefixes.
    select : dict[str, str]
        Column mapping from source (context.column) to output names.
    mode : str, default "inner"
        Join mode: "inner", "left", "right", "outer".
    left_where : str | None
        Pre-join filter for left table (applied before joining).
    right_where : str | None
        Pre-join filter for right table (applied before joining).
    result_where : str | None
        Post-join filter on the result (uses output column names from select).
    result_limit : int, default 100
        Maximum rows to return (must be <= 1000).
    result_offset : int, default 0
        Pagination offset.
    tmp_context_prefix : str
        Prefix for temporary context (e.g., "Data" or base context path).

    Returns
    -------
    list[dict[str, Any]]
        Joined and filtered rows.

    Raises
    ------
    ValueError
        If result_limit > 1000 or tables count != 2.
    """
    import re as _re

    if result_limit > 1000:
        raise ValueError("Limit must be <= 1000")

    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly TWO tables are required.")

    # Validate result_where references columns in select
    if result_where:

        def _qualified_refs(expr: str) -> set:
            return set(
                m.group(0)
                for m in _re.finditer(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", expr)
            )

        missing = _qualified_refs(result_where) - set(select)
        if missing:
            raise ValueError(
                "`result_where` references column(s) that are not present in `select`. "
                "Either add them to `select` or move the predicate to `left_where` / `right_where`. "
                f"Missing: {', '.join(sorted(missing))}",
            )

    logger.debug(
        "Filter join: %s JOIN %s",
        tables[0],
        tables[1],
    )

    dest_context = f"{tmp_context_prefix}/_tmp_join_{uuid.uuid4().hex[:8]}"
    _create_join(
        dest_context=dest_context,
        tables=tables,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )

    try:
        rows: List[Dict[str, Any]] = [
            lg.entries
            for lg in unify.get_logs(
                context=dest_context,
                filter=result_where,
                offset=result_offset,
                limit=result_limit,
                exclude_fields=list_private_fields(dest_context),
            )
        ]
        return rows
    finally:
        try:
            unify.delete_context(dest_context)
        except Exception:
            pass


def reduce_join_impl(
    *,
    tables: Union[str, List[str]],
    join_expr: str,
    select: Dict[str, str],
    metric: str,
    columns: Union[str, List[str]],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
    result_where: Optional[str] = None,
    group_by: Optional[Union[str, List[str]]] = None,
    tmp_context_prefix: str,
) -> Any:
    """
    Join two contexts and aggregate the result.

    Mirrors filter_join_impl but calls reduce_impl instead of get_logs.

    Parameters
    ----------
    tables : str | list[str]
        Exactly TWO fully-qualified context paths to join.
    join_expr : str
        Join condition expression using context paths as prefixes.
    select : dict[str, str]
        Column mapping from source (context.column) to output names.
    metric : str
        Reduction metric (count, sum, mean, etc.).
    columns : str | list[str]
        Column(s) to compute the metric on (output aliases from select).
    mode : str, default "inner"
        Join mode.
    left_where, right_where : str | None
        Pre-join filters for left/right tables.
    result_where : str | None
        Post-join filter on the result (uses output column names).
    group_by : str | list[str] | None
        Column(s) to group by before aggregation (output aliases).
    tmp_context_prefix : str
        Prefix for temporary context.

    Returns
    -------
    Any
        Scalar value or grouped dict depending on group_by.
    """
    import re as _re

    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly TWO tables are required.")

    if result_where:

        def _qualified_refs(expr: str) -> set:
            return set(
                m.group(0)
                for m in _re.finditer(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", expr)
            )

        missing = _qualified_refs(result_where) - set(select)
        if missing:
            raise ValueError(
                "`result_where` references column(s) that are not present in `select`. "
                "Either add them to `select` or move the predicate to `left_where` / `right_where`. "
                f"Missing: {', '.join(sorted(missing))}",
            )

    dest_context = f"{tmp_context_prefix}/_tmp_join_{uuid.uuid4().hex[:8]}"
    _create_join(
        dest_context=dest_context,
        tables=tables,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )

    try:
        return reduce_impl(
            dest_context,
            metric=metric,
            columns=columns,
            filter=result_where,
            group_by=group_by,
        )
    finally:
        try:
            unify.delete_context(dest_context)
        except Exception:
            pass


def search_join_impl(
    *,
    tables: Union[str, List[str]],
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
    tmp_context_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Join two contexts and perform semantic search.

    Follows the exact same pattern as KnowledgeManager.search_join.

    Parameters
    ----------
    tables : str | list[str]
        Exactly TWO fully-qualified context paths to join.
    join_expr : str
        Join condition expression using context paths.
    select : dict[str, str]
        Column mapping from source to output names.
    mode : str, default "inner"
        Join mode.
    left_where : str | None
        Pre-join filter for left table.
    right_where : str | None
        Pre-join filter for right table.
    references : dict[str, str] | None
        Mapping of column → reference text for semantic similarity.
    k : int, default 10
        Number of results (1..1000).
    filter : str | None
        Post-join filter before semantic search.
    tmp_context_prefix : str
        Prefix for temporary context.

    Returns
    -------
    list[dict[str, Any]]
        Top-k rows ranked by semantic similarity.
    """
    if k < 1 or k > 1000:
        raise ValueError("k must be between 1 and 1000")

    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly TWO tables are required.")

    logger.debug(
        "Search join: %s JOIN %s",
        tables[0],
        tables[1],
    )

    dest_context = f"{tmp_context_prefix}/_tmp_join_{uuid.uuid4().hex[:8]}"
    _create_join(
        dest_context=dest_context,
        tables=tables,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )

    try:
        normalized = normalize_filter_expr(filter)
        rows: List[Dict[str, Any]] = table_search_top_k(
            context=dest_context,
            references=references,
            k=k,
            row_filter=normalized,
        )
        return rows
    finally:
        try:
            unify.delete_context(dest_context)
        except Exception:
            pass


def filter_multi_join_impl(
    *,
    joins: List[Dict[str, Any]],
    context_resolver: Callable[[str], str],
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
    tmp_context_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Execute a sequence of joins and filter the final result.

    Follows the exact same pattern as KnowledgeManager.filter_multi_join.

    Parameters
    ----------
    joins : list[dict]
        Ordered list of join steps. Each step supports:
        - "tables" (list[str], required): Exactly two table names.
          Use "$prev", "__prev__", or "_" to reference previous result.
        - "join_expr" (str, required): Join predicate.
        - "select" (dict[str, str], required): Column mappings.
        - "mode" (str, optional): Join mode (default: "inner").
        - "left_where", "right_where" (str | None, optional): Pre-join filters.
    context_resolver : callable
        Function that resolves table names to fully-qualified contexts.
        Signature: (table_name: str) -> str
    result_where : str | None
        Post-join filter on the final result.
    result_limit : int, default 100
        Maximum rows to return (must be <= 1000).
    result_offset : int, default 0
        Pagination offset.
    tmp_context_prefix : str
        Prefix for temporary context names.

    Returns
    -------
    list[dict[str, Any]]
        Joined and filtered rows.
    """
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if result_limit > 1000:
        raise ValueError("Limit must be <= 1000")

    logger.debug("Multi-join with %d steps", len(joins))

    tmp_prefix = f"{tmp_context_prefix}/_tmp_mjoin_{uuid.uuid4().hex[:6]}"
    tmp_contexts: List[str] = []
    previous_ctx: Optional[str] = None
    final_select_names: Optional[set] = None

    for idx, step in enumerate(joins):
        local = step.copy()
        raw_tables = local.get("tables")
        raw_tables = [raw_tables] if isinstance(raw_tables, str) else raw_tables
        if not isinstance(raw_tables, list) or len(raw_tables) != 2:
            raise ValueError(
                f"Step {idx} must specify exactly TWO tables – got {raw_tables!r}",
            )

        # Substitute $prev placeholder
        step_tables = [
            (previous_ctx if t in {"$prev", "__prev__", "_"} else t) for t in raw_tables
        ]
        if any(t is None for t in step_tables):
            raise ValueError(
                "Misplaced `$prev` in first join – there is no previous result.",
            )

        # Fix-up join_expr & select that reference $prev
        def _replace_prev(s: Optional[Union[str, List[str], Dict[str, str]]]):
            if s is None or previous_ctx is None:
                return s

            def repl(txt: str) -> str:
                return (
                    txt.replace("$prev", previous_ctx)
                    .replace("__prev__", previous_ctx)
                    .replace("_.", f"{previous_ctx}.")
                )

            if isinstance(s, str):
                return repl(s)
            elif isinstance(s, dict):
                return {repl(k): v for k, v in s.items()}
            return [repl(c) for c in s]

        join_expr = _replace_prev(local.get("join_expr"))
        select = _replace_prev(local.get("select"))

        # Resolve contexts (skip resolution for $prev which is already a context)
        resolved_tables = []
        for i, t in enumerate(step_tables):
            if t == previous_ctx:
                resolved_tables.append(t)
            else:
                resolved_tables.append(context_resolver(t))

        # Destination context for this hop
        is_last = idx == len(joins) - 1
        dest_ctx = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
        tmp_contexts.append(dest_ctx)

        # Materialise the join
        _create_join(
            dest_context=dest_ctx,
            tables=resolved_tables,
            join_expr=join_expr,
            select=select,
            mode=local.get("mode", "inner"),
            left_where=local.get("left_where"),
            right_where=local.get("right_where"),
        )

        previous_ctx = dest_ctx
        if is_last and isinstance(select, dict):
            try:
                final_select_names = set(select.values())
            except Exception:
                final_select_names = None

    assert previous_ctx is not None

    # Validate result_where references final projection names
    if result_where and final_select_names is not None:
        import re as _re

        tokens = set(
            m.group(0) for m in _re.finditer(r"\b[A-Za-z_]\w*\b", result_where)
        )
        _reserved = {"and", "or", "not", "in", "is", "True", "False", "None"}
        candidate_cols = {t for t in tokens if t not in _reserved and not t.isdigit()}
        missing = candidate_cols - final_select_names
        if missing:
            raise ValueError(
                "`result_where` references column(s) not present in the final step's `select` mapping. "
                f"Missing: {', '.join(sorted(missing))}",
            )

    try:
        rows: List[Dict[str, Any]] = [
            lg.entries
            for lg in unify.get_logs(
                context=previous_ctx,
                filter=result_where,
                offset=result_offset,
                limit=result_limit,
                exclude_fields=list_private_fields(previous_ctx),
            )
        ]
        return rows
    finally:
        # Clean up temporary contexts
        for ctx in tmp_contexts:
            try:
                unify.delete_context(ctx)
            except Exception:
                pass


def search_multi_join_impl(
    *,
    joins: List[Dict[str, Any]],
    context_resolver: Callable[[str], str],
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
    tmp_context_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Execute a sequence of joins and perform semantic search.

    Follows the exact same pattern as KnowledgeManager.search_multi_join.

    Parameters
    ----------
    joins : list[dict]
        Ordered list of join steps (same format as filter_multi_join).
    context_resolver : callable
        Function that resolves table names to contexts.
    references : dict[str, str] | None
        Mapping of column → reference text for semantic similarity.
    k : int, default 10
        Number of results (1..1000).
    filter : str | None
        Post-join filter before semantic search.
    tmp_context_prefix : str
        Prefix for temporary contexts.

    Returns
    -------
    list[dict[str, Any]]
        Top-k rows ranked by semantic similarity.
    """
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if k < 1 or k > 1000:
        raise ValueError("k must be between 1 and 1000")

    logger.debug("Multi-join search with %d steps", len(joins))

    tmp_prefix = f"{tmp_context_prefix}/_tmp_mjoin_{uuid.uuid4().hex[:6]}"
    tmp_contexts: List[str] = []
    previous_ctx: Optional[str] = None

    for idx, step in enumerate(joins):
        local = step.copy()
        raw_tables = local.get("tables")
        raw_tables = [raw_tables] if isinstance(raw_tables, str) else raw_tables
        if not isinstance(raw_tables, list) or len(raw_tables) != 2:
            raise ValueError(
                f"Step {idx} must specify exactly TWO tables – got {raw_tables!r}",
            )

        # Substitute $prev placeholder
        step_tables = [
            (previous_ctx if t in {"$prev", "__prev__", "_"} else t) for t in raw_tables
        ]
        if any(t is None for t in step_tables):
            raise ValueError(
                "Misplaced `$prev` in first join – there is no previous result.",
            )

        # Fix-up join_expr & select that reference $prev
        def _replace_prev(s: Optional[Union[str, List[str], Dict[str, str]]]):
            if s is None or previous_ctx is None:
                return s

            def repl(txt: str) -> str:
                return (
                    txt.replace("$prev", previous_ctx)
                    .replace("__prev__", previous_ctx)
                    .replace("_.", f"{previous_ctx}.")
                )

            if isinstance(s, str):
                return repl(s)
            elif isinstance(s, dict):
                return {repl(k): v for k, v in s.items()}
            return [repl(c) for c in s]

        join_expr = _replace_prev(local.get("join_expr"))
        select = _replace_prev(local.get("select"))

        # Resolve contexts
        resolved_tables = []
        for i, t in enumerate(step_tables):
            if t == previous_ctx:
                resolved_tables.append(t)
            else:
                resolved_tables.append(context_resolver(t))

        # Destination context for this hop
        is_last = idx == len(joins) - 1
        dest_ctx = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
        tmp_contexts.append(dest_ctx)

        # Materialise the join
        _create_join(
            dest_context=dest_ctx,
            tables=resolved_tables,
            join_expr=join_expr,
            select=select,
            mode=local.get("mode", "inner"),
            left_where=local.get("left_where"),
            right_where=local.get("right_where"),
        )

        previous_ctx = dest_ctx

    assert previous_ctx is not None

    try:
        normalized = normalize_filter_expr(filter)
        rows: List[Dict[str, Any]] = table_search_top_k(
            context=previous_ctx,
            references=references,
            k=k,
            row_filter=normalized,
        )
        return rows
    finally:
        # Clean up temporary contexts
        for ctx in tmp_contexts:
            try:
                unify.delete_context(ctx)
            except Exception:
                pass
