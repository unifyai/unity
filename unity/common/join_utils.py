"""
Common join utilities for multi-context operations.

This module provides generic join operations that work with any Unify contexts.
It extracts common join logic used by KnowledgeManager, FileManager, and DataManager.

All functions operate directly on Unify contexts (fully-qualified paths) and
do not perform any manager-specific context resolution.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

import unify

from .filter_utils import normalize_filter_expr
from .search_utils import table_search_top_k
from .embed_utils import list_private_fields


def rewrite_join_paths(
    original_tables: List[str],
    resolved_tables: List[str],
    join_expr: str,
    select: Dict[str, str],
) -> tuple[str, Dict[str, str]]:
    """Rewrite table paths in *join_expr* and *select* keys from originals to resolved.

    Generalised N-table version of :func:`rewrite_join_expr` /
    :func:`rewrite_select`.  Works by simple string replacement for each
    (original → resolved) pair, which is correct because table paths are
    unique prefix strings in these expressions.

    Parameters
    ----------
    original_tables : list[str]
        Table paths as originally provided (e.g. by the actor).
    resolved_tables : list[str]
        Corresponding fully-qualified context paths.
    join_expr : str
        Join condition expression with original table references.
    select : dict[str, str]
        Column mapping whose **keys** reference original table paths.

    Returns
    -------
    tuple[str, dict[str, str]]
        ``(rewritten_join_expr, rewritten_select)``.
    """
    for original, resolved in zip(original_tables, resolved_tables):
        if original != resolved:
            join_expr = join_expr.replace(original, resolved)
            select = {k.replace(original, resolved): v for k, v in select.items()}
    return join_expr, select


def create_join(
    *,
    left_context: str,
    right_context: str,
    dest_context: str,
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
) -> str:
    """
    Create a joined context from two source contexts.

    This is the core join operation that calls `unify.join_logs()` with the
    pair_of_args pattern.

    Parameters
    ----------
    left_context : str
        Fully-qualified Unify context for the left table.
    right_context : str
        Fully-qualified Unify context for the right table.
    dest_context : str
        Fully-qualified Unify context where join results will be stored.
    join_expr : str
        Join condition expression. Column references should use the context
        paths as prefixes (e.g., "LeftCtx.col == RightCtx.col").
    select : dict[str, str]
        Mapping of source columns to output column names.
        Keys should use context paths as prefixes (e.g., "LeftCtx.col": "alias").
    mode : str, default "inner"
        Join mode: "inner", "left", "right", or "outer".
    left_where : str | None
        Optional filter applied to left context before joining.
    right_where : str | None
        Optional filter applied to right context before joining.

    Returns
    -------
    str
        The destination context path.
    """
    unify.join_logs(
        pair_of_args=(
            {
                "context": left_context,
                **({} if left_where is None else {"filter_expr": left_where}),
            },
            {
                "context": right_context,
                **({} if right_where is None else {"filter_expr": right_where}),
            },
        ),
        join_expr=join_expr,
        mode=mode,
        new_context=dest_context,
        columns=select,
    )
    return dest_context


def rewrite_join_expr(
    expr: str,
    left_table: str,
    right_table: str,
    left_context: str,
    right_context: str,
) -> str:
    """
    Rewrite a join expression by replacing table references with context paths.

    Parameters
    ----------
    expr : str
        Original join expression using table names.
    left_table : str
        Logical name of the left table in the expression.
    right_table : str
        Logical name of the right table in the expression.
    left_context : str
        Fully-qualified context for the left table.
    right_context : str
        Fully-qualified context for the right table.

    Returns
    -------
    str
        Rewritten expression with context paths.
    """
    return expr.replace(left_table, left_context).replace(right_table, right_context)


def rewrite_select(
    select: Dict[str, str],
    left_table: str,
    right_table: str,
    left_context: str,
    right_context: str,
) -> Dict[str, str]:
    """
    Rewrite select column mappings by replacing table references with context paths.

    Parameters
    ----------
    select : dict[str, str]
        Original column mapping using table names.
    left_table : str
        Logical name of the left table.
    right_table : str
        Logical name of the right table.
    left_context : str
        Fully-qualified context for the left table.
    right_context : str
        Fully-qualified context for the right table.

    Returns
    -------
    dict[str, str]
        Rewritten column mapping with context paths.
    """
    return {
        c.replace(left_table, left_context).replace(right_table, right_context): v
        for c, v in select.items()
    }


def rewrite_filter(
    expr: Optional[str],
    table: str,
    context: str,
) -> Optional[str]:
    """
    Rewrite a filter expression by replacing table reference with context path.

    Parameters
    ----------
    expr : str | None
        Original filter expression.
    table : str
        Logical table name in the expression.
    context : str
        Fully-qualified context path.

    Returns
    -------
    str | None
        Rewritten expression or None if input was None.
    """
    if expr is None:
        return None
    return expr.replace(table, context)


def substitute_prev_placeholder(
    value: Optional[Union[str, List[str], Dict[str, str]]],
    previous_table: Optional[str],
) -> Optional[Union[str, List[str], Dict[str, str]]]:
    """
    Substitute $prev placeholders with the actual previous table name.

    Parameters
    ----------
    value : str | list[str] | dict[str, str] | None
        Value that may contain $prev placeholders.
    previous_table : str | None
        Name of the previous join result to substitute.

    Returns
    -------
    str | list[str] | dict[str, str] | None
        Value with placeholders substituted.
    """
    if value is None or previous_table is None:
        return value

    def repl(txt: str) -> str:
        return (
            txt.replace("$prev", previous_table)
            .replace("__prev__", previous_table)
            .replace("_.", f"{previous_table}.")
        )

    if isinstance(value, str):
        return repl(value)
    elif isinstance(value, dict):
        return {repl(k): v for k, v in value.items()}
    return [repl(c) for c in value]


def filter_join(
    *,
    left_context: str,
    right_context: str,
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
    tmp_context_prefix: str,
    cleanup: bool = True,
) -> List[Dict[str, Any]]:
    """
    Join two contexts and return filtered rows from the result.

    Creates a temporary join context, queries it, and optionally cleans up.

    Parameters
    ----------
    left_context : str
        Fully-qualified Unify context for the left table.
    right_context : str
        Fully-qualified Unify context for the right table.
    join_expr : str
        Join condition expression (already rewritten to use contexts).
    select : dict[str, str]
        Column mapping (already rewritten to use contexts).
    mode : str, default "inner"
        Join mode.
    left_where, right_where : str | None
        Pre-join filters for each side.
    result_where : str | None
        Post-join filter on the result.
    result_limit : int, default 100
        Maximum rows to return (must be <= 1000).
    result_offset : int, default 0
        Pagination offset.
    tmp_context_prefix : str
        Prefix for temporary context name (e.g., "Assistant/Knowledge/_tmp").
    cleanup : bool, default True
        Whether to delete the temporary context after querying.

    Returns
    -------
    list[dict[str, Any]]
        Rows from the joined result.

    Raises
    ------
    ValueError
        If result_limit > 1000.
    """
    if result_limit > 1000:
        raise ValueError("Limit must be <= 1000")

    dest_context = f"{tmp_context_prefix}/_tmp_join_{uuid.uuid4().hex[:8]}"

    create_join(
        left_context=left_context,
        right_context=right_context,
        dest_context=dest_context,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )

    try:
        excl = list_private_fields(dest_context)
        rows = [
            lg.entries
            for lg in unify.get_logs(
                context=dest_context,
                filter=result_where,
                offset=result_offset,
                limit=result_limit,
                exclude_fields=excl,
            )
        ]
        return rows
    finally:
        if cleanup:
            try:
                unify.delete_context(dest_context)
            except Exception:
                pass


def search_join(
    *,
    left_context: str,
    right_context: str,
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
    tmp_context_prefix: str,
    cleanup: bool = True,
) -> List[Dict[str, Any]]:
    """
    Join two contexts and return top-k semantic matches from the result.

    Creates a temporary join context, performs semantic search, and optionally cleans up.

    Parameters
    ----------
    left_context : str
        Fully-qualified Unify context for the left table.
    right_context : str
        Fully-qualified Unify context for the right table.
    join_expr : str
        Join condition expression (already rewritten to use contexts).
    select : dict[str, str]
        Column mapping (already rewritten to use contexts).
    mode : str, default "inner"
        Join mode.
    left_where, right_where : str | None
        Pre-join filters for each side.
    references : dict[str, str] | None
        Mapping of source_expr → reference_text for semantic similarity.
    k : int, default 10
        Number of rows to return (1..1000).
    filter : str | None
        Post-join filter before semantic search.
    tmp_context_prefix : str
        Prefix for temporary context name.
    cleanup : bool, default True
        Whether to delete the temporary context after querying.

    Returns
    -------
    list[dict[str, Any]]
        Top-k rows ranked by semantic similarity.

    Raises
    ------
    ValueError
        If k < 1 or k > 1000.
    """
    if k < 1 or k > 1000:
        raise ValueError("k must be between 1 and 1000")

    dest_context = f"{tmp_context_prefix}/_tmp_join_{uuid.uuid4().hex[:8]}"

    create_join(
        left_context=left_context,
        right_context=right_context,
        dest_context=dest_context,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )

    try:
        normalized = normalize_filter_expr(filter)
        rows = table_search_top_k(
            context=dest_context,
            references=references,
            k=k,
            row_filter=normalized,
        )
        return rows
    finally:
        if cleanup:
            try:
                unify.delete_context(dest_context)
            except Exception:
                pass


def filter_multi_join(
    *,
    joins: List[Dict[str, Any]],
    context_resolver: Any,  # Callable[[str], str] - resolves table name to context
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
    tmp_context_prefix: str,
    cleanup: bool = True,
) -> List[Dict[str, Any]]:
    """
    Execute a sequence of joins and return filtered rows from the final result.

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
    cleanup : bool, default True
        Whether to delete temporary contexts after querying.

    Returns
    -------
    list[dict[str, Any]]
        Rows from the final joined result.

    Raises
    ------
    ValueError
        If joins is empty, tables count != 2, or misplaced $prev.
    """
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if result_limit > 1000:
        raise ValueError("Limit must be <= 1000")

    tmp_tables: List[str] = []
    previous_table: Optional[str] = None

    for idx, step in enumerate(joins):
        local_step = step.copy()
        raw_tables = local_step.get("tables")
        raw_tables = [raw_tables] if isinstance(raw_tables, str) else raw_tables
        if not isinstance(raw_tables, list) or len(raw_tables) != 2:
            raise ValueError(
                f"Step {idx} must specify exactly TWO tables – got {raw_tables!r}",
            )

        # Substitute $prev placeholder
        step_tables = [
            (previous_table if t in {"$prev", "__prev__", "_"} else t)
            for t in raw_tables
        ]
        if any(t is None for t in step_tables):
            raise ValueError(
                "Misplaced `$prev` in first join – there is no previous result.",
            )

        left_table, right_table = step_tables

        # Resolve to contexts (skip resolution for $prev which is already a context)
        left_context = (
            left_table if left_table == previous_table else context_resolver(left_table)
        )
        right_context = (
            right_table
            if right_table == previous_table
            else context_resolver(right_table)
        )

        # Substitute $prev in join_expr and select
        join_expr = substitute_prev_placeholder(
            local_step.get("join_expr"),
            previous_table,
        )
        select = substitute_prev_placeholder(
            local_step.get("select"),
            previous_table,
        )

        # Rewrite to use contexts
        join_expr = rewrite_join_expr(
            join_expr,  # type: ignore[arg-type]
            left_table,
            right_table,
            left_context,
            right_context,
        )
        select = rewrite_select(
            select,  # type: ignore[arg-type]
            left_table,
            right_table,
            left_context,
            right_context,
        )

        # Rewrite pre-filters
        left_where = rewrite_filter(
            local_step.get("left_where"),
            left_table,
            left_context,
        )
        right_where = rewrite_filter(
            local_step.get("right_where"),
            right_table,
            right_context,
        )

        # Destination context for this step
        is_last = idx == len(joins) - 1
        dest_table = (
            f"{tmp_context_prefix}/_tmp_mjoin_{uuid.uuid4().hex[:6]}_final"
            if is_last
            else f"{tmp_context_prefix}/_tmp_mjoin_{uuid.uuid4().hex[:6]}_{idx}"
        )
        tmp_tables.append(dest_table)

        create_join(
            left_context=left_context,
            right_context=right_context,
            dest_context=dest_table,
            join_expr=join_expr,
            select=select,
            mode=local_step.get("mode", "inner"),
            left_where=left_where,
            right_where=right_where,
        )

        previous_table = dest_table

    assert previous_table is not None

    try:
        excl = list_private_fields(previous_table)
        rows = [
            lg.entries
            for lg in unify.get_logs(
                context=previous_table,
                filter=result_where,
                offset=result_offset,
                limit=result_limit,
                exclude_fields=excl,
            )
        ]
        return rows
    finally:
        if cleanup:
            for ctx in tmp_tables:
                try:
                    unify.delete_context(ctx)
                except Exception:
                    pass


def search_multi_join(
    *,
    joins: List[Dict[str, Any]],
    context_resolver: Any,  # Callable[[str], str]
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
    tmp_context_prefix: str,
    cleanup: bool = True,
) -> List[Dict[str, Any]]:
    """
    Execute a sequence of joins and return top-k semantic matches from the final result.

    Parameters
    ----------
    joins : list[dict]
        Ordered list of join steps (same format as filter_multi_join).
    context_resolver : callable
        Function that resolves table names to fully-qualified contexts.
    references : dict[str, str] | None
        Mapping of source_expr → reference_text for semantic similarity.
    k : int, default 10
        Number of rows to return (1..1000).
    filter : str | None
        Post-join filter before semantic search.
    tmp_context_prefix : str
        Prefix for temporary context names.
    cleanup : bool, default True
        Whether to delete temporary contexts after querying.

    Returns
    -------
    list[dict[str, Any]]
        Top-k rows ranked by semantic similarity.

    Raises
    ------
    ValueError
        If k < 1 or k > 1000, or join configuration is invalid.
    """
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if k < 1 or k > 1000:
        raise ValueError("k must be between 1 and 1000")

    tmp_tables: List[str] = []
    previous_table: Optional[str] = None

    for idx, step in enumerate(joins):
        local_step = step.copy()
        raw_tables = local_step.get("tables")
        raw_tables = [raw_tables] if isinstance(raw_tables, str) else raw_tables
        if not isinstance(raw_tables, list) or len(raw_tables) != 2:
            raise ValueError(
                f"Step {idx} must specify exactly TWO tables – got {raw_tables!r}",
            )

        # Substitute $prev placeholder
        step_tables = [
            (previous_table if t in {"$prev", "__prev__", "_"} else t)
            for t in raw_tables
        ]
        if any(t is None for t in step_tables):
            raise ValueError(
                "Misplaced `$prev` in first join – there is no previous result.",
            )

        left_table, right_table = step_tables

        # Resolve to contexts
        left_context = (
            left_table if left_table == previous_table else context_resolver(left_table)
        )
        right_context = (
            right_table
            if right_table == previous_table
            else context_resolver(right_table)
        )

        # Substitute $prev in join_expr and select
        join_expr = substitute_prev_placeholder(
            local_step.get("join_expr"),
            previous_table,
        )
        select = substitute_prev_placeholder(
            local_step.get("select"),
            previous_table,
        )

        # Rewrite to use contexts
        join_expr = rewrite_join_expr(
            join_expr,  # type: ignore[arg-type]
            left_table,
            right_table,
            left_context,
            right_context,
        )
        select = rewrite_select(
            select,  # type: ignore[arg-type]
            left_table,
            right_table,
            left_context,
            right_context,
        )

        # Rewrite pre-filters
        left_where = rewrite_filter(
            local_step.get("left_where"),
            left_table,
            left_context,
        )
        right_where = rewrite_filter(
            local_step.get("right_where"),
            right_table,
            right_context,
        )

        # Destination context for this step
        is_last = idx == len(joins) - 1
        dest_table = (
            f"{tmp_context_prefix}/_tmp_mjoin_{uuid.uuid4().hex[:6]}_final"
            if is_last
            else f"{tmp_context_prefix}/_tmp_mjoin_{uuid.uuid4().hex[:6]}_{idx}"
        )
        tmp_tables.append(dest_table)

        create_join(
            left_context=left_context,
            right_context=right_context,
            dest_context=dest_table,
            join_expr=join_expr,
            select=select,
            mode=local_step.get("mode", "inner"),
            left_where=left_where,
            right_where=right_where,
        )

        previous_table = dest_table

    assert previous_table is not None

    try:
        normalized = normalize_filter_expr(filter)
        rows = table_search_top_k(
            context=previous_table,
            references=references,
            k=k,
            row_filter=normalized,
        )
        return rows
    finally:
        if cleanup:
            for ctx in tmp_tables:
                try:
                    unify.delete_context(ctx)
                except Exception:
                    pass
