from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import table_search_top_k
from ..common.grouping_helpers import maybe_group_rows
from ..common.embed_utils import list_private_fields
from .storage import ctx_for_table

if TYPE_CHECKING:
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager


def filter(
    knowledge_manager: "KnowledgeManager",
    *,
    filter: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    tables: Optional[Union[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter rows from one or more tables using a Python boolean expression.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    filter : str | None
        Row-level predicate evaluated with column names as variables.
        ``None`` returns all rows.
    offset : int
        Pagination offset (0-based).
    limit : int
        Maximum rows per table. Must be <= 1000.
    tables : str | list[str] | None
        Subset of tables to scan; ``None`` -> all tables.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Mapping ``table_name -> [row_dict, ...]``.

    Raises
    ------
    ValueError
        If limit exceeds 1000.
    """
    if limit > 1000:
        raise ValueError("Limit must be less than 1000")

    dm = knowledge_manager._data_manager

    # Resolve target tables without triggering per-table field reads
    if tables is None:
        km_prefix = f"{knowledge_manager._ctx}/"
        ctx_list = dm.list_tables(prefix=km_prefix, include_column_info=False)
        if isinstance(ctx_list, dict):
            resolved_tables = [k[len(km_prefix) :] for k in ctx_list.keys()]
        else:
            resolved_tables = [k[len(km_prefix) :] for k in ctx_list]

        # Optionally expose root-level Contacts when linkage is enabled
        if (
            getattr(knowledge_manager, "_include_contacts", False)
            and getattr(knowledge_manager, "_contacts_ctx", None) is not None
        ):
            resolved_tables.append("Contacts")
    elif isinstance(tables, str):
        resolved_tables = [tables]
    else:
        resolved_tables = list(tables)

    def _fetch_one(table_name: str) -> tuple[str, List[Dict[str, Any]]]:
        ctx = ctx_for_table(knowledge_manager, table_name)
        excl = list_private_fields(ctx)
        normalized = normalize_filter_expr(filter)
        # Delegate to DataManager.filter
        rows = dm.filter(
            ctx,
            filter=normalized,
            limit=limit,
            offset=offset,
        )
        # Exclude private fields from results
        filtered_rows = [
            {k: v for k, v in row.items() if k not in excl} for row in rows
        ]
        return table_name, filtered_rows

    results: Dict[str, List[Dict[str, Any]]] = {}

    # Parallelise when scanning multiple tables to reduce wall-clock time
    if len(resolved_tables) <= 1 and resolved_tables:
        name, rows = _fetch_one(resolved_tables[0])
        ctx = ctx_for_table(knowledge_manager, name)
        results[name] = maybe_group_rows(
            rows=rows,
            exclude_fields=list_private_fields(ctx),
            enabled=getattr(knowledge_manager, "_group_results", False),
        )
        return results

    max_workers = min(8, max(1, len(resolved_tables)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, table_name): table_name
            for table_name in resolved_tables
        }
        for fut in as_completed(futures):
            name, rows = fut.result()
            ctx = ctx_for_table(knowledge_manager, name)
            results[name] = maybe_group_rows(
                rows=rows,
                exclude_fields=list_private_fields(ctx),
                enabled=getattr(knowledge_manager, "_group_results", False),
            )

    return dict(sorted(results.items()))


def search(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search within a single knowledge table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        The table to search within.
    references : dict[str, str] | None
        Mapping from source expression to reference text for comparison.
    k : int
        Maximum number of rows to return. Must be between 1 and 1000.
    filter : str | None
        Optional row-level predicate.

    Returns
    -------
    list[dict[str, Any]]
        Up to ``k`` rows sorted by ascending semantic distance.

    Raises
    ------
    ValueError
        If k is not between 1 and 1000.
    """
    if k > 1000 or k < 1:
        raise ValueError("k must be between 1 and 1000")

    context = ctx_for_table(knowledge_manager, table)
    normalized = normalize_filter_expr(filter)

    # Use common search utility (already delegates appropriately)
    rows: List[Dict[str, Any]] = table_search_top_k(
        context=context,
        references=references,
        k=k,
        row_filter=normalized,
    )
    return maybe_group_rows(
        rows=rows,
        exclude_fields=list_private_fields(context),
        enabled=getattr(knowledge_manager, "_group_results", False),
    )


def _create_join(
    knowledge_manager: "KnowledgeManager",
    *,
    dest_table: str,
    tables: Union[str, List[str]],
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
) -> str:
    """
    Create a join and store results in a destination table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    dest_table : str
        Name for the destination table.
    tables : str | list[str]
        Exactly TWO table names.
    join_expr : str
        Join condition expression.
    select : dict[str, str]
        Column selection mapping.
    mode : str
        Join mode (inner, left, right, outer).
    left_where : str | None
        Pre-filter for left table.
    right_where : str | None
        Pre-filter for right table.

    Returns
    -------
    str
        The destination context path.
    """
    dm = knowledge_manager._data_manager

    # Resolve & validate the inputs
    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly TWO tables are required.")

    left_table, right_table = tables
    left_ctx = ctx_for_table(knowledge_manager, left_table)
    right_ctx = ctx_for_table(knowledge_manager, right_table)

    # Rewrite pre-filters to fully-qualified contexts
    def _rewrite_filter(expr: Optional[str], table: str, ctx: str) -> Optional[str]:
        return None if expr is None else expr.replace(table, ctx)

    left_where = _rewrite_filter(left_where, left_table, left_ctx)
    right_where = _rewrite_filter(right_where, right_table, right_ctx)

    # Fully-qualify the join expression and selected columns
    join_expr = join_expr.replace(left_table, left_ctx).replace(right_table, right_ctx)
    select = {
        c.replace(left_table, left_ctx).replace(right_table, right_ctx): v
        for c, v in select.items()
    }

    # Destination context
    dest_ctx = ctx_for_table(knowledge_manager, dest_table)

    # Delegate to DataManager.join_tables
    dm.join_tables(
        left_table=left_ctx,
        right_table=right_ctx,
        join_expr=join_expr,
        dest_table=dest_ctx,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )
    return dest_ctx


def search_join(
    knowledge_manager: "KnowledgeManager",
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
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search over the result of joining two tables.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    tables : list[str]
        Exactly two table names.
    join_expr : str
        Boolean join condition.
    select : dict[str, str]
        Mapping of source columns to output column names.
    mode : str
        Join mode (inner, left, right, outer).
    left_where : str | None
        Pre-filter for left table.
    right_where : str | None
        Pre-filter for right table.
    references : dict[str, str] | None
        Mapping of expressions to reference text for semantic similarity.
    k : int
        Maximum number of rows to return.
    filter : str | None
        Row-level predicate on the joined result.

    Returns
    -------
    list[dict[str, Any]]
        Up to ``k`` rows from the joined result.

    Raises
    ------
    ValueError
        If k is not between 1 and 1000.
    """
    if k > 1000 or k < 1:
        raise ValueError("k must be between 1 and 1000")

    dm = knowledge_manager._data_manager
    dest_table = f"_tmp_join_{uuid.uuid4().hex[:8]}"
    dest_ctx = _create_join(
        knowledge_manager,
        dest_table=dest_table,
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
            context=dest_ctx,
            references=references,
            k=k,
            row_filter=normalized,
        )
        return maybe_group_rows(
            rows=rows,
            exclude_fields=list_private_fields(dest_ctx),
            enabled=getattr(knowledge_manager, "_group_results", False),
        )
    finally:
        dm.delete_table(dest_ctx, dangerous_ok=True)


def search_multi_join(
    knowledge_manager: "KnowledgeManager",
    *,
    joins: List[Dict[str, Any]],
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search over the result of chaining multiple joins.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    joins : list[dict]
        Ordered list of join steps.
    references : dict[str, str] | None
        Mapping of expressions to reference text for semantic similarity.
    k : int
        Maximum number of rows to return.
    filter : str | None
        Row-level predicate on the final result.

    Returns
    -------
    list[dict[str, Any]]
        Up to ``k`` rows from the final joined result.

    Raises
    ------
    ValueError
        If joins is empty or k is out of range.
    """
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if k > 1000 or k < 1:
        raise ValueError("k must be between 1 and 1000")

    dm = knowledge_manager._data_manager
    tmp_prefix = f"_tmp_mjoin_{uuid.uuid4().hex[:6]}"
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

        # Substitute `$prev` placeholder
        step_tables = [
            (previous_table if t in {"$prev", "__prev__", "_"} else t)
            for t in raw_tables
        ]
        if any(t is None for t in step_tables):
            raise ValueError(
                "Misplaced `$prev` in first join – there is no previous result.",
            )

        # Fix-up join_expr & columns that reference `$prev`
        def _replace_prev(s: Optional[Union[str, List[str], Dict[str, str]]]):
            if s is None or previous_table is None:
                return s

            def repl(txt: str) -> str:
                return (
                    txt.replace("$prev", previous_table)
                    .replace("__prev__", previous_table)
                    .replace("_.", f"{previous_table}.")
                )

            if isinstance(s, str):
                return repl(s)
            elif isinstance(s, dict):
                return {repl(k): v for k, v in s.items()}
            return [repl(c) for c in s]

        join_expr_step = _replace_prev(local_step.get("join_expr"))
        select_step = _replace_prev(local_step.get("select"))

        # Destination table for this hop
        is_last = idx == len(joins) - 1
        dest_table = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
        tmp_tables.append(dest_table)

        # Materialise the join
        _create_join(
            knowledge_manager,
            dest_table=dest_table,
            tables=step_tables,
            join_expr=join_expr_step,  # type: ignore[arg-type]
            select=select_step,  # type: ignore[arg-type]
            mode=local_step.get("mode", "inner"),
            left_where=local_step.get("left_where"),
            right_where=local_step.get("right_where"),
        )

        previous_table = dest_table

    assert previous_table is not None

    final_ctx = ctx_for_table(knowledge_manager, previous_table)
    try:
        normalized = normalize_filter_expr(filter)
        rows: List[Dict[str, Any]] = table_search_top_k(
            context=final_ctx,
            references=references,
            k=k,
            row_filter=normalized,
        )
        return maybe_group_rows(
            rows=rows,
            exclude_fields=list_private_fields(final_ctx),
            enabled=getattr(knowledge_manager, "_group_results", False),
        )
    finally:
        # Clean up intermediate tables
        for tbl in tmp_tables:
            try:
                ctx = ctx_for_table(knowledge_manager, tbl)
                dm.delete_table(ctx, dangerous_ok=True)
            except Exception:
                pass


def filter_join(
    knowledge_manager: "KnowledgeManager",
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
) -> List[Dict[str, Any]]:
    """
    Join two tables and return rows with optional filtering.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    tables : list[str]
        Exactly two table names.
    join_expr : str
        Boolean join condition.
    select : dict[str, str]
        Mapping of source columns to output column names.
    mode : str
        Join mode (inner, left, right, outer).
    left_where : str | None
        Pre-filter for left table.
    right_where : str | None
        Pre-filter for right table.
    result_where : str | None
        Filter on the joined result.
    result_limit : int
        Maximum rows to return.
    result_offset : int
        Pagination offset.

    Returns
    -------
    list[dict[str, Any]]
        Rows from the joined result.

    Raises
    ------
    ValueError
        If result_limit exceeds 1000 or result_where references invalid columns.
    """
    import re as _re

    if result_limit > 1000:
        raise ValueError("Limit must be less than 1000")

    dm = knowledge_manager._data_manager

    # Helper to catch mismatches early
    def _qualified_refs(expr: str) -> set[str]:
        return set(
            m.group(0) for m in _re.finditer(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", expr)
        )

    if result_where:
        missing = _qualified_refs(result_where) - set(select)
        if missing:
            raise ValueError(
                "`result_where` references column(s) that are not present in `select`. "
                "Either add them to `select` or move the predicate to `left_where` / `right_where`. "
                f"Missing: {', '.join(sorted(missing))}",
            )

    dest_table = f"_tmp_join_{uuid.uuid4().hex[:8]}"
    dest_ctx = _create_join(
        knowledge_manager,
        dest_table=dest_table,
        tables=tables,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )

    try:
        # Delegate to DataManager.filter
        rows = dm.filter(
            dest_ctx,
            filter=result_where,
            limit=result_limit,
            offset=result_offset,
        )
        # Exclude private fields
        excl = list_private_fields(dest_ctx)
        filtered_rows = [
            {k: v for k, v in row.items() if k not in excl} for row in rows
        ]
        return maybe_group_rows(
            rows=filtered_rows,
            exclude_fields=excl,
            enabled=getattr(knowledge_manager, "_group_results", False),
        )
    finally:
        dm.delete_table(dest_ctx, dangerous_ok=True)


def filter_multi_join(
    knowledge_manager: "KnowledgeManager",
    *,
    joins: List[Dict[str, Any]],
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Chain together multiple joins, then filter the final result.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    joins : list[dict]
        Ordered list of join steps.
    result_where : str | None
        Filter on the final result.
    result_limit : int
        Maximum rows to return.
    result_offset : int
        Pagination offset.

    Returns
    -------
    list[dict[str, Any]]
        Rows from the final joined result.

    Raises
    ------
    ValueError
        If joins is empty or result_limit exceeds 1000.
    """
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if result_limit > 1000:
        raise ValueError("Limit must be less than 1000")

    dm = knowledge_manager._data_manager
    tmp_prefix = f"_tmp_mjoin_{uuid.uuid4().hex[:6]}"
    tmp_tables: List[str] = []
    previous_table: Optional[str] = None
    final_select_names: Optional[set[str]] = None

    for idx, step in enumerate(joins):
        local = step.copy()
        raw_tables = local.get("tables")
        raw_tables = [raw_tables] if isinstance(raw_tables, str) else raw_tables
        if not isinstance(raw_tables, list) or len(raw_tables) != 2:
            raise ValueError(
                f"Step {idx} must specify exactly TWO tables – got {raw_tables!r}",
            )

        # Substitute `$prev` placeholder
        step_tables = [
            (previous_table if t in {"$prev", "__prev__", "_"} else t)
            for t in raw_tables
        ]
        if any(t is None for t in step_tables):
            raise ValueError(
                "Misplaced `$prev` in first join – there is no previous result.",
            )

        # Fix-up join_expr & columns that reference `$prev`
        def _replace_prev(s: Optional[Union[str, List[str], Dict[str, str]]]):
            if s is None or previous_table is None:
                return s

            def repl(txt: str) -> str:
                return (
                    txt.replace("$prev", previous_table)
                    .replace("__prev__", previous_table)
                    .replace("_.", f"{previous_table}.")
                )

            if isinstance(s, str):
                return repl(s)
            elif isinstance(s, dict):
                return {repl(k): v for k, v in s.items()}
            return [repl(c) for c in s]

        join_expr_step = _replace_prev(local.get("join_expr"))
        select_step = _replace_prev(local.get("select"))

        # Destination table for this hop
        is_last = idx == len(joins) - 1
        dest_table = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
        tmp_tables.append(dest_table)

        # Materialise the join
        _create_join(
            knowledge_manager,
            dest_table=dest_table,
            tables=step_tables,
            join_expr=join_expr_step,  # type: ignore[arg-type]
            select=select_step,  # type: ignore[arg-type]
            mode=local.get("mode", "inner"),
            left_where=local.get("left_where"),
            right_where=local.get("right_where"),
        )

        previous_table = dest_table
        if is_last and isinstance(select_step, dict):
            try:
                final_select_names = set(select_step.values())
            except Exception:
                final_select_names = None

    assert previous_table is not None

    # Validate that result_where only references the final projection names
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
                "`result_where` references column(s) not present in the final step's "
                f"`select` mapping. Missing: {', '.join(sorted(missing))}",
            )

    final_ctx = ctx_for_table(knowledge_manager, previous_table)

    try:
        # Delegate to DataManager.filter
        rows = dm.filter(
            final_ctx,
            filter=result_where,
            limit=result_limit,
            offset=result_offset,
        )
        excl = list_private_fields(final_ctx)
        filtered_rows = [
            {k: v for k, v in row.items() if k not in excl} for row in rows
        ]
        return maybe_group_rows(
            rows=filtered_rows,
            exclude_fields=excl,
            enabled=getattr(knowledge_manager, "_group_results", False),
        )
    finally:
        # Clean up intermediate tables
        for tbl in tmp_tables:
            try:
                ctx = ctx_for_table(knowledge_manager, tbl)
                dm.delete_table(ctx, dangerous_ok=True)
            except Exception:
                pass
