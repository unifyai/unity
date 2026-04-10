from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _resolve_and_rewrite(
    knowledge_manager: "KnowledgeManager",
    tables: Union[str, List[str]],
    join_expr: str,
    select: Dict[str, str],
    left_where: Optional[str],
    right_where: Optional[str],
) -> tuple[
    List[str],
    str,
    Dict[str, str],
    Optional[str],
    Optional[str],
]:
    """Resolve KM table names to full ``Knowledge/…`` paths and rewrite expressions."""
    tables_list = [tables] if isinstance(tables, str) else list(tables)
    if len(tables_list) != 2:
        raise ValueError("Exactly TWO tables are required.")

    left, right = tables_list
    left_ctx = ctx_for_table(knowledge_manager, left)
    right_ctx = ctx_for_table(knowledge_manager, right)

    join_expr = join_expr.replace(left, left_ctx).replace(right, right_ctx)
    select = {
        c.replace(left, left_ctx).replace(right, right_ctx): v
        for c, v in select.items()
    }
    if left_where is not None:
        left_where = left_where.replace(left, left_ctx)
    if right_where is not None:
        right_where = right_where.replace(right, right_ctx)

    return [left_ctx, right_ctx], join_expr, select, left_where, right_where


def _resolve_joins(
    knowledge_manager: "KnowledgeManager",
    joins: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve KM table names in multi-join steps, rewriting expressions."""
    _PREV = {"$prev", "__prev__", "_"}
    resolved: List[Dict[str, Any]] = []
    for step in joins:
        s = dict(step)
        tables = s.get("tables", [])
        tables = [tables] if isinstance(tables, str) else list(tables)

        mapping: Dict[str, str] = {}
        for t in tables:
            if t not in _PREV:
                mapping[t] = ctx_for_table(knowledge_manager, t)

        s["tables"] = [mapping.get(t, t) for t in tables]

        for field in ("join_expr", "left_where", "right_where"):
            val = s.get(field)
            if val is not None:
                for short, full in mapping.items():
                    val = val.replace(short, full)
                s[field] = val

        sel = s.get("select")
        if isinstance(sel, dict):
            new_sel: Dict[str, str] = {}
            for k, v in sel.items():
                new_k = k
                for short, full in mapping.items():
                    new_k = new_k.replace(short, full)
                new_sel[new_k] = v
            s["select"] = new_sel

        resolved.append(s)
    return resolved


def _private_field_names(rows: List[Dict[str, Any]]) -> set[str]:
    """Derive private (``_``-prefixed) field names directly from row data."""
    return {k for row in rows for k in row if k.startswith("_")}


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

    Delegates to :pymethod:`DataManager.search_join` after resolving table
    names to fully-qualified ``Knowledge/…`` context paths.

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
    resolved_tables, join_expr, select, left_where, right_where = _resolve_and_rewrite(
        knowledge_manager,
        tables,
        join_expr,
        select,
        left_where,
        right_where,
    )

    rows = knowledge_manager._data_manager.search_join(
        tables=resolved_tables,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
        references=references or {},
        k=k,
        filter=filter,
    )
    return maybe_group_rows(
        rows=rows,
        exclude_fields=_private_field_names(rows),
        enabled=getattr(knowledge_manager, "_group_results", False),
    )


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

    Delegates to :pymethod:`DataManager.search_multi_join` after resolving
    table names in each step to fully-qualified ``Knowledge/…`` context paths.

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
    resolved_joins = _resolve_joins(knowledge_manager, joins)

    rows = knowledge_manager._data_manager.search_multi_join(
        joins=resolved_joins,
        references=references or {},
        k=k,
        filter=filter,
    )
    return maybe_group_rows(
        rows=rows,
        exclude_fields=_private_field_names(rows),
        enabled=getattr(knowledge_manager, "_group_results", False),
    )


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

    Delegates to :pymethod:`DataManager.filter_join` (which uses the fused
    ``join_query`` endpoint — a single server round-trip with no temporary
    table materialisation) after resolving table names.

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
    resolved_tables, join_expr, select, left_where, right_where = _resolve_and_rewrite(
        knowledge_manager,
        tables,
        join_expr,
        select,
        left_where,
        right_where,
    )

    rows = knowledge_manager._data_manager.filter_join(
        tables=resolved_tables,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
        result_where=result_where,
        result_limit=result_limit,
        result_offset=result_offset,
    )
    return maybe_group_rows(
        rows=rows,
        exclude_fields=_private_field_names(rows),
        enabled=getattr(knowledge_manager, "_group_results", False),
    )


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

    Delegates to :pymethod:`DataManager.filter_multi_join` after resolving
    table names in each step to fully-qualified ``Knowledge/…`` context paths.

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
    resolved_joins = _resolve_joins(knowledge_manager, joins)

    rows = knowledge_manager._data_manager.filter_multi_join(
        joins=resolved_joins,
        result_where=result_where,
        result_limit=result_limit,
        result_offset=result_offset,
    )
    return maybe_group_rows(
        rows=rows,
        exclude_fields=_private_field_names(rows),
        enabled=getattr(knowledge_manager, "_group_results", False),
    )
