from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import table_search_top_k
from ..common.grouping_helpers import maybe_group_rows
from ..common.embed_utils import list_private_fields
from .storage import contexts_for_table, table_contexts_for_read, ctx_for_table

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

    if tables is None:
        resolved_tables = list(table_contexts_for_read(knowledge_manager).keys())

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

    def _fetch_one(table_name: str, ctx: str) -> tuple[str, List[Dict[str, Any]], str]:
        excl = list_private_fields(ctx)
        normalized = normalize_filter_expr(filter)
        # Delegate to DataManager.filter
        rows = dm.filter(
            ctx,
            filter=normalized,
            limit=limit + offset,
            offset=0,
        )
        # Exclude private fields from results
        filtered_rows = [
            {k: v for k, v in row.items() if k not in excl} for row in rows
        ]
        return table_name, filtered_rows, ctx

    results: Dict[str, List[Dict[str, Any]]] = {}

    # Parallelise when scanning multiple tables to reduce wall-clock time
    if len(resolved_tables) <= 1 and resolved_tables:
        name = resolved_tables[0]
        merged_rows: List[Dict[str, Any]] = []
        exclude_fields: set[str] = set()
        for ctx in contexts_for_table(knowledge_manager, name):
            _, rows, row_context = _fetch_one(name, ctx)
            merged_rows.extend(rows)
            exclude_fields.update(list_private_fields(row_context))
        results[name] = maybe_group_rows(
            rows=merged_rows[offset : offset + limit],
            exclude_fields=exclude_fields,
            enabled=getattr(knowledge_manager, "_group_results", False),
        )
        return results

    max_workers = min(8, max(1, len(resolved_tables)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, table_name, context): table_name
            for table_name in resolved_tables
            for context in contexts_for_table(knowledge_manager, table_name)
        }
        for fut in as_completed(futures):
            name, rows, row_context = fut.result()
            existing = results.setdefault(name, [])
            existing.extend(rows)

    return {
        name: maybe_group_rows(
            rows=rows[offset : offset + limit],
            exclude_fields=set().union(
                *(
                    set(list_private_fields(ctx))
                    for ctx in contexts_for_table(knowledge_manager, name)
                ),
            ),
            enabled=getattr(knowledge_manager, "_group_results", False),
        )
        for name, rows in sorted(results.items())
    }


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

    normalized = normalize_filter_expr(filter)

    rows: List[Dict[str, Any]] = []
    exclude_fields: set[str] = set()
    for context in contexts_for_table(knowledge_manager, table):
        rows.extend(
            table_search_top_k(
                context=context,
                references=references,
                k=k,
                row_filter=normalized,
            ),
        )
        exclude_fields.update(list_private_fields(context))
    sort_key = next((key for row in rows for key in row if key.startswith("_")), None)
    if sort_key:
        rows.sort(key=lambda row: row.get(sort_key, float("inf")))

    return maybe_group_rows(
        rows=rows[:k],
        exclude_fields=exclude_fields,
        enabled=getattr(knowledge_manager, "_group_results", False),
    )


def _resolve_and_rewrite(
    knowledge_manager: "KnowledgeManager",
    tables: Union[str, List[str]],
    join_expr: str,
    select: Dict[str, str],
    left_where: Optional[str],
    right_where: Optional[str],
    *,
    namespace: str | None = None,
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
    left_ctx = (
        f"{namespace}/{left}"
        if namespace is not None
        else ctx_for_table(knowledge_manager, left)
    )
    right_ctx = (
        f"{namespace}/{right}"
        if namespace is not None
        else ctx_for_table(knowledge_manager, right)
    )

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
    *,
    namespace: str | None = None,
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
                mapping[t] = (
                    f"{namespace}/{t}"
                    if namespace is not None
                    else ctx_for_table(knowledge_manager, t)
                )

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


def _read_knowledge_namespaces(knowledge_manager: "KnowledgeManager") -> list[str]:
    """Return readable Knowledge namespaces for join-style read tools."""
    if hasattr(knowledge_manager, "_read_knowledge_namespaces"):
        return knowledge_manager._read_knowledge_namespaces()  # type: ignore[attr-defined]
    return [knowledge_manager._ctx]


def _namespace_has_tables(
    knowledge_manager: "KnowledgeManager",
    namespace: str,
    tables: list[str],
) -> bool:
    """Return whether every joined table exists under a Knowledge namespace."""
    prefix = f"{namespace}/"
    ctx_info = knowledge_manager._data_manager.list_tables(
        prefix=prefix,
        include_column_info=False,
    )
    available = ctx_info.keys() if isinstance(ctx_info, dict) else ctx_info
    available_set = set(available)
    return all(f"{namespace}/{table}" in available_set for table in tables)


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
    rows: List[Dict[str, Any]] = []
    joined_tables = [tables] if isinstance(tables, str) else list(tables)
    for namespace in _read_knowledge_namespaces(knowledge_manager):
        if not _namespace_has_tables(knowledge_manager, namespace, joined_tables):
            continue
        (
            resolved_tables,
            resolved_join_expr,
            resolved_select,
            resolved_left_where,
            resolved_right_where,
        ) = _resolve_and_rewrite(
            knowledge_manager,
            tables,
            join_expr,
            select,
            left_where,
            right_where,
            namespace=namespace,
        )

        rows.extend(
            knowledge_manager._data_manager.search_join(
                tables=resolved_tables,
                join_expr=resolved_join_expr,
                select=resolved_select,
                mode=mode,
                left_where=resolved_left_where,
                right_where=resolved_right_where,
                references=references or {},
                k=k,
                filter=filter,
            ),
        )
    rows = rows[:k]
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
    rows: List[Dict[str, Any]] = []
    join_tables = sorted(
        {
            table
            for join in joins
            for table in (
                [join.get("tables")]
                if isinstance(join.get("tables"), str)
                else list(join.get("tables", []))
            )
            if table not in {"$prev", "__prev__", "_"}
        },
    )
    for namespace in _read_knowledge_namespaces(knowledge_manager):
        if not _namespace_has_tables(knowledge_manager, namespace, join_tables):
            continue
        resolved_joins = _resolve_joins(knowledge_manager, joins, namespace=namespace)
        rows.extend(
            knowledge_manager._data_manager.search_multi_join(
                joins=resolved_joins,
                references=references or {},
                k=k,
                filter=filter,
            ),
        )
    rows = rows[:k]
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
    rows: List[Dict[str, Any]] = []
    joined_tables = [tables] if isinstance(tables, str) else list(tables)
    for namespace in _read_knowledge_namespaces(knowledge_manager):
        if not _namespace_has_tables(knowledge_manager, namespace, joined_tables):
            continue
        (
            resolved_tables,
            resolved_join_expr,
            resolved_select,
            resolved_left_where,
            resolved_right_where,
        ) = _resolve_and_rewrite(
            knowledge_manager,
            tables,
            join_expr,
            select,
            left_where,
            right_where,
            namespace=namespace,
        )

        rows.extend(
            knowledge_manager._data_manager.filter_join(
                tables=resolved_tables,
                join_expr=resolved_join_expr,
                select=resolved_select,
                mode=mode,
                left_where=resolved_left_where,
                right_where=resolved_right_where,
                result_where=result_where,
                result_limit=result_limit + result_offset,
                result_offset=0,
            ),
        )
    rows = rows[result_offset : result_offset + result_limit]
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
    rows: List[Dict[str, Any]] = []
    join_tables = sorted(
        {
            table
            for join in joins
            for table in (
                [join.get("tables")]
                if isinstance(join.get("tables"), str)
                else list(join.get("tables", []))
            )
            if table not in {"$prev", "__prev__", "_"}
        },
    )
    for namespace in _read_knowledge_namespaces(knowledge_manager):
        if not _namespace_has_tables(knowledge_manager, namespace, join_tables):
            continue
        resolved_joins = _resolve_joins(knowledge_manager, joins, namespace=namespace)
        rows.extend(
            knowledge_manager._data_manager.filter_multi_join(
                joins=resolved_joins,
                result_where=result_where,
                result_limit=result_limit + result_offset,
                result_offset=0,
            ),
        )
    rows = rows[result_offset : result_offset + result_limit]
    return maybe_group_rows(
        rows=rows,
        exclude_fields=_private_field_names(rows),
        enabled=getattr(knowledge_manager, "_group_results", False),
    )
