from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from typing import Any, Dict, List, Optional, Union

import unify

from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import table_search_top_k
from ..common.grouping_helpers import maybe_group_rows
from ..common.embed_utils import list_private_fields
from .storage import ctx_for_table


def filter(
    self,
    *,
    filter: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    tables: Optional[Union[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if limit > 1000:
        raise ValueError("Limit must be less than 1000")

    # Resolve target tables without triggering per-table field reads.
    # When the caller does not specify tables, list contexts directly
    # rather than calling `_tables_overview(include_column_info=True)`,
    # which would fetch columns for every table (unnecessary here).
    if tables is None:
        km_prefix = f"{self._ctx}/"
        ctxs = unify.get_contexts(prefix=km_prefix)
        resolved_tables: List[str] = [k[len(km_prefix) :] for k in ctxs.keys()]
        # Optionally expose root-level Contacts when linkage is enabled
        if (
            getattr(self, "_include_contacts", False)
            and getattr(self, "_contacts_ctx", None) is not None
        ):
            try:
                contacts_info = unify.get_context(self._contacts_ctx)  # type: ignore[attr-defined]
                if isinstance(contacts_info, dict):
                    resolved_tables.append("Contacts")
            except Exception:
                pass
    elif isinstance(tables, str):
        resolved_tables = [tables]
    else:
        resolved_tables = list(tables)

    # Fetch private-field lists and rows per table without serial stalls.
    # Each table performs at most two backend reads: fields (once) and logs.
    def _fetch_one(table_name: str) -> tuple[str, List[Dict[str, Any]]]:
        ctx = ctx_for_table(self, table_name)
        excl = list_private_fields(ctx)
        normalized = normalize_filter_expr(filter)
        rows: List[Dict[str, Any]] = [
            log.entries
            for log in unify.get_logs(
                context=ctx,
                filter=normalized,
                offset=offset,
                limit=limit,
                exclude_fields=excl,
            )
        ]
        return table_name, rows

    results: Dict[str, List[Dict[str, Any]]] = {}
    # Parallelise when scanning multiple tables to reduce wall-clock time.
    max_workers = min(8, max(1, len(resolved_tables)))
    if len(resolved_tables) <= 1:
        # Avoid thread-pool overhead for the common single-table case
        name, rows = _fetch_one(resolved_tables[0])
        ctx = ctx_for_table(self, name)
        results[name] = maybe_group_rows(
            rows=rows,
            exclude_fields=list_private_fields(ctx),
            enabled=getattr(self, "_group_results", False),
        )
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, table_name): table_name
            for table_name in resolved_tables
        }
        for fut in as_completed(futures):
            name, rows = fut.result()
            ctx = ctx_for_table(self, name)
            results[name] = maybe_group_rows(
                rows=rows,
                exclude_fields=list_private_fields(ctx),
                enabled=getattr(self, "_group_results", False),
            )

    return results


def search(
    self,
    *,
    table: str,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if k > 1000 or k < 1:
        raise ValueError("k must be between 1 and 1000")
    context = ctx_for_table(self, table)
    normalized = normalize_filter_expr(filter)
    rows: List[Dict[str, Any]] = table_search_top_k(
        context=context,
        references=references,
        k=k,
        row_filter=normalized,
    )
    return maybe_group_rows(
        rows=rows,
        exclude_fields=list_private_fields(context),
        enabled=getattr(self, "_group_results", False),
    )


def _create_join(
    self,
    *,
    dest_table: str,
    tables: Union[str, List[str]],
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
) -> str:
    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("❌  Exactly TWO tables are required.")

    left_table, right_table = tables
    left_ctx, right_ctx = ctx_for_table(self, left_table), ctx_for_table(
        self,
        right_table,
    )

    def _rewrite_filter(expr: Optional[str], table: str, ctx: str) -> Optional[str]:
        return None if expr is None else expr.replace(table, ctx)

    left_where = _rewrite_filter(left_where, left_table, left_ctx)
    right_where = _rewrite_filter(right_where, right_table, right_ctx)

    join_expr = join_expr.replace(left_table, left_ctx).replace(right_table, right_ctx)
    select = {
        c.replace(left_table, left_ctx).replace(right_table, right_ctx): v
        for c, v in select.items()
    }

    dest_ctx = ctx_for_table(self, dest_table)
    unify.join_logs(
        pair_of_args=(
            {
                "context": left_ctx,
                **({} if left_where is None else {"filter_expr": left_where}),
            },
            {
                "context": right_ctx,
                **({} if right_where is None else {"filter_expr": right_where}),
            },
        ),
        join_expr=join_expr,
        mode=mode,
        new_context=dest_ctx,
        columns=select,
    )
    return dest_ctx


def search_join(
    self,
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
    if k > 1000 or k < 1:
        raise ValueError("k must be between 1 and 1000")
    dest_table = f"_tmp_join_{uuid.uuid4().hex[:8]}"
    dest_ctx = _create_join(
        self,
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
            enabled=getattr(self, "_group_results", False),
        )
    finally:
        try:
            unify.delete_context(dest_ctx)
        except Exception:
            pass


def search_multi_join(
    self,
    *,
    joins: List[Dict[str, Any]],
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if k > 1000 or k < 1:
        raise ValueError("k must be between 1 and 1000")

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

        join_expr = _replace_prev(local_step.get("join_expr"))
        select = _replace_prev(local_step.get("select"))

        # Destination table for this hop
        is_last = idx == len(joins) - 1
        dest_table = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
        tmp_tables.append(dest_table)

        # Materialise the join (no reads yet)
        _create_join(
            self,
            dest_table=dest_table,
            tables=step_tables,
            join_expr=join_expr,  # type: ignore[arg-type]
            select=select,  # type: ignore[arg-type]
            mode=local_step.get("mode", "inner"),
            left_where=local_step.get("left_where"),
            right_where=local_step.get("right_where"),
        )

        previous_table = dest_table

    assert previous_table is not None  # mypy guard

    final_ctx = ctx_for_table(self, previous_table)
    try:
        # 1) Primary similarity-ranked results from the final joined context
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
            enabled=getattr(self, "_group_results", False),
        )
    finally:
        try:
            # Clean up intermediate tables
            from .storage import delete_tables as _del_tbl

            _del_tbl(self, tables=tmp_tables)
        except Exception:
            pass


def filter_join(
    self,
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
    import re as _re

    if result_limit > 1000:
        raise ValueError("Limit must be less than 1000")

    # ── helper to catch mismatches early ────────────────────────────
    def _qualified_refs(expr: str) -> set[str]:
        return set(
            m.group(0) for m in _re.finditer(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", expr)
        )

    if result_where:
        missing = _qualified_refs(result_where) - set(select)
        if missing:
            raise ValueError(
                "❌  `result_where` references column(s) that are not present in `select`.  "
                "Either add them to `select` or move the predicate to `left_where` / `right_where`.  "
                f"Missing: {', '.join(sorted(missing))}",
            )

    dest_table = f"_tmp_join_{uuid.uuid4().hex[:8]}"
    dest_ctx = _create_join(
        self,
        dest_table=dest_table,
        tables=tables,
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )

    rows: List[Dict[str, Any]] = [
        lg.entries
        for lg in unify.get_logs(
            context=dest_ctx,
            filter=result_where,
            offset=result_offset,
            limit=result_limit,
            exclude_fields=list_private_fields(dest_ctx),
        )
    ]
    try:
        unify.delete_context(dest_ctx)
    except Exception:
        pass

    grouped = maybe_group_rows(
        rows=rows,
        exclude_fields=list_private_fields(dest_ctx),
        enabled=getattr(self, "_group_results", False),
    )
    return grouped


def filter_multi_join(
    self,
    *,
    joins: List[Dict[str, Any]],
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
) -> List[Dict[str, Any]]:
    if not joins:
        raise ValueError("`joins` must contain at least one join step.")
    if result_limit > 1000:
        raise ValueError("Limit must be less than 1000")

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

        join_expr = _replace_prev(local.get("join_expr"))
        select = _replace_prev(local.get("select"))

        # Destination table for this hop
        is_last = idx == len(joins) - 1
        dest_table = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
        tmp_tables.append(dest_table)

        # Materialise the join (no reads yet)
        _create_join(
            self,
            dest_table=dest_table,
            tables=step_tables,
            join_expr=join_expr,  # type: ignore[arg-type]
            select=select,  # type: ignore[arg-type]
            mode=local.get("mode", "inner"),
            left_where=local.get("left_where"),
            right_where=local.get("right_where"),
        )

        previous_table = dest_table
        if is_last and isinstance(select, dict):
            try:
                final_select_names = set(select.values())
            except Exception:
                final_select_names = None

    assert previous_table is not None
    # Validate that result_where only references the final projection names
    if result_where and final_select_names is not None:
        import re as _re

        tokens = set(
            m.group(0) for m in _re.finditer(r"\b[A-Za-z_]\w*\b", result_where)
        )
        # Remove common operators/keywords/names that aren't column refs
        _reserved = {"and", "or", "not", "in", "is", "True", "False", "None"}
        candidate_cols = {t for t in tokens if t not in _reserved and not t.isdigit()}
        missing = candidate_cols - final_select_names
        if missing:
            raise ValueError(
                "❌  `result_where` references column(s) not present in the final step's `select` mapping. "
                f"Missing: {', '.join(sorted(missing))}",
            )
    final_ctx = ctx_for_table(self, previous_table)

    rows: List[Dict[str, Any]] = [
        lg.entries
        for lg in unify.get_logs(
            context=final_ctx,
            filter=result_where,
            offset=result_offset,
            limit=result_limit,
            exclude_fields=list_private_fields(final_ctx),
        )
    ]

    try:
        # do not delete the user-requested *persistent* table
        from .storage import delete_tables as _del_tbl

        _del_tbl(self, tables=tmp_tables)
    except Exception:
        pass

    return maybe_group_rows(
        rows=rows,
        exclude_fields=list_private_fields(final_ctx),
        enabled=getattr(self, "_group_results", False),
    )
