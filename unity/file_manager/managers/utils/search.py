from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import uuid

import unify

from unity.common.log_utils import create_logs as unity_create_logs
from unity.common.search_utils import table_search_top_k
from unity.common.filter_utils import normalize_filter_expr
from unity.common.embed_utils import list_private_fields


def ctx_for_table(self, table: str) -> str:
    """
    Resolve a table reference to a fully-qualified Unify context.

    Preferred forms (path-first):
    - "<file_path>" → per-file Content context for that exact file path
    - "<file_path>.Tables.<label>" → per-file Tables context for the given table label

    Also accepted for backward compatibility:
    - "FileRecords" → global index context (`self._ctx`)
    - "<root>" → legacy alias for a per-file Content root
    - "<root>.Tables.<label>" → legacy per-file Tables form

    Notes
    -----
    - Use fully qualified absolute file paths whenever possible, as stored in
      FileRecords and returned by identity helpers. Avoid legacy "<root>" forms.
    - All returned contexts are built by prefixing the manager base +
      "/Files/<alias>/…" via the ops helpers.
    """
    t = (table or "").strip()
    if not t:
        raise ValueError("table must be non-empty")
    if t.lower() == "filerecords":
        return self._ctx
    # Per-file Content
    if ".tables." not in t.lower():
        from .storage import ctx_for_file as _ctx_for_file

        return _ctx_for_file(self, file_path=t)
    # Per-file table
    root, label = t.split(".Tables.", 1)
    from .storage import ctx_for_file_table as _ctx_for_file_table

    return _ctx_for_file_table(self, file_path=root, table=label)


def resolve_table_ref(self, ref: str) -> str:
    """
    Resolve a table reference to a fully-qualified context.

    Preferred forms (path-first):
    - "<file_path>" → per-file Content context for that exact file path
    - "<file_path>.Tables.<label>" → per-file Tables context for the given table label
    - "FileRecords" → global index context

    Also accepted for backward compatibility:
    - "<root>" → legacy alias for a per-file Content root (deprecated; use <file_path>)
    - "<root>.Tables.<label>" → legacy per-file Tables form (deprecated; use <file_path>.Tables.<label>)
    - "<file_path>:<table>" (legacy colon-separated form)
    - "id=<file_id>:<table>" or "#<file_id>:<table>" (id-based addressing)
    """
    # If the ref already looks like a fully-qualified context under this manager,
    # return as-is using known manager roots (no hard-coded labels).
    try:
        _index_ctx = getattr(self, "_ctx")
    except Exception:
        _index_ctx = None
    try:
        _files_root = getattr(self, "_per_file_root")
    except Exception:
        _files_root = None
    if isinstance(ref, str):
        r = ref.strip()
        if _index_ctx and r == _index_ctx:
            return r
        if _files_root and r.startswith(f"{_files_root}/"):
            return r
        # Check if ref is already a fully-qualified context path (e.g., temporary join contexts)
        # Temporary contexts are under FileRecords (e.g., "Assistant/FileRecords/Local/_tmp_mjoin_...")
        # Check if it starts with the index context prefix (which includes FileRecords)
        if _index_ctx and r.startswith(_index_ctx + "/"):
            return r
    if ":" not in ref:
        # Treat as logical name from tables_overview
        return ctx_for_table(self, ref)
    left, tbl = ref.split(":", 1)
    key = left.strip()

    # Allow id-based addressing: "id=123:Table" or "#123:Table"
    def _lookup_path_by_id(fid: int) -> str:
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=f"file_id == {int(fid)}",
                limit=1,
                from_fields=["file_path"],
            )
        except Exception:
            rows = []
        if not rows:
            raise ValueError(f"No file found with file_id={fid}")
        return rows[0].entries.get("file_path")

    if key.startswith("id="):
        file_id = int(key.split("=", 1)[1])
        file_path = _lookup_path_by_id(file_id)
        from .storage import ctx_for_file_table as _ctx_for_file_table

        return _ctx_for_file_table(self, file_path=file_path, table=tbl)
    if key.startswith("#") and key[1:].isdigit():
        file_id = int(key[1:])
        file_path = _lookup_path_by_id(file_id)
        from .storage import ctx_for_file_table as _ctx_for_file_table

        return _ctx_for_file_table(self, file_path=file_path, table=tbl)
    # Fallback: treat as file path / display name
    file_path = key
    from .storage import ctx_for_file_table as _ctx_for_file_table

    return _ctx_for_file_table(self, file_path=file_path, table=tbl)


# --------------------- Index-level filter/search (FileRecords) ---------------- #


def filter_files(
    self,
    *,
    filter: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    tables: Optional[Union[str, List[str]]] = None,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter rows from one or more contexts (index by default).

    Parameters
    ----------
    filter : str | None
        Python boolean expression evaluated with column names in scope.
    offset : int
        Zero-based pagination offset per context.
    limit : int
        Maximum rows per context (<= 1000).
    tables : list[str] | str | None
        Table references to filter. Accepted forms:
        - Full context path (preferred): Use describe() to get exact paths
        - Path-first: "<file_path>" for per-file Content,
          "<file_path>.Tables.<label>" for per-file tables
        - Logical names from `tables_overview()`: "FileRecords" for index
        When None, only the FileRecords index is scanned.
    columns : list[str] | None
        Specific columns to return. When None, returns all columns
        (excluding private fields).

    Returns
    -------
    list[dict]
        Flat list of rows collected from the index (when tables=None) or
        concatenated rows from all resolved contexts.
    """
    normalized = normalize_filter_expr(filter)
    if tables is None:
        excl = list_private_fields(self._ctx) if columns is None else None
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            exclude_fields=excl,
            from_fields=columns,
        )
        return [lg.entries for lg in logs]

    # Normalize tables to list
    if isinstance(tables, str):
        table_names = [tables]
    else:
        table_names = list(tables)

    # Resolve contexts for each table name
    to_contexts: List[tuple[str, str]] = []
    for name in table_names:
        ctx = ctx_for_table(self, name)
        to_contexts.append((name, ctx))

    # Parallel filter across contexts (KM-style shape but flattened)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []

    def _fetch(_ctx: str) -> List[Dict[str, Any]]:
        excl = list_private_fields(_ctx) if columns is None else None
        rows = [
            lg.entries
            for lg in unify.get_logs(
                context=_ctx,
                filter=normalized,
                offset=offset,
                limit=limit,
                exclude_fields=excl,
                from_fields=columns,
            )
        ]
        return rows

    max_workers = min(8, max(1, len(to_contexts)))
    if len(to_contexts) <= 1:
        name, ctx = to_contexts[0]
        results.extend(_fetch(ctx))
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, ctx): name for name, ctx in to_contexts}
        for fut in as_completed(futures):
            rows = fut.result()
            results.extend(rows)

    return results


# Removed the separate `filter(...)` – multi-table support is now in filter_files


def search_files(
    self,
    *,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    table: Optional[str] = None,
    filter: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Semantic search over a resolved context and return entry dicts.

    Parameters
    ----------
    references : dict[str, str] | None
        Mapping of source_expr → reference_text used to compute similarity.
    k : int
        Number of rows to return (1..1000).
    table: str | None
        Table reference to search. Accepted forms:
        - Full context path (preferred): Use describe() to get exact paths
        - Path-first: "<file_path>" for per-file Content,
          "<file_path>.Tables.<label>" for per-file tables
        - Logical names: "FileRecords" for index
        When None, defaults to the global FileRecords index.
    filter: str | None
        Row-level predicate (evaluated with column names as variables).
        *None* returns all rows.
    columns : list[str] | None
        Specific columns to return. When None, returns all columns
        (excluding private fields).
    """
    normalized = normalize_filter_expr(filter)

    if table is None:
        context = getattr(self, "_ctx", None)
        unique_id_field = "file_id"
    else:
        context = ctx_for_table(self, table)
        # Choose unique id based on context category
        if isinstance(context, str) and context.endswith("/Content"):
            unique_id_field = "row_id"
        elif isinstance(context, str) and "/Tables/" in context:
            unique_id_field = "row_id"
        else:
            unique_id_field = None
    rows = table_search_top_k(
        context=context,
        references=references,
        k=k,
        row_filter=normalized,
        unique_id_field=unique_id_field,  # type: ignore[arg-type]
        allowed_fields=columns,
    )
    return rows


def create_join(
    self,
    *,
    dest_table_ctx: str,
    left_ref: str,
    right_ref: str,
    join_expr: str,
    select: Dict[str, str],
    mode: str = "inner",
    left_where: Optional[str] = None,
    right_where: Optional[str] = None,
) -> str:
    """Create a derived table by joining two per-file tables."""
    left_ctx = resolve_table_ref(self, left_ref)
    right_ctx = resolve_table_ref(self, right_ref)

    # Rewrite join/select to use table aliases A. and B. for join expressions
    # Unify's join_logs API requires column references to be prefixed with A. or B.
    import re

    # For join expressions, replace column references with aliased versions
    # Unify requires A. and B. prefixes for columns in join expressions
    def _rewrite_join_expr_with_aliases(
        expr: str,
        left_ref: str,
        right_ref: str,
    ) -> str:
        """
        Rewrite join expression to use table aliases A. and B.
        For expressions like 'rid == rid', convert to 'A.rid == B.rid'
        For expressions like '/path/to/file.row_id == /path/to/file.row_id', convert to 'A.row_id == B.row_id'
        """
        # First, replace explicit table refs (including file paths) with aliases
        left_ref_escaped = re.escape(left_ref)
        right_ref_escaped = re.escape(right_ref)

        # Replace left_ref.column with A.column (handle paths with /)
        # Match the full left_ref followed by .column_name
        expr = re.sub(rf"{left_ref_escaped}\.([a-zA-Z_][a-zA-Z0-9_]*)", r"A.\1", expr)
        # Replace right_ref.column with B.column (handle paths with /)
        expr = re.sub(rf"{right_ref_escaped}\.([a-zA-Z_][a-zA-Z0-9_]*)", r"B.\1", expr)

        # For bare column names in join expressions, split by comparison operators
        # and assign A. to left side, B. to right side
        # Pattern: column == column, column != column, etc.
        operators = r"(==|!=|<=|>=|<|>)"
        parts = re.split(operators, expr)

        if len(parts) >= 3:
            # We have a comparison: left_expr operator right_expr
            left_expr = parts[0].strip()
            op = parts[1]
            right_expr = parts[2].strip()

            # Add A. prefix to bare column names on left side (but not if already prefixed)
            # Only match identifiers that aren't already prefixed with A. or B.
            left_expr = re.sub(
                r"(?<!\.)\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\.)",
                lambda m: (
                    f"A.{m.group(1)}"
                    if not m.group(1).startswith(("A.", "B."))
                    and m.group(1) not in ("A", "B")
                    else m.group(1)
                ),
                left_expr,
            )
            # Add B. prefix to bare column names on right side
            right_expr = re.sub(
                r"(?<!\.)\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\.)",
                lambda m: (
                    f"B.{m.group(1)}"
                    if not m.group(1).startswith(("A.", "B."))
                    and m.group(1) not in ("A", "B")
                    else m.group(1)
                ),
                right_expr,
            )

            # Reconstruct expression
            remaining = "".join(parts[3:]) if len(parts) > 3 else ""
            expr = f"{left_expr} {op} {right_expr}{remaining}"
        else:
            # No comparison operator found, just add A. prefix to bare columns
            expr = re.sub(
                r"(?<!\.)\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\.)",
                lambda m: (
                    f"A.{m.group(1)}"
                    if not m.group(1).startswith(("A.", "B."))
                    and m.group(1) not in ("A", "B")
                    else m.group(1)
                ),
                expr,
            )

        return expr

    # Rewrite join expression to use aliases
    join_expr = _rewrite_join_expr_with_aliases(join_expr, left_ref, right_ref)

    # For select, replace table refs with aliases
    def _rewrite_select_with_aliases(
        select_dict: Dict[str, str],
        left_ref: str,
        right_ref: str,
    ) -> Dict[str, str]:
        result = {}
        left_ref_escaped = re.escape(left_ref)
        right_ref_escaped = re.escape(right_ref)
        for col_expr, alias in select_dict.items():
            # Replace left_ref.column with A.column (handle paths with /)
            col_expr = re.sub(
                rf"{left_ref_escaped}\.([a-zA-Z_][a-zA-Z0-9_]*)",
                r"A.\1",
                col_expr,
            )
            # Replace right_ref.column with B.column (handle paths with /)
            col_expr = re.sub(
                rf"{right_ref_escaped}\.([a-zA-Z_][a-zA-Z0-9_]*)",
                r"B.\1",
                col_expr,
            )
            # For bare column names, assume left table (A.)
            if "." not in col_expr and col_expr.strip() not in ("A", "B"):
                col_expr = f"A.{col_expr}"
            result[col_expr] = alias
        return result

    select = _rewrite_select_with_aliases(select, left_ref, right_ref)

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
        new_context=dest_table_ctx,
        columns=select,
    )
    return dest_table_ctx


def ensure_tmp_ctx(self, ctx: str) -> None:
    unify.create_context(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
    )


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
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Join two tables and return filtered rows from the join result.

    Parameters
    ----------
    tables : str | list[str]
        Exactly two table references. Accepted forms:
        - Path-first (preferred): "<file_path>" for per-file Content,
          "<file_path>.Tables.<label>" for per-file tables
        - Logical names from `tables_overview()` or legacy refs
        - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
    join_expr : str
        Join expression using the same refs as provided in `tables`.
    select : dict[str, str]
        Mapping of output column → source column (use refs as in `tables`).
    mode : str
        One of 'inner', 'left', 'right', 'outer'.
    left_where, right_where : str | None
        Optional filter expressions applied to left and right inputs.
    result_where : str | None
        Optional filter applied to the joined result.
    result_limit, result_offset : int
        Pagination parameters.

    Returns
    -------
    dict
        {"rows": list[dict]} from the temporary join context.
    """
    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly two tables are required as 'file_path:table'")

    tmp_ctx = f"{self._ctx}/_tmp_join_{uuid.uuid4().hex[:6]}"
    ensure_tmp_ctx(self, tmp_ctx)
    create_join(
        self,
        dest_table_ctx=tmp_ctx,
        left_ref=tables[0],
        right_ref=tables[1],
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )
    base_excl = list_private_fields(tmp_ctx)
    rows = [
        e.entries
        for e in unify.get_logs(
            context=tmp_ctx,
            filter=result_where,
            offset=result_offset,
            limit=result_limit,
            exclude_fields=base_excl,
        )
    ]
    return {"rows": rows}


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
    """
    Join two tables and return top-k semantic matches from the join result.

    Parameters
    ----------
    tables : str | list[str]
        Exactly two table references. Accepted forms:
        - Path-first (preferred): "<file_path>" for per-file Content,
          "<file_path>.Tables.<label>" for per-file tables
        - Logical names from `tables_overview()` or legacy refs
        - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
    join_expr : str
        Join expression using the same refs as provided in `tables`.
    select : dict[str, str]
        Mapping of output column → source column (use refs as in `tables`).
    mode : str
        One of 'inner', 'left', 'right', 'outer'.
    left_where, right_where : str | None
        Optional filter expressions applied to left and right inputs.
    references : dict[str, str] | None
        Mapping of ref → example text to bias embedding search.
    k : int
        Number of results to return (1..1000).
    filter : str | None
        Optional filter on the joined result before search.

    Returns
    -------
    list[dict[str, Any]]
        Top-k rows ranked by semantic similarity.
    """
    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly two tables are required as 'file_path:table'")

    tmp_ctx = f"{self._ctx}/_tmp_join_{uuid.uuid4().hex[:6]}"
    ensure_tmp_ctx(self, tmp_ctx)
    create_join(
        self,
        dest_table_ctx=tmp_ctx,
        left_ref=tables[0],
        right_ref=tables[1],
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )
    rows = table_search_top_k(
        context=tmp_ctx,
        references=references,
        k=k,
        unique_id_field=None,
        row_filter=filter,
    )
    return rows


def filter_multi_join(
    self,
    *,
    joins: List[Dict[str, Any]],
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Execute a sequence of joins. Each step:
    - tables: [left, right] (logical names or "$prev")
    - join_expr, select, mode, left_where, right_where

    Parameters
    ----------
    joins : list[dict]
        Ordered steps; each step provides ``tables`` (two refs or "$prev"),
        ``join_expr``, ``select`` and optional ``mode``, ``left_where``, ``right_where``.
        Table references in ``tables`` accept:
        - Path-first (preferred): "<file_path>" for per-file Content,
          "<file_path>.Tables.<label>" for per-file tables
        - Logical names from `tables_overview()` or legacy refs
        - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
        - "$prev" to reference the previous join step's result
    result_where : str | None
        Predicate applied to the final joined result over projected columns.
    result_limit, result_offset : int
        Pagination parameters; limit <= 1000.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        {"rows": [...]} from the final join context.
    """
    if not joins:
        return {"rows": []}

    prev_ctx: Optional[str] = None
    for idx, step in enumerate(joins):
        tbls = step.get("tables")
        if isinstance(tbls, str):
            tbls = [tbls]
        if not tbls or len(tbls) != 2:
            raise ValueError("Each join step must provide exactly two tables")

        left_ref, right_ref = tbls
        if prev_ctx is not None:
            if left_ref in ("$prev", "__prev__", "_"):
                left_ref = prev_ctx
            if right_ref in ("$prev", "__prev__", "_"):
                right_ref = prev_ctx

        # Temporary join contexts should be under Files/<alias>, not FileRecords/<alias>
        from .ops import _per_file_root as _get_per_file_root

        per_file_root = _get_per_file_root(self)
        tmp_ctx = f"{per_file_root}/_tmp_mjoin_{uuid.uuid4().hex[:6]}_{idx}"
        ensure_tmp_ctx(self, tmp_ctx)
        create_join(
            self,
            dest_table_ctx=tmp_ctx,
            left_ref=left_ref,
            right_ref=right_ref,
            join_expr=step.get("join_expr", ""),
            select=step.get("select", {}),
            mode=step.get("mode", "inner"),
            left_where=step.get("left_where"),
            right_where=step.get("right_where"),
        )
        prev_ctx = tmp_ctx

    base_excl = list_private_fields(prev_ctx)
    rows = [
        e.entries
        for e in unify.get_logs(
            context=prev_ctx,
            filter=result_where,
            offset=result_offset,
            limit=result_limit,
            exclude_fields=base_excl,
        )
    ]
    return {"rows": rows}


def search_multi_join(
    self,
    *,
    joins: List[Dict[str, Any]],
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a sequence of joins then run semantic search (top-k) over the
    final materialized result.

    Parameters
    ----------
    joins : list[dict]
        Ordered steps; each step provides ``tables`` (two refs or "$prev"),
        ``join_expr``, ``select`` and optional ``mode``, ``left_where``, ``right_where``.
        Table references in ``tables`` accept:
        - Path-first (preferred): "<file_path>" for per-file Content,
          "<file_path>.Tables.<label>" for per-file tables
        - Logical names from `tables_overview()` or legacy refs
        - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
        - "$prev" to reference the previous join step's result
    references : dict[str, str] | None
        Mapping of expressions in the final result → reference text for ranking.
    k : int
        Maximum rows to return (<= 1000).
    filter : str | None
        Optional predicate before ranking.

    Returns
    -------
    list[dict[str, Any]]
        Top-k rows ranked by semantic similarity.
    """
    if not joins:
        return []
    out = filter_multi_join(
        self,
        joins=joins,
        result_where=None,
        result_limit=1000,
        result_offset=0,
    )
    tmp_ctx = f"{self._ctx}/_tmp_search_mjoin_{uuid.uuid4().hex[:6]}"
    ensure_tmp_ctx(self, tmp_ctx)
    rows = out.get("rows", [])
    if rows:
        unity_create_logs(
            context=tmp_ctx,
            entries=rows,
            batched=True,
            add_to_all_context=self.include_in_multi_assistant_table,
        )
    rows = table_search_top_k(
        context=tmp_ctx,
        references=references,
        k=k,
        unique_id_field=None,
        row_filter=filter,
    )
    return rows
