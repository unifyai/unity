from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
import uuid

from unity.common.log_utils import create_logs as unity_create_logs
from unity.common.search_utils import table_search_top_k
from unity.common.filter_utils import normalize_filter_expr

if TYPE_CHECKING:
    from unity.file_manager.managers.file_manager import FileManager


def ctx_for_storage(
    file_manager: "FileManager",
    *,
    storage_id: str,
    table: Optional[str] = None,
) -> str:
    """
    Resolve a storage_id reference to a fully-qualified Unify context.

    Parameters
    ----------
    storage_id : str
        The storage identifier (e.g., str(file_id) or a custom label).
    table : str | None
        If None, returns Content context.
        If provided, returns Tables/<table> context.

    Returns
    -------
    str
        Full Unify context path.
    """
    from .storage import ctx_for_file_content, ctx_for_file_table

    if table:
        return ctx_for_file_table(file_manager, storage_id=storage_id, table=table)
    return ctx_for_file_content(file_manager, storage_id=storage_id)


def resolve_table_ref(file_manager: "FileManager", ref: str) -> str:
    """
    Resolve a table reference to a fully-qualified context.

    Preferred forms (storage_id-first):
    - "s=<storage_id>" → Content context using storage_id
    - "s=<storage_id>.Tables.<table>" → Tables context using storage_id

    Also supported (for convenience):
    - "id=<file_id>" or "#<file_id>" → Content context (resolves file_id to storage_id)
    - "id=<file_id>.Tables.<table>" → Tables context (resolves file_id to storage_id)
    - "<storage_id>" → Content context (direct storage_id string)
    - "<storage_id>.Tables.<table>" → Tables context
    - "FileRecords" → global index context

    Notes
    -----
    storage_id is the stable identifier for context paths. It can be:
    - str(file_id) for files with auto-assigned storage (default)
    - A custom label for files with shared storage

    Use describe(file_path=...) to get the storage_id for a file.
    """
    from .storage import ctx_for_file_content, ctx_for_file_table

    # If the ref already looks like a fully-qualified context under this manager,
    # return as-is.
    try:
        _index_ctx = getattr(file_manager, "_ctx")
    except Exception:
        _index_ctx = None
    try:
        _files_root = getattr(file_manager, "_per_file_root")
    except Exception:
        _files_root = None
    if isinstance(ref, str):
        r = ref.strip()
        if _index_ctx and r == _index_ctx:
            return r
        if _files_root and r.startswith(f"{_files_root}/"):
            return r
        # Check if ref is already a fully-qualified context path (e.g., temporary join contexts)
        if _index_ctx and r.startswith(_index_ctx + "/"):
            return r

    # Handle "FileRecords" special case
    if ref.lower() == "filerecords":
        return file_manager._ctx

    # Helper to resolve file_id to storage_id
    def _get_storage_id_for_file_id(fid: int) -> Optional[str]:
        dm = file_manager._data_manager
        try:
            rows = dm.filter(
                context=file_manager._ctx,
                filter=f"file_id == {int(fid)}",
                limit=1,
                columns=["file_id", "storage_id"],
            )
            if rows:
                storage_id = rows[0].get("storage_id", "")
                file_id = rows[0].get("file_id")
                return storage_id if storage_id else str(file_id)
        except Exception:
            pass
        return None

    # Handle "s=<storage_id>" or "s=<storage_id>.Tables.<table>" forms
    if ref.startswith("s="):
        s_part = ref.split("=", 1)[1]
        if ".tables." in s_part.lower():
            parts = s_part.split(".Tables.", 1)
            storage_id = parts[0]
            tbl = parts[1] if len(parts) > 1 else None
        else:
            storage_id = s_part
            tbl = None
        if tbl:
            return ctx_for_file_table(file_manager, storage_id=storage_id, table=tbl)
        return ctx_for_file_content(file_manager, storage_id=storage_id)

    # Handle "id=<file_id>" or "id=<file_id>.Tables.<table>" forms
    if ref.startswith("id="):
        id_part = ref.split("=", 1)[1]
        if ".tables." in id_part.lower():
            parts = id_part.split(".Tables.", 1)
            file_id = int(parts[0])
            tbl = parts[1] if len(parts) > 1 else None
        else:
            file_id = int(id_part)
            tbl = None
        storage_id = _get_storage_id_for_file_id(file_id)
        if not storage_id:
            raise ValueError(f"No file found with file_id={file_id}")
        if tbl:
            return ctx_for_file_table(file_manager, storage_id=storage_id, table=tbl)
        return ctx_for_file_content(file_manager, storage_id=storage_id)

    # Handle "#<file_id>" or "#<file_id>.Tables.<table>" forms
    if ref.startswith("#"):
        rest = ref[1:]
        if ".tables." in rest.lower():
            parts = rest.split(".Tables.", 1)
            if parts[0].isdigit():
                file_id = int(parts[0])
                tbl = parts[1] if len(parts) > 1 else None
                storage_id = _get_storage_id_for_file_id(file_id)
                if not storage_id:
                    raise ValueError(f"No file found with file_id={file_id}")
                if tbl:
                    return ctx_for_file_table(
                        file_manager,
                        storage_id=storage_id,
                        table=tbl,
                    )
                return ctx_for_file_content(file_manager, storage_id=storage_id)
        elif rest.isdigit():
            file_id = int(rest)
            storage_id = _get_storage_id_for_file_id(file_id)
            if not storage_id:
                raise ValueError(f"No file found with file_id={file_id}")
            return ctx_for_file_content(file_manager, storage_id=storage_id)

    # Handle direct storage_id or storage_id.Tables.<table> forms
    base = getattr(file_manager, "_per_file_root")
    safe_fn = getattr(file_manager, "safe", lambda x: x)

    t = ref.strip()
    if ".tables." in t.lower():
        parts = t.split(".Tables.", 1)
        storage_id = parts[0]
        table_name = parts[1] if len(parts) > 1 else ""
        return ctx_for_file_table(file_manager, storage_id=storage_id, table=table_name)

    # Direct storage_id → Content context
    return ctx_for_file_content(file_manager, storage_id=t)


# --------------------- Index-level filter/search (FileRecords) ---------------- #


def filter_files(
    file_manager: "FileManager",
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
        - Path-first: "<storage_id>" for per-file Content,
          "<storage_id>.Tables.<label>" for per-file tables
        - Logical names from `describe()`: "FileRecords" for index
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
    dm = file_manager._data_manager
    normalized = normalize_filter_expr(filter)

    if tables is None:
        # Query the index directly via DataManager
        return dm.filter(
            context=file_manager._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            columns=columns,
        )

    # Normalize tables to list
    if isinstance(tables, str):
        table_names = [tables]
    else:
        table_names = list(tables)

    # Resolve contexts for each table name
    to_contexts: List[tuple[str, str]] = []
    for name in table_names:
        ctx = resolve_table_ref(file_manager, name)
        to_contexts.append((name, ctx))

    # Parallel filter across contexts via DataManager
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []

    def _fetch(_ctx: str) -> List[Dict[str, Any]]:
        return dm.filter(
            context=_ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            columns=columns,
        )

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
    file_manager: "FileManager",
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
        - Path-first: "<storage_id>" for per-file Content,
          "<storage_id>.Tables.<label>" for per-file tables
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
        context = getattr(file_manager, "_ctx", None)
        unique_id_field = "file_id"
    else:
        context = resolve_table_ref(file_manager, table)
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
    file_manager: "FileManager",
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
    left_ctx = resolve_table_ref(file_manager, left_ref)
    right_ctx = resolve_table_ref(file_manager, right_ref)

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

    # Delegate to DataManager.join_tables
    dm = file_manager._data_manager
    dm.join_tables(
        left_table=left_ctx,
        right_table=right_ctx,
        join_expr=join_expr,
        dest_table=dest_table_ctx,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )
    return dest_table_ctx


def ensure_tmp_ctx(file_manager: "FileManager", ctx: str) -> None:
    """Create a temporary join context via DataManager."""
    dm = file_manager._data_manager
    dm.create_table(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
    )


def filter_join(
    file_manager: "FileManager",
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
        - Path-first (preferred): "<storage_id>" for per-file Content,
          "<storage_id>.Tables.<label>" for per-file tables
        - Logical names from `describe()` or legacy refs
        - Legacy forms: "<storage_id>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
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

    tmp_ctx = f"{file_manager._ctx}/_tmp_join_{uuid.uuid4().hex[:6]}"
    ensure_tmp_ctx(file_manager, tmp_ctx)
    create_join(
        file_manager,
        dest_table_ctx=tmp_ctx,
        left_ref=tables[0],
        right_ref=tables[1],
        join_expr=join_expr,
        select=select,
        mode=mode,
        left_where=left_where,
        right_where=right_where,
    )
    # Fetch results via DataManager
    dm = file_manager._data_manager
    rows = dm.filter(
        context=tmp_ctx,
        filter=result_where,
        offset=result_offset,
        limit=result_limit,
    )
    return {"rows": rows}


def search_join(
    file_manager: "FileManager",
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
        - Path-first (preferred): "<storage_id>" for per-file Content,
          "<storage_id>.Tables.<label>" for per-file tables
        - Logical names from `describe()` or legacy refs
        - Legacy forms: "<storage_id>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
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

    tmp_ctx = f"{file_manager._ctx}/_tmp_join_{uuid.uuid4().hex[:6]}"
    ensure_tmp_ctx(file_manager, tmp_ctx)
    create_join(
        file_manager,
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
    file_manager: "FileManager",
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
        - Path-first (preferred): "<storage_id>" for per-file Content,
          "<storage_id>.Tables.<label>" for per-file tables
        - Logical names from `describe()` or legacy refs
        - Legacy forms: "<storage_id>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
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

        per_file_root = _get_per_file_root(file_manager)
        tmp_ctx = f"{per_file_root}/_tmp_mjoin_{uuid.uuid4().hex[:6]}_{idx}"
        ensure_tmp_ctx(file_manager, tmp_ctx)
        create_join(
            file_manager,
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

    # Fetch results via DataManager
    dm = file_manager._data_manager
    rows = dm.filter(
        context=prev_ctx,
        filter=result_where,
        offset=result_offset,
        limit=result_limit,
    )
    return {"rows": rows}


def search_multi_join(
    file_manager: "FileManager",
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
        - Path-first (preferred): "<storage_id>" for per-file Content,
          "<storage_id>.Tables.<label>" for per-file tables
        - Logical names from `describe()` or legacy refs
        - Legacy forms: "<storage_id>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
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
        file_manager,
        joins=joins,
        result_where=None,
        result_limit=1000,
        result_offset=0,
    )
    tmp_ctx = f"{file_manager._ctx}/_tmp_search_mjoin_{uuid.uuid4().hex[:6]}"
    ensure_tmp_ctx(file_manager, tmp_ctx)
    rows = out.get("rows", [])
    if rows:
        unity_create_logs(
            context=tmp_ctx,
            entries=rows,
            batched=True,
            add_to_all_context=file_manager.include_in_multi_assistant_table,
        )
    rows = table_search_top_k(
        context=tmp_ctx,
        references=references,
        k=k,
        unique_id_field=None,
        row_filter=filter,
    )
    return rows
