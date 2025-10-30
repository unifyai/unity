from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import uuid

import unify

from unity.common.search_utils import table_search_top_k


def resolve_table_ref(self, ref: str) -> str:
    """
    Resolve a table reference to a fully-qualified context.

    Accepted forms:
    - "FileRecords" → global index context (self._ctx)
    - "<file_path>:<table>" → per-table context for the given file
    - "id=<file_id>:<table>" or "#<file_id>:<table>" → resolve file_path by id then per-table context
    """
    if ":" not in ref:
        # Special token to reference the FileRecords/<alias> index context
        if ref.strip().lower() in {"filerecords", "records"}:
            return self._ctx
        raise ValueError("Table reference must be 'filename:table' or 'FileRecords'")
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
        filename = _lookup_path_by_id(file_id)
        return self._ctx_for_table(filename, tbl)
    if key.startswith("#") and key[1:].isdigit():
        file_id = int(key[1:])
        filename = _lookup_path_by_id(file_id)
        return self._ctx_for_table(filename, tbl)
    # Fallback: treat as file path / display name
    filename = key
    return self._ctx_for_table(filename, tbl)


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

    # Rewrite join/select to fully-qualified contexts
    join_expr = join_expr.replace(left_ref, left_ctx).replace(right_ref, right_ctx)
    select = {
        c.replace(left_ref, left_ctx).replace(right_ref, right_ctx): v
        for c, v in select.items()
    }

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
    try:
        unify.create_context(
            ctx,
            unique_keys={"row_id": "int"},
            auto_counting={"row_id": None},
        )
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
) -> Dict[str, List[Dict[str, Any]]]:
    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly two tables are required as 'filename:table'")

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
    rows = [
        e.entries
        for e in unify.get_logs(
            context=tmp_ctx,
            filter=result_where,
            offset=result_offset,
            limit=result_limit,
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
    if isinstance(tables, str):
        tables = [tables]
    if len(tables) != 2:
        raise ValueError("Exactly two tables are required as 'filename:table'")

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
    return table_search_top_k(
        context=tmp_ctx,
        references=references,
        k=k,
        allowed_fields=None,
        unique_id_field=None,
        filter_expr=filter,
    )


def filter_multi_join(
    self,
    *,
    joins: List[Dict[str, Any]],
    result_where: Optional[str] = None,
    result_limit: int = 100,
    result_offset: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
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

        tmp_ctx = f"{self._ctx}/_tmp_mjoin_{uuid.uuid4().hex[:6]}_{idx}"
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

    rows = [
        e.entries
        for e in unify.get_logs(
            context=prev_ctx,
            filter=result_where,
            offset=result_offset,
            limit=result_limit,
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
        unify.create_logs(context=tmp_ctx, entries=rows, batched=True)
    return table_search_top_k(
        context=tmp_ctx,
        references=references,
        k=k,
        allowed_fields=None,
        unique_id_field=None,
        filter_expr=filter,
    )
