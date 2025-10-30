from __future__ import annotations

from typing import Any, Dict, List, Optional

import unify
from ...common.context_store import TableStore
from ...common.model_to_fields import model_to_fields
from ..types.file import File as _PerFileRow


# ---------- FileRecords root (FileRecords/<alias>) helpers ---------------------


def add_or_replace_file_row(
    self,
    *,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create or replace a file row in the FileRecords/<alias> context.

    Returns a short outcome dict.
    """
    # Try to find an existing row by file_path; replace if found
    fp = entry.get("file_path")
    if fp:
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"file_path == {fp!r}",
            limit=2,
            from_fields=["file_id", "file_path"],
        )

        if len(rows) > 1:
            raise ValueError(
                f"Multiple index rows found for file_path {fp!r} (data integrity)",
            )
        if len(rows) == 1:
            # Update existing row
            unify.update_logs(
                logs=[rows[0].id],
                context=self._ctx,
                entries=entry,
                overwrite=True,
            )
            return {
                "outcome": "file updated successfully",
                "details": {
                    "file_id": rows[0].entries.get("file_id"),
                    "file_path": fp,
                },
            }

    # Create new row when no existing match
    log = unify.log(context=self._ctx, **entry, new=True, mutable=True)
    return {
        "outcome": "file created successfully",
        "details": {
            "file_id": log.entries.get("file_id"),
            "file_path": entry.get("file_path"),
        },
    }


def delete_file_rows_by_ids(
    self,
    *,
    log_ids: List[int],
) -> Dict[str, Any]:
    if not log_ids:
        return {"status": "no-op"}
    return unify.delete_logs(
        logs=list(log_ids),
        context=self._ctx,
        project=unify.active_project(),
        delete_empty_logs=True,
    )


def per_file_table_ctx(self, *, filename: str, table: str) -> str:
    """Return the fully-qualified context for a per-file table.

    Shape: <base>/File/<alias>/<safe_filename>/Tables/<safe_table>
    """
    safe_fn = getattr(self, "_safe") if hasattr(self, "_safe") else (lambda x: x)
    base = None
    try:
        _base_attr = getattr(self, "_per_file_root")
        if isinstance(_base_attr, str) and _base_attr:
            base = _base_attr
    except Exception:
        base = None
    if base is None:
        # Derive from FileRecords ctx: <prefix>/FileRecords/<alias> → <prefix>/File/<alias>
        try:
            ctx = getattr(self, "_ctx")
            if isinstance(ctx, str) and "/FileRecords/" in ctx:
                prefix, alias = ctx.split("/FileRecords/", 1)
                base = f"{prefix}/File/{alias}"
        except Exception:
            base = None
    if base is None:
        base = "File"
    return f"{base}/{safe_fn(filename)}/Tables/{safe_fn(table)}"


def per_file_ctx(self, *, filename: str) -> str:
    """Return the fully-qualified context for a per-file root (no Tables suffix)."""
    safe_fn = getattr(self, "_safe") if hasattr(self, "_safe") else (lambda x: x)
    base = None
    try:
        _base_attr = getattr(self, "_per_file_root")
        if isinstance(_base_attr, str) and _base_attr:
            base = _base_attr
    except Exception:
        base = None
    if base is None:
        # Derive from FileRecords ctx: <prefix>/FileRecords/<alias> → <prefix>/File/<alias>
        try:
            ctx = getattr(self, "_ctx")
            if isinstance(ctx, str) and "/FileRecords/" in ctx:
                prefix, alias = ctx.split("/FileRecords/", 1)
                base = f"{prefix}/File/{alias}"
        except Exception:
            base = None
    if base is None:
        base = "File"
    return f"{base}/{safe_fn(filename)}/Content"


def ensure_per_file_context(self, *, filename: str) -> None:
    """
    Ensure a per-file context exists with the File schema and hierarchical counters.
    """
    ctx = per_file_ctx(self, filename=filename)
    fields = model_to_fields(_PerFileRow)
    store = TableStore(
        ctx,
        unique_keys={"content_id": "int"},
        auto_counting={
            "content_id": None,
            "document_id": None,
            "section_id": "document_id",
            "image_id": "section_id",
            "table_id": "section_id",
            "paragraph_id": "section_id",
            "sentence_id": "paragraph_id",
        },
        fields=fields,
        description=f"Per-file context for '{filename}' using the File schema",
    )
    try:
        store.ensure_context()
    except Exception as e:
        print(f"Error ensuring per-file context: {e}")
        # Best-effort provisioning; do not fail the caller on ensure errors


def ensure_per_file_table_context(
    self,
    *,
    filename: str,
    table: str,
    unique_key: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    columns: Optional[List[str]] = None,
    example_row: Optional[Dict[str, Any]] = None,
) -> None:
    ctx = per_file_table_ctx(self, filename=filename, table=table)
    # Build fields from columns or example_row keys when provided
    fields_map = []
    try:
        if columns:
            for c in columns:
                fields_map.append(c)
        elif example_row:
            for k in example_row.keys():
                fields_map.append(k)
    except Exception:
        fields_map = []
    try:
        unify.create_context(
            ctx,
            unique_keys={unique_key: "int"},
            auto_counting=(
                auto_counting if auto_counting is not None else {unique_key: None}
            ),
            fields=fields_map,
        )
    except Exception:
        # Best-effort provisioning; safe to ignore when context already exists
        pass


def delete_per_file_table_rows_by_filter(
    self,
    *,
    filename: str,
    table: str,
    filter_expr: Optional[str],
) -> int:
    ctx = per_file_table_ctx(self, filename=filename, table=table)
    if filter_expr is None:
        ids = unify.get_logs(context=ctx, return_ids_only=True)
    else:
        ids = unify.get_logs(context=ctx, filter=filter_expr, return_ids_only=True)
    ids = list(ids)
    if not ids:
        return 0
    unify.delete_logs(
        logs=ids,
        context=ctx,
        project=unify.active_project(),
        delete_empty_logs=True,
    )
    return len(ids)


def batch_insert_per_file_table_rows(
    self,
    *,
    filename: str,
    table: str,
    rows: List[Dict[str, Any]],
) -> List[int]:
    if not rows:
        return []
    ctx = per_file_table_ctx(self, filename=filename, table=table)
    res = unify.create_logs(context=ctx, entries=rows, batched=True)
    return [lg.id for lg in res]


def delete_per_file_rows_by_filter(
    self,
    *,
    filename: str,
    filter_expr: Optional[str],
) -> int:
    ctx = per_file_ctx(self, filename=filename)
    if filter_expr is None:
        ids = unify.get_logs(context=ctx, return_ids_only=True)
    else:
        ids = unify.get_logs(context=ctx, filter=filter_expr, return_ids_only=True)
    ids = list(ids)
    if not ids:
        return 0
    unify.delete_logs(
        logs=ids,
        context=ctx,
        project=unify.active_project(),
        delete_empty_logs=True,
    )
    return len(ids)


def batch_insert_per_file_rows(
    self,
    *,
    filename: str,
    rows: List[Dict[str, Any]],
) -> List[int]:
    if not rows:
        return []
    ctx = per_file_ctx(self, filename=filename)
    res = unify.create_logs(context=ctx, entries=rows, batched=True)
    return [lg.id for lg in res]


# ---------- High-level create helpers (ensure + insert) ------------------------


def create_file_record(
    self,
    *,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Create or update a FileRecord row in the global index (idempotent)."""
    return add_or_replace_file_row(self, entry=entry)


def create_file(
    self,
    *,
    filename: str,
    rows: List[Dict[str, Any]],
) -> List[int]:
    """Ensure per-file context then insert rows (batched)."""
    ensure_per_file_context(self, filename=filename)
    return batch_insert_per_file_rows(self, filename=filename, rows=rows)


def create_file_table(
    self,
    *,
    filename: str,
    table: str,
    rows: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    example_row: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Ensure per-file table context (with fields) then insert rows (batched)."""
    ensure_per_file_table_context(
        self,
        filename=filename,
        table=table,
        columns=columns,
        example_row=example_row,
    )
    return batch_insert_per_file_table_rows(
        self,
        filename=filename,
        table=table,
        rows=rows,
    )
