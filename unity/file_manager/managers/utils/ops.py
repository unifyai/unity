from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import unify

from unity.common.log_utils import log as unity_log, create_logs as unity_create_logs
from unity.file_manager.types.file import FileRecordRow

if TYPE_CHECKING:
    from unity.file_manager.types.file import FileContentRow


def safe(self):
    return getattr(self, "safe") if hasattr(self, "safe") else (lambda x: x)


def _per_file_root(self) -> str:
    base = None
    try:
        _base_attr = getattr(self, "_per_file_root")
        if isinstance(_base_attr, str) and _base_attr:
            base = _base_attr
    except Exception:
        base = None
    if base is None:
        try:
            ctx = getattr(self, "_ctx")
            if isinstance(ctx, str) and "/FileRecords/" in ctx:
                prefix, alias = ctx.split("/FileRecords/", 1)
                base = f"{prefix}/Files/{alias}"
        except Exception:
            base = None
    return base or "Files"


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
    # Use file_path as-is (adapters handle both relative and absolute paths)

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
    log = unity_log(
        context=self._ctx,
        **entry,
        new=True,
        mutable=True,
        add_to_all_context=self.include_in_multi_assistant_table,
    )
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


def rename_file_contexts(
    self,
    *,
    old_file_path: str,
    new_file_path: str,
) -> Dict[str, Any]:
    """
    Rename per-file Content context and any per-file Tables contexts from the
    old file path to the new file path. Respects ingest layout:

    - per_file: rename <base>/Files/<alias>/<old>/Content → <base>/Files/<alias>/<new>/Content
      and every <base>/Files/<alias>/<old>/Tables/<label> accordingly.
    - unified: unified Content remains under the unified label; only per-file Tables
      contexts (keyed by safe(old_file_path)) are renamed to safe(new_file_path).

    Returns a dict with counts of renamed contexts.
    """
    from .storage import _resolve_file_target as _res

    old_info = _res(self, old_file_path)
    ingest_mode = old_info.get("ingest_mode")
    base = _per_file_root(self)
    safe = safe(self)
    old_root = safe(old_info.get("target_name") or old_file_path)
    new_root = safe(new_file_path)

    renamed = {"content": 0, "tables": 0}

    # Content context rename only for per_file
    if ingest_mode == "per_file":
        old_ctx = f"{base}/{old_root}/Content"
        new_ctx = f"{base}/{new_root}/Content"
        try:
            unify.rename_context(old_ctx, new_ctx)
            renamed["content"] += 1
        except Exception:
            pass

    # Tables contexts may exist under old_root
    try:
        prefix = f"{base}/{old_root}/Tables/"
        ctxs = unify.get_contexts(prefix=prefix)
    except Exception:
        ctxs = {}
    for full in (ctxs or {}).keys():
        try:
            label = full.split("/Tables/", 1)[-1]
            new_tbl = f"{base}/{new_root}/Tables/{safe(label)}"
            unify.rename_context(full, new_tbl)
            renamed["tables"] += 1
        except Exception:
            continue

    return {"renamed": renamed}


def update_file_record_by_path(
    self,
    *,
    old_file_path: str,
    new_file_path: str,
    **extra_updates: Any,
) -> Dict[str, Any]:
    """
    Update the FileRecords row identified by old_file_path to the new_file_path,
    applying any extra field updates (e.g., file_name, display_path, source_uri).
    """
    # Use paths as-is (adapters handle both relative and absolute paths)

    try:
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"file_path == {old_file_path!r}",
            limit=2,
            from_fields=["file_id"],
        )
    except Exception:
        rows = []
    if not rows:
        return {"outcome": "no-op", "reason": "not found"}
    if len(rows) > 1:
        raise ValueError(f"Multiple rows found for file_path={old_file_path!r}")
    log_id = rows[0].id
    updates = {"file_path": new_file_path}
    updates.update(extra_updates or {})
    unify.update_logs(
        logs=[log_id],
        context=self._ctx,
        entries=updates,
        overwrite=True,
    )
    return {
        "outcome": "updated",
        "details": {
            "file_id": rows[0].entries.get("file_id"),
            "file_path": new_file_path,
        },
    }


def delete_file_contexts(
    self,
    *,
    file_path: str,
) -> Dict[str, Any]:
    """
    Delete all contexts and rows associated with a file according to its ingest layout.

    - per_file: drop the per-file Content context and all per-file Tables contexts.
    - unified: do not drop the unified Content context; delete only rows in that
      context where source_uri matches this file's source_uri; drop this file's
      per-file Tables contexts if present.
    """
    from .storage import _resolve_file_target as _res

    info = _res(self, file_path)
    base = _per_file_root(self)
    safe = safe(self)
    ingest_mode = info.get("ingest_mode")
    unified_label = info.get("unified_label")

    purged = {"content_rows": 0, "tables": 0}

    # Fetch file_id for unified deletions
    # Use path as-is for querying
    try:
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"file_path == {file_path!r}",
            limit=1,
            from_fields=["file_id", "table_ingest"],
        )
    except Exception:
        rows = []
    file_id = rows[0].entries.get("file_id") if rows else None
    table_ingest = bool(rows[0].entries.get("table_ingest", True)) if rows else True

    if ingest_mode == "per_file":
        root = safe(file_path)
        ctx = f"{base}/{root}/Content"
        unify.delete_context(ctx)
        if table_ingest:
            prefix = f"{base}/{root}/Tables/"
            for tctx in (unify.get_contexts(prefix=prefix) or {}).keys():
                try:
                    unify.delete_context(tctx, missing_ok=False)
                    purged["tables"] += 1
                except Exception:
                    pass
    else:
        # unified: delete only rows with matching file_id from unified Content
        uctx = f"{base}/{safe(str(unified_label or 'Unified'))}/Content"
        try:
            filt = None if file_id is None else f"file_id == {int(file_id)}"
            ids = list(unify.get_logs(context=uctx, filter=filt, return_ids_only=True))
            if ids:
                unify.delete_logs(
                    logs=ids,
                    context=uctx,
                    project=unify.active_project(),
                    delete_empty_logs=True,
                )
                purged["content_rows"] += len(ids)
        except Exception:
            pass
        if table_ingest:
            root = safe(file_path)
            prefix = f"{base}/{root}/Tables/"
            for tctx in (unify.get_contexts(prefix=prefix) or {}).keys():
                try:
                    unify.delete_context(tctx, missing_ok=False)
                    purged["tables"] += 1
                except Exception:
                    pass

    return {"purged": purged}


def per_file_table_ctx(self, *, file_path: str, table: str) -> str:
    """Compatibility wrapper → storage.ctx_for_file_table."""
    from .storage import ctx_for_file_table as _ctx

    return _ctx(self, file_path=file_path, table=table)


def per_file_ctx(self, *, file_path: str) -> str:
    """Compatibility wrapper → storage.ctx_for_file."""
    from .storage import ctx_for_file as _ctx

    return _ctx(self, file_path=file_path)


def ensure_per_file_context(
    self,
    *,
    file_path: str,
) -> None:
    """Compatibility wrapper → storage.ensure_file_context."""
    from .storage import ensure_file_context as _ensure

    _ensure(self, file_path=file_path)


def ensure_per_file_table_context(
    self,
    *,
    file_path: str,
    table: str,
    unique_key: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    columns: Optional[List[str]] = None,
    example_row: Optional[Dict[str, Any]] = None,
    business_context: Optional[Any] = None,
) -> None:
    """Compatibility wrapper → storage.ensure_file_table_context."""
    from .storage import ensure_file_table_context as _ensure_tbl

    _ensure_tbl(
        self,
        file_path=file_path,
        table=table,
        unique_key=unique_key,
        business_context=business_context,
        auto_counting=auto_counting,
        columns=columns,
        example_row=example_row,
    )


def delete_per_file_table_rows_by_filter(
    self,
    *,
    file_path: str,
    table: str,
    filter_expr: Optional[str],
) -> int:
    ctx = per_file_table_ctx(self, file_path=file_path, table=table)
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
    file_path: str,
    table: str,
    rows: List[Dict[str, Any]],
) -> List[int]:
    if not rows:
        return []
    ctx = per_file_table_ctx(self, file_path=file_path, table=table)
    res = unity_create_logs(
        context=ctx,
        entries=rows,
        batched=True,
        add_to_all_context=self.include_in_multi_assistant_table,
    )
    return [lg.id for lg in res]


def delete_per_file_rows_by_filter(
    self,
    *,
    file_path: str,
    filter_expr: Optional[str],
) -> int:
    ctx = per_file_ctx(self, file_path=file_path)
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
    file_path: str,
    rows: List[Dict[str, Any]],
) -> List[int]:
    if not rows:
        return []
    ctx = per_file_ctx(self, file_path=file_path)
    res = unity_create_logs(
        context=ctx,
        entries=rows,
        batched=True,
        add_to_all_context=self.include_in_multi_assistant_table,
    )
    return [lg.id for lg in res]


# ---------- High-level create helpers (ensure + insert) ------------------------


def create_file_record(
    self,
    *,
    entry: FileRecordRow,
) -> Dict[str, Any]:
    """Create or update a FileRecord row in the global index (idempotent)."""
    return add_or_replace_file_row(
        self,
        entry=entry.model_dump(mode="json", exclude_none=True),
    )


def create_file_content(
    self,
    *,
    file_path: str,
    rows: List["FileContentRow"],
) -> List[int]:
    """Ensure per-file context then insert rows (batched)."""
    from unity.file_manager.types.file import FileContentRow

    ensure_per_file_context(
        self,
        file_path=file_path,
    )
    entries: List[Dict[str, Any]] = [
        (
            r.model_dump(mode="json", exclude_none=True)
            if isinstance(r, FileContentRow)
            else dict(r)
        )  # type: ignore[arg-type]
        for r in list(rows or [])
    ]
    return batch_insert_per_file_rows(self, file_path=file_path, rows=entries)


def create_file_table(
    self,
    *,
    file_path: str,
    table: str,
    rows: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    example_row: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Ensure per-file table context (with fields) then insert rows (batched)."""
    ensure_per_file_table_context(
        self,
        file_path=file_path,
        table=table,
        columns=columns,
        example_row=example_row,
    )
    return batch_insert_per_file_table_rows(
        self,
        file_path=file_path,
        table=table,
        rows=rows,
    )


# ----------------------------- Mutator helpers ------------------------------- #


def rename_file(
    self,
    *,
    file_id_or_path: Union[str, int],
    new_name: str,
) -> Dict[str, Any]:
    """Rename a file via adapter and propagate changes across contexts/index.

    Parameters
    ----------
    file_id_or_path : str | int
        Either the file_id (int) as preserved in the FileRecords index, or the
        fully-qualified file_path (str) as stored in the FileRecords index/context.
        When a file_id is provided, it is resolved to the corresponding file_path.
    new_name : str
        New file name; adapter determines path semantics.
    """
    if not getattr(self._adapter.capabilities, "can_rename", False):  # type: ignore[attr-defined]
        raise PermissionError("Rename not permitted by backend policy")

    # Resolve file_id_or_path to file_path
    if isinstance(file_id_or_path, int):
        try:
            logs = unify.get_logs(
                context=self._ctx,  # type: ignore[attr-defined]
                filter=f"file_id == {file_id_or_path}",
                limit=1,
                from_fields=["file_path"],
            )
            if not logs:
                raise ValueError(f"No file found with file_id {file_id_or_path}")
            raw_file_path = logs[0].entries.get("file_path")
            if not raw_file_path:
                raise ValueError(
                    f"File record with file_id {file_id_or_path} has no file_path",
                )
            # Use path as-is from index
            file_path = str(raw_file_path)
        except Exception as e:
            raise ValueError(f"Failed to resolve file_id {file_id_or_path}: {e}")
    else:
        # Use path as-is (only convert to string if needed)
        file_path = str(file_id_or_path)

    new_name = str(new_name)

    # Query index with path as-is
    try:
        logs = unify.get_logs(
            context=self._ctx,  # type: ignore[attr-defined]
            filter=f"file_path == {file_path!r}",
            limit=2,
            from_fields=["file_id", "file_path"],
        )
    except Exception:
        logs = []
    if not logs:
        raise ValueError(f"File '{file_path}' not found in Unify logs.")
    if len(logs) > 1:
        raise ValueError(
            f"Multiple files found with file_path '{file_path}'. Data integrity issue.",
        )

    old_file_path = logs[0].entries.get("file_path", file_path)
    # Use path as-is (adapters handle both relative and absolute via _abspath)
    ref = self._adapter.rename(old_file_path, new_name)  # type: ignore[attr-defined]
    new_path = ref.path

    try:
        rename_file_contexts(self, old_file_path=old_file_path, new_file_path=new_path)
        base_name = new_name.rsplit(".", 1)[0] if "." in new_name else new_name
        update_file_record_by_path(
            self,
            old_file_path=old_file_path,
            new_file_path=new_path,
        )
    except Exception:
        pass

    return getattr(ref, "model_dump", lambda: {"path": ref.path, "name": new_name})()


def move_file(
    self,
    *,
    file_id_or_path: Union[str, int],
    new_parent_path: str,
) -> Dict[str, Any]:
    """Move a file via adapter and propagate changes across contexts/index.

    Parameters
    ----------
    file_id_or_path : str | int
        Either the file_id (int) as preserved in the FileRecords index, or the
        fully-qualified file_path (str) as stored in the FileRecords index/context.
        When a file_id is provided, it is resolved to the corresponding file_path.
    new_parent_path : str
        Destination directory in adapter-native form.
    """
    if not getattr(self._adapter.capabilities, "can_move", False):  # type: ignore[attr-defined]
        raise PermissionError("Move not permitted by backend policy")

    # Resolve file_id_or_path to file_path
    if isinstance(file_id_or_path, int):
        try:
            logs = unify.get_logs(
                context=self._ctx,  # type: ignore[attr-defined]
                filter=f"file_id == {file_id_or_path}",
                limit=1,
                from_fields=["file_path"],
            )
            if not logs:
                raise ValueError(f"No file found with file_id {file_id_or_path}")
            raw_file_path = logs[0].entries.get("file_path")
            if not raw_file_path:
                raise ValueError(
                    f"File record with file_id {file_id_or_path} has no file_path",
                )
            # Use path as-is from index
            file_path = str(raw_file_path)
        except Exception as e:
            raise ValueError(f"Failed to resolve file_id {file_id_or_path}: {e}")
    else:
        # Use path as-is (only convert to string if needed)
        file_path = str(file_id_or_path)

    new_parent_path = str(new_parent_path)

    # Query index with path as-is
    try:
        logs = unify.get_logs(
            context=self._ctx,  # type: ignore[attr-defined]
            filter=f"file_path == {file_path!r}",
            limit=2,
            from_fields=["file_id", "file_path"],
        )
    except Exception:
        logs = []
    if not logs:
        raise ValueError(f"File '{file_path}' not found in Unify logs.")
    if len(logs) > 1:
        raise ValueError(
            f"Multiple files found with file_path '{file_path}'. Data integrity issue.",
        )
    old_file_path = logs[0].entries.get("file_path", file_path)

    # Use paths as-is (adapters handle both relative and absolute via _abspath)
    ref = self._adapter.move(old_file_path, new_parent_path)  # type: ignore[attr-defined]
    new_path = ref.path

    try:
        rename_file_contexts(self, old_file_path=old_file_path, new_file_path=new_path)
        base_name = new_path.rsplit("/", 1)[-1]
        base_name = base_name.rsplit(".", 1)[0] if "." in base_name else base_name
        update_file_record_by_path(
            self,
            old_file_path=old_file_path,
            new_file_path=new_path,
        )
    except Exception:
        pass

    return getattr(
        ref,
        "model_dump",
        lambda: {"path": ref.path, "parent": new_parent_path},
    )()


def delete_file(
    self,
    *,
    file_id_or_path: Union[str, int],
    _log_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Delete a file record and its contexts per ingest-mode rules.

    Parameters
    ----------
    file_id_or_path : str | int
        Either the file_id (int) as preserved in the FileRecords index, or the
        fully-qualified file_path (str) as stored in the FileRecords index/context.
        When a file_id is provided, it is resolved to the corresponding file_path.
    _log_id : int | None
        Optional existing log ID to delete (speeds up deletion).
    """
    # Resolve file_id_or_path to file_path and file_id
    if isinstance(file_id_or_path, int):
        file_id = file_id_or_path
        try:
            logs = unify.get_logs(
                context=self._ctx,  # type: ignore[attr-defined]
                filter=f"file_id == {file_id}",
                limit=1,
                from_fields=["file_id", "file_path"],
            )
            if not logs:
                raise ValueError(f"No file found with file_id {file_id}")
            raw_file_path = logs[0].entries.get("file_path", "")
            if not raw_file_path:
                raise ValueError(f"File record with file_id {file_id} has no file_path")
            # Use path as-is from index
            file_path = str(raw_file_path)
        except Exception as e:
            raise ValueError(f"Failed to resolve file_id {file_id}: {e}")
    else:
        # Use path as-is (only convert to string if needed)
        file_path = str(file_id_or_path)
        # Query index with path as-is
        try:
            logs = unify.get_logs(
                context=self._ctx,  # type: ignore[attr-defined]
                filter=f"file_path == {file_path!r}",
                limit=2,
                from_fields=["file_id", "file_path"],
            )
        except Exception:
            logs = []
        if not logs:
            raise ValueError(f"File '{file_path}' not found in Unify logs.")
        if len(logs) > 1:
            raise ValueError(
                f"Multiple files found with file path '{file_path}'. Data integrity issue.",
            )
        file_path = logs[0].entries.get("file_path", file_path)
        file_id = logs[0].entries.get("file_id")
        if file_id is None:
            raise ValueError(f"File record for '{file_path}' has no file_id")

    # 1) Resolve log id
    if _log_id is None:
        try:
            log_ids = unify.get_logs(
                context=self._ctx,  # type: ignore[attr-defined]
                filter=f"file_id == {file_id}",
                limit=2,
                return_ids_only=True,
            )
        except Exception:
            log_ids = []
        if not log_ids:
            raise ValueError(f"No file found with file_id {file_id}")
        if len(log_ids) > 1:
            raise RuntimeError(
                f"Multiple files found with file_id {file_id}. Data integrity issue.",
            )
        _log_id = log_ids[0]

    # 2) Get file_path for adapter deletion and protected check (already resolved above)

    if getattr(self, "is_protected")(file_path):  # type: ignore[attr-defined]
        raise PermissionError(
            f"'{file_path}' is protected and cannot be deleted by FileManager.",
        )

    # 3) Adapter deletion when supported
    if getattr(self, "_adapter", None) is not None and getattr(  # type: ignore[attr-defined]
        self._adapter.capabilities,
        "can_delete",
        False,
    ):
        try:
            self._adapter.delete(file_path)  # type: ignore[attr-defined]
        except (NotImplementedError, FileNotFoundError):
            pass

    # 4) Ingest-aware purge of contexts/rows
    try:
        delete_file_contexts(self, file_path=file_path)
    except Exception:
        pass

    # 5) Delete index row
    try:
        unify.delete_logs(context=self._ctx, logs=_log_id)  # type: ignore[attr-defined]
    except Exception as e:
        raise ValueError(f"Failed to delete file record: {e}")

    return {
        "outcome": "file deleted",
        "details": {"file_id": file_id, "file_path": file_path},
    }
