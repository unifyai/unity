from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import time

import unify
from unity.file_manager.types.file import FileRecord, FileContent
from unity.file_manager.types.config import (
    FilePipelineConfig as _FilePipelineConfig,
    EmbeddingSpec,
    resolve_callables as _resolve_callables,
)
from unity.common.embed_utils import ensure_vector_column


def _safe(self):
    return getattr(self, "_safe") if hasattr(self, "_safe") else (lambda x: x)


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


def apply_content_ingest_policy(
    records: List[Dict[str, Any]],
    *,
    config: _FilePipelineConfig,
    file_format: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Apply per-format content ingestion policy to parsed content rows before insertion.

    - When no policy exists for the format → return rows unchanged.
    - mode == "none" → return []
    - mode == "document_only" → keep (or synthesize) a single document row
      and drop fields listed in policy.omit_fields.
    - mode == "default" → keep rows but drop fields listed in policy.omit_fields.
    """
    rows: List[Dict[str, Any]] = list(records or [])
    fmt = (file_format or "").strip().lower()
    if not fmt:
        return rows
    policy = (
        getattr(getattr(config, "ingest", None), "content_policy_by_format", {}) or {}
    ).get(fmt)
    if policy is None:
        return rows
    mode = getattr(policy, "mode", "default")
    omit_fields = list(getattr(policy, "omit_fields", []) or [])
    if mode == "none":
        return []
    if mode == "document_only":
        doc_rows = [
            r for r in rows if str(r.get("content_type") or "").lower() == "document"
        ]
        if not doc_rows:
            id_layout = getattr(getattr(config, "ingest", None), "id_layout", "map")
            synthesized: Dict[str, Any] = {"content_type": "document"}
            if id_layout == "columns":
                synthesized["document_id"] = 0
            else:
                synthesized["content_id"] = {"document": 0}
            doc_rows = [synthesized]
        cleaned: List[Dict[str, Any]] = []
        for r in doc_rows:
            r2 = {k: v for k, v in r.items() if k not in omit_fields}
            cleaned.append(r2)
        return cleaned
    # default: keep rows, drop omitted fields if any
    if not omit_fields:
        return rows
    return [{k: v for k, v in r.items() if k not in omit_fields} for r in rows]


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
    safe = _safe(self)
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
    unify.update_logs(logs=[log_id], context=self._ctx, entries=updates, overwrite=True)
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
    safe = _safe(self)
    ingest_mode = info.get("ingest_mode")
    unified_label = info.get("unified_label")

    purged = {"content_rows": 0, "tables": 0}

    # Fetch file_id for unified deletions
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
        try:
            unify.delete_context(ctx)
        except Exception:
            try:
                ids = list(unify.get_logs(context=ctx, return_ids_only=True))
                if ids:
                    unify.delete_logs(
                        logs=ids,
                        context=ctx,
                        project=unify.active_project(),
                        delete_empty_logs=True,
                    )
                    purged["content_rows"] += len(ids)
            except Exception:
                pass
        if table_ingest:
            try:
                prefix = f"{base}/{root}/Tables/"
                for tctx in (unify.get_contexts(prefix=prefix) or {}).keys():
                    try:
                        unify.delete_context(tctx)
                        purged["tables"] += 1
                    except Exception:
                        pass
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
            try:
                prefix = f"{base}/{root}/Tables/"
                for tctx in (unify.get_contexts(prefix=prefix) or {}).keys():
                    try:
                        unify.delete_context(tctx)
                        purged["tables"] += 1
                    except Exception:
                        pass
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
    auto_counting_per_file: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """Compatibility wrapper → storage.ensure_file_context."""
    from .storage import ensure_file_context as _ensure

    _ensure(self, file_path=file_path, auto_counting_per_file=auto_counting_per_file)


def ensure_per_file_table_context(
    self,
    *,
    file_path: str,
    table: str,
    unique_key: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    columns: Optional[List[str]] = None,
    example_row: Optional[Dict[str, Any]] = None,
) -> None:
    """Compatibility wrapper → storage.ensure_file_table_context."""
    from .storage import ensure_file_table_context as _ensure_tbl

    _ensure_tbl(
        self,
        file_path=file_path,
        table=table,
        unique_key=unique_key,
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
    res = unify.create_logs(context=ctx, entries=rows, batched=True)
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


def create_file_content(
    self,
    *,
    file_path: str,
    auto_counting_per_file: Optional[Dict[str, Optional[str]]] = None,
    rows: List[Dict[str, Any]],
) -> List[int]:
    """Ensure per-file context then insert rows (batched)."""
    ensure_per_file_context(
        self,
        file_path=file_path,
        auto_counting_per_file=auto_counting_per_file,
    )
    return batch_insert_per_file_rows(self, file_path=file_path, rows=rows)


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
            file_path = logs[0].entries.get("file_path")
            if not file_path:
                raise ValueError(
                    f"File record with file_id {file_id_or_path} has no file_path",
                )
        except Exception as e:
            raise ValueError(f"Failed to resolve file_id {file_id_or_path}: {e}")
    else:
        file_path = str(file_id_or_path).lstrip("/")

    new_name = str(new_name)

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
            f"Multiple files found with filename '{file_path}'. Data integrity issue.",
        )

    old_file_path = logs[0].entries.get("file_path", file_path)
    ref = self._adapter.rename(file_path, new_name)  # type: ignore[attr-defined]
    new_path = ref.path.lstrip("/")

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
            file_path = logs[0].entries.get("file_path")
            if not file_path:
                raise ValueError(
                    f"File record with file_id {file_id_or_path} has no file_path",
                )
        except Exception as e:
            raise ValueError(f"Failed to resolve file_id {file_id_or_path}: {e}")
    else:
        file_path = str(file_id_or_path).lstrip("/")

    new_parent_path = str(new_parent_path).lstrip("/")

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
            f"Multiple files found with filename '{file_path}'. Data integrity issue.",
        )
    old_file_path = logs[0].entries.get("file_path", file_path)

    ref = self._adapter.move(file_path, new_parent_path)  # type: ignore[attr-defined]
    new_path = ref.path.lstrip("/")

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
            filename = logs[0].entries.get("file_path", "")
            if not filename:
                raise ValueError(f"File record with file_id {file_id} has no file_path")
        except Exception as e:
            raise ValueError(f"Failed to resolve file_id {file_id}: {e}")
    else:
        file_path = str(file_id_or_path).lstrip("/")
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
                f"Multiple files found with filename '{file_path}'. Data integrity issue.",
            )
        filename = logs[0].entries.get("file_path", file_path)
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

    if getattr(self, "is_protected")(filename):  # type: ignore[attr-defined]
        raise PermissionError(
            f"'{filename}' is protected and cannot be deleted by FileManager.",
        )

    # 3) Adapter deletion when supported
    if getattr(self, "_adapter", None) is not None and getattr(  # type: ignore[attr-defined]
        self._adapter.capabilities,
        "can_delete",
        False,
    ):
        try:
            self._adapter.delete(filename)  # type: ignore[attr-defined]
        except (NotImplementedError, FileNotFoundError):
            pass

    # 4) Ingest-aware purge of contexts/rows
    try:
        delete_file_contexts(self, file_path=filename)
    except Exception:
        pass

    # 5) Delete index row
    try:
        unify.delete_logs(context=self._ctx, logs=_log_id)  # type: ignore[attr-defined]
    except Exception as e:
        raise ValueError(f"Failed to delete file record: {e}")

    return {
        "outcome": "file deleted",
        "details": {"file_id": file_id, "file_path": filename},
    }


# ----------------------------- Ingest/Embed helpers (chunked) ------------------- #


def index_file_record(
    self,
    *,
    file_path: str,
    result: Dict[str, Any],
    config: _FilePipelineConfig,
) -> None:
    """
    Create/update the FileRecord index row for a file (idempotent).
    Mirrors the first step of FileManager._ingest.
    """
    ident = getattr(self, "_build_file_identity")(file_path)
    from .ops import (
        create_file_record as _ops_create_file_record,
    )  # local import to avoid cycles

    _ops_create_file_record(
        self,
        entry=FileRecord.to_file_record_entry(
            file_path=file_path,
            source_uri=getattr(ident, "source_uri", None),
            source_provider=getattr(ident, "source_provider", None),
            result=result,
            ingest_mode=(
                getattr(getattr(config, "ingest", None), "mode", "per_file")
                or "per_file"
            ),
            unified_label=(
                getattr(getattr(config, "ingest", None), "unified_label", None)
                if getattr(getattr(config, "ingest", None), "mode", "per_file")
                == "unified"
                else None
            ),
            table_ingest=bool(
                getattr(getattr(config, "ingest", None), "table_ingest", True),
            ),
        ),
    )


def resolve_embed_strategy(
    document: Any,
    result: Dict[str, Any],
    config: _FilePipelineConfig,
) -> str:
    """
    Decide embedding strategy based on config and size heuristic.
    Returns one of: \"off\", \"after\", \"along\".
    """
    strat = getattr(getattr(config, "embed", None), "strategy", "auto")
    if strat in ("off", "after", "along"):
        return str(strat)
    # auto: compute size
    try:
        n_records = int(result.get("total_records", 0) or 0)
    except Exception:
        n_records = 0
    try:
        tables = getattr(getattr(document, "metadata", None), "tables", []) or []
        total_table_rows = 0
        for t in tables:
            rows = getattr(t, "rows", None)
            try:
                total_table_rows += len(rows) if rows is not None else 0
            except Exception:
                continue
    except Exception:
        total_table_rows = 0
    size = max(n_records, total_table_rows)
    threshold = int(getattr(getattr(config, "embed", None), "large_threshold", 2000))
    return "along" if size >= threshold else "after"


def iter_ingest_content_rows(
    self,
    *,
    file_path: str,
    records: List[Dict[str, Any]],
    config: _FilePipelineConfig,
    batch_size: int,
    replace_existing: bool = True,
) -> Iterator[List[int]]:
    """
    Incrementally ingest per-file Content rows in chunks; yields inserted ids per chunk.
    Performs delete-once when replace_existing=True.
    """
    rows: List[Dict[str, Any]] = list(records or [])
    # Filter allowed columns
    allowed = (
        set(config.ingest.allowed_columns) if config.ingest.allowed_columns else None
    )
    if allowed:
        rows = [{k: v for k, v in rec.items() if k in allowed} for rec in rows]

    # destination context display name
    dest_name = (
        file_path
        if config.ingest.mode == "per_file"
        else (config.ingest.unified_label or "Unified")
    )
    # lookup file_id from index
    _rows = unify.get_logs(
        context=self._ctx,  # type: ignore[attr-defined]
        filter=f"file_path == {file_path!r}",
        limit=1,
        from_fields=["file_id"],
    )
    _fid = _rows[0].entries.get("file_id") if _rows else None
    if _fid is None:
        raise ValueError(f"File ID not found for file_path: {file_path}")

    # delete once if requested
    did_delete = False
    if replace_existing:
        try:
            if config.ingest.mode == "per_file":
                delete_per_file_rows_by_filter(
                    self,
                    file_path=dest_name,
                    filter_expr=None,
                )
            else:
                filt = f"file_id == {_fid}"
                delete_per_file_rows_by_filter(
                    self,
                    file_path=dest_name,
                    filter_expr=filt,
                )
            did_delete = True  # noqa: F841
        except Exception:
            pass

    # chunk and insert
    if batch_size <= 0:
        batch_size = 1000
    id_layout = getattr(getattr(config, "ingest", None), "id_layout", "map")
    auto_cnt = (
        getattr(config.ingest, "id_hierarchy", None) if id_layout == "columns" else None
    )

    for i in range(0, len(rows), int(batch_size)):
        chunk = rows[i : i + int(batch_size)]
        file_content_entries: List[Dict[str, Any]] = (
            FileContent.to_file_content_entries(
                file_id=int(_fid),
                rows=chunk,
                id_layout=id_layout,
            )
        )
        ids = create_file_content(
            self,
            file_path=dest_name,
            auto_counting_per_file=auto_cnt,
            rows=file_content_entries,
        )
        yield list(ids or [])


def iter_ingest_tables_for_document(
    self,
    *,
    file_path: str,
    document: Any,
    table_rows_batch_size: int = 100,
) -> Iterator[Tuple[str, List[int]]]:
    """
    Incrementally ingest per-file tables for a document; yields (table_ctx, inserted_ids) per batch.
    """
    tables = getattr(getattr(document, "metadata", None), "tables", []) or []
    if not tables:
        return

    for idx, tbl in enumerate(tables, start=1):
        columns = getattr(tbl, "columns", None)
        rows = getattr(tbl, "rows", None)
        if not rows:
            continue

        # derive columns when missing
        if not columns:
            first = rows[0]
            if isinstance(first, dict):
                columns = list(first.keys())
            else:
                columns = [str(val) for val in first]
            # treat first row as header row and drop it
            rows = rows[1:]

        # label
        sheet_name = getattr(tbl, "sheet_name", None)
        if sheet_name:
            table_label = f"{sheet_name}"
        else:
            section_path = getattr(tbl, "section_path", None)
            if section_path:
                try:
                    table_label = _safe(self)(str(section_path))  # type: ignore
                except Exception:
                    table_label = f"{idx:02d}"
            else:
                table_label = f"{idx:02d}"

        # ensure table context
        ensure_per_file_table_context(
            self,
            file_path=file_path,
            table=table_label,
            columns=list(columns) if columns else None,
            example_row=(rows[0] if (rows and isinstance(rows[0], dict)) else None),
        )
        table_ctx = per_file_table_ctx(self, file_path=file_path, table=table_label)

        # batch insert
        batch: List[Dict[str, Any]] = []
        for r in rows:
            if isinstance(r, dict):
                entry = {
                    str(k): (str(v) if v is not None else "") for k, v in r.items()
                }
            else:
                entry = {
                    str(col): (str(val) if val is not None else "")
                    for col, val in zip(columns, r)
                }
            batch.append(entry)
            if len(batch) >= max(1, int(table_rows_batch_size)):
                ids = create_file_table(
                    self,
                    file_path=file_path,
                    table=table_label,
                    rows=batch,
                    columns=list(columns) if columns else None,
                    example_row=(
                        rows[0] if (rows and isinstance(rows[0], dict)) else None
                    ),
                )
                yield table_ctx, list(ids or [])
                batch = []

        if batch:
            ids = create_file_table(
                self,
                file_path=file_path,
                table=table_label,
                rows=batch,
                columns=list(columns) if columns else None,
                example_row=(rows[0] if (rows and isinstance(rows[0], dict)) else None),
            )
            yield table_ctx, list(ids or [])


def embed_content_chunk_for_ids(
    ctx_name: str,
    specs: List[EmbeddingSpec],
    inserted_ids: List[int] | None,
    *,
    enable_progress: bool = False,
) -> None:
    """
    Apply embedding specs targeting content contexts, scoped to inserted ids when provided.
    Expects `specs` to already be filtered to the appropriate context kind by the caller.
    """
    if not specs:
        return
    if enable_progress:
        try:
            print(
                f"[EmbedOps] Content chunk: ctx={ctx_name} specs={len(specs)} from_ids={len(inserted_ids or [])}",
            )
        except Exception:
            pass
    # Filter applicable specs (content contexts only)
    applicable_specs = [sp for sp in specs if sp.context in ("per_file", "unified")]
    if not applicable_specs:
        return

    batch_start = time.perf_counter()

    def _run_embed(sp: EmbeddingSpec) -> str:
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   -> ensure_vector_column target={sp.target_column} source={sp.source_column}",
                )
            except Exception:
                pass
        t0 = time.perf_counter()
        ensure_vector_column(
            ctx_name,
            embed_column=sp.target_column,
            source_column=sp.source_column,
            from_ids=list(inserted_ids or []),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   <- done target={sp.target_column} time_ms={dt:.1f}",
                )
            except Exception:
                pass
        return sp.target_column

    max_workers = min(8, max(1, len(applicable_specs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_embed, sp): sp for sp in applicable_specs}
        for fut in as_completed(futures):
            # Surface exceptions to fail fast; otherwise ignore
            fut.result()
    if enable_progress:
        try:
            print(
                f"[EmbedOps] Content chunk complete: specs={len(applicable_specs)} total_time_ms={(time.perf_counter()-batch_start)*1000.0:.1f}",
            )
        except Exception:
            pass


def embed_table_chunk_for_ids(
    self,
    *,
    table_ctx: str,
    specs: List[EmbeddingSpec],
    inserted_ids: List[int] | None = None,
    enable_progress: bool = False,
) -> None:
    """
    Apply embedding specs targeting a per-file-table context for a specific table chunk.
    """
    if not specs:
        return
    # extract tail label after /Tables/
    tail = None
    try:
        if "/Tables/" in table_ctx:
            tail = table_ctx.split("/Tables/", 1)[-1]
    except Exception:
        tail = None
    safe = _safe(self)
    if enable_progress:
        try:
            print(
                f"[EmbedOps] Table chunk: ctx={table_ctx} specs={len(specs)} from_ids={len(inserted_ids or [])}",
            )
        except Exception:
            pass
    applicable: list[EmbeddingSpec] = []
    for sp in specs:
        if sp.context != "per_file_table":
            continue
        table_filter = getattr(sp, "table", None)
        if table_filter in (None, "*"):
            applicable.append(sp)
            continue
        try:
            safe_target = safe(str(table_filter))  # type: ignore[arg-type]
        except Exception:
            safe_target = str(table_filter)
        if tail and tail == safe_target:
            applicable.append(sp)

    if not applicable:
        return

    batch_start = time.perf_counter()

    def _run_embed(sp: EmbeddingSpec) -> str:
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   -> ensure_vector_column target={sp.target_column} source={sp.source_column}",
                )
            except Exception:
                pass
        t0 = time.perf_counter()
        ensure_vector_column(
            table_ctx,
            embed_column=sp.target_column,
            source_column=sp.source_column,
            from_ids=list(inserted_ids or []),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   <- done target={sp.target_column} time_ms={dt:.1f}",
                )
            except Exception:
                pass
        return sp.target_column

    max_workers = min(8, max(1, len(applicable)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_embed, sp): sp for sp in applicable}
        for fut in as_completed(futures):
            fut.result()
    if enable_progress:
        try:
            print(
                f"[EmbedOps] Table chunk complete: specs={len(applicable)} total_time_ms={(time.perf_counter()-batch_start)*1000.0:.1f}",
            )
        except Exception:
            pass


def embed_chunk_with_hooks(
    self,
    *,
    target_ctx_name: str,
    specs: List[EmbeddingSpec],
    inserted_ids: List[int],
    file_path: str,
    result: Dict[str, Any],
    document: Any,
    config: _FilePipelineConfig,
    enable_progress: bool,
    chunk_index: Optional[int] = None,
    chunk_type: str = "content",
    table_ctx: Optional[str] = None,
) -> None:
    """
    Helper to run pre-embed hooks, embedding, and post-embed hooks for a chunk.

    This function wraps the embedding logic with hooks and is designed to be
    submitted to a thread pool for asynchronous execution.

    Parameters
    ----------
    self
        FileManager instance (passed as first parameter for compatibility).
    target_ctx_name : str
        Context name for content embedding (used when chunk_type="content").
    specs : list[EmbeddingSpec]
        Embedding specifications to apply.
    inserted_ids : list[int]
        List of inserted log IDs to embed.
    file_path : str
        File path identifier.
    result : dict
        Parse result dictionary.
    document : Any
        Parsed document object.
    config : FilePipelineConfig
        Pipeline configuration.
    enable_progress : bool
        Whether to enable progress logging.
    chunk_index : int | None
        Optional chunk index for progress logging.
    chunk_type : str
        Either "content" or "table".
    table_ctx : str | None
        Table context name (required when chunk_type="table").
    """
    # Pre-embed hooks (per chunk when enabled)
    if bool(getattr(getattr(config, "embed", None), "hooks_per_chunk", True)):
        try:
            for fn in _resolve_callables(config.plugins.pre_embed):
                try:
                    fn(
                        manager=self,
                        filename=file_path,
                        result=result,
                        document=document,
                        config=config,
                    )
                except Exception:
                    continue
        except Exception:
            pass

    # Embed the chunk (scoped to inserted ids)
    try:
        if chunk_type == "content":
            embed_content_chunk_for_ids(
                target_ctx_name,
                specs,
                inserted_ids,
                enable_progress=enable_progress,
            )
            if enable_progress and chunk_index is not None:
                print(
                    f"[Along] Embedded content chunk {chunk_index} (ids={len(inserted_ids)})",
                )
        elif chunk_type == "table":
            if table_ctx is None:
                raise ValueError("table_ctx is required when chunk_type='table'")
            embed_table_chunk_for_ids(
                self,
                table_ctx=table_ctx,
                specs=specs,
                inserted_ids=inserted_ids,
                enable_progress=enable_progress,
            )
    except Exception:
        pass

    # Post-embed hooks (per chunk when enabled)
    if bool(getattr(getattr(config, "embed", None), "hooks_per_chunk", True)):
        try:
            for fn in _resolve_callables(config.plugins.post_embed):
                try:
                    fn(
                        manager=self,
                        filename=file_path,
                        result=result,
                        document=document,
                        config=config,
                    )
                except Exception:
                    continue
        except Exception:
            pass


def embed_content_chunks_async(
    self,
    *,
    file_path: str,
    records: List[Dict[str, Any]],
    target_ctx_name: str,
    content_specs: List[EmbeddingSpec],
    result: Dict[str, Any],
    document: Any,
    config: _FilePipelineConfig,
    batch_size: int,
    enable_progress: bool,
) -> None:
    """
    Ingest content rows in chunks and embed them asynchronously.

    Ingestion happens sequentially (chunk N+1 starts only after chunk N finishes),
    but embedding jobs run asynchronously without blocking subsequent ingestion.
    All embedding jobs are tracked and awaited at the end.

    Parameters
    ----------
    self
        FileManager instance.
    file_path : str
        File path identifier.
    records : list[dict]
        Parsed content records to ingest.
    target_ctx_name : str
        Target context name for content embedding.
    content_specs : list[EmbeddingSpec]
        Embedding specifications for content.
    result : dict
        Parse result dictionary.
    document : Any
        Parsed document object.
    config : FilePipelineConfig
        Pipeline configuration.
    batch_size : int
        Batch size for ingestion.
    enable_progress : bool
        Whether to enable progress logging.
    """
    total_records = int(
        (result or {}).get("total_records")
        or len(list(result.get("records", []) or [])),
    )
    processed_records = 0
    chunk_index = 0

    # Track embedding futures for async execution
    embedding_futures: List[Future] = []

    # Use thread pool for embedding jobs
    max_workers = min(8, max(1, len(content_specs) if content_specs else 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Ingest chunks sequentially, submit embedding jobs asynchronously
        for inserted_ids in iter_ingest_content_rows(
            self,
            file_path=file_path,
            records=records,
            config=config,
            batch_size=batch_size,
            replace_existing=bool(
                getattr(getattr(config, "ingest", None), "replace_existing", True),
            ),
        ):
            if not inserted_ids:
                continue
            chunk_index += 1
            processed_records += len(inserted_ids)
            if enable_progress:
                print(
                    f"[Along] Content chunk {chunk_index}: inserted={len(inserted_ids)} processed={processed_records}/{total_records}",
                )

            # Submit embedding job asynchronously (non-blocking)
            future = executor.submit(
                embed_chunk_with_hooks,
                self,
                target_ctx_name=target_ctx_name,
                specs=content_specs,
                inserted_ids=inserted_ids,
                file_path=file_path,
                result=result,
                document=document,
                config=config,
                enable_progress=enable_progress,
                chunk_index=chunk_index,
                chunk_type="content",
            )
            embedding_futures.append(future)

        # Wait for all content embedding jobs to complete
        for future in embedding_futures:
            try:
                future.result()
            except Exception:
                pass


def embed_table_chunks_async(
    self,
    *,
    file_path: str,
    document: Any,
    target_ctx_name: str,
    table_specs: List[EmbeddingSpec],
    result: Dict[str, Any],
    config: _FilePipelineConfig,
    table_rows_batch_size: int,
    enable_progress: bool,
) -> None:
    """
    Ingest table rows in chunks and embed them asynchronously.

    Ingestion happens sequentially (chunk N+1 starts only after chunk N finishes),
    but embedding jobs run asynchronously without blocking subsequent ingestion.
    All embedding jobs are tracked and awaited at the end.

    Parameters
    ----------
    self
        FileManager instance.
    file_path : str
        File path identifier.
    document : Any
        Parsed document object with tables.
    target_ctx_name : str
        Target context name (not used for tables but kept for consistency).
    table_specs : list[EmbeddingSpec]
        Embedding specifications for tables.
    result : dict
        Parse result dictionary.
    config : FilePipelineConfig
        Pipeline configuration.
    table_rows_batch_size : int
        Batch size for table row ingestion.
    enable_progress : bool
        Whether to enable progress logging.
    """
    # Derive per-table total row counts for progress reporting
    table_row_totals: Dict[str, int] = {}
    try:
        tables = getattr(getattr(document, "metadata", None), "tables", []) or []
        for t in tables:
            rows = getattr(t, "rows", None) or []
            # Account for header row removal when columns are inferred
            table_row_totals[
                str(
                    getattr(t, "sheet_name", "")
                    or getattr(t, "section_path", "")
                    or "",
                )
            ] = max(0, len(rows))
    except Exception:
        table_row_totals = {}
    table_progress: Dict[str, int] = {}

    # Track table embedding futures for async execution
    table_embedding_futures: List[Future] = []

    # Use thread pool for table embedding jobs
    table_max_workers = min(8, max(1, len(table_specs) if table_specs else 4))
    with ThreadPoolExecutor(max_workers=table_max_workers) as table_executor:
        # Ingest table chunks sequentially, submit embedding jobs asynchronously
        for table_ctx, inserted_ids in iter_ingest_tables_for_document(
            self,
            file_path=file_path,
            document=document,
            table_rows_batch_size=table_rows_batch_size,
        ):
            if enable_progress:
                label = table_ctx.split("/Tables/", 1)[-1]
                table_progress[label] = table_progress.get(label, 0) + len(
                    inserted_ids or [],
                )
                tot = table_row_totals.get(label, 0)
                if tot > 0:
                    print(
                        f"[Along] Table '{label}' chunk: inserted={len(inserted_ids or [])} processed={table_progress[label]}/{tot}",
                    )
                else:
                    print(
                        f"[Along] Table '{label}' chunk: inserted={len(inserted_ids or [])}",
                    )

            # Submit table embedding job asynchronously (non-blocking)
            future = table_executor.submit(
                embed_chunk_with_hooks,
                self,
                target_ctx_name=target_ctx_name,  # Not used for tables but required
                specs=table_specs,
                inserted_ids=inserted_ids,
                file_path=file_path,
                result=result,
                document=document,
                config=config,
                enable_progress=enable_progress,
                chunk_index=None,  # Table chunks don't use chunk_index
                chunk_type="table",
                table_ctx=table_ctx,
            )
            table_embedding_futures.append(future)

        # Wait for all table embedding jobs to complete
        for future in table_embedding_futures:
            try:
                future.result()
            except Exception:
                pass
