from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Iterator, Tuple, AsyncIterator
import asyncio
import threading

import time

import unify
from unity.file_manager.types.file import FileContent
from unity.file_manager.types.config import (
    FilePipelineConfig as _FilePipelineConfig,
    FileEmbeddingSpec,
    TableEmbeddingSpec,
    resolve_callables as _resolve_callables,
)
from unity.common.embed_utils import ensure_vector_column

try:
    from unity.file_manager.managers.progress_display import FileProgressManager
except ImportError:
    FileProgressManager = None  # type: ignore


def _safe(self):
    return getattr(self, "_safe") if hasattr(self, "_safe") else (lambda x: x)


def _run_async_from_sync(coro):
    """
    Run an async coroutine from a sync context, handling both cases:
    - If no event loop is running, use asyncio.run()
    - If an event loop is running, create a new loop in a thread and wait
    """
    try:
        # Try to get the running loop
        loop = asyncio.get_running_loop()
        # We're in an async context - run in a separate thread with new event loop
        result_container = {"value": None, "exception": None}
        event = threading.Event()

        def run_in_thread():
            try:
                # Create a new event loop in this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result_container["value"] = new_loop.run_until_complete(coro)
            except Exception as e:
                result_container["exception"] = e
            finally:
                event.set()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        event.wait()
        thread.join()

        if result_container["exception"]:
            raise result_container["exception"]
        return result_container["value"]
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)


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


# ----------------------------- Ingest/Embed helpers (chunked) ------------------- #


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
    file_format: Optional[str] = None,
) -> Iterator[List[int]]:
    """
    Incrementally ingest per-file Content rows in chunks; yields inserted ids per chunk.
    Performs delete-once when replace_existing=True.

    Parameters
    ----------
    file_format : str | None
        File format (e.g., 'xlsx', 'csv') for applying content ingest policy.
    """
    rows: List[Dict[str, Any]] = list(records or [])

    # Apply content ingest policy (same as in _ingest)
    if file_format:
        rows = apply_content_ingest_policy(
            rows,
            config=config,
            file_format=file_format,
        )

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
    # Use path as-is for querying
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

    enable_progress = bool(
        getattr(getattr(config, "diagnostics", None), "enable_progress", False),
    )

    total_rows = len(rows)
    total_chunks = (total_rows + batch_size - 1) // batch_size if total_rows > 0 else 0

    # Get progress manager from FileManager instance if available
    progress_manager = (
        getattr(self, "_progress_manager", None)
        if hasattr(self, "_progress_manager")
        else None
    )

    # Start content ingestion tracking
    if enable_progress and progress_manager and FileProgressManager:
        progress_manager.start_content_ingest(file_path, total_chunks)

    chunk_iter = range(0, len(rows), int(batch_size))
    chunk_num = 0

    for i in chunk_iter:
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
        inserted_ids = list(ids or [])

        chunk_num += 1
        # Update progress manager
        if enable_progress and progress_manager and FileProgressManager:
            progress_manager.update_content_ingest(file_path, chunk_num)

        yield inserted_ids


def iter_ingest_tables_for_document(
    self,
    *,
    file_path: str,
    document: Any,
    table_rows_batch_size: int = 100,
    config: Optional[_FilePipelineConfig] = None,
) -> Iterator[Tuple[str, List[int]]]:
    """
    Incrementally ingest per-file tables for a document; yields (table_ctx, inserted_ids) per batch.
    """
    # Get enable_progress from config (needed for logging)
    enable_progress = (
        bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        )
        if config
        else False
    )

    tables = getattr(getattr(document, "metadata", None), "tables", []) or []
    if not tables:
        if enable_progress:
            try:
                print(f"[IngestOps] No tables found in document for {file_path}")
            except Exception:
                pass
        return

    if enable_progress:
        try:
            print(
                f"[IngestOps] Found {len(tables)} table(s) in document for {file_path}",
            )
        except Exception:
            pass

    for idx, tbl in enumerate(tables, start=1):
        columns = getattr(tbl, "columns", None)
        rows = getattr(tbl, "rows", None)
        sheet_name = getattr(tbl, "sheet_name", None)
        if enable_progress:
            try:
                print(
                    f"[IngestOps] Processing table {idx}/{len(tables)}: sheet_name={sheet_name}, rows={len(rows) if rows else 0}",
                )
            except Exception:
                pass
        if not rows:
            if enable_progress:
                try:
                    print(
                        f"[IngestOps] Skipping table {idx} (sheet_name={sheet_name}): no rows",
                    )
                except Exception:
                    pass
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

        if enable_progress:
            try:
                print(
                    f"[IngestOps] Table label determined: '{table_label}' (sheet_name={sheet_name}, section_path={getattr(tbl, 'section_path', None)})",
                )
            except Exception:
                pass

        # Resolve business context from config
        # Find matching BusinessContextSpec by file_path, then find matching TableBusinessContextSpec by table
        business_context = None
        config_table_names = []
        if (
            config
            and hasattr(config, "ingest")
            and hasattr(config.ingest, "business_contexts")
        ):
            for bc in config.ingest.business_contexts:
                if bc.file_path == file_path:
                    # Found matching file, now find matching table spec
                    config_table_names = [ts.table for ts in bc.tables]
                    for table_spec in bc.tables:
                        if table_spec.table == table_label:
                            business_context = table_spec
                            if enable_progress:
                                try:
                                    print(
                                        f"[IngestOps] Matched business context for table '{table_label}'",
                                    )
                                except Exception:
                                    pass
                            break
                    if business_context:
                        break

        if enable_progress and not business_context and config_table_names:
            try:
                print(
                    f"[IngestOps] No business context match for table '{table_label}'. Config tables: {config_table_names}",
                )
            except Exception:
                pass

        # ensure table context
        ensure_per_file_table_context(
            self,
            file_path=file_path,
            table=table_label,
            columns=list(columns) if columns else None,
            example_row=(rows[0] if (rows and isinstance(rows[0], dict)) else None),
            business_context=business_context,
        )
        table_ctx = per_file_table_ctx(self, file_path=file_path, table=table_label)

        # batch insert
        total_rows_for_table = len(rows)
        total_chunks_for_table = (
            (total_rows_for_table + table_rows_batch_size - 1) // table_rows_batch_size
            if total_rows_for_table > 0
            else 0
        )

        # Get progress manager from FileManager instance if available
        progress_manager = (
            getattr(self, "_progress_manager", None)
            if hasattr(self, "_progress_manager")
            else None
        )

        # Register table with progress manager
        embed_strategy = (
            getattr(getattr(config, "embed", None), "strategy", "off")
            if config
            else "off"
        )
        embed_total = total_chunks_for_table if embed_strategy == "along" else 0

        if enable_progress and progress_manager and FileProgressManager:
            progress_manager.register_table(
                file_path,
                table_label,
                total_chunks_for_table,
                embed_total,
            )

        batch: List[Dict[str, Any]] = []
        chunk_num = 0
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
                inserted_ids = list(ids or [])
                chunk_num += 1

                # Update progress manager
                if enable_progress and progress_manager and FileProgressManager:
                    progress_manager.update_table_ingest(
                        file_path,
                        table_label,
                        chunk_num,
                    )

                yield table_ctx, inserted_ids
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
            inserted_ids = list(ids or [])
            chunk_num += 1

            # Update progress manager
            if enable_progress and progress_manager and FileProgressManager:
                progress_manager.update_table_ingest(file_path, table_label, chunk_num)

            yield table_ctx, inserted_ids

        # Complete table ingestion
        if enable_progress and progress_manager and FileProgressManager:
            progress_manager.complete_table_ingest(file_path, table_label)


def embed_content_chunk_for_ids(
    ctx_name: str,
    specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]],
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
    applicable_specs = [
        (file_spec, table_spec)
        for file_spec, table_spec in specs
        if file_spec.context in ("per_file", "unified")
    ]
    if not applicable_specs:
        return

    batch_start = time.perf_counter()

    def _run_embed_column_pair(source_col: str, target_col: str) -> str:
        """Embed a single column pair."""
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   -> ensure_vector_column target={target_col} source={source_col}",
                )
            except Exception:
                pass
        t0 = time.perf_counter()
        ensure_vector_column(
            ctx_name,
            embed_column=target_col,
            source_column=source_col,
            from_ids=list(inserted_ids or []),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   <- done target={target_col} time_ms={dt:.1f}",
                )
            except Exception:
                pass
        return target_col

    # Create tasks for all column pairs across all applicable specs
    embed_tasks: List[Tuple[str, str]] = []
    for file_spec, table_spec in applicable_specs:
        # Iterate over source_columns and target_columns pairs
        for source_col, target_col in zip(
            table_spec.source_columns,
            table_spec.target_columns,
        ):
            embed_tasks.append((source_col, target_col))

    if not embed_tasks:
        return

    async def _run_embeds_async() -> None:
        """Run embedding tasks using asyncio."""
        loop = asyncio.get_event_loop()
        max_workers = min(8, max(1, len(embed_tasks)))
        semaphore = asyncio.Semaphore(max_workers)

        async def run_embed_task(src: str, tgt: str) -> None:
            async with semaphore:
                await loop.run_in_executor(None, _run_embed_column_pair, src, tgt)

        tasks = [
            asyncio.create_task(run_embed_task(src, tgt)) for src, tgt in embed_tasks
        ]
        # Wait for all tasks, surfacing exceptions to fail fast
        for task in tasks:
            try:
                await task
            except Exception:
                pass

    _run_async_from_sync(_run_embeds_async())
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
    specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]],
    inserted_ids: List[int] | None = None,
    enable_progress: bool = False,
) -> None:
    """
    Apply embedding specs targeting a per-file-table context for a specific table chunk.
    """
    if not specs:
        return
    # extract tail label after /Tables/ (this is already sanitized from ingestion)
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
                f"[EmbedOps] Table chunk: ctx={table_ctx} tail={tail} specs={len(specs)} from_ids={len(inserted_ids or [])}",
            )
        except Exception:
            pass
    applicable: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]] = []
    for file_spec, table_spec in specs:
        if file_spec.context != "per_file_table":
            continue
        table_filter = table_spec.table
        if table_filter in (None, "*"):
            applicable.append((file_spec, table_spec))
            continue
        # Match: compare sanitized config table name with sanitized tail from context
        # The tail is already sanitized from ingestion, so we need to sanitize the config table name
        try:
            safe_target = safe(str(table_filter))  # type: ignore[arg-type]
        except Exception:
            safe_target = str(table_filter)
        # Also try direct comparison in case sanitization differs
        if tail and (tail == safe_target or tail == str(table_filter)):
            applicable.append((file_spec, table_spec))
            if enable_progress:
                try:
                    print(
                        f"[EmbedOps] Matched table spec: config_table={table_filter} safe_target={safe_target} tail={tail}",
                    )
                except Exception:
                    pass

    if not applicable:
        if enable_progress:
            try:
                print(
                    f"[EmbedOps] No matching specs for table_ctx={table_ctx} tail={tail}. Available specs: {[s.table for _, s in specs if _.context == 'per_file_table']}",
                )
            except Exception:
                pass
        return

    batch_start = time.perf_counter()

    def _run_embed_column_pair(source_col: str, target_col: str) -> str:
        """Embed a single column pair."""
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   -> ensure_vector_column target={target_col} source={source_col}",
                )
            except Exception:
                pass
        t0 = time.perf_counter()
        ensure_vector_column(
            table_ctx,
            embed_column=target_col,
            source_column=source_col,
            from_ids=list(inserted_ids or []),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if enable_progress:
            try:
                print(
                    f"[EmbedOps]   <- done target={target_col} time_ms={dt:.1f}",
                )
            except Exception:
                pass
        return target_col

    # Create tasks for all column pairs across all applicable specs
    embed_tasks: List[Tuple[str, str]] = []
    for file_spec, table_spec in applicable:
        # Iterate over source_columns and target_columns pairs
        for source_col, target_col in zip(
            table_spec.source_columns,
            table_spec.target_columns,
        ):
            embed_tasks.append((source_col, target_col))

    if not embed_tasks:
        return

    async def _run_table_embeds_async() -> None:
        """Run table embedding tasks using asyncio."""
        loop = asyncio.get_event_loop()
        max_workers = min(8, max(1, len(embed_tasks)))
        semaphore = asyncio.Semaphore(max_workers)

        async def run_embed_task(src: str, tgt: str) -> None:
            async with semaphore:
                await loop.run_in_executor(None, _run_embed_column_pair, src, tgt)

        tasks = [
            asyncio.create_task(run_embed_task(src, tgt)) for src, tgt in embed_tasks
        ]
        # Wait for all tasks
        for task in tasks:
            try:
                await task
            except Exception:
                pass

    _run_async_from_sync(_run_table_embeds_async())
    if enable_progress:
        try:
            print(
                f"[EmbedOps] Table chunk complete: specs={len(applicable)} total_time_ms={(time.perf_counter()-batch_start)*1000.0:.1f}",
            )
        except Exception:
            pass


def embed_chunk_with_hooks(
    self,
    target_ctx_name: str,
    specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]],
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
    specs : list[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]]
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
                        file_path=file_path,
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
            if enable_progress:
                try:
                    print(
                        f"[EmbedOps] Starting content chunk embedding: chunk_index={chunk_index}, ids={len(inserted_ids)}, specs={len(specs)}",
                    )
                except Exception:
                    pass
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
            if enable_progress:
                try:
                    print(
                        f"[EmbedOps] Starting table chunk embedding: table_ctx={table_ctx}, ids={len(inserted_ids)}, specs={len(specs)}",
                    )
                except Exception:
                    pass
            embed_table_chunk_for_ids(
                self,
                table_ctx=table_ctx,
                specs=specs,
                inserted_ids=inserted_ids,
                enable_progress=enable_progress,
            )
            if enable_progress:
                try:
                    print(
                        f"[EmbedOps] Completed table chunk embedding: table_ctx={table_ctx}",
                    )
                except Exception:
                    pass
    except Exception as e:
        if enable_progress:
            try:
                print(
                    f"[EmbedOps] Error embedding chunk: chunk_type={chunk_type}, error={e}",
                )
                import traceback

                traceback.print_exc()
            except Exception:
                pass
        raise

    # Post-embed hooks (per chunk when enabled)
    if bool(getattr(getattr(config, "embed", None), "hooks_per_chunk", True)):
        try:
            for fn in _resolve_callables(config.plugins.post_embed):
                try:
                    fn(
                        manager=self,
                        file_path=file_path,
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
    content_specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]],
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
    content_specs : list[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]]
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

    # Calculate total chunks for diagnostics
    total_chunks = (
        (total_records + batch_size - 1) // batch_size if total_records > 0 else 0
    )

    # Get progress manager from FileManager instance if available
    progress_manager = (
        getattr(self, "_progress_manager", None)
        if hasattr(self, "_progress_manager")
        else None
    )

    # Start content embedding tracking
    if enable_progress and progress_manager and FileProgressManager:
        progress_manager.start_content_embed(file_path, total_chunks)

    async def _embed_chunks_async() -> None:
        """Internal async function to run embedding jobs."""
        loop = asyncio.get_event_loop()

        # Create semaphore to limit concurrent embedding jobs
        max_workers = min(8, max(1, len(content_specs) if content_specs else 4))
        semaphore = asyncio.Semaphore(max_workers)

        embedding_tasks: List[asyncio.Task] = []

        # Extract file_format from result (same as in _ingest)
        _fmt = result.get("file_format")
        file_format = getattr(_fmt, "value", _fmt)
        file_format_str = str(file_format).lower().strip() if file_format else None

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
            file_format=file_format_str,
        ):
            if not inserted_ids:
                continue
            nonlocal chunk_index, processed_records
            chunk_index += 1
            processed_records += len(inserted_ids)

            # Submit embedding job asynchronously (non-blocking)
            chunk_id = f"chunk_{chunk_index}"
            if enable_progress and progress_manager and FileProgressManager:
                progress_manager.update_content_embed_chunk(
                    file_path,
                    chunk_id,
                    "started",
                )

            async def embed_chunk_task(ids: List[int], idx: int, cid: str) -> None:
                async with semaphore:
                    try:
                        await loop.run_in_executor(
                            None,
                            embed_chunk_with_hooks,
                            self,
                            target_ctx_name,
                            content_specs,
                            ids,
                            file_path,
                            result,
                            document,
                            config,
                            enable_progress,  # Pass enable_progress so we can see what's happening
                            idx,
                            "content",
                        )
                    finally:
                        # Mark chunk as completed
                        if enable_progress and progress_manager and FileProgressManager:
                            progress_manager.update_content_embed_chunk(
                                file_path,
                                cid,
                                "completed",
                            )

            task = asyncio.create_task(
                embed_chunk_task(inserted_ids, chunk_index, chunk_id),
            )
            embedding_tasks.append(task)

            # Yield control to event loop so tasks can start running immediately
            # This is critical: without this, tasks won't start until all ingestion completes
            await asyncio.sleep(0)

        # Wait for all content embedding jobs to complete
        for task in embedding_tasks:
            await task

        # Complete content embedding
        if enable_progress and progress_manager and FileProgressManager:
            progress_manager.complete_content_embed(file_path)

    # Run async function from sync context
    _run_async_from_sync(_embed_chunks_async())


def embed_table_chunks_async(
    self,
    *,
    file_path: str,
    document: Any,
    target_ctx_name: str,
    table_specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]],
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
    table_specs : list[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]]
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
    table_chunk_counts: Dict[str, int] = {}  # Track chunk counts per table

    # Get progress manager from FileManager instance if available
    progress_manager = (
        getattr(self, "_progress_manager", None)
        if hasattr(self, "_progress_manager")
        else None
    )

    async def _embed_table_chunks_async() -> None:
        """Internal async function to run table embedding jobs."""
        loop = asyncio.get_event_loop()

        if enable_progress:
            try:
                print(
                    f"[EmbedOps] Starting table embedding for {file_path} with {len(table_specs)} table spec(s)",
                )
                for file_spec, table_spec in table_specs:
                    print(
                        f"[EmbedOps]   Spec: file_path={file_spec.file_path}, context={file_spec.context}, table={table_spec.table}",
                    )
            except Exception:
                pass

        # Create semaphore to limit concurrent table embedding jobs
        table_max_workers = min(8, max(1, len(table_specs) if table_specs else 4))
        semaphore = asyncio.Semaphore(table_max_workers)

        table_embedding_tasks: List[asyncio.Task] = []
        table_chunk_info: Dict[str, List[int]] = {}  # Track chunks per table
        table_chunk_indices: Dict[str, int] = {}  # Track chunk index per table

        # Note: Table embedding tasks are already created by register_table during ingestion
        # No need to call start_table_embed here as it's a no-op anyway

        # Ingest table chunks sequentially, submit embedding jobs asynchronously
        for table_ctx, inserted_ids in iter_ingest_tables_for_document(
            self,
            file_path=file_path,
            document=document,
            table_rows_batch_size=table_rows_batch_size,
            config=config,
        ):
            if not inserted_ids:
                continue

            label = table_ctx.split("/Tables/", 1)[-1]
            if label not in table_chunk_info:
                table_chunk_info[label] = []
                table_chunk_indices[label] = 0

            table_chunk_indices[label] += 1
            chunk_id = f"{label}_chunk_{table_chunk_indices[label]}"
            table_chunk_info[label].append(len(inserted_ids))
            table_progress[label] = table_progress.get(label, 0) + len(inserted_ids)

            # Mark embedding chunk as started
            if enable_progress and progress_manager and FileProgressManager:
                progress_manager.update_table_embed_chunk(
                    file_path,
                    label,
                    chunk_id,
                    "started",
                )

            # Submit table embedding job asynchronously (non-blocking)
            async def embed_table_chunk_task(
                ctx: str,
                ids: List[int],
                tbl_label: str,
                cid: str,
            ) -> None:
                async with semaphore:
                    try:
                        await loop.run_in_executor(
                            None,
                            embed_chunk_with_hooks,
                            self,
                            target_ctx_name,  # Not used for tables but required
                            table_specs,
                            ids,
                            file_path,
                            result,
                            document,
                            config,
                            enable_progress,  # Pass enable_progress so we can see what's happening
                            None,  # Table chunks don't use chunk_index
                            "table",
                            ctx,
                        )
                    finally:
                        # Mark chunk as completed
                        if enable_progress and progress_manager and FileProgressManager:
                            progress_manager.update_table_embed_chunk(
                                file_path,
                                tbl_label,
                                cid,
                                "completed",
                            )

            task = asyncio.create_task(
                embed_table_chunk_task(table_ctx, inserted_ids, label, chunk_id),
            )
            table_embedding_tasks.append((task, label))

            # Yield control to event loop so tasks can start running immediately
            # This is critical: without this, tasks won't start until all ingestion completes
            await asyncio.sleep(0)

        # Wait for all table embedding jobs to complete
        for task, label in table_embedding_tasks:
            try:
                await task
            except Exception as e:
                if enable_progress and progress_manager and FileProgressManager:
                    progress_manager.fail_file(
                        file_path,
                        f"Table '{label}' embedding failed: {e}",
                    )

        # Complete table embedding for each table
        if enable_progress and progress_manager and FileProgressManager:
            for label in table_chunk_info.keys():
                progress_manager.complete_table_embed(file_path, label)

    # Run async function from sync context
    _run_async_from_sync(_embed_table_chunks_async())


def _process_single_file_core(
    self,
    *,
    idx: int,
    document: Any,
    original_path: str,
    config: _FilePipelineConfig,
    enable_progress: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Core logic for processing a single file: post-parse hooks, ingest, embed, and build result.

    This is the shared helper function used by all processing variants (parallel/sequential, sync/async).

    Parameters
    ----------
    self
        FileManager instance.
    idx : int
        Document index.
    document : Any
        Parsed document object.
    original_path : str
        Original file path.
    config : FilePipelineConfig
        Pipeline configuration.
    enable_progress : bool
        Whether to enable progress logging.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        (original_path, result_dict) tuple.
    """
    from unity.file_manager.parser.types.document import Document as _Doc
    from unity.file_manager.managers.file_ops import (
        build_compact_parse_model as _build_compact_parse_model,
    )

    try:
        # Post-parse hooks (document available)
        try:
            for fn in _resolve_callables(config.plugins.post_parse):
                try:
                    fn(
                        manager=self,
                        file_path=original_path,
                        result=None,
                        document=document,
                        config=config,
                    )
                except Exception as e:
                    if enable_progress:
                        print(
                            f"[FileManager] ⚠️  Post-parse hook failed for {original_path}: {e}",
                        )
        except Exception as e:
            if enable_progress:
                print(
                    f"[FileManager] ⚠️  Post-parse hooks failed for {original_path}: {e}",
                )

        result = document.to_parse_result(
            original_path,
            auto_counting=(
                config.ingest.auto_counting_per_file
                if config.ingest.id_layout == "columns"
                else None
            ),
            document_index=idx,
            id_layout=getattr(config.ingest, "id_layout", "map"),
            id_string_format=getattr(config.ingest, "id_string_format", None),
        )

        # Choose strategy per-file
        try:
            strategy = resolve_embed_strategy(document, result, config)
        except Exception:
            strategy = "after"

        if enable_progress:
            print(f"[FileManager] 📄 Processing {original_path} (strategy={strategy})")

        if strategy == "along":
            getattr(self, "_ingest_and_embed")(
                file_path=original_path,
                document=document,
                result=result,
                config=config,
            )
        else:
            inserted_ids = getattr(self, "_ingest")(
                file_path=original_path,
                document=document,
                result=result,
                config=config,
            )
            if strategy != "off":
                getattr(self, "_embed")(
                    file_path=original_path,
                    document=document,
                    result=result,
                    inserted_ids=inserted_ids,
                    config=config,
                )

        # Decide return mode
        mode = getattr(getattr(config, "output", None), "return_mode", "compact")
        if mode == "full":
            return (original_path, result)
        elif mode == "none":
            return (
                original_path,
                {
                    "file_path": original_path,
                    "status": result.get("status"),
                    "error": result.get("error"),
                    "total_records": result.get("total_records"),
                    "file_format": result.get("file_format"),
                },
            )
        else:  # compact
            return (
                original_path,
                _build_compact_parse_model(
                    self,
                    file_path=original_path,
                    document=document,
                    result=result,
                    config=config,
                ),
            )
    except Exception as e:
        if enable_progress:
            print(f"[FileManager] ❌ Error processing {original_path}: {e}")
        from unity.file_manager.parser.types.document import Document as _Doc

        return (
            original_path,
            _Doc.error_result(original_path, f"processing failed: {e}"),
        )
    finally:
        # Mark file complete in progress manager
        try:
            if enable_progress and hasattr(self, "_progress_manager"):
                pm = getattr(self, "_progress_manager", None)
                if pm is not None:
                    pm.complete_file(original_path)
        except Exception:
            pass


def process_files_parallel(
    self,
    *,
    documents: List[Any],
    exported_paths: List[str],
    exported_paths_to_original_paths: Dict[str, str],
    config: _FilePipelineConfig,
    enable_progress: bool,
) -> Dict[str, Any]:
    """
    Process multiple files in parallel when ingest_mode is per_file.

    Parameters
    ----------
    self
        FileManager instance.
    documents : list[Any]
        Parsed document objects.
    exported_paths : list[str]
        Exported file paths.
    exported_paths_to_original_paths : dict[str, str]
        Mapping of exported paths to original paths.
    config : FilePipelineConfig
        Pipeline configuration.
    enable_progress : bool
        Whether to enable progress logging.

    Returns
    -------
    dict[str, Any]
        Mapping of original_path -> result dict.
    """
    from unity.file_manager.parser.types.document import Document as _Doc

    results: Dict[str, Any] = {}

    def _process_single_file(
        idx: int,
        document: Any,
        exported_path: str,
        original_path: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Wrapper for thread pool execution."""
        return _process_single_file_core(
            self,
            idx=idx,
            document=document,
            original_path=original_path,
            config=config,
            enable_progress=enable_progress,
        )

    # Process files using asyncio
    async def _process_files_async() -> Dict[str, Any]:
        """Process files in parallel using asyncio."""
        loop = asyncio.get_event_loop()
        max_workers = min(len(documents), 8)  # Limit concurrent workers
        semaphore = asyncio.Semaphore(max_workers)

        async def process_file_task(
            idx: int,
            doc: Any,
            exp_path: str,
            orig_path: str,
        ) -> Tuple[int, str, Dict[str, Any]]:
            async with semaphore:
                result = await loop.run_in_executor(
                    None,
                    _process_single_file,
                    idx,
                    doc,
                    exp_path,
                    orig_path,
                )
                return (idx, orig_path, result[1])

        tasks = []
        for idx, document in enumerate(documents):
            if idx >= len(exported_paths):
                continue
            exported_path = exported_paths[idx]
            original_path = exported_paths_to_original_paths.get(
                exported_path,
                exported_path,
            )
            tasks.append(
                asyncio.create_task(
                    process_file_task(idx, document, exported_path, original_path),
                ),
            )

        # Collect results as they complete
        completed = 0
        if enable_progress:
            print(f"[Ops] 🔄 Processing {len(tasks)} file(s) in parallel")

        for coro in asyncio.as_completed(tasks):
            completed += 1
            try:
                idx, original_path, result_dict = await coro
                results[original_path] = result_dict
                if enable_progress:
                    print(
                        f"[Ops] ✅ Completed {completed}/{len(tasks)}: {original_path}",
                    )
            except Exception as e:
                # Try to get original_path from task if possible
                try:
                    # Get the task that raised the exception
                    for task in tasks:
                        if task.done() and task.exception() == e:
                            # Extract original_path from task args if possible
                            # Fallback to index-based lookup
                            break
                except Exception:
                    pass
                # Use index-based fallback
                try:
                    idx = next(i for i, t in enumerate(tasks) if t == coro)
                    original_path = exported_paths_to_original_paths.get(
                        exported_paths[idx] if idx < len(exported_paths) else "",
                        exported_paths[idx] if idx < len(exported_paths) else "",
                    )
                except Exception:
                    original_path = "unknown"
                results[original_path] = _Doc.error_result(
                    original_path,
                    f"processing failed: {e}",
                )
                if enable_progress:
                    print(
                        f"[FileManager] ❌ Failed {completed}/{len(tasks)}: {original_path}",
                    )

        return results

    return _run_async_from_sync(_process_files_async())


def process_files_sequential(
    self,
    *,
    documents: List[Any],
    exported_paths: List[str],
    exported_paths_to_original_paths: Dict[str, str],
    config: _FilePipelineConfig,
    enable_progress: bool,
) -> Dict[str, Any]:
    """
    Process multiple files sequentially (unified mode or single file).

    Parameters
    ----------
    self
        FileManager instance.
    documents : list[Any]
        Parsed document objects.
    exported_paths : list[str]
        Exported file paths.
    exported_paths_to_original_paths : dict[str, str]
        Mapping of exported paths to original paths.
    config : FilePipelineConfig
        Pipeline configuration.
    enable_progress : bool
        Whether to enable progress logging.

    Returns
    -------
    dict[str, Any]
        Mapping of original_path -> result dict.
    """
    results: Dict[str, Any] = {}

    if enable_progress:
        if len(documents) > 1:
            print(
                f"[Ops] 🔄 Processing {len(documents)} file(s) sequentially (unified mode)",
            )
        else:
            print(f"[Ops] 🔄 Processing 1 file sequentially")

    for idx, document in enumerate(documents):
        fp = exported_paths[idx] if idx < len(exported_paths) else None
        if fp is None:
            continue
        original_path = exported_paths_to_original_paths.get(fp, fp)

        if enable_progress:
            print(
                f"[Ops] 📄 Processing file {idx + 1}/{len(documents)}: {original_path}",
            )

        original_path, result_dict = _process_single_file_core(
            self,
            idx=idx,
            document=document,
            original_path=original_path,
            config=config,
            enable_progress=enable_progress,
        )
        results[original_path] = result_dict

        if enable_progress:
            print(f"[Ops] ✅ Completed {idx + 1}/{len(documents)}: {original_path}")

    if enable_progress:
        print(
            f"[Ops] ✅ Sequential processing complete: {len(results)} file(s) processed",
        )

    return results


async def _process_single_file_core_async(
    self,
    *,
    idx: int,
    document: Any,
    original_path: str,
    config: _FilePipelineConfig,
    enable_progress: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Async wrapper for core single-file processing logic.

    Uses asyncio.to_thread for CPU-bound operations (ingest/embed).

    Parameters
    ----------
    self
        FileManager instance.
    idx : int
        Document index.
    document : Any
        Parsed document object.
    original_path : str
        Original file path.
    config : FilePipelineConfig
        Pipeline configuration.
    enable_progress : bool
        Whether to enable progress logging.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        (original_path, result_dict) tuple.
    """
    import asyncio
    from unity.file_manager.parser.types.document import Document as _Doc
    from unity.file_manager.managers.file_ops import (
        build_compact_parse_model as _build_compact_parse_model,
    )

    try:
        # Post-parse hooks (document available)
        try:
            for fn in _resolve_callables(config.plugins.post_parse):
                try:
                    fn(
                        manager=self,
                        file_path=original_path,
                        result=None,
                        document=document,
                        config=config,
                    )
                except Exception as e:
                    if enable_progress:
                        print(
                            f"[FileManager] ⚠️  Post-parse hook failed for {original_path}: {e}",
                        )
        except Exception as e:
            if enable_progress:
                print(
                    f"[FileManager] ⚠️  Post-parse hooks failed for {original_path}: {e}",
                )

        result = document.to_parse_result(
            original_path,
            auto_counting=(
                config.ingest.auto_counting_per_file
                if config.ingest.id_layout == "columns"
                else None
            ),
            document_index=idx,
            id_layout=getattr(config.ingest, "id_layout", "map"),
            id_string_format=getattr(config.ingest, "id_string_format", None),
        )

        # Choose strategy per-file
        try:
            strategy = resolve_embed_strategy(document, result, config)
        except Exception:
            strategy = "after"

        if enable_progress:
            print(f"[FileManager] 📄 Processing {original_path} (strategy={strategy})")

        # Run ingest/embed in thread pool (CPU-bound operations)
        if strategy == "along":
            await asyncio.to_thread(
                getattr(self, "_ingest_and_embed"),
                file_path=original_path,
                document=document,
                result=result,
                config=config,
            )
        else:
            inserted_ids = await asyncio.to_thread(
                getattr(self, "_ingest"),
                file_path=original_path,
                document=document,
                result=result,
                config=config,
            )
            if strategy != "off":
                await asyncio.to_thread(
                    getattr(self, "_embed"),
                    file_path=original_path,
                    document=document,
                    result=result,
                    inserted_ids=inserted_ids,
                    config=config,
                )

        # Decide return mode
        mode = getattr(getattr(config, "output", None), "return_mode", "compact")
        if mode == "full":
            return (original_path, result)
        elif mode == "none":
            return (
                original_path,
                {
                    "file_path": original_path,
                    "status": result.get("status"),
                    "error": result.get("error"),
                    "total_records": result.get("total_records"),
                    "file_format": result.get("file_format"),
                },
            )
        else:  # compact
            _model = await asyncio.to_thread(
                _build_compact_parse_model,
                self,
                file_path=original_path,
                document=document,
                result=result,
                config=config,
            )
            # Ensure a stable dict payload with file_path for async tests/consumers
            if isinstance(_model, dict):
                _out = dict(_model)
                _out.setdefault("file_path", original_path)
                return (original_path, _out)
            else:
                try:
                    _as_dict = getattr(
                        _model,
                        "model_dump",
                        lambda: dict(_model),
                    )()
                except Exception:
                    _as_dict = {"value": _model}
                _as_dict.setdefault("file_path", original_path)
                return (original_path, _as_dict)
    except Exception as e:
        if enable_progress:
            print(f"[FileManager] ❌ Error processing {original_path}: {e}")
        return (
            original_path,
            _Doc.error_result(original_path, f"processing failed: {e}"),
        )
    finally:
        # Mark file complete in progress manager
        try:
            if enable_progress and hasattr(self, "_progress_manager"):
                pm = getattr(self, "_progress_manager", None)
                if pm is not None:
                    pm.complete_file(original_path)
        except Exception:
            pass


async def process_files_parallel_async(
    self,
    *,
    documents: List[Any],
    exported_paths: List[str],
    exported_paths_to_original_paths: Dict[str, str],
    config: _FilePipelineConfig,
    enable_progress: bool,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Process multiple files in parallel asynchronously when ingest_mode is per_file.

    Parameters
    ----------
    self
        FileManager instance.
    documents : list[Any]
        Parsed document objects.
    exported_paths : list[str]
        Exported file paths.
    exported_paths_to_original_paths : dict[str, str]
        Mapping of exported paths to original paths.
    config : FilePipelineConfig
        Pipeline configuration.
    enable_progress : bool
        Whether to enable progress logging.

    Returns
    -------
    list[Tuple[str, dict[str, Any]]]
        List of (original_path, result_dict) tuples in order.
    """
    import asyncio
    from unity.file_manager.parser.types.document import Document as _Doc

    tasks = []
    for idx, document in enumerate(documents):
        if idx >= len(exported_paths):
            continue
        exported_path = exported_paths[idx]
        original_path = exported_paths_to_original_paths.get(
            exported_path,
            exported_path,
        )
        tasks.append(
            _process_single_file_core_async(
                self,
                idx=idx,
                document=document,
                original_path=original_path,
                config=config,
                enable_progress=enable_progress,
            ),
        )

    # Execute all tasks concurrently
    completed_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Return results in order
    return [
        (
            result_data
            if not isinstance(result_data, Exception)
            else (
                exported_paths_to_original_paths.get(
                    exported_paths[idx] if idx < len(exported_paths) else "",
                    exported_paths[idx] if idx < len(exported_paths) else "",
                ),
                _Doc.error_result(
                    exported_paths_to_original_paths.get(
                        exported_paths[idx] if idx < len(exported_paths) else "",
                        exported_paths[idx] if idx < len(exported_paths) else "",
                    ),
                    f"processing failed: {result_data}",
                ),
            )
        )
        for idx, result_data in enumerate(completed_results)
    ]


async def process_files_sequential_async(
    self,
    *,
    documents: List[Any],
    exported_paths: List[str],
    exported_paths_to_original_paths: Dict[str, str],
    config: _FilePipelineConfig,
    enable_progress: bool,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Process multiple files sequentially asynchronously (unified mode or single file).

    Parameters
    ----------
    self
        FileManager instance.
    documents : list[Any]
        Parsed document objects.
    exported_paths : list[str]
        Exported file paths.
    exported_paths_to_original_paths : dict[str, str]
        Mapping of exported paths to original paths.
    config : FilePipelineConfig
        Pipeline configuration.
    enable_progress : bool
        Whether to enable progress logging.

    Yields
    ------
    dict[str, Any]
        Result dict per file.
    """
    if enable_progress and len(documents) > 1:
        print(
            f"[FileManager] Sequential processing for {len(documents)} files (unified mode)",
        )

    for idx, document in enumerate(documents):
        fp = exported_paths[idx] if idx < len(exported_paths) else None
        if fp is None:
            continue
        original_path = exported_paths_to_original_paths.get(fp, fp)
        original_path, result_dict = await _process_single_file_core_async(
            self,
            idx=idx,
            document=document,
            original_path=original_path,
            config=config,
            enable_progress=enable_progress,
        )
        yield result_dict
