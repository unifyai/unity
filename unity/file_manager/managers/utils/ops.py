from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING


from unity.file_manager.types.file import FileRecordRow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.types.file import FileContentRow


def _per_file_root(file_manager: "FileManager") -> str:
    base = None
    try:
        _base_attr = getattr(file_manager, "_per_file_root")
        if isinstance(_base_attr, str) and _base_attr:
            base = _base_attr
    except Exception:
        base = None
    if base is None:
        try:
            ctx = getattr(file_manager, "_ctx")
            if isinstance(ctx, str) and "/FileRecords/" in ctx:
                prefix, alias = ctx.split("/FileRecords/", 1)
                base = f"{prefix}/Files/{alias}"
        except Exception:
            base = None
    return base or "Files"


# ---------- FileRecords root (FileRecords/<alias>) helpers ---------------------


def add_or_replace_file_row(
    file_manager: "FileManager",
    *,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create or replace a file row in the FileRecords/<alias> context.

    Delegates Unify operations to DataManager.

    Returns a short outcome dict.
    """
    dm = file_manager._data_manager

    # Try to find an existing row by file_path; replace if found
    fp = entry.get("file_path")
    if fp:
        rows = dm.filter(
            context=file_manager._ctx,
            filter=f"file_path == {fp!r}",
            limit=2,
            columns=["file_id", "file_path"],
        )

        if len(rows) > 1:
            raise ValueError(
                f"Multiple index rows found for file_path {fp!r} (data integrity)",
            )
        if len(rows) == 1:
            # Update existing row via DataManager
            file_id = rows[0].get("file_id")
            dm.update_rows(
                context=file_manager._ctx,
                updates=entry,
                filter=f"file_id == {file_id}",
            )
            return {
                "outcome": "file updated successfully",
                "details": {
                    "file_id": file_id,
                    "file_path": fp,
                },
            }

    # Create new row via DataManager
    file_id_before = getattr(entry, "file_id", None)
    log_ids = dm.insert_rows(
        context=file_manager._ctx,
        rows=[entry],
        add_to_all_context=file_manager.include_in_multi_assistant_table,
    )
    file_id_after = getattr(entry, "file_id", None)
    logger.info(
        "[ops] insert_rows for %s: file_id before=%s after=%s, "
        "add_to_all=%s, log_ids=%s",
        fp,
        file_id_before,
        file_id_after,
        file_manager.include_in_multi_assistant_table,
        log_ids,
    )
    return {
        "outcome": "file created successfully",
        "details": {
            "file_id": getattr(entry, "file_id", None),
            "file_path": getattr(entry, "file_path", None),
        },
    }


def delete_file_contexts(
    file_manager: "FileManager",
    *,
    storage_id: str,
    file_id: Optional[int] = None,
    is_shared_storage: bool = False,
) -> Dict[str, Any]:
    """
    Delete all contexts and rows associated with a storage_id.

    Delegates Unify operations to DataManager.

    Parameters
    ----------
    storage_id : str
        The storage identifier for context paths.
    file_id : int | None
        The file_id for row-level deletion in shared contexts.
    is_shared_storage : bool
        If True, only deletes rows matching file_id from the shared Content context
        (does not drop the entire context). If False, drops the entire Content context.

    Behavior:
    - is_shared_storage=False: drop the Content context and all Tables contexts.
      Context paths: Files/<alias>/<storage_id>/Content, Files/<alias>/<storage_id>/Tables/*
    - is_shared_storage=True: delete only rows with matching file_id from the Content
      context (does not drop); drop this file's per-file Tables contexts if present.
    """
    dm = file_manager._data_manager
    base = _per_file_root(file_manager)

    purged = {"content": 0, "content_rows": 0, "tables": 0}

    # Get table_ingest setting from file record
    table_ingest = True
    if file_id is not None:
        try:
            rows = dm.filter(
                context=file_manager._ctx,
                filter=f"file_id == {file_id}",
                limit=1,
                columns=["table_ingest"],
            )
            if rows:
                table_ingest = bool(rows[0].get("table_ingest", True))
        except Exception:
            pass

    if not is_shared_storage:
        # Drop the entire Content context
        content_ctx = f"{base}/{storage_id}/Content"
        try:
            dm.delete_table(content_ctx, dangerous_ok=True)
            purged["content"] += 1
        except Exception:
            pass

        # Delete all Tables contexts for this storage_id
        if table_ingest:
            tables_prefix = f"{base}/{storage_id}/Tables/"
            try:
                all_tables = dm.list_tables(prefix=tables_prefix)
                for tctx in all_tables:
                    try:
                        dm.delete_table(tctx, dangerous_ok=True)
                        purged["tables"] += 1
                    except Exception:
                        pass
            except Exception:
                pass
    else:
        # Shared storage: delete only rows with matching file_id
        if file_id is not None:
            content_ctx = f"{base}/{storage_id}/Content"
            try:
                deleted = dm.delete_rows(
                    context=content_ctx,
                    filter=f"file_id == {file_id}",
                )
                purged["content_rows"] += deleted
            except Exception:
                pass

        # Delete per-file Tables contexts (shared storage mode still has per-file tables
        # under str(file_id) path, not storage_id)
        if table_ingest and file_id is not None:
            tables_prefix = f"{base}/{file_id}/Tables/"
            try:
                all_tables = dm.list_tables(prefix=tables_prefix)
                for tctx in all_tables:
                    try:
                        dm.delete_table(tctx, dangerous_ok=True)
                        purged["tables"] += 1
                    except Exception:
                        pass
            except Exception:
                pass

    return {"purged": purged}


# ----------------------- storage_id-based context wrappers -------------------- #


def ensure_file_context(
    file_manager: "FileManager",
    *,
    storage_id: str,
) -> None:
    """Ensure Content context exists for storage_id."""
    from .storage import ensure_file_context as _ensure

    _ensure(file_manager, storage_id=storage_id)


def ensure_file_table_context(
    file_manager: "FileManager",
    *,
    storage_id: str,
    table: str,
    unique_key: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    columns: Optional[List[str]] = None,
    example_row: Optional[Dict[str, Any]] = None,
    business_context: Optional[Any] = None,
) -> None:
    """Ensure table context exists for storage_id."""
    from .storage import ensure_file_table_context as _ensure_tbl

    _ensure_tbl(
        file_manager,
        storage_id=storage_id,
        table=table,
        unique_key=unique_key,
        business_context=business_context,
        auto_counting=auto_counting,
        columns=columns,
        example_row=example_row,
    )


# ------------------- storage_id-based row operations ------------------------- #


def delete_file_table_rows(
    file_manager: "FileManager",
    *,
    storage_id: str,
    table: str,
    filter_expr: Optional[str],
) -> int:
    """Delete rows from a table context using storage_id.

    Delegates to DataManager for Unify operations.
    """
    from .storage import ctx_for_file_table as _ctx_for_table

    dm = file_manager._data_manager
    ctx = _ctx_for_table(file_manager, storage_id=storage_id, table=table)

    if filter_expr is None:
        # Delete all rows via a filter that matches everything
        return dm.delete_rows(context=ctx, filter="1 == 1")
    else:
        return dm.delete_rows(context=ctx, filter=filter_expr)


def batch_insert_file_table_rows(
    file_manager: "FileManager",
    *,
    storage_id: str,
    table: str,
    rows: List[Dict[str, Any]],
) -> List[int]:
    """Insert rows into a table context using storage_id.

    Delegates to DataManager for Unify operations.

    Returns list of log IDs for the inserted rows.
    """
    if not rows:
        return []
    dm = file_manager._data_manager
    from .storage import ctx_for_file_table as _ctx_for_table

    ctx = _ctx_for_table(file_manager, storage_id=storage_id, table=table)
    return dm.insert_rows(
        context=ctx,
        rows=rows,
        add_to_all_context=file_manager.include_in_multi_assistant_table,
    )


def delete_file_content_rows(
    file_manager: "FileManager",
    *,
    storage_id: str,
    filter_expr: Optional[str],
) -> int:
    """Delete rows from a Content context using storage_id.

    Delegates to DataManager for Unify operations.
    """
    from .storage import ctx_for_file_content as _ctx_for_content

    dm = file_manager._data_manager
    ctx = _ctx_for_content(file_manager, storage_id=storage_id)

    if filter_expr is None:
        # Delete all rows
        return dm.delete_rows(context=ctx, filter="1 == 1")
    else:
        return dm.delete_rows(context=ctx, filter=filter_expr)


# ---------- High-level create helpers (ensure + insert) ------------------------


def create_file_record(
    file_manager: "FileManager",
    *,
    entry: FileRecordRow,
) -> Dict[str, Any]:
    """Create or update a FileRecord row in the global index (idempotent)."""
    return add_or_replace_file_row(
        file_manager,
        entry=entry.model_dump(mode="json", exclude_none=True),
    )


def create_file_content(
    file_manager: "FileManager",
    *,
    storage_id: str,
    rows: List["FileContentRow"],
) -> List[int]:
    """Ensure Content context then insert rows using storage_id."""
    from unity.file_manager.types.file import FileContentRow
    from .storage import ctx_for_file_content as _ctx_for_content

    ensure_file_context(
        file_manager,
        storage_id=storage_id,
    )
    entries: List[Dict[str, Any]] = [
        (
            r.model_dump(mode="json", exclude_none=True)
            if isinstance(r, FileContentRow)
            else dict(r)
        )  # type: ignore[arg-type]
        for r in list(rows or [])
    ]
    # Insert rows via DataManager
    dm = file_manager._data_manager
    ctx = _ctx_for_content(file_manager, storage_id=storage_id)
    if not entries:
        return []
    return dm.insert_rows(
        context=ctx,
        rows=entries,
        add_to_all_context=file_manager.include_in_multi_assistant_table,
    )


def create_file_table(
    file_manager: "FileManager",
    *,
    storage_id: str,
    table: str,
    rows: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    example_row: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Ensure table context then insert rows using storage_id."""
    ensure_file_table_context(
        file_manager,
        storage_id=storage_id,
        table=table,
        columns=columns,
        example_row=example_row,
    )
    return batch_insert_file_table_rows(
        file_manager,
        storage_id=storage_id,
        table=table,
        rows=rows,
    )


# ----------------------------- Mutator helpers ------------------------------- #


def rename_file(
    file_manager: "FileManager",
    *,
    file_id_or_path: Union[str, int],
    new_name: str,
) -> Dict[str, Any]:
    """Rename a file via adapter and update FileRecords index.

    Since context paths use stable file_id identifiers, no context rename is needed.
    Only the FileRecords row is updated with the new file_path, file_name, source_uri.

    Delegates Unify operations to DataManager.

    Parameters
    ----------
    file_id_or_path : str | int
        Either the file_id (int) as preserved in the FileRecords index, or the
        fully-qualified file_path (str) as stored in the FileRecords index/context.
        When a file_id is provided, it is resolved to the corresponding file_path.
    new_name : str
        New file name; adapter determines path semantics.
    """
    if not getattr(file_manager._adapter.capabilities, "can_rename", False):  # type: ignore[attr-defined]
        raise PermissionError("Rename not permitted by backend policy")

    dm = file_manager._data_manager
    file_id: Optional[int] = None

    # Resolve file_id_or_path to file_path and file_id using DataManager
    if isinstance(file_id_or_path, int):
        file_id = file_id_or_path
        rows = dm.filter(
            context=file_manager._ctx,
            filter=f"file_id == {file_id}",
            limit=1,
            columns=["file_id", "file_path", "source_uri"],
        )
        if not rows:
            raise ValueError(f"No file found with file_id {file_id}")
        file_path = rows[0].get("file_path")
        if not file_path:
            raise ValueError(f"File record with file_id {file_id} has no file_path")
        file_path = str(file_path)
    else:
        file_path = str(file_id_or_path)
        rows = dm.filter(
            context=file_manager._ctx,
            filter=f"file_path == {file_path!r}",
            limit=2,
            columns=["file_id", "file_path", "source_uri"],
        )
        if not rows:
            raise ValueError(f"File '{file_path}' not found in index.")
        if len(rows) > 1:
            raise ValueError(
                f"Multiple files found with file_path '{file_path}'. Data integrity issue.",
            )
        file_id = rows[0].get("file_id")

    new_name = str(new_name)
    old_file_path = rows[0].get("file_path", file_path)
    old_source_uri = rows[0].get("source_uri", "")

    # Perform filesystem rename via adapter
    ref = file_manager._adapter.rename(old_file_path, new_name)  # type: ignore[attr-defined]
    new_path = ref.path

    # Update FileRecords row with new path info (no context rename needed)
    # Contexts use stable file_id, so they remain unchanged
    sync_errors = []
    try:
        # Compute new source_uri by replacing the filename in the old source_uri
        new_source_uri = old_source_uri
        if old_source_uri:
            # Replace old filename with new filename in source_uri
            old_name = (
                old_file_path.rsplit("/", 1)[-1]
                if "/" in old_file_path
                else old_file_path
            )
            if old_name in old_source_uri:
                new_source_uri = old_source_uri.replace(old_name, new_name)

        dm.update_rows(
            context=file_manager._ctx,
            updates={
                "file_path": new_path,
                "file_name": new_name,
                "source_uri": new_source_uri,
            },
            filter=f"file_id == {file_id}",
        )
    except Exception as e:
        sync_errors.append(("index_update", str(e)))

    result = getattr(ref, "model_dump", lambda: {"path": ref.path, "name": new_name})()
    if sync_errors:
        result["_sync_warnings"] = sync_errors

    return result


def move_file(
    file_manager: "FileManager",
    *,
    file_id_or_path: Union[str, int],
    new_parent_path: str,
) -> Dict[str, Any]:
    """Move a file via adapter and update FileRecords index.

    Since context paths use stable file_id identifiers, no context rename is needed.
    Only the FileRecords row is updated with the new file_path, file_name (if changed),
    and source_uri.

    Delegates Unify operations to DataManager.

    Parameters
    ----------
    file_id_or_path : str | int
        Either the file_id (int) as preserved in the FileRecords index, or the
        fully-qualified file_path (str) as stored in the FileRecords index/context.
        When a file_id is provided, it is resolved to the corresponding file_path.
    new_parent_path : str
        Destination directory in adapter-native form.
    """
    if not getattr(file_manager._adapter.capabilities, "can_move", False):  # type: ignore[attr-defined]
        raise PermissionError("Move not permitted by backend policy")

    dm = file_manager._data_manager
    file_id: Optional[int] = None

    # Resolve file_id_or_path to file_path and file_id using DataManager
    if isinstance(file_id_or_path, int):
        file_id = file_id_or_path
        rows = dm.filter(
            context=file_manager._ctx,
            filter=f"file_id == {file_id}",
            limit=1,
            columns=["file_id", "file_path", "file_name", "source_uri"],
        )
        if not rows:
            raise ValueError(f"No file found with file_id {file_id}")
        file_path = rows[0].get("file_path")
        if not file_path:
            raise ValueError(f"File record with file_id {file_id} has no file_path")
        file_path = str(file_path)
    else:
        file_path = str(file_id_or_path)
        rows = dm.filter(
            context=file_manager._ctx,
            filter=f"file_path == {file_path!r}",
            limit=2,
            columns=["file_id", "file_path", "file_name", "source_uri"],
        )
        if not rows:
            raise ValueError(f"File '{file_path}' not found in index.")
        if len(rows) > 1:
            raise ValueError(
                f"Multiple files found with file_path '{file_path}'. Data integrity issue.",
            )
        file_id = rows[0].get("file_id")

    new_parent_path = str(new_parent_path)
    old_file_path = rows[0].get("file_path", file_path)
    old_file_name = rows[0].get("file_name", "")
    old_source_uri = rows[0].get("source_uri", "")

    # Perform filesystem move via adapter
    ref = file_manager._adapter.move(old_file_path, new_parent_path)  # type: ignore[attr-defined]
    new_path = ref.path

    # Extract new file name from the new path (move might change the name too)
    new_file_name = new_path.rsplit("/", 1)[-1] if "/" in new_path else new_path

    # Update FileRecords row with new path info (no context rename needed)
    # Contexts use stable file_id, so they remain unchanged
    sync_errors = []
    try:
        # Compute new source_uri by replacing the directory path
        new_source_uri = old_source_uri
        if old_source_uri and old_file_path:
            # Replace old path with new path in source_uri
            old_dir = old_file_path.rsplit("/", 1)[0] if "/" in old_file_path else ""
            new_dir = new_parent_path.rstrip("/")
            if old_dir and old_dir in old_source_uri:
                new_source_uri = old_source_uri.replace(old_dir, new_dir)

        updates: Dict[str, Any] = {
            "file_path": new_path,
            "source_uri": new_source_uri,
        }
        # Update file_name if it changed (move can also rename)
        if new_file_name != old_file_name:
            updates["file_name"] = new_file_name

        dm.update_rows(
            context=file_manager._ctx,
            updates=updates,
            filter=f"file_id == {file_id}",
        )
    except Exception as e:
        sync_errors.append(("index_update", str(e)))

    result = getattr(
        ref,
        "model_dump",
        lambda: {"path": ref.path, "parent": new_parent_path},
    )()
    if sync_errors:
        result["_sync_warnings"] = sync_errors

    return result


def delete_file(
    file_manager: "FileManager",
    *,
    file_id_or_path: Union[str, int],
) -> Dict[str, Any]:
    """Delete a file record and its contexts.

    Delegates Unify operations to DataManager.

    Parameters
    ----------
    file_id_or_path : str | int
        Either the file_id (int) as preserved in the FileRecords index, or the
        fully-qualified file_path (str) as stored in the FileRecords index/context.
        When a file_id is provided, it is resolved to the corresponding file_path.
    """
    dm = file_manager._data_manager

    # Resolve file_id_or_path to file_path, file_id, and storage_id using DataManager
    if isinstance(file_id_or_path, int):
        file_id = file_id_or_path
        rows = dm.filter(
            context=file_manager._ctx,
            filter=f"file_id == {file_id}",
            limit=1,
            columns=["file_id", "file_path", "storage_id"],
        )
        if not rows:
            raise ValueError(f"No file found with file_id {file_id}")
        file_path = rows[0].get("file_path", "")
        if not file_path:
            raise ValueError(f"File record with file_id {file_id} has no file_path")
        file_path = str(file_path)
        storage_id = rows[0].get("storage_id", "")
    else:
        file_path = str(file_id_or_path)
        rows = dm.filter(
            context=file_manager._ctx,
            filter=f"file_path == {file_path!r}",
            limit=2,
            columns=["file_id", "file_path", "storage_id"],
        )
        if not rows:
            raise ValueError(f"File '{file_path}' not found in index.")
        if len(rows) > 1:
            raise ValueError(
                f"Multiple files found with file_path '{file_path}'. Data integrity issue.",
            )
        file_id = rows[0].get("file_id")
        if file_id is None:
            raise ValueError(f"File record for '{file_path}' has no file_id")
        storage_id = rows[0].get("storage_id", "")

    # Compute effective storage_id (use file_id if empty)
    effective_storage_id = storage_id if storage_id else str(file_id)

    # Determine if this is shared storage (storage_id != str(file_id))
    is_shared_storage = storage_id and storage_id != str(file_id)

    # Check if file is protected
    if getattr(file_manager, "is_protected")(file_path):  # type: ignore[attr-defined]
        raise PermissionError(
            f"'{file_path}' is protected and cannot be deleted by FileManager.",
        )

    sync_errors = []

    # 1) Adapter deletion when supported (filesystem operation)
    if getattr(file_manager, "_adapter", None) is not None and getattr(
        file_manager._adapter.capabilities,
        "can_delete",
        False,
    ):
        try:
            file_manager._adapter.delete(file_path)  # type: ignore[attr-defined]
        except (NotImplementedError, FileNotFoundError):
            pass
        except Exception as e:
            sync_errors.append(("adapter_delete", str(e)))

    # 2) Delete contexts/rows via DataManager
    try:
        delete_file_contexts(
            file_manager,
            storage_id=effective_storage_id,
            file_id=file_id,
            is_shared_storage=is_shared_storage,
        )
    except Exception as e:
        sync_errors.append(("context_delete", str(e)))

    # 3) Delete FileRecords index row via DataManager
    try:
        dm.delete_rows(
            context=file_manager._ctx,
            filter=f"file_id == {file_id}",
        )
    except Exception as e:
        raise ValueError(f"Failed to delete file record: {e}")

    result: Dict[str, Any] = {
        "outcome": "file deleted",
        "details": {"file_id": file_id, "file_path": file_path},
    }
    if sync_errors:
        result["_sync_warnings"] = sync_errors

    return result
