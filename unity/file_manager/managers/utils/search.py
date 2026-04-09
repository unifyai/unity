from __future__ import annotations

from typing import Optional, TYPE_CHECKING

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
