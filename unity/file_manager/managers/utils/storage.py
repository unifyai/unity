from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

from ....common.context_store import TableStore
from ....common.model_to_fields import model_to_fields
from ...types.file import FileRecord as FileRow, FileContent as _PerFileContent
from ...types.config import TableBusinessContextSpec

if TYPE_CHECKING:
    from ..file_manager import FileManager
    from ...types.describe import ColumnInfo, FileStorageMap


def provision_storage(file_manager: "FileManager") -> None:
    """Ensure FileRecords/<alias> context, schema and local view exist (idempotent)."""
    # Create the TableStore if not already initialised by the manager
    if (
        not hasattr(file_manager, "_store")
        or getattr(file_manager, "_store", None) is None
    ):
        file_manager._store = TableStore(  # type: ignore[attr-defined]
            file_manager._ctx,  # type: ignore[attr-defined]
            unique_keys={"file_id": "int"},
            auto_counting={"file_id": None},
            description=(
                "FileRecords index for a single filesystem; per-file content lives under Files/<alias>/<safe(file_path)>/Tables/<table>."
            ),
            fields=model_to_fields(FileRow),
        )
    try:
        file_manager._store.ensure_context()  # type: ignore[attr-defined]
    except Exception:
        # Best-effort
        pass


def get_columns(
    file_manager: "FileManager",
    table: Optional[str] = None,
) -> Dict[str, str]:
    """Return {column_name: column_type} for index or any resolved context.

    Resolution rules when ``table`` is provided:
    - "FileRecords" → return index columns
    - Context paths from describe() (preferred):
      "<root>/Content", "<root>/Tables/<label>" (per-file table)
    - Legacy refs: "<storage_id>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
    - Fully-qualified context: use as-is
    """
    if not table:
        return file_manager._store.get_columns()  # type: ignore[attr-defined]
    t = str(table).strip()
    if not t:
        return {}
    if t.lower() == "filerecords":
        return file_manager._store.get_columns()  # type: ignore[attr-defined]
    resolved_ctx: Optional[str] = None
    # If the input already targets a context under this manager, use it directly.
    try:
        _index_ctx = getattr(file_manager, "_ctx")
    except Exception:
        _index_ctx = None
    try:
        _files_root = getattr(file_manager, "_per_file_root")
    except Exception:
        _files_root = None
    if (_index_ctx and t == _index_ctx) or (
        _files_root and t.startswith(f"{_files_root}/")
    ):
        resolved_ctx = t
    else:
        try:
            # Resolve table reference to full context path
            from .search import (
                resolve_table_ref as _res_ref,
            )  # local import to avoid cycles

            resolved_ctx = _res_ref(file_manager, t)
        except Exception:
            resolved_ctx = t
    try:
        # Use DataManager for field retrieval
        dm = file_manager._data_manager
        columns = dm.get_columns(str(resolved_ctx))
        return {k: v.get("data_type") for k, v in columns.items()}
    except Exception:
        # Best-effort: return empty mapping on failure
        return {}


# ----------------------- Context getters (strings only) ----------------------- #


def ctx_for_file_index(file_manager: "FileManager") -> str:
    """Return the fully-qualified context for FileRecords/<alias>."""
    return getattr(file_manager, "_ctx")


def ctx_for_file_content(file_manager: "FileManager", *, storage_id: str) -> str:
    """Return the fully-qualified Content context for a storage_id.

    Shape: <base>/Files/<alias>/<storage_id>/Content

    Parameters
    ----------
    storage_id : str
        The context path identifier. Can be str(file_id) for per-file storage
        or a custom label for shared storage across multiple files.

    Returns
    -------
    str
        Full Unify context path for document content.

    Notes
    -----
    Using storage_id provides:
    - Stable references that survive file renames
    - Shorter, LLM-friendly context paths
    - Flexible grouping (single file or shared context)
    """
    base = getattr(file_manager, "_per_file_root")
    return f"{base}/{storage_id}/Content"


def ctx_for_file_table(
    file_manager: "FileManager",
    *,
    storage_id: str,
    table: str,
) -> str:
    """Return the fully-qualified table context for a storage_id.

    Shape: <base>/Files/<alias>/<storage_id>/Tables/<safe(table)>

    Parameters
    ----------
    storage_id : str
        The context path identifier.
    table : str
        The logical table name (e.g., 'Sheet1', 'extracted_table_1').

    Returns
    -------
    str
        Full Unify context path for the table.
    """
    safe = (
        getattr(file_manager, "safe")
        if hasattr(file_manager, "safe")
        else (lambda x: x)
    )
    base = getattr(file_manager, "_per_file_root")
    return f"{base}/{storage_id}/Tables/{safe(table)}"


def ctx_tables_prefix(file_manager: "FileManager", *, storage_id: str) -> str:
    """Return the prefix for all table contexts under a storage_id.

    Shape: <base>/Files/<alias>/<storage_id>/Tables/

    Parameters
    ----------
    storage_id : str
        The context path identifier.

    Returns
    -------
    str
        Prefix path for listing/discovering table contexts.
    """
    base = getattr(file_manager, "_per_file_root")
    return f"{base}/{storage_id}/Tables/"


# ------------------------ Context provisioners (ensure) ---------------------- #


def ensure_file_context(
    file_manager: "FileManager",
    *,
    storage_id: str,
) -> None:
    """Ensure a Content context exists for a storage_id (idempotent).

    Parameters
    ----------
    storage_id : str
        The context path identifier (e.g., str(file_id) or a custom label).
    """
    ctx = ctx_for_file_content(file_manager, storage_id=storage_id)
    fields = model_to_fields(_PerFileContent)
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        fields=fields,
        description=f"Content context for storage_id={storage_id}",
    )
    try:
        store.ensure_context()
    except Exception:
        # Best-effort provisioning
        pass


def ensure_file_table_context(
    file_manager: "FileManager",
    *,
    storage_id: str,
    table: str,
    unique_key: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    columns: Optional[Union[List[str], Dict[str, Any]]] = None,
    example_row: Optional[Dict[str, Any]] = None,
    business_context: Optional[TableBusinessContextSpec] = None,
) -> None:
    """Ensure a Tables/<label> context exists for a storage_id.

    Parameters
    ----------
    storage_id : str
        The context path identifier.
    table : str
        The logical table name (e.g., 'Sheet1', 'extracted_table_1').
    unique_key : str
        Name of the unique key column (default: 'row_id').
    auto_counting : dict | None
        Auto-counting configuration for columns.
    columns : list[str] | dict[str, Any] | None
        Column specifications.
    example_row : dict | None
        Example row to infer columns from.
    business_context : TableBusinessContextSpec | None
        Business context metadata for the table.
    """
    ctx = ctx_for_file_table(file_manager, storage_id=storage_id, table=table)
    # Build fields from columns or example_row keys when provided
    fields_map: Dict[str, Any] = {}
    if columns is not None:
        if isinstance(columns, dict):
            fields_map = dict(columns)
        else:
            names = list(columns)
            fields_map = {str(name): {"type": "Any", "mutable": True} for name in names}
    elif example_row:
        names = [str(k) for k in example_row.keys()]
        fields_map = {name: {"type": "Any", "mutable": True} for name in names}

    # Apply business context: merge column descriptions into fields_map
    if business_context:
        column_descriptions = business_context.column_descriptions or {}
        for field_name in fields_map:
            if field_name in column_descriptions:
                desc = column_descriptions[field_name]
                if not isinstance(fields_map[field_name], dict):
                    fields_map[field_name] = {"type": "Any", "mutable": True}
                elif "type" not in fields_map[field_name]:
                    fields_map[field_name]["type"] = "Any"
                fields_map[field_name]["description"] = desc

    # Determine context description
    context_description = None
    if business_context and business_context.table_description:
        context_description = business_context.table_description

    store = TableStore(
        ctx,
        unique_keys={unique_key: "int"},
        auto_counting={unique_key: None, **(auto_counting or {})},
        fields=fields_map,
        description=context_description or "",
    )
    try:
        store.ensure_context()
    except Exception:
        # Best-effort
        pass


def resolve_storage_id(
    file_manager: "FileManager",
    *,
    file_path: Optional[str] = None,
    file_id: Optional[int] = None,
) -> Optional[str]:
    """
    Resolve the storage_id for a file from FileRecords via DataManager.

    Parameters
    ----------
    file_path : str, optional
        The file path to look up.
    file_id : int, optional
        The file_id to look up.

    Returns
    -------
    str | None
        The storage_id, or None if the file is not found.
        If the stored storage_id is empty, returns str(file_id).
    """
    if file_id is None and file_path is None:
        return None

    dm = file_manager._data_manager

    try:
        if file_id is not None:
            rows = dm.filter(
                context=file_manager._ctx,
                filter=f"file_id == {file_id}",
                limit=1,
                columns=["file_id", "storage_id"],
            )
        else:
            rows = dm.filter(
                context=file_manager._ctx,
                filter=f"file_path == {file_path!r}",
                limit=1,
                columns=["file_id", "storage_id"],
            )
    except Exception:
        return None

    if not rows:
        return None

    entry = rows[0]
    stored_id = entry.get("storage_id", "")
    resolved_file_id = entry.get("file_id")

    # If storage_id is empty, use str(file_id)
    if not stored_id and resolved_file_id is not None:
        return str(resolved_file_id)
    return stored_id or None


# ----------------------- describe() implementation ----------------------- #


def describe_file(
    file_manager: "FileManager",
    *,
    file_path: Optional[str] = None,
    file_id: Optional[int] = None,
) -> "FileStorageMap":
    """
    Return a complete storage representation of a file in the Unify backend.

    This is the primary discovery tool for understanding a file's status and
    storage. It returns existence/status info, all context paths, schemas,
    and identifiers needed for accurate filter/search/reduce operations.

    Parameters
    ----------
    file_path : str, optional
        The filesystem path of the file. Either file_path or file_id must be provided.
    file_id : int, optional
        The stable unique identifier from FileRecords. Either file_path or file_id
        must be provided.

    Returns
    -------
    FileStorageMap
        Complete status and storage representation including:
        - Status: filesystem_exists, indexed_exists, parsed_status
        - Identity: file_id, file_path, storage_id, source_uri, source_provider
        - Config: table_ingest, file_format
        - Storage: document (/Content), tables (/Tables/<name>), index_context

    Raises
    ------
    ValueError
        If neither file_path nor file_id is provided.

    Examples
    --------
    >>> # Describe by file path - works even if not indexed
    >>> storage = file_manager.describe(file_path="/reports/Q4.csv")
    >>> if not storage.indexed_exists:
    ...     print("File exists but not indexed yet")
    ...     file_manager.ingest_files("/reports/Q4.csv")
    >>> elif storage.parsed_status != "success":
    ...     print(f"Parsing failed: {storage.parsed_status}")
    >>> else:
    ...     print(f"Tables: {storage.table_names}")

    >>> # Use the context path for queries
    >>> if storage.has_tables:
    ...     results = data_manager.filter(
    ...         context=storage.tables[0].context_path,
    ...         filter="revenue > 1000000"
    ...     )

    >>> # Describe by file_id (faster, no path resolution needed)
    >>> storage = file_manager.describe(file_id=42)

    Notes
    -----
    - describe() never raises for missing files; check indexed_exists instead.
    - filesystem_exists is checked via the adapter if file_path is provided.
    - Storage info (document/tables) is only populated when parsed_status='success'.
    - Context paths use storage_id for stability across renames.
    """
    from ...types.describe import (
        FileStorageMap,
        DocumentInfo,
        TableInfo,
        ContextSchema,
    )

    if file_path is None and file_id is None:
        raise ValueError("Either file_path or file_id must be provided")

    dm = file_manager._data_manager

    # Initialize all status fields
    filesystem_exists: bool = False
    indexed_exists: bool = False
    parsed_status: Optional[str] = None
    storage_id: str = ""
    table_ingest: bool = True
    file_format: Optional[str] = None

    resolved_file_id: Optional[int] = file_id
    resolved_file_path: Optional[str] = file_path
    source_uri: Optional[str] = None
    source_provider: Optional[str] = None

    # Fields to fetch from index
    index_fields = [
        "file_id",
        "file_path",
        "source_uri",
        "source_provider",
        "status",
        "storage_id",
        "table_ingest",
        "file_format",
    ]

    # Helper to extract entry from rows returned by DataManager
    def _process_index_entry(rows: List[Dict[str, Any]]) -> bool:
        """Process index rows and update local vars. Returns True if found."""
        nonlocal indexed_exists, resolved_file_id, resolved_file_path
        nonlocal source_uri, source_provider, parsed_status
        nonlocal storage_id, table_ingest, file_format

        if not rows:
            return False

        entry = rows[0]
        indexed_exists = True
        resolved_file_id = entry.get("file_id", file_id)
        resolved_file_path = entry.get("file_path", file_path)
        source_uri = entry.get("source_uri")
        source_provider = entry.get("source_provider")
        parsed_status = entry.get("status")
        storage_id = entry.get("storage_id", "")
        table_ingest = bool(entry.get("table_ingest", True))
        file_format = entry.get("file_format")
        return True

    # Resolve from file_id first (most direct)
    if file_id is not None:
        try:
            rows = dm.filter(
                context=file_manager._ctx,
                filter=f"file_id == {file_id}",
                limit=1,
                columns=index_fields,
            )
            _process_index_entry(rows)
        except Exception as e:
            logger.warning(f"Failed to lookup file_id={file_id}: {e}")

    # If not found by file_id, try file_path
    if not indexed_exists and file_path is not None:
        # Try by source_uri first (more reliable)
        resolve_to_uri = getattr(file_manager, "_resolve_to_uri", None)
        if resolve_to_uri:
            try:
                source_uri = resolve_to_uri(file_path)
            except Exception:
                pass

        # Try by source_uri if we have one
        if source_uri:
            try:
                rows = dm.filter(
                    context=file_manager._ctx,
                    filter=f"source_uri == {source_uri!r}",
                    limit=1,
                    columns=index_fields,
                )
                _process_index_entry(rows)
            except Exception:
                pass

        # Fallback to file_path match
        if not indexed_exists:
            try:
                rows = dm.filter(
                    context=file_manager._ctx,
                    filter=f"file_path == {file_path!r}",
                    limit=1,
                    columns=index_fields,
                )
                _process_index_entry(rows)
            except Exception as e:
                logger.warning(f"Failed to lookup file_path={file_path!r}: {e}")

    # Check filesystem existence (only for path-based lookups)
    if resolved_file_path:
        try:
            exists_fn = getattr(file_manager, "exists", None)
            if exists_fn:
                filesystem_exists = bool(exists_fn(resolved_file_path))
        except Exception:
            pass

    # Get adapter provider if not already set
    if not source_provider:
        try:
            adapter = getattr(file_manager, "_adapter", None)
            source_provider = getattr(adapter, "name", None) or getattr(
                file_manager,
                "_fs_type",
                None,
            )
        except Exception:
            pass

    # Compute effective storage_id (use file_id if empty)
    effective_storage_id = storage_id
    if not effective_storage_id and resolved_file_id is not None:
        effective_storage_id = str(resolved_file_id)

    # If not indexed, return early with status info only
    if not indexed_exists:
        return FileStorageMap(
            filesystem_exists=filesystem_exists,
            indexed_exists=False,
            parsed_status=None,
            storage_id="",
            table_ingest=True,
            file_format=None,
            file_id=None,
            file_path=resolved_file_path or file_path or "",
            source_uri=source_uri,
            source_provider=source_provider,
            document=None,
            tables=[],
            index_context=file_manager._ctx,
            has_document=False,
            has_tables=False,
        )

    # Build context paths using storage_id
    base = getattr(file_manager, "_per_file_root")
    content_ctx = f"{base}/{effective_storage_id}/Content"
    tables_prefix = f"{base}/{effective_storage_id}/Tables/"

    # Only fetch storage info if parsing was successful
    document_info: Optional[DocumentInfo] = None
    table_infos: List[TableInfo] = []

    if parsed_status == "success":
        # Check if document context exists and get its schema
        try:
            content_columns = dm.get_columns(content_ctx)
            if content_columns:
                columns = _fields_to_column_info(content_columns)
                # Try to get context description
                content_description: Optional[str] = None
                try:
                    content_meta = dm.get_table(content_ctx)
                    if isinstance(content_meta, dict):
                        content_description = content_meta.get("description")
                except Exception:
                    pass
                document_info = DocumentInfo(
                    context_path=content_ctx,
                    description=content_description,
                    column_schema=ContextSchema(columns=columns),
                    row_count=None,  # Fetch on-demand with reduce()
                )
        except Exception:
            # Content context doesn't exist
            pass

        # Discover table contexts via DataManager
        try:
            # List all contexts under the tables prefix
            all_contexts = dm.list_tables(
                prefix=tables_prefix,
                include_column_info=True,
            )
            # all_contexts is a dict[str, Any] when include_column_info=True
            if isinstance(all_contexts, dict):
                for ctx_path, ctx_info in all_contexts.items():
                    if ctx_path == tables_prefix:
                        continue  # Skip the prefix itself
                    # Extract table name from path
                    table_name = ctx_path.replace(tables_prefix, "").split("/")[0]
                    if not table_name:
                        continue

                    # Extract table description from context info
                    table_description: Optional[str] = None
                    if isinstance(ctx_info, dict):
                        table_description = ctx_info.get("description")

                    # Get table schema
                    try:
                        table_columns = dm.get_columns(ctx_path)
                        columns = _fields_to_column_info(table_columns)
                        table_infos.append(
                            TableInfo(
                                name=table_name,
                                context_path=ctx_path,
                                column_schema=ContextSchema(columns=columns),
                                row_count=None,  # Fetch on-demand with reduce()
                                description=table_description,
                            ),
                        )
                    except Exception:
                        # Include table even without schema
                        table_infos.append(
                            TableInfo(
                                name=table_name,
                                context_path=ctx_path,
                                column_schema=ContextSchema(columns=[]),
                                row_count=None,
                                description=table_description,
                            ),
                        )
        except Exception as e:
            logger.warning(
                f"Failed to discover tables for storage_id={effective_storage_id}: {e}",
            )

    return FileStorageMap(
        # Status fields
        filesystem_exists=filesystem_exists,
        indexed_exists=indexed_exists,
        parsed_status=parsed_status,
        # Storage config
        storage_id=effective_storage_id,
        table_ingest=table_ingest,
        file_format=file_format,
        # Identity
        file_id=resolved_file_id,
        file_path=resolved_file_path or "",
        source_uri=source_uri,
        source_provider=source_provider,
        # Storage
        document=document_info,
        tables=table_infos,
        index_context=file_manager._ctx,
        has_document=document_info is not None,
        has_tables=len(table_infos) > 0,
    )


def _fields_to_column_info(fields: Dict[str, Any]) -> List["ColumnInfo"]:
    """Convert Unify fields dict to list of ColumnInfo."""
    from ...types.describe import ColumnInfo

    columns: List[ColumnInfo] = []
    embedding_columns: Dict[str, str] = {}

    # First pass: identify embedding columns
    for fname, finfo in fields.items():
        if fname.startswith("_") and fname.endswith("_emb"):
            # This is an embedding column for the source column
            source_col = fname[1:-4]  # Strip leading _ and trailing _emb
            embedding_columns[source_col] = fname

    # Second pass: build column info
    for fname, finfo in fields.items():
        if fname.startswith("_"):
            continue  # Skip internal columns

        data_type = "unknown"
        description = None

        if isinstance(finfo, dict):
            data_type = finfo.get("data_type", finfo.get("type", "unknown"))
            description = finfo.get("description")

        is_searchable = fname in embedding_columns
        embedding_col = embedding_columns.get(fname)

        columns.append(
            ColumnInfo(
                name=fname,
                data_type=str(data_type),
                description=description,
                is_searchable=is_searchable,
                embedding_column=embedding_col,
            ),
        )

    return columns
