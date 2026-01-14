from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import unify

logger = logging.getLogger(__name__)

from ....common.context_store import TableStore
from ....common.model_to_fields import model_to_fields
from ...types.file import FileRecord as FileRow, FileContent as _PerFileContent
from ...types.config import TableBusinessContextSpec

if TYPE_CHECKING:
    from ...types.describe import ColumnInfo, FileStorageMap


def provision_storage(self) -> None:
    """Ensure FileRecords/<alias> context, schema and local view exist (idempotent)."""
    # Create the TableStore if not already initialised by the manager
    if not hasattr(self, "_store") or getattr(self, "_store", None) is None:
        self._store = TableStore(  # type: ignore[attr-defined]
            self._ctx,  # type: ignore[attr-defined]
            unique_keys={"file_id": "int"},
            auto_counting={"file_id": None},
            description=(
                "FileRecords index for a single filesystem; per-file content lives under Files/<alias>/<safe(file_path)>/Tables/<table>."
            ),
            fields=model_to_fields(FileRow),
        )
    try:
        self._store.ensure_context()  # type: ignore[attr-defined]
    except Exception:
        # Best-effort
        pass


def get_columns(self, table: Optional[str] = None) -> Dict[str, str]:
    """Return {column_name: column_type} for index or any resolved context.

    Resolution rules when ``table`` is provided:
    - "FileRecords" → return index columns
    - Logical names from tables_overview (preferred):
      "<root>" (Content), "<root>.Tables.<label>" (per-file table)
    - Legacy refs: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
    - Fully-qualified context: use as-is
    """
    if not table:
        return self._store.get_columns()  # type: ignore[attr-defined]
    t = str(table).strip()
    if not t:
        return {}
    if t.lower() == "filerecords":
        return self._store.get_columns()  # type: ignore[attr-defined]
    resolved_ctx: Optional[str] = None
    # If the input already targets a context under this manager, use it directly.
    try:
        _index_ctx = getattr(self, "_ctx")
    except Exception:
        _index_ctx = None
    try:
        _files_root = getattr(self, "_per_file_root")
    except Exception:
        _files_root = None
    if (_index_ctx and t == _index_ctx) or (
        _files_root and t.startswith(f"{_files_root}/")
    ):
        resolved_ctx = t
    else:
        try:
            # Prefer full resolver (supports legacy forms)
            from .search import (
                resolve_table_ref as _res_ref,
            )  # local import to avoid cycles

            resolved_ctx = _res_ref(self, t)
        except Exception:
            try:
                # Fallback to logical name resolver
                from .search import ctx_for_table as _ctx_for_table  # type: ignore

                resolved_ctx = _ctx_for_table(self, t)
            except Exception:
                resolved_ctx = t
    try:
        fields = unify.get_fields(context=str(resolved_ctx))
        return {k: v.get("data_type") for k, v in fields.items()}
    except Exception as e:
        # Best-effort: return empty mapping on failure
        return {}


# ----------------------- Context getters (strings only) ----------------------- #


def ctx_for_file_index(self) -> str:
    """Return the fully-qualified context for FileRecords/<alias>."""
    return getattr(self, "_ctx")


def ctx_for_file(self, *, file_path: str) -> str:
    """Return the fully-qualified per-file Content context.

    Shape: <base>/Files/<alias>/<safe(file_path)>/Content

    .. deprecated::
        Use ctx_for_file_by_id for new code. file_path-based paths
        are being replaced with file_id-based paths for stability.
    """
    safe = getattr(self, "safe") if hasattr(self, "safe") else (lambda x: x)
    base = getattr(self, "_per_file_root")
    return f"{base}/{safe(file_path)}/Content"


def ctx_for_file_table(self, *, file_path: str, table: str) -> str:
    """Return the fully-qualified per-file table context.

    Shape: <base>/Files/<alias>/<safe(file_path)>/Tables/<safe(table)>

    .. deprecated::
        Use ctx_for_file_table_by_id for new code. file_path-based paths
        are being replaced with file_id-based paths for stability.
    """
    safe = getattr(self, "safe") if hasattr(self, "safe") else (lambda x: x)
    base = getattr(self, "_per_file_root")
    return f"{base}/{safe(file_path)}/Tables/{safe(table)}"


# ----------------------- file_id-based context getters ----------------------- #


def ctx_for_file_by_id(self, *, file_id: int) -> str:
    """Return the fully-qualified per-file Content context using file_id.

    Shape: <base>/Files/<alias>/<file_id>/Content

    Parameters
    ----------
    file_id : int
        The stable unique identifier for the file from FileRecords.

    Returns
    -------
    str
        Full Unify context path for the file's document content.

    Notes
    -----
    Using file_id instead of file_path provides:
    - Stable references that survive file renames
    - Shorter, LLM-friendly context paths
    - Direct join capability with FileRecords.file_id
    """
    base = getattr(self, "_per_file_root")
    return f"{base}/{file_id}/Content"


def ctx_for_file_table_by_id(self, *, file_id: int, table: str) -> str:
    """Return the fully-qualified per-file table context using file_id.

    Shape: <base>/Files/<alias>/<file_id>/Tables/<safe(table)>

    Parameters
    ----------
    file_id : int
        The stable unique identifier for the file from FileRecords.
    table : str
        The logical table name (e.g., 'Sheet1', 'extracted_table_1').

    Returns
    -------
    str
        Full Unify context path for the file's table.

    Notes
    -----
    Using file_id instead of file_path provides:
    - Stable references that survive file renames
    - Shorter, LLM-friendly context paths
    - Direct join capability with FileRecords.file_id
    """
    safe = getattr(self, "safe") if hasattr(self, "safe") else (lambda x: x)
    base = getattr(self, "_per_file_root")
    return f"{base}/{file_id}/Tables/{safe(table)}"


def ctx_tables_prefix_by_id(self, *, file_id: int) -> str:
    """Return the prefix for all table contexts under a file.

    Shape: <base>/Files/<alias>/<file_id>/Tables/

    Parameters
    ----------
    file_id : int
        The stable unique identifier for the file from FileRecords.

    Returns
    -------
    str
        Prefix path for listing/discovering table contexts.
    """
    base = getattr(self, "_per_file_root")
    return f"{base}/{file_id}/Tables/"


# ------------------------ Context provisioners (ensure) ---------------------- #


def ensure_file_context(
    self,
    *,
    file_path: str,
) -> None:
    """Ensure a per-file Content context exists (idempotent)."""
    ctx = ctx_for_file(self, file_path=file_path)
    fields = model_to_fields(_PerFileContent)
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        fields=fields,
        description=f"Per-file context for '{file_path}' using the File schema",
    )
    try:
        store.ensure_context()
    except Exception:
        # Best-effort provisioning
        pass


def ensure_file_table_context(
    self,
    *,
    file_path: str,
    table: str,
    unique_key: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    columns: Optional[Union[List[str], Dict[str, Any]]] = None,
    example_row: Optional[Dict[str, Any]] = None,
    business_context: Optional[TableBusinessContextSpec] = None,
) -> None:
    """Ensure a per-file Tables/<label> context exists with initial fields.

    - Unique key defaults to ``row_id`` with auto-counting enabled.
    - When ``columns`` is provided:
      * If a list, create fields with ``{"mutable": True}`` (no strict type).
      * If a dict mapping names→types, pass through as-is.
    - Else, when ``example_row`` is provided, infer field names from its keys and
      create fields with ``{"mutable": True}`` (no strict type).
    - When ``business_context`` is provided:
      * Merge column descriptions into field definitions (add ``"description"`` key).
      * Use ``table_description`` as the context description.
    """
    ctx = ctx_for_file_table(self, file_path=file_path, table=table)
    # Build fields from columns or example_row keys when provided
    fields_map: Dict[str, Any] = {}
    if columns is not None:
        if isinstance(columns, dict):
            # Pass through mapping of column names to types as-is
            fields_map = dict(columns)
        else:
            # Treat as an iterable of column names
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
                # Ensure field entry is a dict with required 'type' field
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


def _resolve_file_target(self, file: str) -> Dict[str, Any]:
    """
    Resolve the ingest target for a file path or unified label.

    Returns keys:
    - ingest_mode: "per_file" | "unified"
    - unified_label: str | None
    - table_ingest: bool
    - content_ctx: str (fully-qualified Content context)
    - tables_prefix: str (prefix for per-file Tables contexts)
    - target_name: str (safe root used in per-file context building)
    """
    # Use path as-is for querying
    try:
        rows = unify.get_logs(
            context=self._ctx,  # type: ignore[attr-defined]
            filter=f"file_path == {file!r}",
            limit=1,
            from_fields=["ingest_mode", "unified_label", "table_ingest"],
        )
    except Exception:
        rows = []

    ingest_mode = "per_file"
    unified_label = None
    table_ingest = True
    safe = getattr(self, "safe") if hasattr(self, "safe") else (lambda x: x)

    if rows:
        e = rows[0].entries
        ingest_mode = (e.get("ingest_mode") or "per_file").strip() or "per_file"
        unified_label = e.get("unified_label")
        table_ingest = bool(e.get("table_ingest", True))
        root = (
            safe(file)
            if ingest_mode == "per_file"
            else safe(str(unified_label or "Unified"))
        )
    else:
        # Treat as a unified label candidate
        try:
            rows2 = unify.get_logs(
                context=self._ctx,  # type: ignore[attr-defined]
                filter=f"unified_label == {file!r}",
                limit=1,
                from_fields=["unified_label"],
            )
        except Exception:
            rows2 = []
        if rows2:
            ingest_mode = "unified"
            unified_label = file
            root = safe(str(unified_label))
        else:
            root = safe(file)

    base = getattr(self, "_per_file_root")
    content_ctx = f"{base}/{root}/Content"
    tables_prefix = f"{base}/{root}/Tables/"

    return {
        "ingest_mode": ingest_mode,
        "unified_label": unified_label,
        "table_ingest": table_ingest,
        "content_ctx": content_ctx,
        "tables_prefix": tables_prefix,
        "target_name": root,
    }


def tables_overview(
    self,
    *,
    include_column_info: bool = True,
    file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return an ingest-aware overview of contexts managed by the FileManager.

    Parameters
    ----------
    include_column_info : bool, default True
        When True, include the FileRecords column schema in the global view.
    file : str | None
        When provided, return a file-scoped view that respects the file's
        ingest layout (per_file vs unified).

    Returns
    -------
    dict
        - Global view (file=None):
          { "FileRecords": { context, description, columns? } }
        - File-scoped view (per_file mode):
          {
            "FileRecords": {...},
            "<safe(file_path)>": {
              "Content": { context, description },
              "Tables": { "<safe(label)>": { context, description }, ... }
            }
          }
        - File-scoped view (unified mode):
          {
            "FileRecords": {...},
            "<unified_label>": { "Content": { context, description } },
            "<safe(file_path)>": { "Tables": { "<safe(label)>": { context, description }, ... } }
          }

    Notes
    -----
    - Keys under "Tables" are always passed through the manager's safe() helper.
    - All resolved contexts are prefixed with the base + "/Files/<alias>/…" path.
    """
    # Global index overview (KM-style)
    if file is None:
        try:
            ctx_info = unify.get_context(self._ctx)  # type: ignore[attr-defined]
        except Exception:
            ctx_info = {}

        out: Dict[str, Dict[str, Any]] = {
            "FileRecords": {
                "context": self._ctx,
                "description": (
                    ctx_info.get("description") if isinstance(ctx_info, dict) else ""
                ),
            },
        }
        if include_column_info:
            try:
                out["FileRecords"]["columns"] = get_columns(self)
            except Exception:
                out["FileRecords"]["columns"] = {}
        return out

    # Ingest-aware nested overview
    info = _resolve_file_target(self, file)
    ingest_mode = info.get("ingest_mode")
    unified_label = info.get("unified_label")
    content_ctx = info.get("content_ctx")
    tables_prefix = info.get("tables_prefix")
    table_ingest = bool(info.get("table_ingest", True))
    safe = getattr(self, "safe") if hasattr(self, "safe") else (lambda x: x)

    # Always include FileRecords
    out: Dict[str, Dict[str, Any]] = {"FileRecords": {"context": self._ctx}}
    try:
        ci = unify.get_context(self._ctx)
        out["FileRecords"]["description"] = (
            ci.get("description") if isinstance(ci, dict) else ""
        )
        if include_column_info:
            out["FileRecords"]["columns"] = get_columns(self)
    except Exception:
        pass

    # Use local ctx builder for canonical table context paths

    def _tables_map(prefix: str, root_name: str) -> Dict[str, Dict[str, Any]]:
        try:
            ctxs = unify.get_contexts(prefix=prefix)
        except Exception:
            ctxs = {}
        m: Dict[str, Dict[str, Any]] = {}
        for full, desc in (ctxs or {}).items():
            try:
                raw = full.split("/Tables/", 1)[-1]
                key = safe(raw)
                m[key] = {
                    "context": ctx_for_file_table(self, file_path=root_name, table=raw),  # type: ignore[arg-type]
                    "description": desc,
                }
            except Exception:
                continue
        return m

    if ingest_mode == "per_file":
        root = safe(file)
        out[root] = {"Content": {"context": content_ctx}}
        try:
            cinfo = unify.get_context(content_ctx)
            out[root]["Content"]["description"] = (
                cinfo.get("description") if isinstance(cinfo, dict) else ""
            )
        except Exception:
            out[root]["Content"]["description"] = ""
        if table_ingest:
            out[root]["Tables"] = _tables_map(tables_prefix, info.get("target_name"))
        return out

    # unified
    ulabel = safe(str(unified_label or "Unified"))
    out[ulabel] = {"Content": {"context": content_ctx}}
    try:
        cinfo = unify.get_context(content_ctx)
        out[ulabel]["Content"]["description"] = (
            cinfo.get("description") if isinstance(cinfo, dict) else ""
        )
    except Exception:
        out[ulabel]["Content"]["description"] = ""
    if table_ingest:
        leaf = safe(file)
        # If leaf matches ulabel (file path matches unified label), merge Tables into existing entry
        if leaf == ulabel:
            out[ulabel]["Tables"] = _tables_map(tables_prefix, info.get("target_name"))
        else:
            out[leaf] = {"Tables": _tables_map(tables_prefix, info.get("target_name"))}
    return out


def file_info(
    self,
    *,
    identifier: Union[str, int],
):
    """
    Core implementation for retrieving comprehensive file information.

    Parameters
    ----------
    self : FileManager
        The FileManager instance (for context resolution, adapter access, etc.).
    identifier : str | int
        File identifier. Accepted forms:
        - Absolute file path: "/path/to/file.pdf"
        - Provider URI: "local:///path/to/file.pdf", "gdrive://fileId"
        - File ID (int): The numeric file_id from FileRecords

    Returns
    -------
    FileInfo (Pydantic model)
        Structured output with filesystem/index status and ingest layout.
    """
    from unity.file_manager.types.file import FileInfo as _FileInfo

    # Start with defaults
    file_path = str(identifier)
    filesystem_exists = False
    indexed_exists = False
    parsed_status = None
    source_provider = None
    source_uri = None
    ingest_mode = "per_file"
    unified_label = None
    table_ingest = True
    file_format = None

    # Resolve file_id to file_path first
    try:
        fid = int(identifier)
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"file_id == {fid}",
            limit=1,
            from_fields=[
                "file_path",
                "source_uri",
                "status",
                "ingest_mode",
                "unified_label",
                "table_ingest",
                "file_format",
            ],
        )
        if logs:
            indexed_exists = True
            entry = logs[0].entries
            file_path = entry.get("file_path", str(identifier))
            source_uri = entry.get("source_uri")
            parsed_status = entry.get("status")
            ingest_mode = entry.get("ingest_mode", "per_file")
            unified_label = entry.get("unified_label")
            table_ingest = bool(entry.get("table_ingest", True))
            file_format = entry.get("file_format")
    except (ValueError, TypeError):
        # Not an int, continue with string resolution
        pass

    # For string identifiers, resolve canonical URI
    if not indexed_exists:
        resolve_to_uri = getattr(self, "_resolve_to_uri", None)
        if resolve_to_uri:
            source_uri = resolve_to_uri(identifier)

    # Filesystem existence (only for path-like identifiers)
    try:
        exists_fn = getattr(self, "exists", None)
        if exists_fn:
            filesystem_exists = bool(exists_fn(file_path))
    except Exception:
        pass

    # Get adapter provider
    try:
        adapter = getattr(self, "_adapter", None)
        source_provider = getattr(adapter, "name", None) or getattr(
            self,
            "_fs_type",
            None,
        )
        try:
            adapter_get = getattr(self, "_adapter_get", None)
            if adapter_get:
                ref = adapter_get(file_id_or_path=file_path)
                source_provider = str(ref.get("provider") or source_provider)
        except Exception:
            pass
    except Exception:
        pass

    # Index lookup if not already found via file_id
    if not indexed_exists:
        try:
            logs = []
            # Try by source_uri first
            if source_uri:
                try:
                    logs = unify.get_logs(
                        context=self._ctx,
                        filter=f"source_uri == {source_uri!r}",
                        limit=1,
                        from_fields=[
                            "status",
                            "file_path",
                            "ingest_mode",
                            "unified_label",
                            "table_ingest",
                            "file_format",
                        ],
                    )
                except Exception:
                    pass
            # Fallback to file_path match
            if not logs:
                try:
                    logs = unify.get_logs(
                        context=self._ctx,
                        filter=f"file_path == {file_path!r}",
                        limit=1,
                        from_fields=[
                            "status",
                            "file_path",
                            "ingest_mode",
                            "unified_label",
                            "table_ingest",
                            "file_format",
                        ],
                    )
                except Exception:
                    pass

            if logs:
                indexed_exists = True
                entry = logs[0].entries
                # Update file_path if found in index
                file_path = entry.get("file_path", file_path)
                parsed_status = entry.get("status")
                ingest_mode = entry.get("ingest_mode", "per_file")
                unified_label = entry.get("unified_label")
                table_ingest = bool(entry.get("table_ingest", True))
                file_format = entry.get("file_format")
        except Exception:
            pass

    return _FileInfo(
        file_path=file_path,
        filesystem_exists=filesystem_exists,
        indexed_exists=indexed_exists,
        parsed_status=parsed_status,
        source_provider=source_provider,
        source_uri=source_uri,
        ingest_mode=ingest_mode,
        unified_label=unified_label,
        table_ingest=table_ingest,
        file_format=file_format,
    )


def schema_explain(
    self,
    *,
    table: str,
) -> str:
    """
    Return a natural-language explanation of a table's structure and purpose.

    Parameters
    ----------
    self : FileManager
        The FileManager instance (for context resolution).
    table : str
        Table reference (path-first preferred):
        - "<file_path>" for per-file Content
        - "<file_path>.Tables.<label>" for per-file tables
        - "FileRecords" for the global file index

    Returns
    -------
    str
        Compact natural-language explanation including:
        - What the table represents
        - Key fields and their meanings
        - Row count
    """
    # Resolve table reference to context
    ctx = table
    resolve_table_ref = getattr(self, "_resolve_table_ref", None)
    if resolve_table_ref is not None:
        try:
            ctx = resolve_table_ref(table)
        except Exception:
            ctx = table

    # Get fields with descriptions from Unify
    try:
        fields = unify.get_fields(context=ctx) or {}
    except Exception:
        fields = {}

    if not fields:
        return f"No schema information available for table '{table}'."

    # Build structured explanation
    parts: List[str] = []

    # Table identity
    parts.append(f"Table: {table}")
    parts.append("")

    # Categorize columns (skip internal/private columns)
    regular_cols: List[str] = []

    for fname in fields.keys():
        if not isinstance(fname, str):
            continue
        if not fname.startswith("_"):
            regular_cols.append(fname)

    # Key fields with descriptions
    parts.append("Columns:")
    for col in regular_cols:
        field_info = fields.get(col, {})
        desc = None
        if isinstance(field_info, dict):
            desc = field_info.get("description")
        if desc:
            parts.append(f"  - {col}: {desc}")
        else:
            parts.append(f"  - {col}")
    parts.append("")

    # Row count hint
    try:
        count = unify.get_logs_metric(
            metric="count",
            key=regular_cols[0] if regular_cols else "row_id",
            context=ctx,
        )
        parts.append(f"Row count: {count}")
    except Exception:
        pass

    return "\n".join(parts)


# ----------------------- describe() implementation ----------------------- #


def describe_file(
    self,
    *,
    file_path: Optional[str] = None,
    file_id: Optional[int] = None,
) -> "FileStorageMap":
    """
    Return a complete storage representation of a file in the Unify backend.

    This is the primary discovery tool for understanding how a file's data
    is stored. It returns all context paths, schemas, and identifiers needed
    for accurate filter/search/reduce operations.

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
        Complete storage representation including:
        - file_id: Stable identifier for cross-referencing
        - file_path: Original filesystem path
        - document: Info about /Content context (if present)
        - tables: List of /Tables/<name> contexts with schemas
        - index_context: Path to FileRecords index

    Raises
    ------
    ValueError
        If neither file_path nor file_id is provided, or if the file is not found.

    Examples
    --------
    >>> # Describe by file path
    >>> storage = file_manager.describe(file_path="/reports/Q4.csv")
    >>> print(storage.file_id)  # 42
    >>> print(storage.tables[0].context_path)
    'Files/Local/42/Tables/Sheet1'

    >>> # Use the context path for queries
    >>> results = data_manager.filter(
    ...     context=storage.tables[0].context_path,
    ...     filter="revenue > 1000000"
    ... )

    >>> # Describe by file_id (faster, no path resolution needed)
    >>> storage = file_manager.describe(file_id=42)

    Notes
    -----
    - The describe() method queries the backend live for fresh schema information.
    - Context paths use file_id (not file_path) for stability across renames.
    - Row counts are not included by default; use reduce(metric='count') when needed.
    """
    from ...types.describe import (
        FileStorageMap,
        DocumentInfo,
        TableInfo,
        ContextSchema,
    )

    if file_path is None and file_id is None:
        raise ValueError("Either file_path or file_id must be provided")

    resolved_file_id: Optional[int] = file_id
    resolved_file_path: Optional[str] = file_path
    source_uri: Optional[str] = None
    source_provider: Optional[str] = None

    # Resolve file_id from file_path if needed
    if resolved_file_id is None and file_path is not None:
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=f"file_path == {file_path!r}",
                limit=1,
                from_fields=["file_id", "source_uri", "source_provider"],
            )
            if rows:
                entry = rows[0].entries
                resolved_file_id = entry.get("file_id")
                source_uri = entry.get("source_uri")
                source_provider = entry.get("source_provider")
        except Exception as e:
            logger.warning(f"Failed to lookup file_id for {file_path}: {e}")

    # Resolve file_path from file_id if needed
    if resolved_file_path is None and resolved_file_id is not None:
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=f"file_id == {resolved_file_id}",
                limit=1,
                from_fields=["file_path", "source_uri", "source_provider"],
            )
            if rows:
                entry = rows[0].entries
                resolved_file_path = entry.get("file_path")
                source_uri = entry.get("source_uri")
                source_provider = entry.get("source_provider")
        except Exception as e:
            logger.warning(
                f"Failed to lookup file_path for file_id={resolved_file_id}: {e}",
            )

    if resolved_file_id is None:
        raise ValueError(
            f"File not found in index: file_path={file_path!r}, file_id={file_id}",
        )

    # Build context paths using file_id
    base = getattr(self, "_per_file_root")
    content_ctx = f"{base}/{resolved_file_id}/Content"
    tables_prefix = f"{base}/{resolved_file_id}/Tables/"

    # Check if document context exists and get its schema
    document_info: Optional[DocumentInfo] = None
    try:
        content_fields = unify.get_fields(context=content_ctx)
        if content_fields:
            columns = _fields_to_column_info(content_fields)
            document_info = DocumentInfo(
                context_path=content_ctx,
                column_schema=ContextSchema(columns=columns),
                row_count=None,  # Fetch on-demand with reduce()
            )
    except Exception:
        # Content context doesn't exist
        pass

    # Discover table contexts
    table_infos: List[TableInfo] = []
    try:
        # List all contexts under the tables prefix
        all_contexts = unify.get_contexts(prefix=tables_prefix)
        for ctx_path, ctx_info in all_contexts.items():
            if ctx_path == tables_prefix:
                continue  # Skip the prefix itself
            # Extract table name from path
            table_name = ctx_path.replace(tables_prefix, "").split("/")[0]
            if not table_name:
                continue

            # Get table schema
            try:
                table_fields = unify.get_fields(context=ctx_path)
                columns = _fields_to_column_info(table_fields)
                table_infos.append(
                    TableInfo(
                        name=table_name,
                        context_path=ctx_path,
                        column_schema=ContextSchema(columns=columns),
                        row_count=None,  # Fetch on-demand with reduce()
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
                    ),
                )
    except Exception as e:
        logger.warning(f"Failed to discover tables for file_id={resolved_file_id}: {e}")

    return FileStorageMap(
        file_id=resolved_file_id,
        file_path=resolved_file_path or "",
        source_uri=source_uri,
        source_provider=source_provider,
        document=document_info,
        tables=table_infos,
        index_context=self._ctx,
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
