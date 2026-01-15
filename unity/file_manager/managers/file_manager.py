from __future__ import annotations

import json
import functools
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

import unify

from unity.common.tool_spec import manager_tool, read_only
from unity.file_manager.base import BaseFileManager
from unity.file_manager.file_parsers import FileParser
from unity.file_manager.types.file import FileRecord
from unity.file_manager.types.config import (
    FilePipelineConfig as _FilePipelineConfig,
)
from unity.file_manager.types.ingest import (
    IngestPipelineResult,
    BaseIngestedFile,
    IngestedMinimal,
    ContentRef,
    FileMetrics,
    FileResultType,
)
from unity.file_manager.filesystem_adapters.base import BaseFileSystemAdapter
from unity.file_manager.prompt_builders import (
    build_file_manager_ask_about_file_prompt,
)
from unity.common.llm_helpers import (
    methods_to_tool_dict,
)
from unity.common.clarification_tools import add_clarification_tool_with_events
from unity.common.async_tool_loop import (
    TOOL_LOOP_LINEAGE,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.settings import SETTINGS
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.events.manager_event_logging import log_manager_call
from unity.common.context_store import TableStore
from unity.common.model_to_fields import model_to_fields
from unity.common.llm_client import new_llm_client
from .utils.search import (
    resolve_table_ref as _srch_resolve_table_ref,
    create_join as _srch_create_join,
)
from .utils.storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
    tables_overview as _storage_tables_overview,
    schema_explain as _storage_schema_explain,
    ctx_for_file as _storage_ctx_for_file,
    ctx_for_file_table as _storage_ctx_for_file_table,
)
from unity.data_manager.types import PlotResult as _VizPlotResult


if TYPE_CHECKING:
    from unity.file_manager.types.describe import FileStorageMap
    from unity.data_manager.base import BaseDataManager


class FileManager(BaseFileManager):
    """A FileManager bound to exactly one filesystem adapter.

    - Discovery/search index is backed by adapter capabilities
    - Bytes are retrieved from the adapter and parsed on demand
    - Exposes unified tools for read-only inspection and safe rename/move
    """

    def __init__(
        self,
        adapter: Optional[BaseFileSystemAdapter] = None,
        *,
        parser: Optional[FileParser] = None,
        rolling_summary_in_prompts: bool = True,
        data_manager: Optional["BaseDataManager"] = None,
    ) -> None:
        """
        Construct a FileManager bound to a single filesystem adapter.

        Parameters
        ----------
        adapter : BaseFileSystemAdapter | None, default None
            Filesystem adapter (e.g., Local, CodeSandbox, Interact). When None,
            adapter-backed operations will raise.
        parser : BaseParser | None, default None
            Parser used for extracting tables/text/metadata from bytes. Defaults
            to ``DoclingParser`` with table and image extraction enabled.
        rolling_summary_in_prompts : bool, default True
            Whether to include the rolling activity summary in prompts.
        data_manager : BaseDataManager | None, default None
            Optional DataManager instance for data operations delegation.
            If None, a DataManager will be lazily instantiated when needed.
        """
        super().__init__()
        self.include_in_multi_assistant_table = False
        self._adapter = adapter
        self.__parser: Optional[FileParser] = parser
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self.__data_manager: Optional["BaseDataManager"] = data_manager

        # Derive a stable alias and context
        try:
            raw_alias = (
                getattr(self._adapter, "name", "Local").strip()
                if self._adapter is not None
                else "Local"
            )
        except Exception:
            raw_alias = "Local"

        self._fs_alias = self.safe(str(raw_alias))

        # Extract clean filesystem type for LLM prompts (without path/details)
        self._fs_type = self._extract_filesystem_type(raw_alias)

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from ... import ensure_initialised as _ensure_initialised  # type: ignore  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
            except Exception:
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a FileManager."
        base_ctx = read_ctx or "default"
        # Use a single Files namespace and a per‑filesystem suffix
        # Root contexts
        # - FileRecords: index of files (lightweight per file row)
        # - File:        per-file content roots (one subcontext per safe filepath)
        self._ctx = f"{base_ctx}/FileRecords/{self._fs_alias}"
        self._per_file_root = f"{base_ctx}/Files/{self._fs_alias}"

        # Ensure context and fields exist
        self._store = TableStore(
            self._ctx,
            unique_keys={"file_id": "int"},
            auto_counting={"file_id": None},
            description=(
                "FileRecords index for a single filesystem; per-file content lives under Files/<alias>/<safe_filepath>/Tables/<table>."
            ),
            fields=model_to_fields(FileRecord),
        )
        try:
            self._store.ensure_context()
        except unify.RequestError as e:
            body = getattr(e.response, "text", "") or ""
            # Treat duplicate context as success and do not emit error output
            if "already exists" in body:
                pass

        # Ensure storage via shared helper (idempotent)
        try:
            self._provision_storage()
        except Exception:
            pass

        # Immutable built-in fields derived from the FileRecord model
        self._BUILTIN_FIELDS: tuple[str, ...] = tuple(FileRecord.model_fields.keys())

        # Public tool dictionaries, mirroring other managers
        # Multi-table tools (joins across per-file tables)
        ask_multi_table_tools: Dict[str, Callable] = methods_to_tool_dict(
            self.filter_join,
            self.search_join,
            self.filter_multi_join,
            self.search_multi_join,
            include_class_name=False,
        )
        ask_about_file_tools: Dict[str, Callable] = methods_to_tool_dict(
            # Read-only helpers (no ingest_files - this is read-only)
            self.file_info,
            self.list_columns,
            self.tables_overview,
            self.schema_explain,
            self.filter_files,
            self.search_files,
            self.reduce,
            # Visualization
            self.visualize,
            include_class_name=False,
        )
        self.add_tools("ask_about_file", ask_about_file_tools)
        self.add_tools("ask_about_file.multi_table", ask_multi_table_tools)

    @functools.cached_property
    def _parser(self):
        if self.__parser is None:
            self.__parser = FileParser()
        return self.__parser

    @property
    def _data_manager(self) -> "BaseDataManager":
        """
        Lazily instantiated DataManager for data operations delegation.

        When FileManager delegates filter, search, reduce, join, and plot
        operations, it routes through this DataManager instance.

        Returns
        -------
        BaseDataManager
            The DataManager instance (real or simulated based on settings).
        """
        if self.__data_manager is None:
            from unity.manager_registry import ManagerRegistry

            self.__data_manager = ManagerRegistry.get_data_manager()
        return self.__data_manager

    def _resolve_table_refs(
        self,
        tables: Union[str, List[str]],
    ) -> List[str]:
        """
        Resolve FileManager table references to full Unify context paths.

        Handles various input formats:
        - Full context paths (unchanged)
        - File path references: "/reports/Q4.csv" → Files/Local/{file_id}/Content
        - Table references: "/reports/Q4.csv.Tables.Sheet1" → Files/Local/{file_id}/Tables/Sheet1

        Parameters
        ----------
        tables : str | list[str]
            Table reference(s) to resolve.

        Returns
        -------
        list[str]
            List of resolved full context paths.
        """
        if isinstance(tables, str):
            tables = [tables]

        resolved = []
        for t in tables:
            # If it's already a full context path, use it directly
            if t.startswith(("Files/", "Data/", "Knowledge/")):
                resolved.append(t)
            else:
                # Resolve using the existing table ref resolution logic
                resolved.append(_srch_resolve_table_ref(self, t))
        return resolved

    def _resolve_joins_table_refs(
        self,
        joins: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Resolve table references in a list of join step definitions.

        Handles $prev references specially - they should not be resolved.

        Parameters
        ----------
        joins : list[dict]
            List of join step definitions with 'tables' keys.

        Returns
        -------
        list[dict]
            Join steps with resolved table references.
        """
        resolved_joins = []
        for step in joins:
            resolved_step = dict(step)  # Copy to avoid mutation
            if "tables" in step:
                tables = step["tables"]
                resolved_tables = []
                for t in tables:
                    if t == "$prev":
                        resolved_tables.append(t)  # Keep $prev as-is
                    elif t.startswith(("Files/", "Data/", "Knowledge/")):
                        resolved_tables.append(t)  # Already resolved
                    else:
                        resolved_tables.append(_srch_resolve_table_ref(self, t))
                resolved_step["tables"] = resolved_tables
            resolved_joins.append(resolved_step)
        return resolved_joins

    def _provision_storage(self) -> None:
        """
        Idempotently provision storage for this manager's index context.

        Behaviour
        ---------
        - Ensures the global index context for this filesystem exists and has
          the correct unique key and auto-counting configuration.
        - Safe to call repeatedly; no-ops when the context already exists.

        Returns
        -------
        None
        """
        _storage_provision(self)

    # ------------------------- File info helper ----------------------------- #
    @read_only
    def file_info(self, *, identifier: Union[str, int]):
        """
        Return comprehensive information about a file's status and ingest identity.

        .. deprecated::
            Use describe(file_path=...) or describe(file_id=...) instead for
            comprehensive file discovery. describe() returns FileStorageMap with
            exact context paths, schemas, and stable file_id for cross-referencing.

        Use this to get a complete picture of a file including filesystem presence,
        index status, parse status, and ingest layout (per_file vs unified mode).

        Parameters
        ----------
        identifier : str | int
            File identifier. Accepted forms:
            - Absolute file path: "/path/to/file.pdf"
            - Provider URI: "local:///path/to/file.pdf", "gdrive://fileId"
            - File ID (int): The numeric file_id from FileRecords

        Returns
        -------
        FileInfo (Pydantic model)
            - file_path: str - The resolved/queried path
            - filesystem_exists: bool - True if file exists on disk
            - indexed_exists: bool - True if file is in FileRecords index
            - parsed_status: "success" | "error" | None - Parse result
            - source_provider: str | None - Adapter/provider name
            - source_uri: str | None - Canonical provider URI
            - ingest_mode: "per_file" | "unified" - Storage layout
            - unified_label: str | None - Bucket label (when unified)
            - table_ingest: bool - Whether tables are in per-file /Tables/
            - file_format: str | None - File format (pdf, xlsx, etc.)

        Ingest modes
        ------------
        per_file mode:
          - Content: <base>/Files/<alias>/<file>/Content
          - Tables: <base>/Files/<alias>/<file>/Tables/<label>

        unified mode:
          - Content: <base>/Files/<alias>/<unified_label>/Content
          - Tables: <base>/Files/<alias>/<file>/Tables/<label>
          (Tables always remain per-file regardless of mode)

        Usage Examples
        --------------
        # Check by file path:
        info = file_info(identifier="/path/to/document.pdf")
        if info.indexed_exists and info.parsed_status == "success":
            filter_files(tables=["/path/to/document.pdf"], ...)

        # Check by file_id:
        info = file_info(identifier=42)

        # Check by provider URI:
        info = file_info(identifier="gdrive://abc123")

        # File exists on disk but not indexed yet:
        info = file_info(identifier="/path/to/new_file.xlsx")
        if info.filesystem_exists and not info.indexed_exists:
            # Needs to be ingested first

        # Check ingest layout:
        info = file_info(identifier="/path/to/data.xlsx")
        if info.ingest_mode == "unified":
            # Content is under unified_label context

        Anti-patterns
        -------------
        - WRONG: Assuming a file is queryable without checking parsed_status
          CORRECT: Verify parsed_status == "success" before querying content

        - WRONG: Querying per-file Content for a unified-mode file
          CORRECT: Check ingest_mode and use unified_label context if unified
        """
        warnings.warn(
            "file_info() is deprecated. Use describe(file_path=...) or describe(file_id=...) "
            "instead for comprehensive file discovery with exact context paths and schemas.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .utils.storage import file_info as _storage_file_info

        return _storage_file_info(self, identifier=identifier)

    # ------------------------- Describe helper ------------------------------ #
    @manager_tool
    @read_only
    def describe(
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

        Use describe() BEFORE calling filter_files(), search_files(), or reduce()
        to obtain the exact context paths for your queries.

        Parameters
        ----------
        file_path : str, optional
            The filesystem path of the file as used in FileRecords.
            Either file_path or file_id must be provided.

        file_id : int, optional
            The stable unique identifier from FileRecords. Either file_path or
            file_id must be provided. Using file_id is preferred when available
            as it's more efficient and survives file renames.

        Returns
        -------
        FileStorageMap
            Complete storage representation including:

            - **file_id** (int): Stable identifier for cross-referencing and joins.
            - **file_path** (str): Original filesystem path.
            - **source_uri** (str | None): Canonical provider URI.
            - **source_provider** (str | None): Provider/adapter name.
            - **document** (DocumentInfo | None): Info about /Content context including:
              - context_path: Full path for filter/search/reduce operations
              - schema: Columns with types and searchability info
            - **tables** (list[TableInfo]): List of /Tables/<name> contexts including:
              - name: Logical table name (e.g., 'Sheet1')
              - context_path: Full path for filter/search/reduce operations
              - schema: Columns with types and searchability info
            - **index_context** (str): Path to FileRecords index.
            - **has_document** (bool): Quick check if /Content exists.
            - **has_tables** (bool): Quick check if any /Tables exist.

        Raises
        ------
        ValueError
            If neither file_path nor file_id is provided, or if the file is not found.

        Usage Examples
        --------------
        Basic discovery by file path:

        >>> storage = file_manager.describe(file_path="/reports/Q4.csv")
        >>> print(f"File ID: {storage.file_id}")
        File ID: 42
        >>> print(f"Has tables: {storage.has_tables}")
        Has tables: True
        >>> print(f"Table names: {storage.table_names}")
        Table names: ['Sheet1']

        Get context path for querying:

        >>> storage = file_manager.describe(file_path="/reports/Q4.csv")
        >>> # Use the exact context path from describe()
        >>> results = file_manager.filter_files(
        ...     context=storage.tables[0].context_path,
        ...     filter="revenue > 1000000",
        ...     columns=["region", "revenue"]
        ... )

        Check schema before building queries:

        >>> storage = file_manager.describe(file_path="/docs/report.pdf")
        >>> if storage.has_document:
        ...     print("Searchable columns:", storage.document.column_schema.searchable_columns)
        ...     # Use semantic search on document content
        ...     results = file_manager.search_files(
        ...         context=storage.document.context_path,
        ...         references="quarterly performance metrics"
        ...     )

        Describe by file_id (faster, no resolution needed):

        >>> storage = file_manager.describe(file_id=42)

        Anti-patterns
        -------------
        - WRONG: Guessing context paths without calling describe()

          >>> # Don't do this - path format may vary
          >>> filter_files(context="Files/Local/reports/Q4.csv/Tables/Sheet1")

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/reports/Q4.csv")
          >>> filter_files(context=storage.tables[0].context_path)

        - WRONG: Assuming all files have both document and tables

          >>> # Will fail if file has no tables
          >>> storage.tables[0].context_path

          CORRECT: Check has_tables/has_document first

          >>> if storage.has_tables:
          ...     # Safe to access tables
          ...     storage.tables[0].context_path

        - WRONG: Using file_path in context paths directly

          >>> # Unstable - breaks if file is renamed
          >>> "Files/Local/" + file_path + "/Content"

          CORRECT: Context paths use file_id internally for stability

          >>> storage.document.context_path  # Uses file_id

        Notes
        -----
        - Context paths use file_id (not file_path) for stability across renames.
        - The describe() method queries the backend live for fresh schema information.
        - Row counts are not included by default; use reduce(metric='count', ...) when needed.
        - For CSV/spreadsheet files, primary data is in tables (has_document may be False).
        - For PDF/DOCX files, primary data is in document with extracted tables optional.
        - Schema includes searchability info - check column_schema.searchable_columns before
          using search_files() to ensure the columns you need are embedded.

        See Also
        --------
        filter_files : Filter rows from a context path obtained from describe()
        search_files : Semantic search on a context path obtained from describe()
        reduce : Aggregate data from a context path obtained from describe()
        file_info : Get file status without full storage map (lighter weight)
        """
        from .utils.storage import describe_file as _storage_describe_file

        return _storage_describe_file(self, file_path=file_path, file_id=file_id)

    # ------------------------- Sync helper ---------------------------------- #
    def _sync(self, *, file_path: str) -> Dict[str, Any]:
        """
        Synchronize a previously ingested file with the underlying filesystem.

        This tool purges existing Unify rows for the file and re-parses the
        source, re-ingesting content (and tables when configured). It avoids
        duplications by deleting before re-inserting.

        Parameters
        ----------
        file_path : str
            The file identifier/path as used in FileRecords.file_path.

        Returns
        -------
        dict
            Outcome with details about purge counts and the new ingest status.
        """
        # Resolve identity and layout via file_info
        info = self.file_info(identifier=file_path)
        ingest_mode = info.get("ingest_mode", "per_file")
        unified_label = info.get("unified_label")
        table_ingest = info.get("table_ingest", True)

        overview = self.tables_overview(file=file_path)
        purged = {"content_rows": 0, "table_rows": 0}

        # Purge content rows
        try:
            from .utils.ops import delete_per_file_rows_by_filter as _ops_del_content

            dest_path = (
                file_path if ingest_mode == "per_file" else (unified_label or "Unified")
            )
            purged["content_rows"] = int(
                _ops_del_content(self, file_path=dest_path, filter_expr=None),
            )
        except Exception:
            pass

        # Purge per-file tables (when present)
        try:
            if table_ingest:
                from .utils.ops import (
                    delete_per_file_table_rows_by_filter as _ops_del_tbl,
                )

                for key, val in overview.items():
                    if key in ("FileRecords", str(unified_label or "")):
                        continue
                    tables = val.get("Tables") if isinstance(val, dict) else None
                    if not tables:
                        continue
                    for tlabel in list(tables.keys()):
                        try:
                            purged["table_rows"] += int(
                                _ops_del_tbl(
                                    self,
                                    file_path=key,
                                    table=tlabel,
                                    filter_expr=None,
                                ),
                            )
                        except Exception:
                            continue
        except Exception:
            pass

        # Re-parse (ingest + embed). Use compact/none to avoid token usage
        try:
            cfg = _FilePipelineConfig()
            cfg.output.return_mode = "none"
            self.ingest_files(file_path, config=cfg)
        except Exception as e:
            return {"outcome": "sync failed", "error": str(e), "purged": purged}

        return {"outcome": "sync complete", "purged": purged}

    # Public wrapper (exposed under organize)
    def sync(self, *, file_path: str) -> Dict[str, Any]:
        """
        Synchronize a previously ingested file with the underlying filesystem.

        Use this when a file's contents have changed on disk and you need to
        update the index. This tool purges existing Content/Table rows and
        re-parses the file from scratch, ensuring the index reflects current
        file contents.

        Parameters
        ----------
        file_path : str
            The file identifier/path as used in FileRecords.file_path.
            Use absolute paths for reliability.

        Returns
        -------
        dict
            Outcome with details:
            - "outcome": "sync complete" or "sync failed"
            - "purged": {"content_rows": int, "table_rows": int}
            - "error": str (only on failure)

        Usage Examples
        --------------
        # Sync a single file after it was updated:
        sync(file_path="/path/to/updated_document.pdf")
        # Returns: {"outcome": "sync complete", "purged": {"content_rows": 45, "table_rows": 3}}

        # Re-index after external modifications:
        modified_files = ["/path/to/a.pdf", "/path/to/b.xlsx"]
        for f in modified_files:
            result = sync(file_path=f)
            if result["outcome"] == "sync failed":
                print(f"Failed to sync {f}: {result.get('error')}")

        # Verify sync worked:
        result = sync(file_path="/path/to/file.pdf")
        if result["outcome"] == "sync complete":
            # Query the updated content
            search_files(table="/path/to/file.pdf", ...)

        Anti-patterns
        -------------
        - WRONG: Using sync on a file that doesn't exist in the filesystem
          CORRECT: Verify exists() returns True before syncing

        - WRONG: Syncing a file that hasn't changed (unnecessary work)
          CORRECT: Only sync when you know the source file was modified

        - WRONG: Expecting sync to preserve manual index modifications
          CORRECT: Sync purges ALL existing rows and re-ingests from source

        - WRONG: Using sync for initial ingestion
          CORRECT: Use ingest_files for new files; sync is for updates only
        """
        return self._sync(file_path=file_path)

    def _resolve_to_uri(self, identifier: str | int) -> str | None:
        """Resolve user-provided identifier (uri | absolute path | file_id) to canonical source_uri.

        Resolution order:
        1) If already looks like a URI ("scheme://"), return as-is.
        2) If numeric → treat as file_id and look up index row.
        3) Try adapter.get_file to obtain uri.
        4) If absolute local path, build local uri via adapter.uri_name.
        """
        try:
            s = str(identifier)
        except Exception:
            return None
        # URI fast-path
        if "://" in s:
            return s
        # file_id lookup
        try:
            fid = int(s)
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"file_id == {fid}",
                limit=1,
                from_fields=["source_uri"],
            )
            if logs:
                uri = logs[0].entries.get("source_uri")
                if isinstance(uri, str) and uri:
                    return uri
        except Exception:
            pass
        # Adapter lookup
        try:
            # Use path as-is (adapters handle both relative and absolute paths)
            ref = self._adapter_get(file_id_or_path=s)
            uri = ref.get("uri")
            if isinstance(uri, str) and uri:
                return uri
        except Exception:
            pass
        # Absolute local path fallback
        try:
            from pathlib import Path as _P

            p = _P(s)
            if p.is_absolute():
                uri_name = getattr(self._adapter, "uri_name", None) or "local"
                _pp = p.resolve().as_posix().lstrip("/")
                return f"{uri_name}://{_pp}"
        except Exception:
            pass
        return None

    @staticmethod
    def safe(value: Any) -> str:
        """
        Uniform sanitizer for a single context path component.

        Parameters
        ----------
        value : Any
            Value to sanitize for safe inclusion in a context path component.

        Returns
        -------
        str
            A lowercase-safe string containing only [a-zA-Z0-9_-], with other
            characters replaced by '_'. The result is truncated to 64
            characters; returns 'item' when empty.
        """
        try:
            import re as _re

            s = str(value)

            # Detect OS-invariant path separators; split into head and tail
            last_slash = max(s.rfind("/"), s.rfind("\\"))
            if last_slash >= 0:
                head_raw, tail_raw = s[:last_slash], s[last_slash + 1 :]
            else:
                head_raw, tail_raw = "", s

            def _sanitize(part: str) -> str:
                # Replace non [a-zA-Z0-9_-] (including dots and path punctuation) with underscores
                return _re.sub(r"[^a-zA-Z0-9_-]", "_", part)

            tail = _sanitize(tail_raw) or "item"
            head = _sanitize(head_raw)

            if not head:
                # No head: return sanitized tail as-is
                return tail

            def _compress_center(text: str, target_len: int) -> str:
                if len(text) <= target_len:
                    return text
                # Use multiple underscores as an ellipsis in the middle
                marker = "____"
                if target_len <= len(marker):
                    return marker[:target_len]
                left = (target_len - len(marker)) // 2
                right = target_len - len(marker) - left
                return text[:left] + marker + text[-right:]

            head_limit = 32
            head_comp = _compress_center(head, head_limit)
            return f"{head_comp}_{tail}"
        except Exception:
            return "item"

    @staticmethod
    def _extract_filesystem_type(adapter_name: str) -> str:
        """
        Extract the filesystem type from an adapter name, stripping path details.

        Parameters
        ----------
        adapter_name : str
            Adapter display name (may include details in brackets).

        Returns
        -------
        str
            The base adapter type (e.g., "Local", "CodeSandbox", "Interact").
        """
        if not adapter_name:
            return "Unknown"
        # Split on '[' and take the first part (the type)
        return adapter_name.split("[")[0].strip() or adapter_name

    # Helpers #
    # --------#

    def _ctx_for_file_table(self, file_path: str, table: str) -> str:
        """
        Return the fully‑qualified Unify context name for a per‑file table.

        Parameters
        ----------
        file_path : str
            The logical identifier/path of the file whose table context is requested.
        table : str
            Logical per‑file table name (e.g. "Products").

        Returns
        -------
        str
            Fully‑qualified context path in the form
            ``<base>/Files/<alias>/<safe_file_path>/Tables/<safe_table>``.
        """
        return _storage_ctx_for_file_table(self, file_path=file_path, table=table)

    def _ctx_for_file(self, file_path: str) -> str:
        """
        Return the fully‑qualified Unify context name for the per‑file root (Content).

        Parameters
        ----------
        file_path : str
            The logical identifier/path of the file whose per‑file context is requested.

        Returns
        -------
        str
            Fully‑qualified context path in the form
            ``<base>/Files/<alias>/<safe_file_path>/Content``.
        """
        return _storage_ctx_for_file(self, file_path=file_path)

    def _resolve_file_target(self, identifier: str) -> Dict[str, Any]:
        """Compatibility wrapper that delegates to storage._resolve_file_target."""
        try:
            from .utils.storage import _resolve_file_target as _res

            return _res(self, identifier)
        except Exception:
            return {
                "ingest_mode": "per_file",
                "unified_label": None,
                "content_ctx": _storage_ctx_for_file(self, file_path=identifier),
                "tables_prefix": f"{_storage_ctx_for_file(self, file_path=identifier)}/Tables/",
                "target_name": identifier,
            }

    # ---------- Join helpers (delegations to search module) ------------------- #
    @read_only
    def _resolve_table_ref(self, ref: str) -> str:
        """
        Resolve a table reference to a fully-qualified Unify context.

        Parameters
        ----------
        ref : str
            Accepted forms:
            - Logical names from `tables_overview()` (preferred):
              "FileRecords" → index; "<file_path>" → per-file Content;
              "<file_path>.Tables.<label>" → per-file table.
            - Legacy forms (backward compatible):
              "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>".

        Returns
        -------
        str
            Fully-qualified Unify context path for the referenced table.

        Examples
        --------
        - Logical name: _resolve_table_ref("Q1_Report") → ".../Files/<alias>/Q1_Report/Content"
        - Per-table: _resolve_table_ref("Q1_Report.Tables.Products") → ".../Tables/Products"
        - Legacy: _resolve_table_ref("/docs/q1.pdf:Products")
        """
        return _srch_resolve_table_ref(self, ref)

    def _create_join(
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
        """
        Create a derived table by joining two sources into ``dest_table_ctx``.

        Parameters
        ----------
        dest_table_ctx : str
            Fully-qualified destination context for the derived table.
        left_ref, right_ref : str
            Logical names (from `tables_overview`) or legacy refs. These are
            resolved to fully-qualified contexts.
        join_expr : str
            Boolean join predicate using the same identifiers as provided in
            ``left_ref`` and ``right_ref``. Identifiers will be rewritten to
            the fully-qualified contexts automatically.
        select : dict[str, str]
            Mapping of source expressions → output column names.
        mode : str, default "inner"
            One of {"inner", "left", "right", "outer"}.
        left_where, right_where : str | None
            Optional predicates applied to inputs before joining.

        Returns
        -------
        str
            The fully-qualified destination context that was created or re-used.

        Notes
        -----
        - Prefer logical names from `tables_overview()` rather than raw contexts.
        - For multi-step joins use the multi-join tools with `$prev`.
        """
        return _srch_create_join(
            self,
            dest_table_ctx=dest_table_ctx,
            left_ref=left_ref,
            right_ref=right_ref,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
        )

    # ---------- Adapter wrappers (bytes + identifiers only) ----------------- #
    @read_only
    def _adapter_list(self) -> List[str]:
        """
        Return adapter file paths/ids for this filesystem.

        Returns
        -------
        list[str]
            File identifiers/paths from the underlying adapter or an empty list
            when no adapter is configured or listing fails.
        """
        try:
            if self._adapter is None:
                return []
            return [ref.path for ref in self._adapter.iter_files()]
        except Exception:
            return []

    def _adapter_get(self, *, file_id_or_path: str) -> Dict[str, Any]:
        """
        Return adapter metadata for a file identified by id or path.

        Parameters
        ----------
        file_id_or_path : str
            Adapter-native identifier or path for the file.

        Returns
        -------
        dict
            Adapter metadata for the file (shape is adapter-specific).
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for direct file lookups")
        # Use path as-is (adapters handle both relative and absolute paths)
        ref = self._adapter.get_file(file_id_or_path)
        return getattr(ref, "model_dump", lambda: ref.__dict__)()

    def _adapter_open_bytes(self, *, file_id_or_path: str) -> Dict[str, Any]:
        """
        Open raw file bytes via the adapter and return a safe payload.

        Parameters
        ----------
        file_id_or_path : str
            Adapter-native identifier or path for the file.

        Returns
        -------
        dict
            Either a base64-encoded payload with key "bytes_b64" or a fallback
            including the byte length.
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for opening file bytes")
        # Use path as-is (adapters handle both relative and absolute paths)
        data = self._adapter.open_bytes(file_id_or_path)
        # Return as base64-ish payload for safety; caller can decide how to use
        try:
            import base64

            return {
                "file_path": file_id_or_path,
                "bytes_b64": base64.b64encode(data).decode("utf-8"),
            }
        except Exception:
            return {"file_path": file_id_or_path, "length": len(data)}

    def _open_bytes_by_filepath(self, file_path: str) -> bytes:
        """
        Return file bytes by consulting the adapter.

        Parameters
        ----------
        file_path : str
            Logical path/identifier used by the index and adapter.

        Returns
        -------
        bytes
            Raw file bytes.

        Raises
        ------
        FileNotFoundError
            When neither metadata resolution nor adapter fallback can provide bytes.
        """
        # Resolve via adapter when available
        if self._adapter is not None:
            # Use path as-is (adapters handle both relative and absolute paths)
            return self._adapter.open_bytes(file_path)
        raise FileNotFoundError(f"Unable to resolve file bytes for '{file_path}'")

    # ---------- Adapter-backed mutators (capability-guarded) --------------- #
    def rename_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_name: str,
    ) -> Dict[str, Any]:
        """
        Rename a file in the underlying filesystem and update index/context metadata.

        This tool renames the file on disk AND updates all associated metadata
        in the index. The file stays in its current directory; only the filename
        changes. For moving to a different directory, use move_file instead.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) from FileRecords, or the fully-qualified
            file_path (str) as stored in the index. Use absolute paths for reliability.
        new_name : str
            New filename (not full path). Must include the file extension.
            Example: "report_2024.pdf", not "/new/path/report.pdf"

        Returns
        -------
        dict
            Result containing the new path and any adapter-specific metadata.
            {"file_path": str, "file_name": str, ...}

        Raises
        ------
        PermissionError
            If the file is protected or rename is not permitted.
        ValueError
            If the file does not exist in the index.

        Usage Examples
        --------------
        # Rename by file path:
        rename_file(
            file_id_or_path="/path/to/old_name.pdf",
            new_name="new_name.pdf"
        )

        # Rename by file_id:
        rename_file(file_id_or_path=42, new_name="renamed_file.xlsx")

        # Rename with extension change (if adapter permits):
        rename_file(
            file_id_or_path="/path/to/document.doc",
            new_name="document.docx"
        )

        Anti-patterns
        -------------
        - WRONG: new_name="/full/path/to/file.pdf" (including directory)
          CORRECT: new_name="file.pdf" (filename only)

        - WRONG: Renaming without verifying the file exists first
          CORRECT: Use stat() or ask() to verify file exists before renaming

        - WRONG: new_name="file" (missing extension)
          CORRECT: new_name="file.pdf" (include extension)

        - WRONG: Renaming protected files
          CORRECT: Check is_protected() first or handle PermissionError
        """
        from .utils.ops import rename_file as _ops_rename

        return _ops_rename(
            self,
            file_id_or_path=file_id_or_path,
            new_name=str(new_name),
        )

    def move_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_parent_path: str,
    ) -> Dict[str, Any]:
        """
        Move a file to a different directory and update index/context metadata.

        This tool moves the file on disk AND updates all associated metadata
        in the index. The filename stays the same; only the directory changes.
        For renaming the file, use rename_file instead.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) from FileRecords, or the fully-qualified
            file_path (str) as stored in the index. Use absolute paths for reliability.
        new_parent_path : str
            Destination directory path. Must be an absolute path to an existing
            directory. Example: "/new/destination/folder"

        Returns
        -------
        dict
            Result containing the new path and any adapter-specific metadata.
            {"file_path": str, ...}

        Raises
        ------
        PermissionError
            If the file is protected or move is not permitted.
        ValueError
            If the file does not exist in the index.

        Usage Examples
        --------------
        # Move file to a different directory:
        move_file(
            file_id_or_path="/path/to/file.pdf",
            new_parent_path="/archive/2024"
        )

        # Move by file_id:
        move_file(file_id_or_path=42, new_parent_path="/processed")

        # Organize files into folders:
        # First query files to move:
        files = filter_files(filter="status == 'success' and file_format == 'pdf'")
        for f in files:
            move_file(file_id_or_path=f["file_path"], new_parent_path="/documents/pdfs")

        Anti-patterns
        -------------
        - WRONG: new_parent_path="/destination/newname.pdf" (including filename)
          CORRECT: new_parent_path="/destination" (directory only)

        - WRONG: Moving to a non-existent directory
          CORRECT: Ensure destination directory exists first

        - WRONG: Moving without verifying the file exists
          CORRECT: Use stat() or ask() to verify file before moving

        - WRONG: Cross-filesystem moves (moving between different adapters)
          CORRECT: Move only within the same filesystem adapter
        """
        from .utils.ops import move_file as _ops_move

        return _ops_move(
            self,
            file_id_or_path=file_id_or_path,
            new_parent_path=str(new_parent_path),
        )

    def delete_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        _log_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Delete a file from the filesystem and purge all related index data.

        This tool removes the file from disk (when adapter supports it) AND
        purges all associated index entries, Content rows, and per-file tables.
        This is a DESTRUCTIVE operation that cannot be undone.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) from FileRecords, or the fully-qualified
            file_path (str) as stored in the index. Use absolute paths for reliability.
        _log_id : int | None
            Internal parameter; do not use directly.

        Returns
        -------
        dict
            {"outcome": "file deleted", "details": {"file_id": int, "file_path": str}}

        Raises
        ------
        ValueError
            If the file does not exist in the index.
        PermissionError
            If the file is protected.

        Usage Examples
        --------------
        # Delete by file path:
        delete_file(file_id_or_path="/path/to/obsolete_file.pdf")
        # Returns: {"outcome": "file deleted", "details": {...}}

        # Delete by file_id:
        delete_file(file_id_or_path=42)

        # Clean up failed ingests:
        failed = filter_files(filter="status == 'error'")
        for f in failed:
            delete_file(file_id_or_path=f["file_path"])

        # Delete outdated files:
        old_files = filter_files(filter="created_at < '2023-01-01'")
        for f in old_files:
            delete_file(file_id_or_path=f["file_path"])

        Anti-patterns
        -------------
        - WRONG: Deleting without verifying the file first
          CORRECT: Use ask() or stat() to confirm file identity before deleting

        - WRONG: Attempting to delete protected files without handling errors
          CORRECT: Check is_protected() first or catch PermissionError

        - WRONG: Deleting files that other processes depend on
          CORRECT: Verify file is not referenced elsewhere before deletion

        - WRONG: Bulk deletion without confirmation
          CORRECT: For bulk operations, verify the filter returns expected files first
        """
        from .utils.ops import delete_file as _ops_delete

        return _ops_delete(self, file_id_or_path=file_id_or_path, _log_id=_log_id)

    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        """
        Check if a file exists in the adapter-backed filesystem.

        This checks ONLY the raw filesystem, NOT the index. Use this to answer
        "does this path currently exist on disk?" regardless of whether it has
        been parsed or indexed.

        For complete status (filesystem + index + parse status), use stat() instead.

        Parameters
        ----------
        file_path : str
            Adapter-native file path/identifier to check. Use absolute paths
            for reliable results.

        Returns
        -------
        bool
            True if the adapter reports the file exists, False otherwise or
            when no adapter is configured.

        Usage Examples
        --------------
        # Check if a file exists on disk:
        if exists("/path/to/document.pdf"):
            # File is present in filesystem
            pass

        # Verify before attempting to ingest:
        if exists("/path/to/new_file.xlsx"):
            ingest_files("/path/to/new_file.xlsx")

        # Check multiple files:
        files = ["/path/to/a.pdf", "/path/to/b.pdf"]
        existing = [f for f in files if exists(f)]

        Anti-patterns
        -------------
        - WRONG: Using exists() to check if a file has been indexed
          CORRECT: Use stat() to check indexed_exists

        - WRONG: Assuming exists() == True means content is queryable
          CORRECT: File must also be indexed; use stat() for complete status

        - WRONG: Using relative paths without knowing the adapter's working directory
          CORRECT: Use absolute paths for reliable results
        """
        if self._adapter is None:
            return False
        try:
            # Use path as-is (adapters handle both relative and absolute paths)
            ok = self._adapter.exists(file_path)
        except Exception:
            return False
        return bool(ok)

    def list(self) -> List[str]:  # type: ignore[override]
        """
        List all files visible to the underlying adapter.

        Returns file paths from the RAW FILESYSTEM, not the index. Use this to
        discover what files are available for ingestion or to compare with
        indexed files.

        For querying indexed files, use filter_files() on the FileRecords table.

        Returns
        -------
        list[str]
            Adapter-native file paths/identifiers discoverable in the underlying
            filesystem, or an empty list if no adapter is configured or listing fails.

        Usage Examples
        --------------
        # List all files in the filesystem:
        files = list()
        # Returns: ["/path/to/file1.pdf", "/path/to/file2.xlsx", ...]

        # Find files not yet indexed:
        filesystem_files = set(list())
        indexed = filter_files()
        indexed_paths = {r["file_path"] for r in indexed}
        not_indexed = filesystem_files - indexed_paths

        # Count available files:
        file_count = len(list())

        Anti-patterns
        -------------
        - WRONG: Using list() to query file metadata (status, parse info)
          CORRECT: Use filter_files() on FileRecords for indexed file queries

        - WRONG: Expecting list() to show only successfully parsed files
          CORRECT: list() shows raw filesystem; filter_files(filter="status == 'success'")
                   shows successfully indexed files

        - WRONG: Using list() for large filesystems then filtering in Python
          CORRECT: Use filter_files() with appropriate filters for efficient queries
        """
        if self._adapter is None:
            return []
        try:
            items = self._adapter.list()
        except Exception:
            return []
        return list(items or [])

    def ingest_files(self, file_paths: Union[str, List[str]], *, config: Optional[_FilePipelineConfig] = None) -> "IngestPipelineResult":  # type: ignore[override]
        """
        Run the complete file processing pipeline: parse, ingest, and embed.

        This method orchestrates the full file processing workflow:
        1. Parse files using the configured parser to extract structured content
        2. Ingest parsed content into storage contexts (per-file or unified)
        3. Create embeddings based on the configured strategy (along, after, or off)

        Parameters
        ----------
        file_paths : str | list[str]
            One or more logical file paths to process.
        config : FilePipelineConfig | None
            Pipeline configuration controlling parser routing/concurrency,
            ingest layout, table ingestion and embedding behavior. When None,
            defaults are used (equivalent to ``FilePipelineConfig()``).

        Returns
        -------
        IngestPipelineResult
            Structured container with per-file ingest results (IngestedPDF, IngestedXlsx, etc.)
            and global pipeline statistics. Supports dict-like access: result[file_path].
        """
        cfg = config or _FilePipelineConfig()

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Get return mode for error handling
        return_mode = getattr(getattr(cfg, "output", None), "return_mode", "compact")

        # Track error results for files that fail early (before pipeline)
        error_results: Dict[str, FileResultType] = {}
        exported_paths: List[str] = []
        exported_paths_to_original_paths: Dict[str, str] = {}

        temp_dir: Optional[str] = None
        try:
            # Initialize progress reporter when enabled
            from .utils.progress import create_reporter, ProgressReporter

            _reporter: Optional[ProgressReporter] = None

            import tempfile as _tempfile

            temp_dir = _tempfile.mkdtemp(prefix="filemanager_ingest_files_")

            # Resolve parse inputs: prefer in-place local path; otherwise export via adapter
            from pathlib import Path as _P

            for path in file_paths:
                try:
                    p = _P(str(path)).expanduser()
                    if p.is_absolute() and p.exists():
                        exported_path = str(p)
                        exported_paths.append(exported_path)
                        exported_paths_to_original_paths[exported_path] = path
                        continue
                except Exception:
                    pass
                try:
                    exported_path = self.export_file(path, temp_dir)
                    exported_paths.append(exported_path)
                    exported_paths_to_original_paths[exported_path] = path
                except Exception as e:
                    # Per-file export failure → do not fail the entire tool call
                    # ALL returns are Pydantic models
                    if return_mode == "full":
                        from unity.file_manager.types.ingest import IngestedFullFile

                        error_results[path] = IngestedFullFile(
                            file_path=path,
                            status="error",
                            error=f"export failed: {e}",
                        )
                    elif return_mode == "none":
                        error_results[path] = IngestedMinimal(
                            file_path=path,
                            status="error",
                            error=f"export failed: {e}",
                            total_records=0,
                            file_format=None,
                        )
                    else:
                        error_results[path] = BaseIngestedFile(
                            file_path=path,
                            status="error",
                            error=f"export failed: {e}",
                            content_ref=ContentRef(
                                context="",
                                record_count=0,
                                text_chars=0,
                            ),
                            metrics=FileMetrics(),
                        )

            # Nothing exported successfully
            if not exported_paths:
                return IngestPipelineResult.from_results(error_results)

            enable_progress = bool(
                getattr(getattr(cfg, "diagnostics", None), "enable_progress", False),
            )

            # Create progress reporter based on config
            if enable_progress:
                progress_mode = getattr(cfg.diagnostics, "progress_mode", "json_file")
                progress_file = getattr(cfg.diagnostics, "progress_file", None)
                verbosity = getattr(cfg.diagnostics, "verbosity", "low")
                try:
                    _reporter = create_reporter(
                        mode=progress_mode,
                        file_path=progress_file,
                        verbosity=verbosity,
                        append=False,  # Clear file at pipeline start for fresh logs
                    )
                except Exception:
                    _reporter = None

            # Parse files using the new FileParser facade.
            # Note: we parse from the exported (local) path, but the FileParseResult logical_path
            # is always the ORIGINAL logical path so contexts and business-context matching work.
            from unity.file_manager.file_parsers.types.contracts import FileParseResult

            parse_results: list[FileParseResult] = []
            try:
                import time as _time

                parse_start_time = _time.perf_counter()

                # Report parse started for all files
                if enable_progress and _reporter is not None:
                    from .utils.progress import create_progress_event

                    for exp in exported_paths:
                        orig = exported_paths_to_original_paths.get(exp, exp)
                        _reporter.report(
                            create_progress_event(
                                orig,
                                "parse",
                                "started",
                                duration_ms=0.0,
                                elapsed_ms=0.0,
                            ),
                        )

                from unity.file_manager.file_parsers.types.contracts import (
                    FileParseRequest,
                )

                parse_requests: list[FileParseRequest] = []
                for exp in exported_paths:
                    orig = exported_paths_to_original_paths.get(exp, exp)
                    parse_requests.append(
                        FileParseRequest(
                            logical_path=orig,
                            source_local_path=exp,
                        ),
                    )

                parse_results = self._parser.parse_batch(
                    parse_requests,
                    raises_on_error=False,
                    parse_config=cfg.parse,
                )

                parse_duration_ms = (_time.perf_counter() - parse_start_time) * 1000

                # Report per-file parse completion/failure
                if enable_progress and _reporter is not None:
                    from .utils.progress import create_progress_event

                    for pr in parse_results:
                        try:
                            orig = str(getattr(pr, "logical_path", "") or "")
                            status = str(getattr(pr, "status", "error"))
                            if status == "success":
                                _reporter.report(
                                    create_progress_event(
                                        orig,
                                        "parse",
                                        "completed",
                                        duration_ms=parse_duration_ms,
                                        elapsed_ms=parse_duration_ms,
                                    ),
                                )
                            else:
                                _reporter.report(
                                    create_progress_event(
                                        orig,
                                        "parse",
                                        "failed",
                                        duration_ms=parse_duration_ms,
                                        elapsed_ms=parse_duration_ms,
                                        error=str(getattr(pr, "error", "") or ""),
                                    ),
                                )
                        except Exception:
                            continue

            except Exception as e:
                parse_duration_ms = (
                    (_time.perf_counter() - parse_start_time) * 1000
                    if "parse_start_time" in dir()
                    else 0.0
                )
                # Catastrophic parse failure → mark all remaining as errors
                from unity.file_manager.file_parsers.types.contracts import (
                    FileParseResult,
                )

                for exp in exported_paths:
                    original_path = exported_paths_to_original_paths.get(exp, exp)
                    if original_path in error_results:
                        continue
                    if return_mode == "full":
                        error_results[original_path] = FileParseResult(
                            logical_path=original_path,
                            status="error",
                            error=f"parse failed: {e}",
                        )
                    elif return_mode == "none":
                        error_results[original_path] = IngestedMinimal(
                            file_path=original_path,
                            status="error",
                            error=f"parse failed: {e}",
                            total_records=0,
                            file_format=None,
                        )
                    else:
                        error_results[original_path] = BaseIngestedFile(
                            file_path=original_path,
                            status="error",
                            error=f"parse failed: {e}",
                            content_ref=ContentRef(
                                context="",
                                record_count=0,
                                text_chars=0,
                            ),
                            metrics=FileMetrics(
                                processing_time=parse_duration_ms / 1000.0,
                            ),
                        )
                return IngestPipelineResult.from_results(error_results)

            # Split success vs parse errors (parse errors are handled like export errors)
            successful_parse_results: List[Any] = []
            successful_paths: List[str] = []
            for pr in parse_results:
                orig = str(getattr(pr, "logical_path", "") or "")
                status = str(getattr(pr, "status", "error"))
                if status != "success":
                    if orig and orig not in error_results:
                        if return_mode == "full":
                            error_results[orig] = pr
                        elif return_mode == "none":
                            error_results[orig] = IngestedMinimal(
                                file_path=orig,
                                status="error",
                                error=str(getattr(pr, "error", "") or ""),
                                total_records=0,
                                file_format=None,
                            )
                        else:
                            error_results[orig] = BaseIngestedFile(
                                file_path=orig,
                                status="error",
                                error=str(getattr(pr, "error", "") or ""),
                                content_ref=ContentRef(
                                    context="",
                                    record_count=0,
                                    text_chars=0,
                                ),
                                metrics=FileMetrics(
                                    processing_time=(parse_duration_ms / 1000.0),
                                ),
                            )
                    continue
                successful_parse_results.append(pr)
                successful_paths.append(orig)

            if not successful_parse_results:
                return IngestPipelineResult.from_results(error_results)

            # Build results and ingest per-file artifacts using the PipelineExecutor
            from .utils.executor import run_pipeline

            # run_pipeline handles parallel vs sequential execution based on
            # cfg.execution.parallel_files internally
            verbosity = getattr(cfg.diagnostics, "verbosity", "low")
            pipeline_result = run_pipeline(
                self,
                parse_results=successful_parse_results,
                file_paths=successful_paths,
                config=cfg,
                reporter=_reporter,
                enable_progress=enable_progress,
                verbosity=verbosity,
            )

            # Merge error results (from export/parse failures) with pipeline results
            if error_results:
                all_files = {**error_results, **pipeline_result.files}
                return IngestPipelineResult.from_results(
                    all_files,
                    total_duration_ms=pipeline_result.statistics.total_duration_ms,
                )

            return pipeline_result
        finally:
            # Clean up temporary directory
            if temp_dir:
                try:
                    import shutil as _shutil2

                    _shutil2.rmtree(temp_dir)
                except Exception:
                    pass
            # Flush progress reporter if created
            try:
                if _reporter is not None:
                    _reporter.flush()
            except Exception:
                pass

    # Note: parse_async has been removed - use ingest_files() which handles
    # async processing internally via the PipelineExecutor when configured.
    # ---------- Unify table helpers (schema + retrieval) ------------------ #
    @read_only
    def _get_columns(self) -> Dict[str, str]:
        """
        Return a mapping of column names to their Unify data types for the index.

        Returns
        -------
        dict[str, str]
            Column name → type mapping for the index context.
        """
        return _storage_get_columns(self)

    @read_only
    def tables_overview(
        self,
        *,
        include_column_info: bool = True,
        file: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return an overview of available tables/contexts managed by this FileManager.

        .. deprecated::
            Use describe(file_path=...) instead for file-specific discovery.
            describe() returns FileStorageMap with exact context paths, schemas,
            and file_id for stable cross-referencing.

        Use this for discovery when you need to understand what data is available.
        Returns logical table names that can be passed to other tools.

        Spreadsheet retrieval flow (XLSX/CSV)
        ------------------------------------
        For spreadsheets, `/Content/` is not a document-section hierarchy. Instead:
        - `content_type='sheet'` rows describe sheets.
        - `content_type='table'` rows are *table catalog rows* (profile + bounded samples + summary)
          that help you discover the right `/Tables/<label>` context to query.

        Recommended pattern:
        1) `tables_overview(file=...)` to see available `/Tables/<label>` contexts
        2) `search_files(..., table='<file_path>', filter=\"content_type in ['sheet','table']\")`
        3) Query the chosen table context: `...table='<file_path>.Tables.<label>'`

        Parameters
        ----------
        include_column_info : bool, default True
            When True and ``file`` is None, include the index schema (columns→types).
        file : str | None, default None
            When None: returns ONLY the global FileRecords index overview.
            When provided: returns file-scoped overview with Content and Tables
            for that specific file (respecting its ingest mode).

        Returns
        -------
        dict[str, dict]
            - file=None: {"FileRecords": {context, description, columns?}}
            - file=<path> (per_file mode):
              {"FileRecords": {...}, "<safe(file)>": {"Content": {...}, "Tables": {...}}}
            - file=<path> (unified mode):
              {"FileRecords": {...}, "<unified_label>": {"Content": {...}},
               "<safe(file)>": {"Tables": {...}}}

        Usage Examples
        --------------
        # Get global FileRecords index info:
        tables_overview()
        # Returns: {"FileRecords": {"context": "...", "columns": {...}}}

        # Get file-specific overview (per_file mode):
        tables_overview(file="/path/to/data.xlsx")
        # Returns: {"FileRecords": {...}, "data_xlsx": {"Content": {...}, "Tables": {"Sheet1": {...}}}}

        # Get file-specific overview (unified mode):
        tables_overview(file="/path/to/data.xlsx")
        # Returns: {"FileRecords": {...}, "Unified": {"Content": {...}}, "data_xlsx": {"Tables": {...}}}

        # Use returned table references in other tools:
        overview = tables_overview(file="/path/to/data.xlsx")
        # Then: filter_files(tables=["/path/to/data.xlsx.Tables.Sheet1"])
        """
        warnings.warn(
            "tables_overview() is deprecated. Use describe(file_path=...) instead for "
            "file-specific discovery with exact context paths, schemas, and file_id.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _storage_tables_overview(
            self,
            include_column_info=include_column_info,
            file=file,
        )

    @read_only
    def schema_explain(self, *, table: str) -> str:
        """
        Return a natural-language explanation of a table's structure and purpose.

        Use this when you need deeper semantic understanding of a table beyond
        what list_columns provides.

        Parameters
        ----------
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
            - Approximate row count

        Usage Examples
        --------------
        # Get explanation of the FileRecords index:
        schema_explain(table="FileRecords")

        # Get explanation of a per-file Table:
        schema_explain(table="/path/to/data.xlsx.Tables.Orders")

        # Get explanation of per-file Content:
        schema_explain(table="/path/to/document.pdf")
        """
        return _storage_schema_explain(
            self,
            table=table,
        )

    @read_only
    def _num_files(self) -> int:
        """
        Return the total number of files present in the index context.

        Returns
        -------
        int
            Count of file rows in the index; 0 on error.
        """
        try:
            ret = unify.get_logs_metric(
                metric="count",
                key="file_id",
                context=self._ctx,
            )
            return int(ret or 0)
        except Exception:
            return 0

    @read_only
    def reduce(
        self,
        *,
        context: Optional[str] = None,
        metric: str,
        column: str,
        filter: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        """
        Compute aggregation metrics over a context.

        This is the PRIMARY tool for any quantitative question (counts, sums,
        averages, statistics). Always use reduce instead of fetching rows and
        computing aggregates in-memory.

        IMPORTANT: Use describe() first to get the exact context path:

        >>> storage = describe(file_path="/reports/sales.xlsx")
        >>> count = reduce(
        ...     context=storage.tables[0].context_path,
        ...     metric='count',
        ...     column='id'
        ... )

        Parameters
        ----------
        context : str, optional
            Full Unify context path to aggregate. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, aggregates over the FileRecords index.

        metric : str
            Reduction metric to compute. Supported values (case-insensitive):
            "count", "sum", "mean", "min", "max", "median", "mode", "var", "std".

        column : str
            Column to aggregate. Check column_schema.column_names from describe()
            for available columns.

        filter : str, optional
            Optional row-level filter expression to narrow the dataset
            BEFORE aggregation. Supports Python expressions evaluated per row.

        group_by : str | list[str], optional
            Column(s) to group by. Single column for one grouping level,
            or a list for hierarchical grouping. Result becomes a nested dict
            keyed by group values.

        Returns
        -------
        Any
            Metric value(s) computed over the context:
            - No grouping → scalar (float/int)
            - With group_by → dict keyed by group values
            - Hierarchical grouping → nested dict

        Usage Examples
        --------------
        First, always call describe() to get context paths:

        >>> storage = describe(file_path="/reports/sales.xlsx")
        >>> table_ctx = storage.tables[0].context_path

        Count rows in a table:

        >>> count = reduce(context=table_ctx, metric='count', column='id')
        >>> # Returns: 42

        Count with filter:

        >>> count = reduce(
        ...     context=table_ctx,
        ...     metric='count',
        ...     column='id',
        ...     filter="status == 'complete'"
        ... )
        >>> # Returns: 38

        Sum a numeric column:

        >>> total = reduce(
        ...     context=table_ctx,
        ...     metric='sum',
        ...     column='amount',
        ...     filter="status == 'complete'"
        ... )
        >>> # Returns: 15420.50

        Average with grouping:

        >>> avg_by_category = reduce(
        ...     context=table_ctx,
        ...     metric='mean',
        ...     column='amount',
        ...     group_by='category'
        ... )
        >>> # Returns: {'Electronics': 245.00, 'Furniture': 890.50, ...}

        Hierarchical grouping:

        >>> breakdown = reduce(
        ...     context=table_ctx,
        ...     metric='sum',
        ...     column='revenue',
        ...     group_by=['region', 'quarter']
        ... )
        >>> # Returns: {'North': {'Q1': 1000, 'Q2': 1200}, 'South': {...}}

        Count files in FileRecords index (no context needed):

        >>> total_files = reduce(metric='count', column='file_id')

        Anti-patterns
        -------------
        - WRONG: Guessing context paths

          >>> reduce(context="Files/Local/sales.xlsx/Tables/Orders", ...)

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/sales.xlsx")
          >>> reduce(context=storage.tables[0].context_path, ...)

        - WRONG: filter_files(...) then count rows in Python

          CORRECT: reduce(metric='count', column='id', filter=...)

        - WRONG: filter_files(...) then sum values in Python

          CORRECT: reduce(metric='sum', column='amount', filter=...)

        - WRONG: Using reduce for text/categorical analysis

          CORRECT: Use search_files for semantic queries

        Notes
        -----
        - Context paths use file_id internally for stability across renames.
        - Check column_schema.column_names from describe() for available columns.
        - Use filter to narrow the dataset BEFORE aggregation for efficiency.

        See Also
        --------
        describe : Get context paths and schema for a file
        filter_files : Fetch rows with exact match filters
        search_files : Semantic search for meaning-based queries
        """
        # Resolve context - default to FileRecords index
        ctx = context if context else self._ctx

        # Delegate to DataManager for the actual reduction
        return self._data_manager.reduce(
            context=ctx,
            metric=metric,
            columns=column,
            filter=filter,
            group_by=group_by,
        )

    @read_only
    def visualize(
        self,
        *,
        tables: Union[str, List[str]],
        plot_type: str,
        x_axis: str,
        y_axis: Optional[str] = None,
        group_by: Optional[str] = None,
        filter: Optional[str] = None,
        title: Optional[str] = None,
        metric: Optional[str] = None,
        aggregate: Optional[str] = None,
        scale_x: Optional[str] = None,
        scale_y: Optional[str] = None,
        bin_count: Optional[int] = None,
        show_regression: Optional[bool] = None,
    ) -> Union["_VizPlotResult", List["_VizPlotResult"]]:
        """
        Generate plot visualizations from table data via the Plot API.

        This tool creates interactive charts from the specified table(s). Before
        calling, use `list_columns` or `tables_overview` to discover available
        columns and their types.

        Parameters
        ----------
        tables : str | list[str]
            Table reference(s) to visualize. Accepted forms:
            - "<file_path>.Tables.<label>" for per-file tables
            - "<file_path>" for per-file Content

            When a LIST is provided, the same plot configuration is applied to
            EACH table. Use this for tables with identical schemas that you want
            to compare (e.g., monthly data tables like "July_2025", "August_2025").
            Each table produces a separate plot with the table name in the title.

            For tables with DIFFERENT schemas, make separate _visualize calls.

        plot_type : str
            Chart type. One of: "bar", "line", "scatter", "histogram".
            - bar: Compare values across categories
            - line: Show trends over time or sequences
            - scatter: Show correlations between two numeric variables
            - histogram: Show distribution of a single variable

        x_axis : str
            Column name for the x-axis. Must exist in the table schema.

        y_axis : str | None
            Column name for the y-axis. Required for bar, line, scatter.
            Optional for histogram (uses count by default).

        group_by : str | None
            Column to group/color data points by. Creates multiple series.

        filter : str | None
            Row-level filter expression. Same syntax as `reduce` and `filter_files`.
            Applied before plotting.

        title : str | None
            Plot title. Auto-generated if not provided.

        metric : str | None
            Metric to aggregate: "sum", "mean", "var", "std", "min", "max", "median", "mode", "count".

        aggregate : str | None
            Aggregation function for grouped data: "sum", "mean", "count", "min", "max",
            "median", "mode", "var", "std". Use when comparing aggregated values across
            groups (e.g., mean completion rate by region).

        scale_x, scale_y : str | None
            Axis scale: "linear" (default) or "log".

        bin_count : int | None
            Number of bins for histogram plots.

        show_regression : bool | None
            Show regression line (scatter plots only).

        Returns
        -------
        PlotResult | list[PlotResult]
            Single table → PlotResult with attrs: url, token, expires_in_hours, title, error
            Multiple tables → list of PlotResult, one per table

            On success: result.url contains the plot URL, result.succeeded is True
            On failure: result.error contains error message, result.succeeded is False

        Usage Examples
        --------------
        # Bar chart of job counts by operative:
        visualize(
            tables='/path/to/data.xlsx.Tables.Jobs',
            plot_type='bar',
            x_axis='OperativeName',
            y_axis='JobCount',
            title='Jobs per Operative'
        )

        # Line chart showing trend over time:
        visualize(
            tables='/path/to/data.xlsx.Tables.Sales',
            plot_type='line',
            x_axis='Date',
            y_axis='Revenue',
            group_by='Region'
        )

        # Same plot across multiple monthly tables (identical schemas):
        visualize(
            tables=[
                '/path/to/data.xlsx.Tables.July_2025',
                '/path/to/data.xlsx.Tables.August_2025',
                '/path/to/data.xlsx.Tables.September_2025',
            ],
            plot_type='bar',
            x_axis='Driver',
            y_axis='TotalDistance',
            metric='sum'
        )
        # Returns: [PlotResult(url="...", title="... (July_2025)"), ...]

        # Histogram of value distribution:
        visualize(
            tables='/path/to/data.xlsx.Tables.Orders',
            plot_type='histogram',
            x_axis='OrderValue',
            bin_count=20
        )

        # Scatter with regression line:
        visualize(
            tables='/path/to/data.xlsx.Tables.Metrics',
            plot_type='scatter',
            x_axis='TimeOnSite',
            y_axis='JobsCompleted',
            show_regression=True
        )

        Anti-patterns
        -------------
        - WRONG: Calling visualize without first checking column names
          CORRECT: Use list_columns(table='...') first to discover schema

        - WRONG: Passing tables with different schemas in same call
          CORRECT: Use separate visualize calls for different schemas

        - WRONG: Using non-existent column names
          CORRECT: Column names must match exactly (case-sensitive)

        - WRONG: histogram with y_axis (histogram uses x_axis distribution)
          CORRECT: For histogram, only specify x_axis
        """
        # Import DataManager types for compatibility
        from unity.data_manager.types import PlotConfig as DMPlotConfig

        # Normalize tables to list
        table_list: List[str] = []
        if isinstance(tables, str):
            if tables:
                table_list = [tables]
        else:
            table_list = [t for t in tables if t]

        if not table_list:
            return _VizPlotResult(error="No tables provided", title=title or "Untitled")

        # Resolve each table to a fully-qualified context
        contexts: List[str] = []
        for tbl in table_list:
            try:
                ctx = self._resolve_table_ref(tbl)
                contexts.append(ctx)
            except Exception as e:
                # If we can't resolve, return error immediately
                return _VizPlotResult(
                    error=f"Failed to resolve table '{tbl}': {e}",
                    title=title or "Untitled",
                    table=tbl,
                )

        # Build plot config using DataManager types
        dm_config = DMPlotConfig(
            plot_type=plot_type,
            x_axis=x_axis,
            y_axis=y_axis,
            group_by=group_by,
            metric=metric,
            aggregate=aggregate,
            scale_x=scale_x,
            scale_y=scale_y,
            bin_count=bin_count,
            show_regression=show_regression,
            title=title,
            filter=filter,
        )

        # Delegate to DataManager for plot generation
        dm_results = self._data_manager.plot_batch(contexts=contexts, config=dm_config)

        # Convert DataManager results to FileManager result type for backward compat
        results: List[_VizPlotResult] = []
        for dm_res in dm_results:
            results.append(
                _VizPlotResult(
                    url=dm_res.url,
                    token=dm_res.token,
                    expires_in_hours=dm_res.expires_in_hours,
                    title=dm_res.title,
                    error=dm_res.error,
                    table=dm_res.context,
                ),
            )

        # Return PlotResult directly (single) or list of PlotResult (multiple)
        if len(results) == 1:
            return results[0]
        return results

    @read_only
    def list_columns(
        self,
        *,
        include_types: bool = True,
        context: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        """
        List columns for the FileRecords index or a specific context.

        Use this to inspect a context's schema before writing filter expressions
        or selecting columns for retrieval.

        IMPORTANT: Use describe() first to get the exact context path, then use
        list_columns() for detailed schema inspection if needed:

        >>> storage = describe(file_path="/reports/Q4.csv")
        >>> columns = list_columns(context=storage.tables[0].context_path)

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column → type (e.g., {"name": "str"}).
            When False, return just the list of column names.
        context : str | None, default None
            Full Unify context path to inspect. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, returns the FileRecords index columns.

        Returns
        -------
        dict[str, str] | list[str]
            Column→type mapping (when include_types=True) or list of column names.

        Usage Examples
        --------------
        # Get FileRecords index schema:
        list_columns()
        # Returns: {"file_id": "int", "file_path": "str", "status": "str", ...}

        # Get columns for a context from describe():
        storage = describe(file_path="/path/to/document.pdf")
        list_columns(context=storage.document.context_path)
        # Returns: {"content_id": "dict", "content_type": "str", "content_text": "str", ...}

        # Get columns for a table context:
        storage = describe(file_path="/path/to/data.xlsx")
        list_columns(context=storage.tables[0].context_path)
        # Returns: {"order_id": "str", "amount": "float", "customer": "str", ...}

        # Get just column names (no types):
        list_columns(context=storage.tables[0].context_path, include_types=False)
        # Returns: ["order_id", "amount", "customer", ...]
        """
        if context is None:
            cols = self._get_columns()
            return cols if include_types else list(cols)
        # Use context path directly (already resolved via describe())
        cols = _storage_get_columns(self, table=context)
        return cols if include_types else list(cols)

    @read_only
    def filter_files(
        self,
        *,
        context: Optional[str] = None,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter rows from a context using exact match expressions.

        Use this tool for exact matches on structured fields (ids, statuses, dates).
        For semantic/meaning-based search, use search_files() instead.

        IMPORTANT: Use describe() first to get the exact context path:

        >>> storage = describe(file_path="/reports/Q4.csv")
        >>> results = filter_files(
        ...     context=storage.tables[0].context_path,
        ...     filter="revenue > 1000000"
        ... )

        Parameters
        ----------
        context : str, optional
            Full Unify context path to filter. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, defaults to the FileRecords index.

        filter : str, optional
            Python expression evaluated per row with column names in scope.
            Any valid Python syntax returning a boolean is supported.
            String values must be quoted. Use .get() for safe dict access.

        offset : int, default 0
            Zero-based pagination offset.

        limit : int, default 100
            Maximum rows to return.

        columns : list[str], optional
            Specific columns to return. When None, returns all columns.
            Use this to reduce response size when only specific fields are needed.

        Returns
        -------
        list[dict]
            Flat list of matching rows from the context.

        Per-file Content: content_id structure
        --------------------------------------
        The `content_id` column in per-file Content encodes document hierarchy:
        - document row: {"document": 0}
        - section row: {"document": 0, "section": 2}
        - paragraph row: {"document": 0, "section": 2, "paragraph": 1}
        - sentence row: {"document": 0, "section": 2, "paragraph": 1, "sentence": 3}
        - table row: {"document": 0, "section": 2, "table": 0}
        - image row: {"document": 0, "section": 2, "image": 0}

        Use .get() accessor for safe filtering:
        - filter="content_id.get('section') == 2"
        - filter="content_id.get('document') == 0 and content_type == 'paragraph'"

        Usage Examples
        --------------
        First, always call describe() to get context paths:

        >>> storage = describe(file_path="/reports/data.xlsx")

        Filter a table from the file:

        >>> results = filter_files(
        ...     context=storage.tables[0].context_path,
        ...     filter="created_at >= '2024-01-01' and created_at < '2024-02-01'",
        ...     columns=["id", "name", "created_at"]
        ... )

        Filter document content by hierarchy:

        >>> storage = describe(file_path="/docs/report.pdf")
        >>> results = filter_files(
        ...     context=storage.document.context_path,
        ...     filter="content_type == 'paragraph' and content_id.get('section') == 2"
        ... )

        Filter the FileRecords index (no context needed):

        >>> results = filter_files(filter="status == 'success'")

        Paginate through results:

        >>> page1 = filter_files(context=ctx, filter="...", offset=0, limit=30)
        >>> page2 = filter_files(context=ctx, filter="...", offset=30, limit=30)

        Anti-patterns
        -------------
        - WRONG: Guessing context paths

          >>> filter_files(context="Files/Local/reports/Q4.csv/Tables/Sheet1")

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/reports/Q4.csv")
          >>> filter_files(context=storage.tables[0].context_path, ...)

        - WRONG: Using filter for meaning-based search

          >>> filter_files(filter="description.contains('budget')")

          CORRECT: Use search_files(references={'description': 'budget'})

        - WRONG: Fetching rows just to count them

          CORRECT: Use reduce(metric='count', keys='id') instead

        - WRONG: filter="content_id['section'] == 2" (direct dict indexing fails)

          CORRECT: filter="content_id.get('section') == 2"

        Notes
        -----
        - Context paths use file_id internally (e.g., Files/Local/42/Tables/Sheet1)
          for stability across file renames. Always use describe() to get paths.
        - For large result sets, use pagination (offset/limit) to avoid memory issues.
        - Check column_schema.column_names from describe() to see available columns.

        See Also
        --------
        describe : Get context paths and schema for a file
        search_files : Semantic search (for meaning-based queries)
        reduce : Aggregate metrics without fetching rows
        """
        # Resolve context - default to FileRecords index
        ctx = context if context else self._ctx

        # Delegate to DataManager for the actual filtering
        return self._data_manager.filter(
            context=ctx,
            filter=filter,
            columns=columns,
            limit=limit,
            offset=offset,
        )

    @read_only
    def search_files(
        self,
        *,
        context: Optional[str] = None,
        references: Optional[Dict[str, str]] = None,
        limit: int = 10,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over a context using reference texts for similarity matching.

        Use this tool when searching by meaning, topics, or concepts in text fields.
        For exact matches on structured fields (ids, statuses), use filter_files instead.

        IMPORTANT: Use describe() first to get the exact context path and check
        which columns are searchable (have embeddings):

        >>> storage = describe(file_path="/docs/report.pdf")
        >>> print(storage.document.column_schema.searchable_columns)
        ['summary', 'content_text']
        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'fire safety regulations'}
        ... )

        Parameters
        ----------
        context : str, optional
            Full Unify context path to search. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, defaults to the FileRecords index.

        references : dict[str, str], optional
            Mapping of column_name → reference_text for semantic matching.
            Only use columns that are searchable (have embeddings) - check
            column_schema.searchable_columns from describe().
            When omitted, returns recent rows without semantic ranking.

        limit : int, default 10
            Number of results to return. Start with 10, increase if needed.
            Maximum 100, but prefer smaller values to avoid context overflow.

        filter : str, optional
            Optional row-level predicate to narrow results before semantic ranking.
            Supports arbitrary Python expressions evaluated per row.

        columns : list[str], optional
            Specific columns to return. When None, returns all columns.
            Use this to reduce response size when only specific fields are needed.

        Returns
        -------
        list[dict]
            Up to `limit` rows ranked by similarity from the context.

        Usage Examples
        --------------
        First, always call describe() to get context paths and check searchability:

        >>> storage = describe(file_path="/docs/report.pdf")
        >>> print(storage.document.column_schema.searchable_columns)

        Search document content:

        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'fire safety regulations'},
        ...     limit=10
        ... )

        Search a specific table (check searchable columns first):

        >>> storage = describe(file_path="/data/telematics.xlsx")
        >>> table = storage.get_table("July_2025")
        >>> results = search_files(
        ...     context=table.context_path,
        ...     references={'Vehicle': 'Stuart Birks'},
        ...     limit=20
        ... )

        Combine semantic search with exact filter:

        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'content_text': 'payment terms'},
        ...     filter="content_type == 'paragraph'",
        ...     limit=5
        ... )

        Tiered search strategy for documents:

        >>> # 1. Paragraphs first (highest precision):
        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'quarterly metrics'},
        ...     filter="content_type == 'paragraph'",
        ...     limit=10
        ... )
        >>> # 2. Sections if paragraphs insufficient:
        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'quarterly metrics'},
        ...     filter="content_type == 'section'",
        ...     limit=5
        ... )

        Search FileRecords index (no context needed):

        >>> results = search_files(references={'summary': 'fire safety'}, limit=10)

        Anti-patterns
        -------------
        - WRONG: Guessing context paths

          >>> search_files(context="Files/Local/docs/report.pdf/Content", ...)

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/docs/report.pdf")
          >>> search_files(context=storage.document.context_path, ...)

        - WRONG: Using non-searchable columns in references

          >>> search_files(references={'id': 'search term'})  # id has no embeddings

          CORRECT: Check searchable_columns first

          >>> storage.document.column_schema.searchable_columns

        - WRONG: Using search_files for exact id/status lookup

          CORRECT: Use filter_files(filter="id == 123") for exact matches

        - WRONG: limit=500 (too many results flood context)

          CORRECT: Start with limit=10, increase only if needed

        Notes
        -----
        - Only columns with embeddings can be used in references. Check
          column_schema.searchable_columns from describe() before searching.
        - Context paths use file_id internally for stability across renames.
        - For best results, combine with filter to narrow the search scope.

        See Also
        --------
        describe : Get context paths, schema, and searchable columns
        filter_files : Exact match filtering (for structured fields)
        reduce : Aggregate metrics without fetching rows
        """
        # Resolve context - default to FileRecords index
        ctx = context if context else self._ctx

        # Delegate to DataManager for the actual search
        return self._data_manager.search(
            context=ctx,
            references=references or {},
            k=limit,
            filter=filter,
            columns=columns,
        )

    # ---------- Per-file join and multi-join tools (read-only) -------------- #
    @read_only
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
        """
        Join two sources and return rows from the joined result with optional filtering.

        Use this tool to correlate data across two tables (e.g., repairs data
        with telematics data). Push filters into left_where/right_where to
        reduce data before joining for efficiency.

        Parameters
        ----------
        tables : list[str] | str
            Exactly two table references. Accepted forms:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names from `tables_overview()`
        join_expr : str
            Join predicate using table references as prefixes.
            Example: "Repairs.operative == Telematics.driver"
            The identifiers in join_expr should match the table references in tables.
        select : dict[str, str]
            Mapping of source expressions → output column names.
            Keys use "TableRef.column" format; values are the output names.
        mode : str
            Join mode: "inner" (default), "left", "right", or "outer".
        left_where, right_where : str | None
            Optional filters applied to left/right tables BEFORE joining.
            Supports arbitrary Python expressions. Use to narrow inputs.
        result_where : str | None
            Filter applied to the joined result. Supports arbitrary Python
            expressions but can only reference column names from `select` output.
        result_limit, result_offset : int
            Pagination parameters for the result. Start with limit=30.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Rows from the materialized join context.

        Usage Examples
        --------------
        # Join repairs with telematics by operative name:
        filter_join(
            tables=[
                '/path/repairs.xlsx.Tables.Jobs',
                '/path/telematics.xlsx.Tables.July_2025'
            ],
            join_expr="Jobs.OperativeWhoCompletedJob == July_2025.Driver",
            select={
                'Jobs.JobTicketReference': 'job_ref',
                'Jobs.FullAddress': 'address',
                'July_2025.Departure': 'trip_start',
                'July_2025.Arrival': 'trip_end'
            },
            mode='inner',
            result_limit=30
        )

        # With input filters to narrow before joining:
        filter_join(
            tables=['TableA', 'TableB'],
            join_expr="TableA.id == TableB.a_id",
            select={'TableA.id': 'id', 'TableB.score': 'score'},
            left_where="TableA.status == 'active'",
            right_where="TableB.score > 0",
            result_limit=50
        )

        # Left join to keep all left rows:
        filter_join(
            tables=['Orders', 'Shipments'],
            join_expr="Orders.order_id == Shipments.order_id",
            select={'Orders.order_id': 'oid', 'Shipments.shipped_date': 'shipped'},
            mode='left',
            result_limit=30
        )

        # Filter the joined result (use output column names from select):
        filter_join(
            tables=['A', 'B'],
            join_expr="A.id == B.id",
            select={'A.name': 'name', 'B.value': 'val'},
            result_where="val > 100",  # Uses 'val' from select, not 'B.value'
            result_limit=30
        )

        Anti-patterns
        -------------
        - WRONG: result_where="B.value > 100" (references original table column)
          CORRECT: result_where="val > 100" (uses output name from select)

        - WRONG: Joining without input filters when tables are large
          CORRECT: Use left_where/right_where to narrow before joining

        - WRONG: result_limit=500 (too many rows)
          CORRECT: Start with result_limit=30, paginate with result_offset
        """
        # Resolve table references to full context paths
        resolved_tables = self._resolve_table_refs(tables)

        # Delegate to DataManager for the actual join
        return self._data_manager.filter_join(
            tables=resolved_tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
        )

    @read_only
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
        """
        Perform a semantic search over the result of joining two sources.

        Use this when you need to join tables AND rank results by semantic
        similarity to a query. Combines the power of joins with semantic search.

        Parameters
        ----------
        tables : list[str] | str
            Exactly two table references. Use path-first format:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names from `tables_overview()`
        join_expr : str
            Join predicate using table references as prefixes.
        select : dict[str, str]
            Mapping of source expressions → output column names.
            Include text columns you want to search semantically.
        mode : str
            Join mode: "inner" (default), "left", "right", or "outer".
        left_where, right_where : str | None
            Optional filters applied to tables BEFORE joining.
            Supports arbitrary Python expressions.
        references : dict[str, str] | None
            Mapping of OUTPUT column names → reference text for semantic ranking.
            Use column names from the `select` values, not original table columns.
        k : int
            Maximum rows to return. Start with 10.
        filter : str | None
            Optional predicate over the joined result before ranking.
            Supports arbitrary Python expressions; uses output column names.

        Returns
        -------
        list[dict[str, Any]]
            Top-k rows ranked by semantic similarity.

        Usage Examples
        --------------
        # Join repairs and telematics, then search by address:
        search_join(
            tables=[
                '/path/repairs.xlsx.Tables.Jobs',
                '/path/telematics.xlsx.Tables.July_2025'
            ],
            join_expr="Jobs.OperativeWhoCompletedJob == July_2025.Driver",
            select={
                'Jobs.JobTicketReference': 'job_ref',
                'Jobs.FullAddress': 'address',
                'July_2025.End location': 'destination'
            },
            references={'address': 'Birmingham city centre'},
            k=10
        )

        # Combine semantic search with exact filter on joined result:
        search_join(
            tables=['Products', 'Reviews'],
            join_expr="Products.id == Reviews.product_id",
            select={
                'Products.name': 'product_name',
                'Reviews.text': 'review_text',
                'Reviews.rating': 'rating'
            },
            references={'review_text': 'excellent quality'},
            filter="rating >= 4",
            k=20
        )

        Anti-patterns
        -------------
        - WRONG: references={'Jobs.FullAddress': 'query'} (uses original column)
          CORRECT: references={'address': 'query'} (uses select output name)

        - WRONG: filter="Jobs.status == 'complete'" (uses original column)
          CORRECT: Include status in select, then filter="status == 'complete'"
        """
        # Resolve table references to full context paths
        resolved_tables = self._resolve_table_refs(tables)

        # Delegate to DataManager for the actual search join
        return self._data_manager.search_join(
            tables=resolved_tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            references=references or {},
            k=k,
            filter=filter,
        )

    @read_only
    def filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chain multiple joins, then return rows from the final joined result.

        Use this for complex queries requiring more than two tables. Each step
        can reference the previous step's result using "$prev".

        Parameters
        ----------
        joins : list[dict]
            Ordered steps. Each step is a dict with:
            - tables: list of 2 refs. Use "$prev" to reference previous step's result.
            - join_expr: join predicate using table refs or "prev" as prefix.
            - select: dict mapping source expressions → output names.
            - mode (optional): "inner", "left", "right", "outer".
            - left_where, right_where (optional): input filters (arbitrary Python).
        result_where : str | None
            Filter applied to the FINAL result. Supports arbitrary Python
            expressions; uses column names from the last step's select output.
        result_limit, result_offset : int
            Pagination parameters. Start with limit=30.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Rows from the final materialized context.

        Usage Examples
        --------------
        # Chain three tables: A → B → C
        filter_multi_join(
            joins=[
                # Step 1: Join A and B
                {
                    'tables': ['TableA', 'TableB'],
                    'join_expr': 'TableA.id == TableB.a_id',
                    'select': {'TableA.id': 'id', 'TableB.value': 'b_value'}
                },
                # Step 2: Join previous result with C
                {
                    'tables': ['$prev', 'TableC'],
                    'join_expr': 'prev.id == TableC.ref_id',
                    'select': {'prev.id': 'id', 'prev.b_value': 'b_val', 'TableC.name': 'c_name'}
                }
            ],
            result_where="b_val > 100",
            result_limit=30
        )

        # Real example: Repairs → Telematics July → Telematics August
        filter_multi_join(
            joins=[
                {
                    'tables': [
                        '/path/repairs.xlsx.Tables.Jobs',
                        '/path/telematics.xlsx.Tables.July_2025'
                    ],
                    'join_expr': 'Jobs.OperativeWhoCompletedJob == July_2025.Driver',
                    'select': {
                        'Jobs.JobTicketReference': 'job_ref',
                        'Jobs.OperativeWhoCompletedJob': 'operative',
                        'July_2025.Trip': 'july_trip'
                    },
                    'left_where': "Jobs.status == 'complete'"
                },
                {
                    'tables': ['$prev', '/path/telematics.xlsx.Tables.August_2025'],
                    'join_expr': 'prev.operative == August_2025.Driver',
                    'select': {
                        'prev.job_ref': 'job_ref',
                        'prev.operative': 'operative',
                        'prev.july_trip': 'july_trip',
                        'August_2025.Trip': 'august_trip'
                    }
                }
            ],
            result_limit=30
        )

        Anti-patterns
        -------------
        - WRONG: Using original table column in result_where
          CORRECT: result_where uses column names from the LAST step's select

        - WRONG: Referencing "$prev" in the first step (there's no previous result)
          CORRECT: First step must use two actual table references

        - WRONG: Not propagating needed columns through each step's select
          CORRECT: Each step's select must include columns needed by later steps
        """
        # Resolve table references in all join steps
        resolved_joins = self._resolve_joins_table_refs(joins)

        # Delegate to DataManager for the actual multi-join
        return self._data_manager.filter_multi_join(
            joins=resolved_joins,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
        )

    @read_only
    def search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search over a chain of joined results.

        Combines multi-step joins with semantic ranking. Use when you need to
        correlate 3+ tables and rank results by meaning.

        Parameters
        ----------
        joins : list[dict]
            Ordered steps. Each step is a dict with:
            - tables: list of 2 refs. Use "$prev" to reference previous step.
            - join_expr: join predicate.
            - select: dict mapping source expressions → output names.
            - mode, left_where, right_where (optional; arbitrary Python for filters).
        references : dict[str, str] | None
            Mapping of FINAL output column names → reference text for ranking.
            Column names must be from the LAST step's select output.
        k : int
            Maximum rows to return. Start with 10.
        filter : str | None
            Optional predicate applied before semantic ranking.
            Supports arbitrary Python expressions; uses column names from the
            LAST step's select output.

        Returns
        -------
        list[dict[str, Any]]
            Top-k rows ranked by semantic similarity.

        Usage Examples
        --------------
        # Join three tables and search semantically:
        search_multi_join(
            joins=[
                {
                    'tables': ['Jobs', 'Visits'],
                    'join_expr': 'Jobs.id == Visits.job_id',
                    'select': {
                        'Jobs.id': 'job_id',
                        'Jobs.description': 'job_desc',
                        'Visits.notes': 'visit_notes'
                    }
                },
                {
                    'tables': ['$prev', 'Operatives'],
                    'join_expr': 'prev.job_id == Operatives.job_id',
                    'select': {
                        'prev.job_id': 'job_id',
                        'prev.job_desc': 'description',
                        'Operatives.name': 'operative_name'
                    }
                }
            ],
            references={'description': 'heating system repair'},
            k=15
        )

        # With filter before semantic ranking:
        search_multi_join(
            joins=[...],
            references={'summary': 'budget updates'},
            filter="status == 'approved'",
            k=10
        )

        Anti-patterns
        -------------
        - WRONG: references={'Jobs.description': 'query'} (uses original column)
          CORRECT: references={'description': 'query'} (uses final select output)

        - WRONG: Not including semantic target column in final step's select
          CORRECT: Ensure the column you want to search is in the last select
        """
        # Resolve table references in all join steps
        resolved_joins = self._resolve_joins_table_refs(joins)

        # Delegate to DataManager for the actual semantic multi-join
        return self._data_manager.search_multi_join(
            joins=resolved_joins,
            references=references or {},
            k=k,
            filter=filter,
        )

    @staticmethod
    def _default_ask_about_file_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """
        Prefer path-first targeting; avoid forcing broad discovery.
        - If the user supplied an explicit path, allow immediate use of read-only tools.
        - Do not require a first-step semantic search; keep the toolset on auto.
        """
        return ("auto", current_tools)

    # ---------- High-level importers (delegated to adapter) ---------------- #
    def import_file(self, file_path: Any) -> str:
        """Import a single file into the underlying filesystem via adapter."""
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for import_file")
        return self._adapter.import_file(str(file_path))

    def import_directory(self, directory: Any) -> List[str]:
        """Import all files within ``directory`` via the adapter."""
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for import_directory")
        return self._adapter.import_directory(str(directory))

    def export_file(self, file_path: str, destination_dir: str) -> str:  # type: ignore[override]
        """
        Export a single adapter-managed file to a local directory.

        This delegates to the adapter's ``export_file``, which reads the file
        from its storage (local or remote) and writes it under ``destination_dir``
        on the local filesystem.

        Parameters
        ----------
        file_path : str
            Adapter-native path/identifier of the file to export.
        destination_dir : str
            Local directory where the exported file should be written.

        Returns
        -------
        str
            Absolute local path of the exported file.

        Raises
        ------
        NotImplementedError
            If no adapter is configured.
        Exception
            Any error from the adapter's ``export_file`` implementation.
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for export_file")
        return self._adapter.export_file(file_path, destination_dir)

    def export_directory(self, directory: str, destination_dir: str) -> List[str]:  # type: ignore[override]
        """
        Export all files from an adapter-managed directory to a local directory.

        This delegates to the adapter's ``export_directory``, which enumerates
        files under ``directory`` in its namespace and writes them into
        ``destination_dir`` on the local filesystem.

        Parameters
        ----------
        directory : str
            Adapter-native directory path to export from.
        destination_dir : str
            Local directory where files should be written.

        Returns
        -------
        list[str]
            Absolute local paths of all exported files.

        Raises
        ------
        NotImplementedError
            If no adapter is configured.
        Exception
            Any error from the adapter's ``export_directory`` implementation.
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for export_directory")
        return self._adapter.export_directory(directory, destination_dir)

    def register_existing_file(
        self,
        path: Any,
        *,
        display_name: Optional[str] = None,
        protected: bool = False,
    ) -> str:
        """
        Register an already-present file with the adapter without moving it.

        This is a metadata-only operation: it tells the adapter that ``path``
        should be treated as a managed file and returns the identifier to use
        with this FileManager. Parsing/indexing still happens via ``parse``.

        Parameters
        ----------
        path : Any
            Existing file path/identifier in the adapter's namespace.
        display_name : str | None, optional
            Optional human-friendly label the adapter may store.
        protected : bool, default False
            Hint to mark the file as protected against destructive operations
            (semantics are adapter-specific).

        Returns
        -------
        str
            Adapter-native identifier/path for the registered file.

        Raises
        ------
        NotImplementedError
            If no adapter is configured.
        Exception
            Any error from the adapter's ``register_existing_file`` implementation.
        """
        if self._adapter is None:
            raise NotImplementedError(
                "No adapter configured for register_existing_file",
            )
        return self._adapter.register_existing_file(
            str(path),
            display_name=display_name,
            protected=protected,
        )

    def is_protected(self, file_path: str) -> bool:
        """
        Return whether the adapter considers this file protected.

        Protection is an adapter-defined flag used to guard against destructive
        operations (rename/move/delete/overwrite); this method simply forwards
        the query to the adapter.

        Parameters
        ----------
        file_path : str
            Adapter-native file path/identifier to check.

        Returns
        -------
        bool
            True if the adapter reports the file as protected, False otherwise
            or when no adapter is configured.
        """
        if self._adapter is None:
            return False
        return self._adapter.is_protected(file_path)

    def save_file_to_downloads(self, file_path: str, contents: bytes) -> str:
        """
        Save bytes into the adapter's downloads area and return the saved path.

        Use this when you have in-memory file content that should be exposed to
        the user as a downloadable artifact. The adapter decides where the
        downloads area lives and how names are de-duplicated.

        Parameters
        ----------
        file_path : str
            Desired file name or relative path within the downloads area.
        contents : bytes
            Raw bytes to persist.

        Returns
        -------
        str
            Adapter-native path or URL referring to the saved file.

        Raises
        ------
        NotImplementedError
            If no adapter is configured.
        Exception
            Any error from the adapter's ``save_file_to_downloads`` implementation.
        """
        if self._adapter is None:
            raise NotImplementedError(
                "No adapter configured for save_file_to_downloads",
            )
        return self._adapter.save_file_to_downloads(file_path, contents)

    # File-specific Q&A
    @functools.wraps(BaseFileManager.ask_about_file, updated=())
    @manager_tool
    @log_manager_call("FileManager", "ask_about_file", payload_key="question")
    async def ask_about_file(
        self,
        file_path: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        response_format: Optional[Any] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        """
        Ask a question about a specific file.

        Parameters
        ----------
        file_path : str
            Identifier/path of the file in the underlying adapter.
        question : str
            The user's natural-language question about the file.
        _return_reasoning_steps : bool, default False
            When True, wraps handle.result() to return (answer, messages).
        _parent_chat_context, _clarification_up_q, _clarification_down_q, rolling_summary_in_prompts, _call_id
            See ask().

        response_format : Any | None
            Optional structured output contract. Provide a Pydantic model class
            or a JSON Schema dict to request a strictly structured response.

        Returns
        -------
        SteerableToolHandle
            Interactive tool loop handle (read-only). When a Pydantic model is
            supplied via response_format, the final result will adhere to that
            schema.
        """
        if not self.exists(file_path):
            raise FileNotFoundError(file_path)
        client = new_llm_client()

        tools = dict(self.get_tools("ask_about_file"))

        # Expose join/multi-join tools for cross-context retrieval
        tools.update(dict(self.get_tools("ask_about_file.multi_table")))

        if _clarification_up_q is not None and _clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                _clarification_up_q,
                _clarification_down_q,
                manager="FileManager",
                method="ask_about_file",
                call_id=_call_id,
            )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        system_msg = build_file_manager_ask_about_file_prompt(
            tools=tools,
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        # Use filesystem type without exposing absolute paths to LLM
        user_blob = json.dumps(
            {"filesystem": self._fs_type, "file_path": file_path, "question": question},
            indent=2,
        )
        use_semantic_cache = "both" if SETTINGS.UNITY_SEMANTIC_CACHE else None
        tool_policy_fn = (
            None
            if use_semantic_cache in ("read", "both")
            else self._default_ask_about_file_tool_policy
        )
        handle = start_async_tool_loop(
            client,
            user_blob,
            tools,
            loop_id=f"{self.__class__.__name__}.ask_about_file",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=tool_policy_fn,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            semantic_cache=use_semantic_cache,
            semantic_cache_namespace=f"{self.__class__.__name__}.ask_about_file",
            response_format=response_format,
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

    @functools.wraps(BaseFileManager.clear, updated=())
    def clear(self) -> None:  # type: ignore[override]
        """
        Clear ALL contexts under this filesystem alias and local caches, then re‑provision.

        Behaviour
        ---------
        - Drops the per‑file namespace ``Files/<alias>/**`` (Content and Tables contexts).
        - Drops the index context ``FileRecords/<alias>`` for this filesystem.
        - Clears any local DataStore mirrors and TableStore ensure memo.
        - Re-provisions storage so future operations see a consistent schema.

        Returns
        -------
        None
        """
        # 1) Delete all per-file contexts under the alias root
        try:
            per_file_prefix = str(self._per_file_root)
            ctxs = list(unify.get_contexts(prefix=per_file_prefix) or [])
            for ctx in sorted(ctxs, key=len, reverse=True):
                unify.delete_context(ctx)
            unify.delete_context(per_file_prefix)
        except Exception:
            pass

        # 2) Delete the index context for this filesystem
        unify.delete_context(self._ctx)

        try:
            # Drop ensure memo for TableStore if used
            from unity.common.context_store import TableStore as _TS  # local import

            try:
                _TS._ENSURED.discard((unify.active_project(), self._ctx))
            except Exception:
                pass
        except Exception:
            pass
        try:
            _storage_provision(self)
        except Exception:
            pass
