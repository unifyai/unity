from __future__ import annotations

import json
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

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
    build_file_manager_ask_prompt,
    build_file_manager_ask_about_file_prompt,
    build_file_manager_organize_prompt,
)
from unity.common.business_context import BusinessContextPayload
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
from unity.common.filter_utils import normalize_filter_expr
from unity.common.llm_client import new_llm_client
from unity.common.metrics_utils import reduce_logs
from .utils.search import (
    resolve_table_ref as _srch_resolve_table_ref,
    create_join as _srch_create_join,
    filter_join as _srch_filter_join,
    search_join as _srch_search_join,
    filter_multi_join as _srch_filter_multi_join,
    search_multi_join as _srch_search_multi_join,
)
from .utils.storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
    tables_overview as _storage_tables_overview,
    schema_explain as _storage_schema_explain,
    ctx_for_file as _storage_ctx_for_file,
    ctx_for_file_table as _storage_ctx_for_file_table,
)
from .utils.viz_utils import (
    PlotConfig as _VizPlotConfig,
    PlotResult as _VizPlotResult,
    generate_plots_batch as _viz_generate_plots_batch,
)


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
        """
        super().__init__()
        self.include_in_multi_assistant_table = False
        self._adapter = adapter
        self.__parser: Optional[FileParser] = parser
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Derive a stable alias and context
        try:
            raw_alias = (
                getattr(self._adapter, "name", "files").strip()
                if self._adapter is not None
                else "files"
            )
        except Exception:
            raw_alias = "files"

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
        # Ask/AskAboutFile tool surfaces (read-only). No ingest_files - this is read-only.
        ask_tools: Dict[str, Callable] = methods_to_tool_dict(
            # Schema discovery helpers
            self._list_columns,
            self._tables_overview,
            self._schema_explain,
            # File info (combines filesystem + index + ingest identity)
            self._file_info,
            # Retrieval helpers
            self._filter_files,
            self._search_files,
            self._reduce,
            # Visualization
            self._visualize,
            # Inventory listing
            self.list,
            # Delegate to file-scoped Q&A when needed
            self.ask_about_file,
            include_class_name=False,
        )
        # Multi-table tools (joins across per-file tables)
        ask_multi_table_tools: Dict[str, Callable] = methods_to_tool_dict(
            self._filter_join,
            self._search_join,
            self._filter_multi_join,
            self._search_multi_join,
            include_class_name=False,
        )
        self.add_tools("ask", ask_tools)
        self.add_tools("ask.multi_table", ask_multi_table_tools)
        ask_about_file_tools: Dict[str, Callable] = methods_to_tool_dict(
            # Read-only helpers (no ingest_files - this is read-only)
            self._file_info,
            self._list_columns,
            self._tables_overview,
            self._schema_explain,
            self._filter_files,
            self._search_files,
            self._reduce,
            # Visualization
            self._visualize,
            include_class_name=False,
        )
        self.add_tools("ask_about_file", ask_about_file_tools)
        self.add_tools("ask_about_file.multi_table", ask_multi_table_tools)
        # Organize is mutation-focused. It may call ask() to gather context,
        # but should not receive direct read-only retrieval tools itself.
        organize_tools: Dict[str, Callable] = methods_to_tool_dict(
            self.ask,
            self._rename_file,
            self._move_file,
            self._delete_file,
            self.sync,
            include_class_name=False,
        )
        self.add_tools("organize", organize_tools)

    @functools.cached_property
    def _parser(self):
        if self.__parser is None:
            self.__parser = FileParser()
        return self.__parser

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
    def _file_info(self, *, identifier: Union[str, int]):
        """
        Return comprehensive information about a file's status and ingest identity.

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
        from .utils.storage import file_info as _storage_file_info

        return _storage_file_info(self, identifier=identifier)

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
        info = self._file_info(identifier=file_path)
        ingest_mode = info.get("ingest_mode", "per_file")
        unified_label = info.get("unified_label")
        table_ingest = info.get("table_ingest", True)

        overview = self._tables_overview(file=file_path)
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
    def _rename_file(
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

    def _move_file(
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

    def _delete_file(
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
    def _tables_overview(
        self,
        *,
        include_column_info: bool = True,
        file: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return an overview of available tables/contexts managed by this FileManager.

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
        return _storage_tables_overview(
            self,
            include_column_info=include_column_info,
            file=file,
        )

    @read_only
    def _schema_explain(self, *, table: str) -> str:
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
    def _reduce(
        self,
        *,
        table: Optional[str] = None,
        metric: str,
        keys: str | List[str],
        filter: Optional[str | Dict[str, str]] = None,
        group_by: Optional[str | List[str]] = None,
    ) -> Any:
        """
        Compute reduction metrics over the FileRecords index or a resolved table.

        This is the PRIMARY tool for any quantitative question (counts, sums,
        averages, statistics). Always use reduce instead of fetching rows and
        computing aggregates in-memory.

        Parameters
        ----------
        table : str | None, default None
            Table reference to aggregate. Accepted forms:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names: "FileRecords" for index
            When None, aggregates over the main FileRecords index.
        metric : str
            Reduction metric to compute. Supported values (case-insensitive):
            "count", "sum", "mean", "min", "max", "median", "mode", "var", "std".
        keys : str | list[str]
            Column(s) to aggregate. A single column returns a scalar; multiple
            columns return a dict mapping each key to its computed value.
        filter : str | dict[str, str] | None, default None
            Optional row-level filter expression(s). Supports arbitrary Python
            expressions evaluated per row. Apply filters here to narrow the
            dataset BEFORE aggregation for efficiency.
        group_by : str | list[str] | None, default None
            Optional column(s) to group by. Single column for one grouping level,
            or a list for hierarchical grouping. Result becomes a nested dict
            keyed by group values.

        Returns
        -------
        Any
            Metric value(s) computed over the resolved context:
            - Single key, no grouping  → scalar (float/int/str/bool)
            - Multiple keys, no grouping → dict[key -> scalar]
            - With grouping → nested dict keyed by group values

        Usage Examples
        --------------
        # Count all files in the index:
        reduce(metric='count', keys='file_id')
        # Returns: 42

        # Count with filter:
        reduce(
            metric='count',
            keys='file_id',
            filter="status == 'success'"
        )
        # Returns: 38

        # Sum a numeric column in a per-file table:
        reduce(
            table='/path/to/data.xlsx.Tables.Orders',
            metric='sum',
            keys='amount',
            filter="status == 'complete'"
        )
        # Returns: 15420.50

        # Average with grouping (breakdown by category):
        reduce(
            table='/path/to/data.xlsx.Tables.Orders',
            metric='mean',
            keys='amount',
            group_by='category'
        )
        # Returns: {'Electronics': 245.00, 'Furniture': 890.50, ...}

        # Multiple metrics on same table (call reduce multiple times):
        count = reduce(table='...', metric='count', keys='id')
        total = reduce(table='...', metric='sum', keys='amount')
        avg = reduce(table='...', metric='mean', keys='amount')

        # Hierarchical grouping:
        reduce(
            table='/path/to/data.xlsx.Tables.Sales',
            metric='sum',
            keys='revenue',
            group_by=['region', 'quarter']
        )
        # Returns: {'North': {'Q1': 1000, 'Q2': 1200}, 'South': {'Q1': 800, ...}}

        Anti-patterns
        -------------
        - WRONG: filter_files(...) then count rows in Python
          CORRECT: reduce(metric='count', keys='id', filter=...)

        - WRONG: filter_files(...) then sum values in Python
          CORRECT: reduce(metric='sum', keys='amount', filter=...)

        - WRONG: Calling reduce without specifying the table when you need a per-file table
          CORRECT: Always specify table='/path/to/file.xlsx.Tables.TableName'

        - WRONG: Using reduce for text/categorical analysis
          CORRECT: Use search_files for semantic queries, filter_files for exact matches
        """
        if table is None:
            ctx = self._ctx
        else:
            try:
                ctx = self._resolve_table_ref(table)
            except Exception:
                ctx = table

        return reduce_logs(
            context=ctx,
            metric=metric,
            keys=keys,
            filter=filter,
            group_by=group_by,
        )

    @read_only
    def _visualize(
        self,
        *,
        tables: Union[str, List[str]],
        plot_type: str,
        x_axis: str,
        y_axis: Optional[str] = None,
        group_by: Optional[str] = None,
        filter: Optional[str] = None,
        title: Optional[str] = None,
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
            Row-level filter expression. Same syntax as `_reduce` and `_filter_files`.
            Applied before plotting.

        title : str | None
            Plot title. Auto-generated if not provided.

        aggregate : str | None
            Aggregation function when grouping: "sum", "mean", "count", "min", "max".

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
            aggregate='sum'
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

        # Build plot config
        config = _VizPlotConfig(
            plot_type=plot_type,
            x_axis=x_axis,
            y_axis=y_axis,
            group_by=group_by,
            aggregate=aggregate,
            scale_x=scale_x,
            scale_y=scale_y,
            bin_count=bin_count,
            show_regression=show_regression,
            title=title,
        )

        # Generate plots (project_name=None uses active project inside batch fn)
        results = _viz_generate_plots_batch(
            contexts=contexts,
            config=config,
            filter_expr=filter,
        )

        # Return PlotResult directly (single) or list of PlotResult (multiple)
        if len(results) == 1:
            return results[0]
        return results

    @read_only
    def _list_columns(
        self,
        *,
        include_types: bool = True,
        table: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        """
        List columns for the FileRecords index or a resolved logical table.

        Use this to inspect a table's schema before writing filter expressions
        or selecting columns for retrieval.

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column → type (e.g., {"name": "str"}).
            When False, return just the list of column names.
        table : str | None, default None
            Table reference to inspect. Accepted forms:
            - Path-first: "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names: "FileRecords" for global index
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

        # Get per-file Content columns:
        list_columns(table="/path/to/document.pdf")
        # Returns: {"content_id": "dict", "content_type": "str", "content_text": "str", ...}

        # Get per-file table columns:
        list_columns(table="/path/to/data.xlsx.Tables.Orders")
        # Returns: {"order_id": "str", "amount": "float", "customer": "str", ...}

        # Get just column names (no types):
        list_columns(table="/path/to/data.xlsx.Tables.Orders", include_types=False)
        # Returns: ["order_id", "amount", "customer", ...]
        """
        if table is None:
            cols = self._get_columns()
            return cols if include_types else list(cols)
        # Resolve logical name → fully-qualified context then fetch
        try:
            ctx = self._resolve_table_ref(table)
        except Exception:
            ctx = table
        cols = _storage_get_columns(self, table=ctx)
        return cols if include_types else list(cols)

    @read_only
    def _filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter files (index) or resolve-and-filter per-file Content/Tables.

        Use this tool for exact matches on structured fields (ids, statuses, dates).

        Parameters
        ----------
        filter : str | None
            Arbitrary Python expression evaluated with column names in scope.
            The expression is evaluated per row; any valid Python syntax that
            returns a boolean is supported. String values must be quoted.
            Use .get() for safe dict access on fields like content_id.
        offset : int
            Zero-based pagination offset per context.
        limit : int
            Maximum rows per context.
        tables : list[str] | str | None
            Table references to filter. Accepted forms:
            - Path-first: "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - "FileRecords" for the global index
            When None, only the FileRecords index is scanned.

        Returns
        -------
        list[dict]
            Flat list of rows from the resolved contexts.

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
        # Filter FileRecords index by status:
        filter_files(filter="status == 'success'")

        # Filter per-file table with date range:
        filter_files(
            filter="created_at >= '2024-01-01' and created_at < '2024-02-01'",
            tables=['/path/to/file.xlsx.Tables.Orders']
        )

        # Paginate through results:
        filter_files(filter="...", offset=0, limit=30)
        filter_files(filter="...", offset=30, limit=30)

        # Filter per-file Content by hierarchy:
        filter_files(
            filter="content_type == 'table' and content_id.get('section') == 2",
            tables=['/path/to/file.pdf']
        )

        # Get all images from a document:
        filter_files(
            filter="content_type == 'image'",
            tables=['/path/to/doc.pdf']
        )

        Anti-patterns
        -------------
        - WRONG: filter="visit_date == '2024-01'" (partial date equality fails)
          CORRECT: filter="visit_date >= '2024-01-01' and visit_date < '2024-02-01'"

        - WRONG: filter="description.contains('budget')" (substring for meaning)
          CORRECT: Use search_files(references={'description': 'budget'}) instead

        - WRONG: filter="content_id['section'] == 2" (direct dict indexing fails)
          CORRECT: filter="content_id.get('section') == 2"

        - WRONG: Fetching rows just to count them
          CORRECT: Use reduce(metric='count', keys='id') instead
        """
        normalized = normalize_filter_expr(filter)
        from .utils.search import filter_files as _srch_filter_files

        rows = _srch_filter_files(
            self,
            filter=normalized,
            offset=offset,
            limit=limit,
            tables=tables,
        )
        return rows

    @read_only
    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        table: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over a resolved context using one or more reference texts.

        Use this tool when searching by meaning, topics, or concepts in text fields.
        For exact matches on structured fields (ids, statuses), use filter_files instead.

        Parameters
        ----------
        references : dict[str, str] | None
            Mapping of column_name → reference_text for semantic matching.
            When omitted, returns recent files without semantic ranking.
        k : int
            Number of results to return. Start with 10, increase if needed.
            Maximum 100, but prefer smaller values to avoid context overflow.
        table : str | None
            Table reference to search. Accepted forms:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names: "FileRecords" for index
            When None, defaults to the global FileRecords index.
        filter : str | None
            Optional row-level predicate to narrow results before semantic ranking.
            Supports arbitrary Python expressions evaluated per row.

        Returns
        -------
        list[dict]
            Up to k rows ranked by similarity from the resolved context.

        Usage Examples
        --------------
        # Semantic search over FileRecords index by summary:
        search_files(references={'summary': 'fire safety regulations'}, k=10)

        # Search per-file Content for paragraphs about a topic:
        search_files(
            references={'content_text': 'payment terms and conditions'},
            table='/path/to/contract.pdf',
            k=5
        )

        # Combine semantic search with exact filter:
        search_files(
            references={'description': 'heating system repair'},
            filter="status == 'success'",
            k=10
        )

        # Search a specific per-file table:
        search_files(
            references={'Vehicle': 'Stuart Birks'},
            table='/path/to/telematics.xlsx.Tables.July_2025',
            k=20
        )

        # Tiered search strategy for documents:
        # 1. Paragraphs first (highest precision):
        search_files(
            references={'summary': '<query>'},
            filter="content_type == 'paragraph'",
            table='/path/to/doc.pdf',
            k=10
        )
        # 2. Sections if paragraphs insufficient:
        search_files(
            references={'summary': '<query>'},
            filter="content_type == 'section'",
            table='/path/to/doc.pdf',
            k=5
        )

        Anti-patterns
        -------------
        - WRONG: references={'id': 'search term'} (id is not a text column)
          CORRECT: Only use text columns in references

        - WRONG: Using search_files for exact id/status lookup
          CORRECT: Use filter_files(filter="id == 123") for exact matches

        - WRONG: k=500 (too many results flood context)
          CORRECT: Start with k=10, increase only if needed
        """
        from .utils.search import search_files as _srch_search_files

        return _srch_search_files(
            self,
            references=references,
            k=k,
            table=table,
            filter=filter,
        )

    # ---------- Per-file join and multi-join tools (read-only) -------------- #
    @read_only
    def _filter_join(
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
        return _srch_filter_join(
            self,
            tables=tables,
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
    def _search_join(
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
        return _srch_search_join(
            self,
            tables=tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            references=references,
            k=k,
            filter=filter,
        )

    @read_only
    def _filter_multi_join(
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
        return _srch_filter_multi_join(
            self,
            joins=joins,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
        )

    @read_only
    def _search_multi_join(
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
        return _srch_search_multi_join(
            self,
            joins=joins,
            references=references,
            k=k,
            filter=filter,
        )

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """
        Prefer path-first targeting; avoid forcing broad discovery.
        - If the user supplied an explicit path, allow immediate use of read-only tools.
        - Do not require a first-step semantic search; keep the toolset on auto.
        """
        return ("auto", current_tools)

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

    @staticmethod
    def _default_organize_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require ask on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
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

    # Filesystem-level Q&A
    @functools.wraps(BaseFileManager.ask, updated=())
    @manager_tool
    @log_manager_call("FileManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        business_payload: Optional[BusinessContextPayload] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        """
        Ask a question about the filesystem, using read-only tools.

        Parameters
        ----------
        text : str
            The user's natural-language question.
        _return_reasoning_steps : bool, default False
            When True, wraps handle.result() to return (answer, messages).
        _parent_chat_context : list[dict] | None
            Optional chat lineage to prepend for context.
        _clarification_up_q, _clarification_down_q : asyncio.Queue | None
            If both provided, enables an interactive clarification tool.
        rolling_summary_in_prompts : bool | None
            Override whether to include rolling activity summaries in system prompts.
        business_payload : BusinessContextPayload | None
            Structured domain-specific context (role, rules, guidelines, hints)
            injected via slot-filling pattern. Business role appears FIRST in prompt.
        _call_id : str | None
            Correlation ID for event logging.

        Returns
        -------
        SteerableToolHandle
            A handle controlling the interactive tool-use loop (read-only).
        """
        client = new_llm_client()
        tools = dict(self.get_tools("ask"))

        # Expose join/multi-join tools for cross-context retrieval
        tools.update(dict(self.get_tools("ask.multi_table")))

        if _clarification_up_q is not None and _clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                _clarification_up_q,
                _clarification_down_q,
                manager="FileManager",
                method="ask",
                call_id=_call_id,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        system_msg = build_file_manager_ask_prompt(
            tools=tools,
            num_files=self._num_files(),
            include_activity=include_activity,
            business_payload=business_payload,
        )
        # TODO: REMOVE - Debug file dump for prompt inspection
        open("system_msg.txt", "w").write(system_msg)
        client.set_system_message(system_msg)
        use_semantic_cache = "both" if SETTINGS.UNITY_SEMANTIC_CACHE else None
        tool_policy_fn = (
            None
            if use_semantic_cache in ("read", "both")
            else self._default_ask_tool_policy
        )
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.ask",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=tool_policy_fn,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            semantic_cache=use_semantic_cache,
            semantic_cache_namespace=f"{self.__class__.__name__}.ask",
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

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

    # Filesystem reorganization
    @functools.wraps(BaseFileManager.organize, updated=())
    @log_manager_call("FileManager", "organize", payload_key="text")
    async def organize(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        """
        Plan and execute safe reorganization operations (rename/move/delete) with guardrails.

        Parameters
        ----------
        text : str
            Natural-language request (e.g., "Move all PDFs to /docs").
        _return_reasoning_steps, _parent_chat_context, _clarification_up_q, _clarification_down_q,
        rolling_summary_in_prompts, _call_id
            See ask().

        Returns
        -------
        SteerableToolHandle
            Interactive tool loop handle. Mutations are capability-guarded via the adapter.
        """
        client = new_llm_client()
        tools = dict(self.get_tools("organize"))
        if _clarification_up_q is not None and _clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                _clarification_up_q,
                _clarification_down_q,
                manager="FileManager",
                method="organize",
                call_id=_call_id,
            )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        system_msg = build_file_manager_organize_prompt(
            tools=tools,
            num_files=len(self.list()),
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.organize",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_organize_tool_policy,
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
