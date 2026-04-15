from __future__ import annotations

import json
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

import unify

from unity.common.tool_spec import manager_tool, read_only, ToolSpec
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
    make_request_clarification_tool,
    methods_to_tool_dict,
)
from unity.events.event_bus import EVENT_BUS, Event
from unity.common.async_tool_loop import (
    TOOL_LOOP_LINEAGE,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.settings import SETTINGS
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.events.manager_event_logging import log_manager_call
from unity.common.context_store import TableStore
from unity.common.context_registry import ContextRegistry, TableContext
from unity.common.model_to_fields import model_to_fields
from unity.common.llm_client import new_llm_client
from .utils.search import (
    resolve_table_ref as _srch_resolve_table_ref,
)
from .utils.storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
    ctx_for_file_content as _storage_ctx_for_file_content,
    ctx_for_file_table as _storage_ctx_for_file_table,
)

from unity.file_manager.file_parsers.types.formats import (
    extension_to_format as _ext_to_fmt,
    is_image_format as _is_image_format,
)

if TYPE_CHECKING:
    from unity.file_manager.types.describe import FileStorageMap
    from unity.data_manager.base import BaseDataManager


# ---------------------------------------------------------------------------
# Lightweight handle for single-shot vision answers
# ---------------------------------------------------------------------------
class _ImageVisionHandle(SteerableToolHandle):
    """Wraps a single-shot vision LLM call as a SteerableToolHandle."""

    def __init__(self, coro, *, _return_reasoning_steps: bool = False):
        self._coro = coro
        self._return_steps = _return_reasoning_steps
        self._result: Optional[str] = None
        self._done = False

    async def result(self):
        if not self._done:
            self._result = await self._coro
            self._done = True
        if self._return_steps:
            return self._result, []
        return self._result

    def done(self) -> bool:
        return self._done

    async def ask(self, question, *, _parent_chat_context=None):
        return self

    async def interject(self, message, *, _parent_chat_context_cont=None):
        pass

    async def stop(self, reason=None, **kwargs):
        self._done = True

    async def pause(self):
        return "Single-shot vision call cannot be paused."

    async def resume(self):
        return "Single-shot vision call is not paused."

    async def next_clarification(self):
        return {}

    async def next_notification(self):
        return {}

    async def answer_clarification(self, call_id, answer):
        pass


class FileManager(BaseFileManager):
    """A FileManager bound to exactly one filesystem adapter.

    - Discovery/search index is backed by adapter capabilities
    - Bytes are retrieved from the adapter and parsed on demand
    - Exposes unified tools for read-only inspection and safe rename/move
    """

    class Config:
        required_contexts = [
            TableContext(
                name="FileRecords",
                description="Root namespace for file record indices.",
            ),
            TableContext(
                name="Files",
                description="Root namespace for per-file content storage.",
            ),
        ]

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
            Filesystem adapter (e.g., Local). When None, a
            ``LocalFileSystemAdapter`` is created automatically.
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
        self.include_in_multi_assistant_table = True
        if adapter is None:
            from unity.file_manager.filesystem_adapters.local_adapter import (
                LocalFileSystemAdapter,
            )

            adapter = LocalFileSystemAdapter()
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

        file_records_base = ContextRegistry.get_context(FileManager, "FileRecords")
        files_base = ContextRegistry.get_context(FileManager, "Files")
        self._ctx = f"{file_records_base}/{self._fs_alias}"
        self._per_file_root = f"{files_base}/{self._fs_alias}"

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
        self._store.ensure_context()
        self._provision_storage()

        # Public tool dictionaries, mirroring other managers
        # Multi-table tools (joins across per-file tables)
        ask_multi_table_tools: Dict[str, Callable] = methods_to_tool_dict(
            ToolSpec(fn=self.filter_join, display_label="Cross-referencing file data"),
            ToolSpec(fn=self.search_join, display_label="Searching across files"),
            ToolSpec(
                fn=self.filter_multi_join,
                display_label="Cross-referencing multiple files",
            ),
            ToolSpec(
                fn=self.search_multi_join,
                display_label="Searching across multiple files",
            ),
            include_class_name=False,
        )
        ask_about_file_tools: Dict[str, Callable] = methods_to_tool_dict(
            ToolSpec(fn=self.describe, display_label="Describing file structure"),
            ToolSpec(fn=self.list_columns, display_label="Listing file columns"),
            ToolSpec(fn=self.filter_files, display_label="Filtering file data"),
            ToolSpec(fn=self.search_files, display_label="Searching file contents"),
            ToolSpec(fn=self.reduce, display_label="Summarising file data"),
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

    # ------------------------- Describe helper ------------------------------ #
    @functools.wraps(BaseFileManager.describe, updated=())
    @manager_tool
    @read_only
    def describe(
        self,
        file_path: str,
    ) -> "FileStorageMap":
        from .utils.storage import describe_file as _storage_describe_file

        return _storage_describe_file(self, file_path=file_path)

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
        # Resolve identity and layout via describe()
        storage = self.describe(file_path=file_path)
        if not storage.indexed_exists:
            return {"outcome": "sync skipped", "reason": "file not indexed"}

        storage_id = storage.storage_id
        table_ingest = (
            storage.table_ingest if storage.table_ingest is not None else True
        )
        file_id = storage.file_id

        purged = {"content_rows": 0, "table_rows": 0}

        # Purge content rows via DataManager
        dm = self._data_manager
        try:
            if storage.has_document and storage.document:
                purged["content_rows"] = dm.delete_rows(
                    context=storage.document.context_path,
                    filter="1 == 1",  # Delete all rows
                )
        except Exception:
            pass

        # Purge per-file tables (when present)
        try:
            if table_ingest and storage.has_tables and storage.tables:
                for table_info in storage.tables:
                    try:
                        purged["table_rows"] += dm.delete_rows(
                            context=table_info.context_path,
                            filter="1 == 1",  # Delete all rows
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
    @functools.wraps(BaseFileManager.sync, updated=())
    def sync(self, *, file_path: str) -> Dict[str, Any]:
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
        # file_id lookup via DataManager
        try:
            fid = int(s)
            rows = self._data_manager.filter(
                context=self._ctx,
                filter=f"file_id == {fid}",
                limit=1,
                columns=["source_uri"],
            )
            if rows:
                uri = rows[0].get("source_uri")
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
            The base adapter type (e.g., "Local").
        """
        if not adapter_name:
            return "Unknown"
        # Split on '[' and take the first part (the type)
        return adapter_name.split("[")[0].strip() or adapter_name

    # Helpers #
    # --------#

    def _ctx_for_file_content(self, storage_id: str) -> str:
        """
        Return the fully‑qualified Unify context name for file Content.

        Parameters
        ----------
        storage_id : str
            The storage identifier (e.g., str(file_id) or a custom label).

        Returns
        -------
        str
            Fully‑qualified context path in the form
            ``<base>/Files/<alias>/<storage_id>/Content``.
        """
        return _storage_ctx_for_file_content(self, storage_id=storage_id)

    def _ctx_for_file_table(self, storage_id: str, table: str) -> str:
        """
        Return the fully‑qualified Unify context name for a file table.

        Parameters
        ----------
        storage_id : str
            The storage identifier (e.g., str(file_id) or a custom label).
        table : str
            Logical table name (e.g. "Products").

        Returns
        -------
        str
            Fully‑qualified context path in the form
            ``<base>/Files/<alias>/<storage_id>/Tables/<safe_table>``.
        """
        return _storage_ctx_for_file_table(self, storage_id=storage_id, table=table)

    def _resolve_storage_id(
        self,
        *,
        file_path: Optional[str] = None,
        file_id: Optional[int] = None,
    ) -> Optional[str]:
        """Resolve storage_id from file_path or file_id."""
        from .utils.storage import resolve_storage_id as _resolve

        return _resolve(self, file_path=file_path, file_id=file_id)

    # ---------- Join helpers (delegations to search module) ------------------- #
    @read_only
    def _resolve_table_ref(self, ref: str) -> str:
        """
        Resolve a table reference to a fully-qualified Unify context.

        Parameters
        ----------
        ref : str
            Accepted forms:
            - "FileRecords" → index context
            - "<storage_id>" → per-file Content context
            - "<storage_id>.Tables.<label>" → per-file table context
            - "id=<file_id>" or "#<file_id>" → file by ID

        Returns
        -------
        str
            Fully-qualified Unify context path for the referenced table.
        """
        return _srch_resolve_table_ref(self, ref)

    # ---------- Adapter wrappers (bytes + identifiers only) ----------------- #
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
    @functools.wraps(BaseFileManager.rename_file, updated=())
    def rename_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_name: str,
    ) -> Dict[str, Any]:
        from .utils.ops import rename_file as _ops_rename

        return _ops_rename(
            self,
            file_id_or_path=file_id_or_path,
            new_name=str(new_name),
        )

    @functools.wraps(BaseFileManager.move_file, updated=())
    def move_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_parent_path: str,
    ) -> Dict[str, Any]:
        from .utils.ops import move_file as _ops_move

        return _ops_move(
            self,
            file_id_or_path=file_id_or_path,
            new_parent_path=str(new_parent_path),
        )

    @functools.wraps(BaseFileManager.delete_file, updated=())
    def delete_file(
        self,
        *,
        file_id_or_path: Union[str, int],
    ) -> Dict[str, Any]:
        from .utils.ops import delete_file as _ops_delete

        return _ops_delete(self, file_id_or_path=file_id_or_path)

    @functools.wraps(BaseFileManager.exists, updated=())
    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        if self._adapter is None:
            return False
        try:
            # Use path as-is (adapters handle both relative and absolute paths)
            ok = self._adapter.exists(file_path)
        except Exception:
            return False
        return bool(ok)

    @functools.wraps(BaseFileManager.list, updated=())
    def list(self) -> List[str]:  # type: ignore[override]
        if self._adapter is None:
            return []
        try:
            items = self._adapter.list()
        except Exception:
            return []
        return list(items or [])

    @functools.wraps(BaseFileManager.ingest_files, updated=())
    def ingest_files(self, file_paths: Union[str, List[str]], *, config: Optional[_FilePipelineConfig] = None) -> "IngestPipelineResult":  # type: ignore[override]
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
                from uuid import uuid4

                parse_start_time = _time.perf_counter()
                run_id = uuid4().hex

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
                                run_id=run_id,
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
                                        run_id=run_id,
                                        trace_id=(
                                            str(
                                                getattr(
                                                    getattr(pr, "trace", None),
                                                    "trace_id",
                                                    "",
                                                )
                                                or "",
                                            )
                                            or None
                                        ),
                                        duration_ms=parse_duration_ms,
                                        elapsed_ms=parse_duration_ms,
                                        meta={
                                            "parse_backend": getattr(
                                                getattr(pr, "trace", None),
                                                "backend",
                                                None,
                                            ),
                                            "parse_trace_status": (
                                                getattr(
                                                    getattr(
                                                        getattr(pr, "trace", None),
                                                        "status",
                                                        None,
                                                    ),
                                                    "value",
                                                    None,
                                                )
                                            ),
                                            "warnings_count": len(
                                                list(
                                                    getattr(
                                                        getattr(pr, "trace", None),
                                                        "warnings",
                                                        [],
                                                    )
                                                    or [],
                                                ),
                                            ),
                                            "table_count": len(
                                                list(getattr(pr, "tables", []) or []),
                                            ),
                                        },
                                        verbosity=verbosity,
                                    ),
                                )
                            else:
                                _reporter.report(
                                    create_progress_event(
                                        orig,
                                        "parse",
                                        "failed",
                                        run_id=run_id,
                                        trace_id=(
                                            str(
                                                getattr(
                                                    getattr(pr, "trace", None),
                                                    "trace_id",
                                                    "",
                                                )
                                                or "",
                                            )
                                            or None
                                        ),
                                        duration_ms=parse_duration_ms,
                                        elapsed_ms=parse_duration_ms,
                                        error=str(getattr(pr, "error", "") or ""),
                                        meta={
                                            "parse_backend": getattr(
                                                getattr(pr, "trace", None),
                                                "backend",
                                                None,
                                            ),
                                            "parse_trace_status": (
                                                getattr(
                                                    getattr(
                                                        getattr(pr, "trace", None),
                                                        "status",
                                                        None,
                                                    ),
                                                    "value",
                                                    None,
                                                )
                                            ),
                                        },
                                        verbosity=verbosity,
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
                if enable_progress and _reporter is not None:
                    from .utils.progress import create_progress_event

                    for exp in exported_paths:
                        original_path = exported_paths_to_original_paths.get(exp, exp)
                        _reporter.report(
                            create_progress_event(
                                original_path,
                                "parse",
                                "failed",
                                run_id=run_id,
                                duration_ms=parse_duration_ms,
                                elapsed_ms=parse_duration_ms,
                                error=str(e),
                                verbosity=verbosity,
                            ),
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
                        from unity.file_manager.types.ingest import IngestedFullFile

                        error_results[original_path] = IngestedFullFile(
                            file_path=original_path,
                            status="error",
                            error=f"parse failed: {e}",
                            content_rows=[],
                            tables=[],
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
                            from unity.file_manager.types.ingest import IngestedFullFile

                            error_results[orig] = IngestedFullFile(
                                file_path=orig,
                                status="error",
                                error=str(getattr(pr, "error", "") or ""),
                                content_rows=[],
                                tables=[],
                            )
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
                if getattr(getattr(cfg, "cost", None), "enable_cost_ledger", False):
                    from unity.file_manager.pipeline import (
                        JsonlCostLedger,
                        PipelineCostAccumulator,
                        PipelineCostRateCard,
                        build_observability_cost_line_items,
                        build_parse_cost_line_items,
                        generate_cost_ledger_path,
                    )

                    rate_card = PipelineCostRateCard.from_config(cfg.cost)
                    accumulator = PipelineCostAccumulator(
                        run_id=run_id,
                        rate_card=rate_card,
                        environment=getattr(cfg.cost, "environment", "local"),
                        tenant_id=getattr(cfg.cost, "tenant_id", None),
                    )
                    for pr in parse_results:
                        accumulator.add_line_items(
                            build_parse_cost_line_items(
                                run_id=run_id,
                                file_path=str(getattr(pr, "logical_path", "") or ""),
                                parse_result=pr,
                                parse_config=cfg.parse,
                                rate_card=rate_card,
                            ),
                        )
                    accumulator.add_line_items(
                        build_observability_cost_line_items(
                            run_id=run_id,
                            rate_card=rate_card,
                            progress_event_count=(
                                2 * len(parse_results)
                                if enable_progress and _reporter is not None
                                else 0
                            ),
                            run_manifest_count=0,
                            file_manifest_count=0,
                            stage_manifest_count=0,
                            cost_ledger_count=1,
                        ),
                    )
                    cost_ledger = JsonlCostLedger(
                        path=(
                            getattr(cfg.cost, "cost_ledger_file", None)
                            or generate_cost_ledger_path()
                        ),
                    )
                    try:
                        cost_ledger.write(accumulator.build_ledger())
                    finally:
                        cost_ledger.flush()
                        cost_ledger.close()
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
                all_parse_results=parse_results,
                run_id=run_id,
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

    @functools.wraps(BaseFileManager.reduce, updated=())
    @read_only
    def reduce(
        self,
        *,
        context: Optional[str] = None,
        metric: str,
        columns: Union[str, List[str]],
        filter: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        # Resolve context - default to FileRecords index
        ctx = context if context else self._ctx

        # Delegate to DataManager for the actual reduction
        return self._data_manager.reduce(
            context=ctx,
            metric=metric,
            columns=columns,
            filter=filter,
            group_by=group_by,
        )

    @functools.wraps(BaseFileManager.list_columns, updated=())
    @read_only
    def list_columns(
        self,
        *,
        include_types: bool = True,
        context: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        if context is None:
            cols = self._get_columns()
            return cols if include_types else list(cols)
        # Use context path directly (already resolved via describe())
        cols = _storage_get_columns(self, table=context)
        return cols if include_types else list(cols)

    @functools.wraps(BaseFileManager.filter_files, updated=())
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

    @functools.wraps(BaseFileManager.search_files, updated=())
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
    @functools.wraps(BaseFileManager.filter_join, updated=())
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
    ) -> List[Dict[str, Any]]:
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

    @functools.wraps(BaseFileManager.search_join, updated=())
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

    @functools.wraps(BaseFileManager.filter_multi_join, updated=())
    @read_only
    def filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        # Resolve table references in all join steps
        resolved_joins = self._resolve_joins_table_refs(joins)

        # Delegate to DataManager for the actual multi-join
        return self._data_manager.filter_multi_join(
            joins=resolved_joins,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
        )

    @functools.wraps(BaseFileManager.search_multi_join, updated=())
    @read_only
    def search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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

    def save_attachment(
        self,
        attachment_id: str,
        filename: str,
        contents: bytes,
    ) -> str:
        """
        Save bytes into the Attachments directory and return the saved path.

        Parameters
        ----------
        attachment_id : str
            Unique attachment identifier.
        filename : str
            Original filename for the attachment.
        contents : bytes
            Raw bytes to persist.

        Returns
        -------
        str
            Relative path to the saved file (e.g. ``"Attachments/abc123_report.pdf"``).
        """
        if self._adapter is None:
            raise NotImplementedError(
                "No adapter configured for save_attachment",
            )
        display_name = self._adapter.save_attachment(attachment_id, filename, contents)

        from unity.settings import SETTINGS

        if SETTINGS.file.IMPLICIT_INGESTION:
            result = self.ingest_files(display_name)

            # ingest_files only creates FileRecords entries for successfully
            # parsed files.  When parsing fails (corrupt file, unknown format,
            # etc.) the file must still be registered so it is discoverable via
            # describe().
            file_result = result.files.get(display_name)
            if file_result and getattr(file_result, "status", None) == "error":
                from .utils.ops import add_or_replace_file_row

                add_or_replace_file_row(
                    self,
                    entry={
                        "file_path": display_name,
                        "source_uri": self._resolve_to_uri(display_name),
                        "source_provider": getattr(self._adapter, "name", None),
                        "status": "error",
                        "error": getattr(file_result, "error", None)
                        or "file could not be parsed",
                        "storage_id": "",
                    },
                )

        return display_name

    # File-specific Q&A
    @functools.wraps(BaseFileManager.ask_about_file, updated=())
    @manager_tool
    @log_manager_call(
        "FileManager",
        "ask_about_file",
        payload_key="question",
        display_label="Reading file",
    )
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
        from pathlib import Path as _Path

        # Detect image files by extension and route to a single-shot vision path
        ext = _Path(file_path).suffix.lower()
        fmt = _ext_to_fmt(ext)
        if _is_image_format(fmt):
            return self._ask_about_image_file(
                file_path,
                question,
                _return_reasoning_steps=_return_reasoning_steps,
                _parent_chat_context=_parent_chat_context,
            )

        # Check if file is indexed in Unify (not just filesystem existence)
        # This allows asking about files that were ingested but may not exist on local FS
        storage = self.describe(file_path=file_path)
        if not storage.indexed_exists:
            raise FileNotFoundError(file_path)
        client = new_llm_client()

        tools = dict(self.get_tools("ask_about_file"))

        # Expose join/multi-join tools for cross-context retrieval
        tools.update(dict(self.get_tools("ask_about_file.multi_table")))

        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "ask_about_file",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "ask_about_file",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        system_msg = build_file_manager_ask_about_file_prompt(
            tools=tools,
            include_activity=include_activity,
        ).to_list()
        client.set_system_message(system_msg)
        # Use filesystem type without exposing absolute paths to LLM
        user_blob = json.dumps(
            {"filesystem": self._fs_type, "file_path": file_path, "question": question},
            indent=2,
        )
        tool_policy_fn = self._default_ask_about_file_tool_policy
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
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

    # ---------------------------------------------------------------------- #
    # Image-specific vision path                                              #
    # ---------------------------------------------------------------------- #
    def _ask_about_image_file(
        self,
        file_path: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> SteerableToolHandle:
        """Single-shot vision LLM call for image files (JPEG/PNG)."""
        import base64 as _b64

        if not self._adapter.exists(file_path):
            raise FileNotFoundError(file_path)

        image_bytes = self._adapter.open_bytes(file_path)

        head = image_bytes[:10]
        if head.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif head.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        else:
            mime = "application/octet-stream"

        b64_data = _b64.b64encode(image_bytes).decode("ascii")
        content_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64_data}"},
        }

        async def _vision_call() -> str:
            from unity.image_manager.prompt_builders import build_image_ask_prompt

            client = new_llm_client()
            client.set_system_message(
                build_image_ask_prompt(caption=file_path, timestamp=None).to_list(),
            )

            messages: list[dict] = []
            if _parent_chat_context:
                from unity.common.context_dump import (
                    make_messages_safe_for_context_dump,
                )

                parent_ctx_safe = make_messages_safe_for_context_dump(
                    _parent_chat_context,
                )
                messages.append(
                    {
                        "role": "system",
                        "_ctx_header": True,
                        "content": (
                            "You are handling an image analysis request.\n\n"
                            "## Parent Chat Context\n"
                            "This is the broader conversation context from which "
                            "this image question originated.\n\n"
                            f"{json.dumps(parent_ctx_safe, indent=2)}\n\n"
                            "Your task: Analyze the provided image and answer the "
                            "question. Respond with plain text only."
                        ),
                    },
                )

            messages.append(
                {
                    "role": "user",
                    "content": [
                        content_block,
                        {"type": "text", "text": question},
                    ],
                },
            )
            return await client.generate(messages=messages)

        return _ImageVisionHandle(
            _vision_call(),
            _return_reasoning_steps=_return_reasoning_steps,
        )

    @functools.wraps(BaseFileManager.clear, updated=())
    def clear(self) -> None:  # type: ignore[override]
        # 1) Delete all per-file contexts under the alias root via DataManager
        dm = self._data_manager
        try:
            per_file_prefix = str(self._per_file_root)
            # list_tables returns dict by default; we just need the keys
            ctxs_info = dm.list_tables(
                prefix=per_file_prefix,
                include_column_info=False,
            )
            ctxs = (
                list(ctxs_info)
                if isinstance(ctxs_info, list)
                else list(ctxs_info.keys() if isinstance(ctxs_info, dict) else [])
            )
            for ctx in sorted(ctxs, key=len, reverse=True):
                try:
                    dm.delete_table(ctx, dangerous_ok=True)
                except Exception:
                    pass
            try:
                dm.delete_table(per_file_prefix, dangerous_ok=True)
            except Exception:
                pass
        except Exception:
            pass

        # 2) Delete the index context for this filesystem via DataManager
        try:
            dm.delete_table(self._ctx, dangerous_ok=True)
        except Exception:
            pass

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

    @functools.wraps(BaseFileManager.render_excel_sheet, updated=())
    def render_excel_sheet(self, sheet, cell_range=None, scale=1.0):
        from unity.file_manager.rendering import render_excel_sheet

        return render_excel_sheet(sheet, cell_range=cell_range, scale=scale)

    @functools.wraps(BaseFileManager.render_pdf, updated=())
    def render_pdf(self, source, page=0, dpi=150):
        from unity.file_manager.rendering import render_pdf

        return render_pdf(source, page=page, dpi=dpi)
