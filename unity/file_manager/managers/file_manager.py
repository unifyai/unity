from __future__ import annotations

import json
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Union
from typing import AsyncIterator

import unify

from unity.common.tool_spec import manager_tool, read_only
from unity.file_manager.base import BaseFileManager
from unity.file_manager.parser.base import BaseParser
from unity.file_manager.parser.docling_parser import DoclingParser
from unity.file_manager.types.file import FileRecord
from unity.file_manager.types.config import (
    FilePipelineConfig as _FilePipelineConfig,
    resolve_callables as _resolve_callables,
)
from unity.file_manager.parser.types.document import Document as _Doc
from unity.file_manager.fs_adapters.base import BaseFileSystemAdapter
from unity.file_manager.prompt_builders import (
    build_file_manager_ask_prompt,
    build_file_manager_ask_about_file_prompt,
    build_file_manager_organize_prompt,
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
from unity.constants import is_readonly_ask_guard_enabled
from unity.constants import is_semantic_cache_enabled
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.events.manager_event_logging import log_manager_call
from unity.common.context_store import TableStore
from unity.common.model_to_fields import model_to_fields
from unity.common.filter_utils import normalize_filter_expr
from unity.common.data_store import DataStore
from unity.common.llm_client import new_llm_client
from unity.common.embed_utils import ensure_vector_column
from .search import (
    resolve_table_ref as _srch_resolve_table_ref,
    create_join as _srch_create_join,
    filter_join as _srch_filter_join,
    search_join as _srch_search_join,
    filter_multi_join as _srch_filter_multi_join,
    search_multi_join as _srch_search_multi_join,
)
from .ops import (
    per_file_table_ctx as _ops_per_file_table_ctx,
    ensure_per_file_table_context as _ops_ensure_per_file_table_context,
    per_file_ctx as _ops_per_file_ctx,
    ensure_per_file_context as _ops_ensure_per_file_context,
    delete_per_file_rows_by_filter as _ops_delete_per_file_rows_by_filter,
    create_file_record as _ops_create_file_record,
    create_file as _ops_create_file,
    create_file_table as _ops_create_file_table,
)
from .storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
    tables_overview as _storage_tables_overview,
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
        parser: Optional[BaseParser] = None,
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
        self._adapter = adapter
        self._parser: BaseParser = (
            parser
            if parser is not None
            else DoclingParser(
                use_llm_enrichment=True,
                extract_images=True,
                extract_tables=True,
            )
        )
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

        self._fs_alias = self._safe(str(raw_alias))

        # Extract clean filesystem type for LLM prompts (without path/details)
        self._fs_type = self._extract_filesystem_type(raw_alias)

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from .. import ensure_initialised as _ensure_initialised  # type: ignore  # local to avoid cycles

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
        # - File:        per-file content roots (one subcontext per filename)
        self._ctx = f"{base_ctx}/FileRecords/{self._fs_alias}"
        self._per_file_root = f"{base_ctx}/File/{self._fs_alias}"

        # Local DataStore mirror (write-through only)
        self._data_store = DataStore.for_context(self._ctx, key_fields=("file_id",))

        # Ensure context and fields exist
        self._store = TableStore(
            self._ctx,
            unique_keys={"file_id": "int"},
            auto_counting={"file_id": None},
            description=(
                "FileRecords index for a single filesystem; per-file content lives under File/<alias>/<filename>/Tables/<table>."
            ),
            fields=model_to_fields(FileRecord),
        )
        self._store.ensure_context()

        # Ensure storage via shared helper (idempotent)
        try:
            self._provision_storage()
        except Exception:
            pass

        # Immutable built-in fields derived from the FileRecord model
        self._BUILTIN_FIELDS: tuple[str, ...] = tuple(FileRecord.model_fields.keys())

        # Public tool dictionaries, mirroring other managers
        # Ask/AskAboutFile tool surfaces (read-only). Ask has the same tools as
        # ask_about_file, minus low-level adapter byte/open helpers.
        ask_tools: Dict[str, Callable] = methods_to_tool_dict(
            # Retrieval helpers
            self._list_columns,
            self._tables_overview,
            self._filter_files,
            self._search_files,
            # Inventory listing
            self.list,
            # Parse when missing (policy enforced in prompts)
            self.parse,
            # Delegate to file-scoped Q&A when needed
            self.ask_about_file,
            # Simple existence probe (JSON-safe)
            self._exists,
            include_class_name=False,
        )
        self.add_tools("ask", ask_tools)
        # Multi-table tools (joins across per-file tables)
        ask_multi_table_tools: Dict[str, Callable] = methods_to_tool_dict(
            self._filter_join,
            self._search_join,
            self._filter_multi_join,
            self._search_multi_join,
            include_class_name=False,
        )
        self.add_tools("ask.multi_table", ask_multi_table_tools)
        ask_about_file_tools: Dict[str, Callable] = methods_to_tool_dict(
            # Read-only helpers
            self.parse,
            self._list_columns,
            self._tables_overview,
            self._filter_files,
            self._search_files,
            # Join/multi-join tools for file-scoped analysis
            self._filter_join,
            self._search_join,
            self._filter_multi_join,
            self._search_multi_join,
            self._exists,
            include_class_name=False,
        )
        self.add_tools("ask_about_file", ask_about_file_tools)
        # Organize is mutation-focused. It may call ask() to gather context,
        # but should not receive direct read-only retrieval tools itself.
        organize_tools: Dict[str, Callable] = methods_to_tool_dict(
            self.ask,
            self._rename_file,
            self._move_file,
            self._delete_file,
            include_class_name=False,
        )
        self.add_tools("organize", organize_tools)

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

    @staticmethod
    def _safe(value: Any) -> str:
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

            cleaned = _re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))
            cleaned = cleaned[:64]
            return cleaned or "item"
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

    def _ctx_for_file_table(self, filename: str, table: str) -> str:
        """
        Return the fully‑qualified Unify context name for a per‑file table.

        Parameters
        ----------
        filename : str
            The logical identifier/path of the file whose table context is requested.
        table : str
            Logical per‑file table name (e.g. "Products").

        Returns
        -------
        str
            Fully‑qualified context path in the form
            ``<base>/File/<alias>/<safe_filename>/Tables/<safe_table>``.
        """
        return _ops_per_file_table_ctx(self, filename=filename, table=table)

    def _ctx_for_file(self, filename: str) -> str:
        """
        Return the fully‑qualified Unify context name for the per‑file root (Content).

        Parameters
        ----------
        filename : str
            The logical identifier/path of the file whose per‑file context is requested.

        Returns
        -------
        str
            Fully‑qualified context path in the form
            ``<base>/File/<alias>/<safe_filename>/Content``.
        """
        return _ops_per_file_ctx(self, filename=filename)

    # ---------- Join helpers (delegations to search module) ------------------- #
    @read_only
    def _resolve_table_ref(self, ref: str) -> str:
        """
        Resolve a table reference to a fully-qualified Unify context.

        Parameters
        ----------
        ref : str
            A table reference understood by the manager's join tools. Typically
            of the form "<file>:<table>" for per-file tables, or other
            manager-supported aliases.

        Returns
        -------
        str
            Fully-qualified Unify context path for the referenced table.
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
            Manager-level table references to join (resolved by the search module).
        join_expr : str
            Boolean join predicate (uses the table identifiers present in refs).
        select : dict[str, str]
            Mapping of source expressions to output column names for the derived context.
        mode : str, default "inner"
            Join mode (e.g., "inner", "left", "right", "outer").
        left_where, right_where : str | None
            Optional row-level predicates applied before the join on each side.

        Returns
        -------
        str
            The fully-qualified destination context that was created or re-used.
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

    def _adapter_get(self, *, target_id_or_path: str) -> Dict[str, Any]:
        """
        Return adapter metadata for a file identified by id or path.

        Parameters
        ----------
        target_id_or_path : str
            Adapter-native identifier or path for the file.

        Returns
        -------
        dict
            Adapter metadata for the file (shape is adapter-specific).
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for direct file lookups")
        ref = self._adapter.get_file(target_id_or_path)
        return getattr(ref, "model_dump", lambda: ref.__dict__)()

    def _adapter_open_bytes(self, *, target_id_or_path: str) -> Dict[str, Any]:
        """
        Open raw file bytes via the adapter and return a safe payload.

        Parameters
        ----------
        target_id_or_path : str
            Adapter-native identifier or path for the file.

        Returns
        -------
        dict
            Either a base64-encoded payload with key "bytes_b64" or a fallback
            including the byte length.
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for opening file bytes")
        data = self._adapter.open_bytes(target_id_or_path)
        # Return as base64-ish payload for safety; caller can decide how to use
        try:
            import base64

            return {
                "file_path": target_id_or_path,
                "bytes_b64": base64.b64encode(data).decode("utf-8"),
            }
        except Exception:
            return {"file_path": target_id_or_path, "length": len(data)}

    def _open_bytes_by_filename(self, filename: str) -> bytes:
        """
        Return file bytes by consulting stored metadata first, then the adapter.

        Parameters
        ----------
        filename : str
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
            return self._adapter.open_bytes(filename)
        raise FileNotFoundError(f"Unable to resolve file bytes for '{filename}'")

    @read_only
    def _exists(self, *, filename: str) -> Dict[str, Any]:
        """
        Check if a file exists in the filesystem and return a JSON-safe payload.

        Parameters
        ----------
        filename : str
            Logical identifier/path to check.

        Returns
        -------
        dict
            Shape: {"exists": bool}. This wrapper is used for tool calls to
            avoid non-JSON scalar content in tool results.
        """
        return {"exists": bool(self.exists(filename))}

    # ---------- Update/Delete helpers (private, follow manager patterns) ----- #
    def _update_file(
        self,
        *,
        file_id: int,
        _log_id: Optional[int] = None,
        **updates: Any,
    ) -> Dict[str, Any]:
        """
        Update one or more fields of an existing file record in Unify.

        This is a low-level helper that follows the same pattern as
        _update_contact, _update_rows, _update_secret in other managers.

        Parameters
        ----------
        file_id : int
            The unique file ID to update
        _log_id : int | None
            Optional: The specific log ID if already known (avoids lookup)
        **updates : Any
            Field names and new values to update

        Returns
        -------
        dict
            Outcome with 'outcome' and 'details' keys

        Raises
        ------
        ValueError
            If no file found with the given file_id
        """
        if not updates:
            raise ValueError("At least one field must be provided for update")

        # Find the log ID if not provided
        if _log_id is None:
            try:
                log_ids = unify.get_logs(
                    context=self._ctx,
                    filter=f"file_id == {file_id}",
                    limit=2,
                    return_ids_only=True,
                )
            except Exception:
                log_ids = []

            if not log_ids:
                raise ValueError(f"No file found with file_id {file_id}")
            if len(log_ids) > 1:
                raise ValueError(
                    f"Multiple files found with file_id {file_id}. Data integrity issue.",
                )

            _log_id = log_ids[0]

        # Perform the update
        unify.update_logs(
            logs=[_log_id],
            context=self._ctx,
            entries=updates,
            overwrite=True,
        )

        return {
            "outcome": "file updated",
            "details": {"file_id": file_id, "updated_fields": list(updates.keys())},
        }

    # ---------- Adapter-backed mutators (capability-guarded) --------------- #
    def _rename_file(self, *, target_id_or_path: str, new_name: str) -> Dict[str, Any]:
        """
        Rename a file in the underlying filesystem and update index metadata.

        Parameters
        ----------
        target_id_or_path : str
            Adapter-native identifier or path for the file.
        new_name : str
            New filename (adapter-specific semantics apply for paths).

        Returns
        -------
        dict
            Adapter reference or a minimal dict describing the updated path/name.

        Raises
        ------
        PermissionError
            If the adapter indicates rename is not permitted.
        ValueError
            If the target cannot be located in the index or multiple matches exist.
        """
        if not getattr(self._adapter.capabilities, "can_rename", False):
            raise PermissionError("Rename not permitted by backend policy")

        # Ensure string type (LLM sometimes passes integers)
        target_id_or_path = str(target_id_or_path).lstrip("/")
        new_name = str(new_name)

        # Fetch existing log entry with all fields for comprehensive update
        try:
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"file_path == {target_id_or_path!r}",
                limit=2,
                from_fields=["file_id", "file_path", "records", "metadata"],
            )
        except Exception:
            logs = []

        if not logs:
            raise ValueError(f"File '{target_id_or_path}' not found in Unify logs.")
        if len(logs) > 1:
            raise ValueError(
                f"Multiple files found with filename '{target_id_or_path}'. Data integrity issue.",
            )

        log_entry = logs[0].entries
        log_id = logs[0].id
        file_id = log_entry.get("file_id")
        old_file_path = log_entry.get("file_path", target_id_or_path)

        # The file_path field IS the filesystem path for all our adapters
        filesystem_path = target_id_or_path

        # Perform the rename via adapter FIRST
        # Only update Unify if this succeeds (doesn't raise exception)
        ref = self._adapter.rename(filesystem_path, new_name)
        new_path = ref.path.lstrip("/")

        # Filesystem operation succeeded, now update the index path in Unify
        try:
            self._update_file(
                file_id=file_id,
                _log_id=log_id,
                file_path=new_path,
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Filesystem rename succeeded but Unify update failed: {e}",
            )

        return getattr(
            ref,
            "model_dump",
            lambda: {"path": ref.path, "name": new_name},
        )()

    def _move_file(
        self,
        *,
        target_id_or_path: str,
        new_parent_path: str,
    ) -> Dict[str, Any]:
        """
        Move a file to a different directory and update index metadata.

        Parameters
        ----------
        target_id_or_path : str
            Adapter-native identifier or path for the file.
        new_parent_path : str
            Destination directory in adapter-native form.

        Returns
        -------
        dict
            Adapter reference or a minimal dict describing the updated path/parent.

        Raises
        ------
        PermissionError
            If the adapter indicates move is not permitted.
        ValueError
            If the target cannot be located in the index or multiple matches exist.
        """
        if not getattr(self._adapter.capabilities, "can_move", False):
            raise PermissionError("Move not permitted by backend policy")

        # Ensure string type and strip leading slashes (LLM sometimes passes integers or absolute paths)
        target_id_or_path = str(target_id_or_path).lstrip("/")
        new_parent_path = str(new_parent_path).lstrip("/")

        # Fetch existing log entry with all fields for comprehensive update
        try:
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"file_path == {target_id_or_path!r}",
                limit=2,
                from_fields=["file_id", "file_path", "records", "metadata"],
            )
        except Exception:
            logs = []

        if not logs:
            raise ValueError(f"File '{target_id_or_path}' not found in Unify logs.")
        if len(logs) > 1:
            raise ValueError(
                f"Multiple files found with filename '{target_id_or_path}'. Data integrity issue.",
            )
        log_entry = logs[0].entries
        log_id = logs[0].id
        file_id = log_entry.get("file_id")
        old_file_path = log_entry.get("file_path", target_id_or_path)

        # The file_path field IS the filesystem path for all our adapters
        filesystem_path = target_id_or_path

        # Perform the move via adapter FIRST
        # Only update Unify if this succeeds (doesn't raise exception)
        ref = self._adapter.move(filesystem_path, new_parent_path)
        new_path = ref.path.lstrip("/")

        # Filesystem operation succeeded, now update the index path in Unify
        try:
            self._update_file(
                file_id=file_id,
                _log_id=log_id,
                file_path=new_path,
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Filesystem move succeeded but Unify update failed: {e}",
            )

        return getattr(
            ref,
            "model_dump",
            lambda: {"path": ref.path, "parent": new_parent_path},
        )()

    def _delete_file(
        self,
        *,
        file_id: int,
        _log_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Delete a file record from Unify table and optionally from the underlying filesystem.

        This method follows the same pattern as _delete_contact, _delete_secret in other managers.

        Steps:
        1. Lookup file by file_id to get file_path and metadata
        2. Check if the file is protected (raises PermissionError if protected)
        3. If adapter supports deletion (can_delete=True), delete from filesystem
        4. Remove the record from the Unify table using unify.delete_logs with log ID

        Parameters
        ----------
        file_id : int
            Unique file ID from the Unify table
        _log_id : int | None
            Optional: The specific log ID if already known (avoids lookup)

        Returns
        -------
        dict
            Result with 'outcome' and 'details' keys

        Raises
        ------
        ValueError
            If no file with the given file_id exists
        PermissionError
            If the file is protected
        RuntimeError
            If multiple files found with the same file_id (data integrity issue)
        """
        # 1. Lookup file by ID and get log ID if not provided
        if _log_id is None:
            try:
                log_ids = unify.get_logs(
                    context=self._ctx,
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

        # Get file_path for protected check and filesystem deletion
        try:
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"file_id == {file_id}",
                limit=1,
                from_fields=["file_id", "file_path"],
            )
        except Exception:
            logs = []

        if not logs:
            raise ValueError(f"No file found with file_id {file_id}")

        entry = logs[0].entries
        filename = entry.get("file_path", "")

        # 2. Check if protected
        if self.is_protected(filename):
            raise PermissionError(
                f"'{filename}' is protected and cannot be deleted by FileManager.",
            )

        # 3. Delete from filesystem if adapter supports it
        if self._adapter is not None and getattr(
            self._adapter.capabilities,
            "can_delete",
            False,
        ):
            try:
                # The file_path field IS the filesystem path - use it directly
                self._adapter.delete(filename)
            except NotImplementedError:
                # Adapter doesn't support deletion, just remove from table
                pass
            except FileNotFoundError:
                # File already deleted from filesystem, just clean up table
                pass

        # 4. Delete from Unify table using log ID (not filter)
        try:
            unify.delete_logs(
                context=self._ctx,
                logs=_log_id,
            )
        except Exception as e:
            raise ValueError(f"Failed to delete file record: {e}")

        return {
            "outcome": "file deleted",
            "details": {
                "file_id": file_id,
                "file_path": filename,
            },
        }

    # ---------- Unify-backed retrieval + BaseFileManager API --------------- #
    def exists(self, filename: str) -> bool:  # type: ignore[override]
        """Check if a file exists in the underlying filesystem.

        This method delegates to the adapter's exists method to check
        if the file exists in the filesystem. It does NOT check Unify logs,
        which only contain parsed files.

        Parameters
        ----------
        filename : str
            The display name or path of the file to check.

        Returns
        -------
        bool
            True if the file exists in the filesystem, False otherwise.
        """
        if self._adapter is None:
            return False
        try:
            ok = self._adapter.exists(filename)
        except Exception:
            return False
        return bool(ok)

    def list(self) -> List[str]:  # type: ignore[override]
        """List all files in the underlying filesystem.

        This method delegates to the adapter's list method to list
        files in the filesystem. It does NOT query Unify logs, which only
        contain parsed files.

        Returns
        -------
        list[str]
            List of file paths/display names in the filesystem.
        """
        if self._adapter is None:
            return []
        try:
            items = self._adapter.list()
        except Exception:
            return []
        return list(items or [])

    # -------------------- Config + ingestion helpers (private) -------------------- #
    def _ingest(
        self,
        *,
        filename: str,
        document: Any,
        result: Dict[str, Any],
        config: _FilePipelineConfig,
    ) -> List[int]:
        """Index the FileRecord and ingest content + tables according to config.

        Parameters
        ----------
        filename : str
            Logical identifier/path of the file parsed.
        document : Any
            Parsed document object (used for table ingestion).
        result : dict
            Parse result from ``Document.to_parse_result``.
        config : FilePipelineConfig
            Pipeline configuration controlling layout and table ingestion.

        Returns
        -------
        list[int]
            Inserted log event ids for the content context (when available).
        """
        # 1) Index the file record (best-effort)
        try:
            _ops_create_file_record(
                self,
                entry={
                    "file_path": filename,
                    "status": result.get("status"),
                    "error": result.get("error"),
                    "summary": result.get("summary"),
                    "file_type": result.get("file_type"),
                    "file_size": result.get("file_size"),
                    "total_records": result.get("total_records"),
                    "processing_time": result.get("processing_time"),
                    "created_at": result.get("created_at"),
                    "modified_at": result.get("modified_at"),
                    "confidence_score": result.get("confidence_score"),
                    "key_topics": result.get("key_topics"),
                    "named_entities": result.get("named_entities"),
                    "content_tags": result.get("content_tags"),
                },
            )
        except Exception:
            pass

        # 2) Ingest content rows
        inserted_ids = self._ingest_file(
            filename=filename,
            records=list(result.get("records", []) or []),
            config=config,
        )

        # 3) Ingest per-file tables if enabled
        if config.ingest.table_ingest:
            try:
                dest_name = (
                    filename
                    if config.ingest.mode == "per_file"
                    else (config.ingest.unified_label or "Unified")
                )
                self._ingest_tables_for_file(
                    filename=dest_name,
                    document=document,
                    table_rows_batch_size=config.ingest.table_rows_batch_size,
                )
            except Exception:
                pass

        return inserted_ids

    def _embed(
        self,
        *,
        filename: str,
        document: Any,
        result: Dict[str, Any],
        inserted_ids: Optional[List[int]],
        config: _FilePipelineConfig,
    ) -> None:
        """Create embeddings as specified by config for content and/or tables.

        Parameters
        ----------
        filename : str
            Logical identifier/path of the file parsed.
        document : Any
            Parsed document object (unused for content embeddings; may be used
            by future table/image embedding policies).
        result : dict
            Parse result dict (used to identify columns/rows contextually).
        inserted_ids : list[int] | None
            Inserted ids for content rows when available; used to scope embedding.
        config : FilePipelineConfig
            Embeddings configuration.
        """
        if not (config.embed.embed_along and config.embed.specs):
            return
        # Pre-embed hooks
        try:
            for fn in _resolve_callables(config.plugins.pre_embed):
                try:
                    fn(
                        manager=self,
                        filename=filename,
                        result=result,
                        document=document,
                        config=config,
                    )
                except Exception:
                    continue
        except Exception:
            pass

        try:
            ctx_name = (
                filename
                if config.ingest.mode == "per_file"
                else (config.ingest.unified_label or "Unified")
            )
            for spec in config.embed.specs:
                if spec.context == "per_file":
                    ctx = _ops_per_file_ctx(self, filename=ctx_name)
                    ensure_vector_column(
                        ctx,
                        embed_column=spec.target_column,
                        source_column=spec.source_column,
                        from_ids=(inserted_ids or None),
                    )
                elif spec.context == "unified":
                    ctx = _ops_per_file_ctx(
                        self,
                        filename=(config.ingest.unified_label or "Unified"),
                    )
                    ensure_vector_column(
                        ctx,
                        embed_column=spec.target_column,
                        source_column=spec.source_column,
                        from_ids=(inserted_ids or None),
                    )
                elif spec.context == "per_file_table":
                    overview = self._tables_overview(file=ctx_name)
                    for _, meta in overview.items():
                        ctx_label = meta.get("context")
                        if not ctx_label or "/Tables/" not in ctx_label:
                            continue
                        if getattr(spec, "table", None) not in (None, "*"):
                            if spec.table not in (meta.get("label"), meta.get("name")):
                                continue
                        ensure_vector_column(
                            ctx_label,
                            embed_column=spec.target_column,
                            source_column=spec.source_column,
                        )
        except Exception:
            pass

        # Post-embed hooks
        try:
            for fn in _resolve_callables(config.plugins.post_embed):
                try:
                    fn(
                        manager=self,
                        filename=filename,
                        result=result,
                        document=document,
                        config=config,
                    )
                except Exception:
                    continue
        except Exception:
            pass

    def _ingest_file(
        self,
        *,
        filename: str,
        records: List[Dict[str, Any]],
        config: _FilePipelineConfig,
    ) -> List[int]:
        """Ingest flattened content rows for one file into the configured context.

        Parameters
        ----------
        filename : str
            Logical identifier/path of the source file. This is used for
            indexing lookups and as the default `file_path` on rows.
        records : list[dict]
            Flat per-file content rows (Document.to_schema_rows output).
        config : FilePipelineConfig
            Pipeline configuration controlling layout, replace_existing, and
            allowed_columns filtering.

        Returns
        -------
        list[int]
            Inserted Unify log-event ids, when available from `_ops_create_file`.
            Empty list on failure or when the backend does not return ids.
        """
        try:
            rows: List[Dict[str, Any]] = list(records or [])
            # Filter to allowed columns if provided
            allowed = (
                set(config.ingest.allowed_columns)
                if config.ingest.allowed_columns
                else None
            )
            if allowed:
                rows = [{k: v for k, v in rec.items() if k in allowed} for rec in rows]

            # Determine destination context "filename" (per-file or unified bucket)
            dest_name = (
                filename
                if config.ingest.mode == "per_file"
                else (config.ingest.unified_label or "Unified")
            )

            # Ensure context exists
            _ops_ensure_per_file_context(self, filename=dest_name)

            # Optional: delete existing rows for the same file/doc identifiers
            if config.ingest.replace_existing and rows:
                fr = rows[0]
                preds: List[str] = []
                if fr.get("document_id") is not None:
                    preds.append(f"document_id == '{fr['document_id']}'")
                if fr.get("file_path"):
                    preds.append(f"file_path.endswith('{fr['file_path']}')")
                if fr.get("document_fingerprint"):
                    preds.append(
                        f"document_fingerprint == '{fr['document_fingerprint']}'",
                    )
                if preds:
                    try:
                        _ops_delete_per_file_rows_by_filter(
                            self,
                            filename=dest_name,
                            filter_expr=" or ".join(f"({p})" for p in preds),
                        )
                    except Exception:
                        pass

            # Lookup file_id from index to set FK on rows
            try:
                _rows = unify.get_logs(
                    context=self._ctx,
                    filter=f"file_path == {filename!r}",
                    limit=1,
                    from_fields=["file_id"],
                )
                _fid = _rows[0].entries.get("file_id") if _rows else None
            except Exception:
                _fid = None

            to_add: List[Dict[str, Any]] = []
            for rec in rows:
                new_rec = dict(rec)
                new_rec.setdefault("file_path", filename)
                if _fid is not None:
                    new_rec.setdefault("file_id", _fid)
                to_add.append(new_rec)

            try:
                inserted_ids = _ops_create_file(self, filename=dest_name, rows=to_add)
                return list(inserted_ids or [])
            except Exception:
                return []
        except Exception:
            return []

        # ---------- Per-table ingestion for spreadsheets (CSV/XLSX/Sheets) ----- #

    def _ingest_tables_for_file(
        self,
        *,
        filename: str,
        document: Any,
        table_rows_batch_size: int = 100,
    ) -> None:
        """
        Create one sub-context per extracted table and log its rows.

        Parameters
        ----------
        filename : str
            Logical file identifier/path.
        document : Any
            Parsed document object exposing ``metadata.tables`` with ``rows`` and
            optional ``columns`` / ``sheet_name``.
        table_rows_batch_size : int, default 100
            Batch size for logging rows to the per-table context.

        Notes
        -----
        - Context naming: File/<alias>/<safe_filename>/Tables/<safe_table_label>.
        - Schema is dynamic per table; fields are inferred by the backend; a unique
          ``row_id`` is auto-counted.

        Returns
        -------
        None
        """
        try:
            from unity.knowledge_manager.types import ColumnType
        except Exception:
            ColumnType = None  # type: ignore

        tables = getattr(getattr(document, "metadata", None), "tables", []) or []
        if not tables:
            return

        # Only ingest when structured rows/columns are available
        for idx, tbl in enumerate(tables, start=1):
            columns = getattr(tbl, "columns", None)
            rows = getattr(tbl, "rows", None)
            if not rows:
                continue

            # Derive column names. If missing, try to use keys from first row when dict-like
            if not columns:
                first = rows[0]
                if isinstance(first, dict):
                    columns = list(first.keys())
                else:
                    # Fallback to generic headers based on row length
                    try:
                        num_cols = len(first)
                    except Exception:
                        num_cols = 0
                columns = [f"col_{i+1}" for i in range(num_cols)]

            # Build a stable table context name: sheet_name → safe(section_path) → idx
            sheet_name = getattr(tbl, "sheet_name", None)
            if sheet_name:
                table_label = f"{sheet_name}"
            else:
                section_path = getattr(tbl, "section_path", None)
                if section_path:
                    try:
                        table_label = self._safe(str(section_path))
                    except Exception:
                        table_label = f"{idx:02d}"
                else:
                    table_label = f"{idx:02d}"

            # Ensure per-file-table context with fields from columns or first row keys
            try:
                _ops_ensure_per_file_table_context(
                    self,
                    filename=filename,
                    table=table_label,
                    columns=list(columns) if columns else None,
                    example_row=(
                        rows[0] if (rows and isinstance(rows[0], dict)) else None
                    ),
                )
            except Exception as e:
                print(f"Error ensuring per-file table context: {e}")

            # Batch rows for efficient logging via ops
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
                    try:
                        _ops_create_file_table(
                            self,
                            filename=filename,
                            table=table_label,
                            rows=batch,
                            columns=list(columns) if columns else None,
                            example_row=(
                                rows[0]
                                if (rows and isinstance(rows[0], dict))
                                else None
                            ),
                        )
                    except Exception as e:
                        print(f"Error logging table rows batch: {e}")
                    batch = []

            # Flush any remaining rows
            if batch:
                try:
                    _ops_create_file_table(
                        self,
                        filename=filename,
                        table=table_label,
                        rows=batch,
                        columns=list(columns) if columns else None,
                        example_row=(
                            rows[0] if (rows and isinstance(rows[0], dict)) else None
                        ),
                    )
                except Exception as e:
                    print(f"Error logging final table rows batch: {e}")

    def parse(self, filenames: Union[str, List[str]], *, config: Optional[_FilePipelineConfig] = None) -> Dict[str, Dict[str, Any]]:  # type: ignore[override]
        """
        Parse one or more files, then ingest their content and tables according to config.

        Parameters
        ----------
        filenames : str | list[str]
            One or more logical file identifiers to parse.
        config : FilePipelineConfig | None
            Pipeline configuration controlling parser kwargs, ingest layout,
            table ingestion and embedding behavior. When None, defaults are used
            (equivalent to ``FilePipelineConfig()``).

        Returns
        -------
        dict[str, dict]
            Mapping of filename → result dict (status, records, flattened metadata fields).
        """
        cfg = config or _FilePipelineConfig()

        if isinstance(filenames, str):
            filenames = [filenames]

        results: Dict[str, Dict[str, Any]] = {}
        file_paths: List[str] = []
        filename_to_path: Dict[str, str] = {}

        temp_dir: Optional[str] = None
        try:
            import tempfile as _tempfile

            temp_dir = _tempfile.mkdtemp(prefix="filemanager_parse_")
            print(f"[FileManager] Created temporary directory: {temp_dir}")

            # Export files to a local temp directory
            for name in filenames:
                try:
                    exported_path = self.export_file(name, temp_dir)
                    print(f"[FileManager] Exported file to: {exported_path}")
                    file_paths.append(exported_path)
                    filename_to_path[exported_path] = name
                except Exception as e:
                    # Per-file export failure → do not fail the entire tool call
                    results[name] = _Doc.error_result(name, f"export failed: {e}")

            # Nothing exported successfully
            if not file_paths:
                return results

            # Parse exported files
            documents: List[Any] = []
            if len(file_paths) > 1 and hasattr(self._parser, "parse_batch"):
                try:
                    documents = self._parser.parse_batch(
                        file_paths,
                        **cfg.parse.parser_kwargs,
                    )
                except Exception as e:
                    # Batch failure → mark all remaining as errors
                    for fp in file_paths:
                        name = filename_to_path.get(fp)
                        if name and name not in results:
                            results[name] = _Doc.error_result(
                                name,
                                f"parse_batch failed: {e}",
                            )
                    return results
            else:
                try:
                    documents = [
                        self._parser.parse(file_paths[0], **cfg.parse.parser_kwargs),
                    ]
                except Exception as e:
                    name = filename_to_path.get(file_paths[0], file_paths[0])
                    results[name] = _Doc.error_result(name, str(e))
                    return results

            # Build results and ingest per-file artifacts
            for idx, document in enumerate(documents):
                fp = file_paths[idx] if idx < len(file_paths) else None
                if fp is None:
                    continue
                name = filename_to_path.get(fp, fp)
                try:
                    # Post-parse hooks (document available)
                    try:
                        for fn in _resolve_callables(cfg.plugins.post_parse):
                            try:
                                fn(
                                    manager=self,
                                    filename=name,
                                    result=None,
                                    document=document,
                                    config=cfg,
                                )
                            except Exception as e:
                                results[name] = _Doc.error_result(
                                    name,
                                    f"post-parse hook failed: {e}",
                                )
                    except Exception as e:
                        results[name] = _Doc.error_result(
                            name,
                            f"post-parse hooks failed: {e}",
                        )
                    result = document.to_parse_result(
                        name,
                        auto_counting=cfg.ingest.auto_counting_per_file,
                        document_index=idx,
                    )
                    results[name] = result
                except Exception as e:
                    results[name] = _Doc.error_result(name, f"parse failed: {e}")

                # Ingest (index + content + tables) and then embed
                inserted_ids = self._ingest(
                    filename=name,
                    document=document,
                    result=result,
                    config=cfg,
                )
                self._embed(
                    filename=name,
                    document=document,
                    result=result,
                    inserted_ids=inserted_ids,
                    config=cfg,
                )

            return results
        finally:
            # Clean up temporary directory
            if temp_dir:
                try:
                    import shutil as _shutil2

                    _shutil2.rmtree(temp_dir)
                except Exception:
                    pass

    async def parse_async(
        self,
        filenames: Union[str, List[str]],
        *,
        config: Optional[_FilePipelineConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Parse one or more files asynchronously, then ingest according to config.

        Parameters
        ----------
        filenames : str | list[str]
            One or more logical file identifiers to parse.
        config : FilePipelineConfig
            Pipeline configuration controlling parser kwargs, ingest layout,
            table ingestion and embedding behavior.

        Yields
        ------
        dict
            Result dict per file with status, records, and flattened metadata fields.
        """
        cfg = config or _FilePipelineConfig()
        eff_batch_size: int = int(cfg.parse.batch_size or 3)

        if isinstance(filenames, str):
            filenames = [filenames]
        file_paths: List[str] = []
        filename_to_path: Dict[str, str] = {}
        temp_dir: Optional[str] = None

        try:
            import tempfile as _tempfile

            temp_dir = _tempfile.mkdtemp(prefix="filemanager_parse_async_")

            # Export files; yield per-file errors without aborting the stream
            for name in filenames:
                try:
                    exported_path = self.export_file(name, temp_dir)
                    file_paths.append(exported_path)
                    filename_to_path[exported_path] = name
                except Exception as e:
                    yield _Doc.error_result(name, f"export failed: {e}")

            if not file_paths:
                return

            if hasattr(self._parser, "parse_batch_async"):
                produced: set[str] = set()
                try:
                    async for index, document in self._parser.parse_batch_async(
                        file_paths,
                        batch_size=eff_batch_size,
                        **cfg.parse.parser_kwargs,
                    ):
                        fp = file_paths[index]
                        name = filename_to_path[fp]
                        produced.add(fp)
                        # Post-parse hooks
                        try:
                            for fn in _resolve_callables(cfg.plugins.post_parse):
                                try:
                                    fn(
                                        manager=self,
                                        filename=name,
                                        result=None,
                                        document=document,
                                        config=cfg,
                                    )
                                except Exception as e:
                                    yield _Doc.error_result(
                                        name,
                                        f"post-parse hook failed: {e}",
                                    )
                                    continue
                        except Exception:
                            yield _Doc.error_result(
                                name,
                                f"post-parse hooks failed: {e}",
                            )
                        result = document.to_parse_result(
                            name,
                            auto_counting=cfg.ingest.auto_counting_per_file,
                            document_index=index,
                        )
                        # Ingest and embed
                        inserted_ids = self._ingest(
                            filename=name,
                            document=document,
                            result=result,
                            config=cfg,
                        )
                        self._embed(
                            filename=name,
                            document=document,
                            result=result,
                            inserted_ids=inserted_ids,
                            config=cfg,
                        )
                    yield result
                except Exception as e:
                    # Emit error results for any files that didn't produce output yet
                    for fp in file_paths:
                        if fp in produced:
                            continue
                        name = filename_to_path[fp]
                        yield _Doc.error_result(name, f"parse_batch_async failed: {e}")
            else:
                # Fallback: sequential parse in async generator
                for fp in file_paths:
                    name = filename_to_path[fp]
                    try:
                        document = self._parser.parse(fp, **cfg.parse.parser_kwargs)
                        result = document.to_parse_result(
                            name,
                            auto_counting=cfg.ingest.auto_counting_per_file,
                            document_index=0,
                        )
                        # Ingest and embed
                        inserted_ids = self._ingest(
                            filename=name,
                            document=document,
                            result=result,
                            config=cfg,
                        )
                        self._embed(
                            filename=name,
                            document=document,
                            result=result,
                            inserted_ids=inserted_ids,
                            config=cfg,
                        )
                        yield result
                    except Exception as e:
                        yield _Doc.error_result(name, str(e))
        finally:
            # Clean up temporary directory and all files within it
            if temp_dir:
                try:
                    import shutil as _shutil

                    _shutil.rmtree(temp_dir)
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

    @read_only
    def _tables_overview(
        self,
        *,
        include_column_info: bool = True,
        file: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return an overview of contexts managed by this FileManager.

        Parameters
        ----------
        include_column_info : bool, default True
            When True and ``file`` is None, include the index schema (columns→types).
        file : str | None, default None
            When provided, return a file-scoped overview listing the per-file
            context and any extracted per‑table contexts for that file (schemas
            are omitted for file‑scoped entries).

        Returns
        -------
        dict[str, dict]
            Overview mapping of logical names to metadata (e.g., description,
            and optionally columns for the index context).
        """
        return _storage_tables_overview(
            self,
            include_column_info=include_column_info,
            file=file,
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
    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        """
        List index columns, optionally including types.

        Parameters
        ----------
        include_types : bool, default True
            When True, return a dict mapping column → type; otherwise return just
            the column names as a list.

        Returns
        -------
        dict[str, str] | list[str]
            Column→type mapping or a list of column names.
        """
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _allowed_fields(self) -> List[str]:
        """
        Return the set of safe fields for retrieval from the index.

        Notes
        -----
        - Excludes private fields (leading underscore) and vector columns (suffix
          "_emb") to keep payloads lean and avoid accidental large fetches.
        - Ensures built-in fields required by the model remain included.

        Returns
        -------
        list[str]
            Safe column names to request in from_fields.
        """
        cols = self._get_columns()
        allowed = [
            name
            for name in cols.keys()
            if not str(name).startswith("_") and not str(name).endswith("_emb")
        ]
        # Ensure all built-ins present
        for b in self._BUILTIN_FIELDS:
            if b not in allowed:
                allowed.append(b)
        return allowed

    @read_only
    def _filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[FileRecord]:
        """
        Filter files using a boolean Python expression evaluated per row.
        Mirrors ContactManager.filter_contacts.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope. Examples:
            - "file_path.endswith('.pdf')"
            - "status == 'success'"
            - "metadata['file_size'] > 1000000"
            When None, returns all files. String comparisons are case‑sensitive unless
            your expression applies a case‑normalisation.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return. Must be <= 1000.

        Returns
        -------
        List[FileRecord]
            Matching files as File objects in creation order.

        Notes
        -----
        - Be careful with quoting inside the expression. Use single quotes to delimit string
          literals inside the filter string.
        - This tool is brittle for substring searches across text; prefer ``_search_files``
          for that purpose.
        """
        normalized = normalize_filter_expr(filter)
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            from_fields=self._allowed_fields(),
        )

        rows = [FileRecord(**lg.entries) for lg in logs]
        # Write-through cache
        for lg in logs:
            self._data_store.put(lg.entries)
        return rows

    @read_only
    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[FileRecord]:
        """
        Semantic search over files using one or more reference texts.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of ``source_expr → reference_text`` terms that define the search space.
            - ``source_expr`` can be either a simple column name (e.g. ``"bio"``,
              ``"first_name"``) or a full Unify derived‑expression (e.g.
              ``"str({first_name}) + ' ' + str({surname})"``). For expressions, a stable
              derived source column is created automatically if needed.
            - ``reference_text`` is free‑form text which will be embedded using the
              configured embedding model.
              When None or empty dict, returns the most recent files.
        k : int, default 10
            Maximum number of files to return. Must be a positive integer. Must be <= 1000.

        Returns
        -------
        List[File]
            Up to k File objects. When semantic references are provided,
            results are sorted by similarity. When references are omitted,
            returns the most recent files.
        """
        from unity.common.search_utils import table_search_top_k

        # Restrict payload to the File schema to avoid fetching private/vector fields
        allowed_fields = self._allowed_fields()

        rows = table_search_top_k(
            context=self._ctx,
            references=references,
            k=k,
            allowed_fields=allowed_fields,
            unique_id_field="file_id",
        )
        # Write-through cache
        for r in rows:
            self._data_store.put(r)
        return [FileRecord(**r) for r in rows]

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

        Parameters
        ----------
        tables : list[str] | str
            Exactly two table references, e.g., ["A", "B"], or a comma-separated string.
        join_expr : str
            Join predicate using the same identifiers as in ``tables``.
        select : dict[str, str]
            Mapping of source expressions to output names for the result.
        mode : str, default "inner"
            Join mode ("inner", "left", "right", "outer").
        left_where, right_where : str | None
            Optional pre-filters for left/right inputs before joining.
        result_where : str | None, default None
            Predicate applied to the joined result (only over selected output columns).
        result_limit : int, default 100
            Max rows to return from the joined result (<= 1000).
        result_offset : int, default 0
            Pagination offset over the joined result.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping of the derived context label to a list of rows.
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

        Parameters
        ----------
        tables : list[str] | str
            Exactly two table references, e.g., ["A", "B"], or a comma-separated string.
        join_expr : str
            Join predicate using the same identifiers as in ``tables``.
        select : dict[str, str]
            Mapping of source expressions to output names for the result.
        mode : str, default "inner"
            Join mode ("inner", "left", "right", "outer").
        left_where, right_where : str | None
            Optional pre-filters for left/right inputs before joining.
        references : dict[str, str] | None
            Mapping of expressions in the join result to reference text for semantic ranking.
        k : int, default 10
            Maximum number of rows to return (<= 1000).
        filter : str | None
            Predicate over the joined result (only over selected output columns).

        Returns
        -------
        list[dict[str, Any]]
            Up to ``k`` rows from the joined result sorted by semantic similarity.
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

        Parameters
        ----------
        joins : list[dict]
            Ordered join steps. Each step supports keys: "tables", "join_expr",
            "select", optional "mode", "left_where", "right_where".
            Use "$prev" to reference the previous result in later steps.
        result_where : str | None
            Predicate applied over the final result (over selected output columns).
        result_limit : int, default 100
            Max rows to return (<= 1000).
        result_offset : int, default 0
            Pagination offset into the final result set.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping of the final derived context label to its rows.
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

        Parameters
        ----------
        joins : list[dict]
            Ordered join steps (see ``_filter_multi_join`` for shape).
        references : dict[str, str] | None
            Mapping of expressions in the final result to reference text.
        k : int, default 10
            Maximum number of rows to return (<= 1000).
        filter : str | None
            Predicate applied over the final join result.

        Returns
        -------
        list[dict[str, Any]]
            Up to ``k`` rows from the final result, ranked by semantic similarity.
        """
        return _srch_search_multi_join(
            self,
            joins=joins,
            references=references,
            k=k,
            filter=filter,
        )

    # File ingestion / deprecation
    async def _ingest_files(
        self,
        *,
        filenames: Union[str, List[str]],
        config: _FilePipelineConfig,
    ) -> Dict[str, Any]:
        """Parse and ingest multiple files using the config-driven pipeline.

        This is a thin wrapper over `parse_async(..., config=config)` that
        aggregates a simple summary. Ingestion (content + tables + embeddings)
        occurs within the parse pipeline itself.

        Parameters
        ----------
        filenames : str | list[str]
            One or more logical file identifiers to parse and ingest.
        config : FilePipelineConfig
            Pipeline configuration controlling parsing, ingestion, and embeddings.

        Returns
        -------
        dict[str, Any]
            Summary with counts and per-file outcomes.
        """
        try:
            if isinstance(filenames, str):
                filenames = [filenames]
            if not filenames:
                return {"success": False, "error": "No filenames provided"}

            total_inserted = 0
            file_results: Dict[str, Any] = {}

            async for result in self.parse_async(filenames, config=config):
                filename = result.get("file_path") or result.get("filename")
                if result.get("status") == "error":
                    file_results[filename] = {
                        "file_path": filename,
                        "success": False,
                        "error": result.get("error"),
                        "inserted": 0,
                    }
                    continue
                inserted = len(result.get("records", []) or [])
                total_inserted += inserted
                file_results[filename] = {
                    "file_path": filename,
                    "success": True,
                    "inserted": inserted,
                    "error": None,
                }

            successful_files = sum(
                1 for fr in file_results.values() if fr.get("success", False)
            )
            failed_files = len(filenames) - successful_files
            return {
                "success": failed_files == 0,
                "total_files": len(filenames),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_records": total_inserted,
                "total_inserted": total_inserted,
                "file_results": list(file_results.values()),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search_files on the first step; auto thereafter."""
        if step_index < 1 and "search_files" in current_tools:
            return (
                "required",
                {"search_files": current_tools["search_files"]},
            )
        return ("auto", current_tools)

    @staticmethod
    def _default_ask_about_file_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search_files on the first step; auto thereafter."""
        if step_index < 1 and "search_files" in current_tools:
            return (
                "required",
                {"search_files": current_tools["search_files"]},
            )
        return ("auto", current_tools)

    @staticmethod
    def _default_organize_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require ask on the first step; auto thereafter."""
        if step_index < 1 and "ask" in current_tools:
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

    def export_file(self, filename: str, destination_dir: str) -> str:  # type: ignore[override]
        """Export a file from the underlying filesystem to a local destination directory.

        This method delegates to the adapter's export_file method, which:
        - LocalAdapter: copies file preserving metadata
        - CodeSandboxAdapter: downloads via SDK (with download API)
        - InteractAdapter: streams via API

        Parameters
        ----------
        filename : str
            The display name or path of the file to export.
        destination_dir : str
            Local directory path where the file should be exported.

        Returns
        -------
        str
            Full path to the exported file in the destination directory.
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for export_file")
        return self._adapter.export_file(filename, destination_dir)

    def export_directory(self, directory: str, destination_dir: str) -> List[str]:  # type: ignore[override]
        """Export all files from a directory to a local destination directory.

        This method delegates to the adapter's export_directory method, which
        optimizes for batch operations where possible (e.g., zip downloads for CodeSandbox).

        Parameters
        ----------
        directory : str
            The directory path to export files from.
        destination_dir : str
            Local directory path where files should be exported.

        Returns
        -------
        list[str]
            List of full paths to exported files in the destination directory.
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
        """Register a pre-existing file path with the adapter (metadata only)."""
        if self._adapter is None:
            raise NotImplementedError(
                "No adapter configured for register_existing_file",
            )
        return self._adapter.register_existing_file(
            str(path),
            display_name=display_name,
            protected=protected,
        )

    def is_protected(self, filename: str) -> bool:
        """Return True when the file is marked as protected (adapter-specific)."""
        if self._adapter is None:
            return False
        return self._adapter.is_protected(filename)

    def save_file_to_downloads(self, filename: str, contents: bytes) -> str:
        """Save file contents into a downloads area and return the saved path."""
        if self._adapter is None:
            raise NotImplementedError(
                "No adapter configured for save_file_to_downloads",
            )
        return self._adapter.save_file_to_downloads(filename, contents)

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
        overview_json = json.dumps(self._tables_overview(), indent=4)
        system_msg = build_file_manager_ask_prompt(
            tools=tools,
            num_files=self._num_files(),
            columns=self._list_columns(),
            table_schemas_json=overview_json,
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        use_semantic_cache = "both" if is_semantic_cache_enabled() else None
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
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
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
        filename: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        """
        Ask a question about a specific file.

        Parameters
        ----------
        filename : str
            Identifier/path of the file in the underlying adapter.
        question : str
            The user's natural-language question about the file.
        _return_reasoning_steps : bool, default False
            When True, wraps handle.result() to return (answer, messages).
        _parent_chat_context, _clarification_up_q, _clarification_down_q, rolling_summary_in_prompts, _call_id
            See ask().

        Returns
        -------
        SteerableToolHandle
            Interactive tool loop handle (read-only).
        """
        if not self.exists(filename):
            raise FileNotFoundError(filename)
        client = new_llm_client()
        tools = dict(self.get_tools("ask_about_file"))
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
        file_overview_json = json.dumps(self._tables_overview(file=filename), indent=4)
        system_msg = build_file_manager_ask_about_file_prompt(
            tools=tools,
            table_schemas_json=file_overview_json,
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        # Use filesystem type without exposing absolute paths to LLM
        user_blob = json.dumps(
            {"filesystem": self._fs_type, "file_path": filename, "question": question},
            indent=2,
        )
        use_semantic_cache = "both" if is_semantic_cache_enabled() else None
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
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
            semantic_cache=use_semantic_cache,
            semantic_cache_namespace=f"{self.__class__.__name__}.ask_about_file",
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
        overview_json = json.dumps(self._tables_overview(), indent=4)
        system_msg = build_file_manager_organize_prompt(
            tools=tools,
            num_files=len(self.list()),
            columns=self._list_columns(),
            table_schemas_json=overview_json,
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
        Clear this manager's index context and local caches, then re-provision.

        Behaviour
        ---------
        - Drops the index context for this filesystem.
        - Clears any local DataStore mirrors.
        - Re-provisions storage so future operations see a consistent schema.

        Returns
        -------
        None
        """
        try:
            unify.delete_context(self._ctx)
        except Exception:
            pass
        try:
            self._data_store.clear()
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
