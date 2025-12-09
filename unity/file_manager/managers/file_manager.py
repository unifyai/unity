from __future__ import annotations

import json
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import unify

from unity.common.tool_spec import manager_tool, read_only
from unity.file_manager.base import BaseFileManager
from unity.file_manager.parser.base import BaseParser
from unity.file_manager.parser.docling_parser import DoclingParser
from unity.file_manager.types.file import FileRecord
from unity.file_manager.types.config import (
    FilePipelineConfig as _FilePipelineConfig,
)
from unity.file_manager.parser.types.document import Document as _Doc
from unity.file_manager.fs_adapters.base import BaseFileSystemAdapter
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
from unity.constants import is_readonly_ask_guard_enabled
from unity.constants import is_semantic_cache_enabled
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
    ctx_for_file as _storage_ctx_for_file,
    ctx_for_file_table as _storage_ctx_for_file_table,
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
        self.include_in_multi_assistant_table = False
        self._adapter = adapter
        self.__parser: Optional[BaseParser] = parser
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
            # Retrieval helpers
            self._list_columns,
            self._tables_overview,
            self._filter_files,
            self._search_files,
            self._reduce,
            # Inventory listing
            self.list,
            # Unified stat helper (filesystem vs index)
            self.stat,
            # Delegate to file-scoped Q&A when needed
            self.ask_about_file,
            # Simple existence probe
            self.exists,
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
            self.stat,
            self._list_columns,
            self._tables_overview,
            self._filter_files,
            self._search_files,
            self._reduce,
            self.exists,
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
            self.__parser = DoclingParser(
                use_llm_enrichment=True,
                extract_images=True,
                extract_tables=True,
            )
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

    # ------------------------- Identity helpers ----------------------------- #
    def _build_file_identity(self, file_path: str):
        """
        Build a FileIdentity for the given file_path.

        - Uses adapter metadata when available to enrich source_provider and display_path.
        - Resolves source_uri via `_resolve_to_uri` as the canonical identity.
        - Resolves ingest settings (ingest_mode, unified_label, table_ingest) from FileRecords.
        """
        from unity.file_manager.types.file import FileIdentity as _FileIdentity
        from os import path as _os_path

        # Base provider/type and display path
        try:
            source_provider = (
                getattr(self._adapter, "name", None)
                or getattr(self, "_fs_type", None)
                or "Unknown"
            )
        except Exception:
            source_provider = getattr(self, "_fs_type", None) or "Unknown"

        # Canonical URI
        source_uri = self._resolve_to_uri(file_path)

        # Prefer adapter's provider when available
        try:
            # Use path as-is (adapters handle both relative and absolute paths)
            ref = self._adapter_get(file_id_or_path=file_path)
            source_provider = str(ref.get("provider") or source_provider)
        except Exception:
            pass

        # Ingest layout via storage resolver
        ingest_mode = "per_file"
        unified_label = None
        table_ingest = True
        try:
            from .utils.storage import _resolve_file_target as _res_file_target  # type: ignore

            # Use path as-is for querying
            res = _res_file_target(self, file_path)
            ingest_mode = res.get("ingest_mode", "per_file")
            unified_label = res.get("unified_label")
            # Prefer table_ingest from index row when present
            try:
                rows = unify.get_logs(
                    context=self._ctx,
                    filter=f"file_path == {file_path!r}",
                    limit=1,
                    from_fields=["table_ingest"],
                )
                if rows:
                    table_ingest = bool(rows[0].entries.get("table_ingest", True))
            except Exception:
                table_ingest = True
        except Exception:
            pass

        # Compute file_name from path
        base = _os_path.basename(str(file_path))
        name, _ext = _os_path.splitext(base)

        return _FileIdentity(
            file_path=str(file_path),
            source_provider=source_provider,
            source_uri=source_uri,
            ingest_mode=ingest_mode,  # type: ignore[arg-type]
            unified_label=unified_label,
            table_ingest=bool(table_ingest),
        )

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
        # Resolve identity and layout
        ident = self._build_file_identity(file_path)
        overview = self._tables_overview(file=file_path)
        purged = {"content_rows": 0, "table_rows": 0}

        # Purge content rows
        try:
            from .utils.ops import delete_per_file_rows_by_filter as _ops_del_content

            purged["content_rows"] = int(
                _ops_del_content(self, file_path=(file_path if ident.ingest_mode == "per_file" else (ident.unified_label or "Unified")), filter_expr=None),  # type: ignore[arg-type]
            )
        except Exception:
            pass

        # Purge per-file tables (when present)
        try:
            if bool(getattr(ident, "table_ingest", True)):
                from .utils.ops import (
                    delete_per_file_table_rows_by_filter as _ops_del_tbl,
                )

                # tables located under either the per-file root (per_file) or safe(file_path) branch (unified)
                for key, val in overview.items():
                    if key in ("FileRecords", str(getattr(ident, "unified_label", ""))):
                        continue
                    tables = val.get("Tables") if isinstance(val, dict) else None
                    if not tables:
                        # Flat variant: individual table entries at top-level (older shape)
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

        This public tool purges existing rows (respecting ingest layout) and
        re-parses/re-ingests the file, keeping contexts consistent.

        Parameters
        ----------
        file_path : str
            The file identifier/path as used in FileRecords.file_path.

        Returns
        -------
        dict
            Outcome with purge counts and status.
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

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) as preserved in the FileRecords index, or the
            fully-qualified file_path (str) as stored in the FileRecords index/context.
            When a file_id is provided, it is resolved to the corresponding file_path.
        new_name : str
            New file name; adapter determines path semantics.

        Returns
        -------
        dict
            Adapter reference or a minimal dict with the new path/name.

        Behaviour
        ---------
        - Propagates the rename across per-file contexts (and per-file tables) for
          per_file ingest-mode. Unified mode keeps the unified Content context
          unchanged and only renames per-file Tables contexts keyed by the safe
          file path.
        - Updates the FileRecords row (file_path, file_name).

        Raises
        ------
        PermissionError
            If rename is not permitted by the adapter.
        ValueError
            If no or multiple FileRecords match the target.
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

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) as preserved in the FileRecords index, or the
            fully-qualified file_path (str) as stored in the FileRecords index/context.
            When a file_id is provided, it is resolved to the corresponding file_path.
        new_parent_path : str
            Destination directory in adapter-native form.

        Returns
        -------
        dict
            Adapter reference or a minimal dict describing the updated path/parent.

        Behaviour
        ---------
        - Propagates the new path across per-file contexts (and tables) for per_file
          ingest-mode. Unified Content remains under the unified label.
        - Updates the FileRecords row (file_path, file_path) to the new location.

        Raises
        ------
        PermissionError
            If move is not permitted by the adapter.
        ValueError
            If no or multiple FileRecords match the target.
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
        Delete a file record and purge related contexts according to ingest layout.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) as preserved in the FileRecords index, or the
            fully-qualified file_path (str) as stored in the FileRecords index/context.
            When a file_id is provided, it is resolved to the corresponding file_path.
        _log_id : int | None
            Optional existing log ID to delete (speeds up deletion).

        Returns
        -------
        dict
            {"outcome": "file deleted", "details": {"file_id": int, "file_path": str}}

        Behaviour
        ---------
        - Per-file mode: drops the per-file Content context and all per-file Tables contexts.
        - Unified mode: deletes only rows whose source_uri matches from the unified Content
          context. Per-file Tables contexts for the file are removed when present.
        - Adapter deletion is attempted when supported (capability-gated).

        Raises
        ------
        ValueError
            If the file_id_or_path does not exist.
        PermissionError
            If the file is protected.
        RuntimeError
            If multiple rows exist for the same file_id_or_path (integrity issue).
        """
        from .utils.ops import delete_file as _ops_delete

        return _ops_delete(self, file_id_or_path=file_id_or_path, _log_id=_log_id)

    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        """
        Check if a file exists in the adapter-backed filesystem.

        This delegates to the adapter's ``exists`` method and does **not** look
        at Unify logs or parsed/indexed state. Use it to answer "does this
        path currently exist in the underlying filesystem?", regardless of
        whether it has ever been parsed.

        Parameters
        ----------
        file_path : str
            Adapter-native file path/identifier to check.

        Returns
        -------
        bool
            True if the adapter reports the file exists, False otherwise or
            when no adapter is configured.
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

        This delegates to the adapter's ``list`` method and returns the set of
        file paths/identifiers that the adapter can currently see. It does
        **not** query Unify logs, so results are about raw filesystem presence,
        not parsed/indexed state.

        Returns
        -------
        list[str]
            Adapter-native file paths/identifiers discoverable in the underlying
            filesystem, or an empty list if no adapter is configured or listing fails.
        """
        if self._adapter is None:
            return []
        try:
            items = self._adapter.list()
        except Exception:
            return []
        return list(items or [])

    def ingest_files(self, file_paths: Union[str, List[str]], *, config: Optional[_FilePipelineConfig] = None) -> Dict[str, Any]:  # type: ignore[override]
        """
        Run the complete file processing pipeline: parse, ingest, and embed.

        This method orchestrates the full file processing workflow:
        1. Parse files using the configured parser to extract structured content
        2. Ingest parsed content into storage contexts (per-file or unified)
        3. Create embeddings based on the configured strategy (along, after, or off)

        Return payload depends on cfg.output.return_mode:
        - "compact" (default): format-specific Pydantic model (e.g., ParsedPDF, ParsedXlsx)
        - "full": raw ParsedFile from Document.to_parse_result (includes heavy fields)
        - "none": minimal stub with status/format/record counts

        Parameters
        ----------
        file_paths : str | list[str]
            One or more logical file paths to process.
        config : FilePipelineConfig | None
            Pipeline configuration controlling parser kwargs, ingest layout,
            table ingestion and embedding behavior. When None, defaults are used
            (equivalent to ``FilePipelineConfig()``).

        Returns
        -------
        dict[str, Any]
            Mapping of file path → compact Pydantic model, raw ParsedFile, or minimal stub per output mode.
        """
        cfg = config or _FilePipelineConfig()

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        results: Dict[str, Any] = {}
        exported_paths: List[str] = []
        exported_paths_to_original_paths: Dict[str, str] = {}

        temp_dir: Optional[str] = None
        try:
            # Initialize progress reporter when enabled
            from .utils.progress import create_reporter, ProgressReporter

            _reporter: Optional[ProgressReporter] = None

            import tempfile as _tempfile

            temp_dir = _tempfile.mkdtemp(prefix="filemanager_parse_")

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
                    results[path] = _Doc.error_result(path, f"export failed: {e}")

            # Nothing exported successfully
            if not exported_paths:
                return results

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

            # Parse files using parser's batch method (handles parallelization)
            documents: List[Any] = []
            if len(exported_paths) > 1 and hasattr(self._parser, "parse_batch"):
                try:
                    logger.debug(
                        f"[FileManager] Parsing {len(exported_paths)} files in parallel (batch_size={cfg.parse.batch_size})",
                    )
                    # Track parse start time for timing
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

                    documents = self._parser.parse_batch(
                        exported_paths,
                        batch_size=cfg.parse.batch_size,
                        **cfg.parse.parser_kwargs,
                    )

                    # Calculate parse timing
                    parse_duration_ms = (_time.perf_counter() - parse_start_time) * 1000

                    # Report parse completion for all files with timing
                    if enable_progress and _reporter is not None:
                        from .utils.progress import create_progress_event

                        for exp in exported_paths:
                            orig = exported_paths_to_original_paths.get(exp, exp)
                            _reporter.report(
                                create_progress_event(
                                    orig,
                                    "parse",
                                    "completed",
                                    duration_ms=parse_duration_ms,
                                    elapsed_ms=parse_duration_ms,
                                ),
                            )
                except Exception as e:
                    # Calculate timing for failure
                    parse_duration_ms = (
                        (_time.perf_counter() - parse_start_time) * 1000
                        if "parse_start_time" in dir()
                        else 0.0
                    )
                    # Batch failure → mark all remaining as errors
                    for fp in exported_paths:
                        original_path = exported_paths_to_original_paths.get(fp)
                        if original_path and original_path not in results:
                            results[original_path] = _Doc.error_result(
                                original_path,
                                f"parse_batch failed: {e}",
                            )
                            if enable_progress and _reporter is not None:
                                from .utils.progress import create_progress_event

                                _reporter.report(
                                    create_progress_event(
                                        original_path,
                                        "parse",
                                        "failed",
                                        duration_ms=parse_duration_ms,
                                        elapsed_ms=parse_duration_ms,
                                        error=str(e),
                                    ),
                                )
                    return results
            else:
                try:
                    # Track parse start time for timing
                    import time as _time

                    parse_start_time = _time.perf_counter()

                    # Report parse started for single file
                    if enable_progress and _reporter is not None:
                        from .utils.progress import create_progress_event

                        orig0 = exported_paths_to_original_paths.get(
                            exported_paths[0],
                            exported_paths[0],
                        )
                        _reporter.report(
                            create_progress_event(
                                orig0,
                                "parse",
                                "started",
                                duration_ms=0.0,
                                elapsed_ms=0.0,
                            ),
                        )

                    documents = [
                        self._parser.parse(
                            exported_paths[0],
                            **cfg.parse.parser_kwargs,
                        ),
                    ]

                    # Calculate parse timing
                    parse_duration_ms = (_time.perf_counter() - parse_start_time) * 1000

                    if enable_progress and _reporter is not None:
                        from .utils.progress import create_progress_event

                        orig0 = exported_paths_to_original_paths.get(
                            exported_paths[0],
                            exported_paths[0],
                        )
                        _reporter.report(
                            create_progress_event(
                                orig0,
                                "parse",
                                "completed",
                                duration_ms=parse_duration_ms,
                                elapsed_ms=parse_duration_ms,
                            ),
                        )
                except Exception as e:
                    # Calculate timing for failure
                    parse_duration_ms = (
                        (_time.perf_counter() - parse_start_time) * 1000
                        if "parse_start_time" in dir()
                        else 0.0
                    )
                    original_path = exported_paths_to_original_paths.get(
                        exported_paths[0],
                        exported_paths[0],
                    )
                    results[original_path] = _Doc.error_result(original_path, str(e))
                    if enable_progress and _reporter is not None:
                        from .utils.progress import create_progress_event

                        _reporter.report(
                            create_progress_event(
                                original_path,
                                "parse",
                                "failed",
                                duration_ms=parse_duration_ms,
                                elapsed_ms=parse_duration_ms,
                                error=str(e),
                            ),
                        )
                    return results

            # Build results and ingest per-file artifacts using the PipelineExecutor
            from .utils.executor import run_pipeline

            # Map exported paths back to original paths for the pipeline
            original_paths = [
                exported_paths_to_original_paths.get(ep, ep) for ep in exported_paths
            ]

            # run_pipeline handles parallel vs sequential execution based on
            # cfg.execution.parallel_files internally
            verbosity = getattr(cfg.diagnostics, "verbosity", "low")
            results.update(
                run_pipeline(
                    self,
                    documents=documents,
                    file_paths=original_paths,
                    config=cfg,
                    reporter=_reporter,
                    enable_progress=enable_progress,
                    verbosity=verbosity,
                ),
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

        Parameters
        ----------
        table : str | None, default None
            Table reference to aggregate. Accepted forms mirror
            :py:meth:`_filter_files` and :py:meth:`_search_files`:
            logical names from :py:meth:`_tables_overview` such as
            ``\"FileRecords\"``, path-first forms like ``\"<file_path>\"`` or
            ``\"<file_path>.Tables.<label>\"``, or legacy refs such as
            ``\"<file_path>:<table>\"``. When ``None``, aggregates over the
            main FileRecords index.
        metric : str
            Reduction metric to compute. Supported values (case-insensitive) are
            ``\"sum\"``, ``\"mean\"``, ``\"var\"``, ``\"std\"``, ``\"min\"``,
            ``\"max\"``, ``\"median\"``, ``\"mode\"``, and ``\"count\"``.
        keys : str | list[str]
            One or more numeric columns in the resolved context to aggregate. A
            single column name returns a scalar; a list of column names
            computes the metric independently per key and returns a
            ``{key -> value}`` mapping.
        filter : str | dict[str, str] | None, default None
            Optional row-level filter expression(s) using the same Python
            syntax as :py:meth:`_filter_files`. When a string, the expression
            is applied uniformly; when a dict, each key maps to its own filter
            expression.
        group_by : str | list[str] | None, default None
            Optional column(s) to group by. Use a single column name for one
            grouping level, or a list such as ``[\"status\", \"file_id\"]`` to
            group hierarchically in that order. When provided, the result
            becomes a nested mapping keyed by group values, mirroring
            :func:`unify.get_logs_metric` behaviour.

        Returns
        -------
        Any
            Metric value(s) computed over the resolved context:

            * Single key, no grouping  → scalar (float/int/str/bool).
            * Multiple keys, no grouping → ``dict[key -> scalar]``.
            * With grouping             → nested ``dict`` keyed by group values.
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
    def _list_columns(
        self,
        *,
        include_types: bool = True,
        table: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        """
        List columns for the FileRecords index or a resolved logical table.

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column → type; otherwise return the
            list of column names only.
        table : str | None, default None
            When provided, resolve the logical name (e.g., "<file_path>",
            "<file_path>.Tables.<label>") or legacy ref and return that context's
            columns. When None, return the FileRecords index columns.

        Returns
        -------
        dict[str, str] | list[str]
            Column→type mapping or a list of column names.

        Examples
        --------
        - _list_columns() → FileRecords schema
        - _list_columns(table="Q1_Report") → per-file Content schema
        - _list_columns(table="Q1_Report.Tables.Products") → table schema
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

        Parameters
        ----------
        filter : str | None
            Python boolean expression evaluated with column names in scope.
        offset : int
            Zero-based pagination offset per context.
        limit : int
            Maximum rows per context (<= 1000).
        tables : list[str] | str | None
            Table references to filter. Accepted forms:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names from `tables_overview()`: "FileRecords" for index,
              or legacy refs like "<root>" (deprecated)
            - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
            When None, only the FileRecords index is scanned.

        Returns
        -------
        list[dict]
            Flat list of rows collected from the index (when tables=None) or
            concatenated rows from all resolved contexts.
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

        Parameters
        ----------
        references : dict[str, str] | None
            Mapping of source_expr → reference_text. Source expressions can be
            column names or derived expressions. When omitted, returns recent files.
        k : int
            Number of results to return (1..1000).

        table : str | None
            Table reference to search. Accepted forms:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names: "FileRecords" for index, or legacy refs like "<root>" (deprecated)
            - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
            When None, defaults to the global FileRecords index.

        filter : str | None
            Row-level predicate (evaluated with column names as variables).

        Returns
        -------
        list[dict]
            Up to k rows ranked by similarity from the resolved context.
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

        Parameters
        ----------
        tables : list[str] | str
            Exactly two table references. Accepted forms:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names from `tables_overview()` or legacy refs
            - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
        join_expr : str
            Join predicate using the same identifiers as in ``tables`` (auto-rewritten).
        select : dict[str, str]
            Mapping of source expressions → output names.
        mode : str
            One of {"inner", "left", "right", "outer"}.
        left_where, right_where : str | None
            Optional input predicates before joining.
        result_where : str | None
            Predicate applied to the joined result over the projected columns.
        result_limit, result_offset : int
            Pagination parameters; limit <= 1000.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Rows from the materialized join context.
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
            Exactly two table references. Accepted forms:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names from `tables_overview()` or legacy refs
            - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
        join_expr : str
            Join predicate using the same identifiers as in ``tables`` (auto-rewritten).
        select : dict[str, str]
            Mapping of source expressions → output names.
        mode : str
            One of {"inner", "left", "right", "outer"}.
        left_where, right_where : str | None
            Optional input predicates before joining.
        references : dict[str, str] | None
            Mapping of expressions in the join result → reference text for semantic ranking.
        k : int
            Maximum rows to return (<= 1000).
        filter : str | None
            Optional predicate over the joined result before ranking.

        Returns
        -------
        list[dict[str, Any]]
            Top-k rows ranked by semantic similarity.
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
            Ordered steps; each step provides ``tables`` (two refs or "$prev"),
            ``join_expr``, ``select`` and optional ``mode``, ``left_where``, ``right_where``.
            Table references in ``tables`` accept:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names from `tables_overview()` or legacy refs
            - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
            - "$prev" to reference the previous join step's result
        result_where : str | None
            Predicate applied to the final joined result over projected columns.
        result_limit, result_offset : int
            Pagination parameters; limit <= 1000.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Rows from the final materialized context.
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
            Ordered steps; each step provides ``tables`` (two refs or "$prev"),
            ``join_expr``, ``select`` and optional ``mode``, ``left_where``, ``right_where``.
            Table references in ``tables`` accept:
            - Path-first (preferred): "<file_path>" for per-file Content,
              "<file_path>.Tables.<label>" for per-file tables
            - Logical names from `tables_overview()` or legacy refs
            - Legacy forms: "<file_path>:<table>", "id=<file_id>:<table>", "#<file_id>:<table>"
            - "$prev" to reference the previous join step's result
        references : dict[str, str] | None
            Mapping of expressions in the final result → reference text for ranking.
        k : int
            Maximum rows to return (<= 1000).
        filter : str | None
            Optional predicate before ranking.

        Returns
        -------
        list[dict[str, Any]]
            Top-k rows ranked by semantic similarity.
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
        """Require search_files on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_ASK_TOOL_IS_SEARCH
            and step_index < 1
            and "search_files" in current_tools
        ):
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
        overview_json = json.dumps(self._tables_overview(), indent=4)
        system_msg = build_file_manager_ask_prompt(
            tools=tools,
            num_files=self._num_files(),
            columns=self._list_columns(),
            table_schemas_json=overview_json,
            include_activity=include_activity,
            business_payload=business_payload,
        )
        # TODO: REMOVE - Debug file dump for prompt inspection
        open("system_msg.txt", "w").write(system_msg)
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

    # ---------- Unified stat helper (read-only) ------------------------------ #
    @read_only
    def stat(self, path_or_uri: str | int) -> Dict[str, Any]:
        """Return unified status for filesystem vs index existence.

        Returns
        -------
        dict
            {
              "canonical_uri": str | None,
              "filesystem_exists": bool,
              "indexed_exists": bool,
              "parsed_status": str | None,
            }
        """
        canonical_uri = self._resolve_to_uri(path_or_uri)

        fs_exists = False
        try:
            fs_exists = bool(self.exists(path_or_uri))
        except Exception:
            fs_exists = False

        indexed_exists = False
        parsed_status = None
        try:
            # Try by source_uri first when available
            logs = []
            if canonical_uri:
                try:
                    logs = unify.get_logs(
                        context=self._ctx,
                        filter=f"source_uri == {canonical_uri!r}",
                        limit=5,
                        from_fields=["status", "file_path", "source_uri"],
                    )
                except Exception:
                    logs = []
            # Fallback to file_path match
            if not logs:
                try:
                    # Use path as-is for querying
                    logs = unify.get_logs(
                        context=self._ctx,
                        filter=f"file_path == {str(path_or_uri)!r}",
                        limit=5,
                        from_fields=["status", "file_path", "source_uri"],
                    )
                except Exception:
                    logs = []
            indexed_exists = bool(logs)
            if logs:
                # If any 'success' exists, report success
                st = next(
                    (
                        lg.entries.get("status")
                        for lg in logs
                        if lg.entries.get("status")
                    ),
                    None,
                )
                parsed_status = st
        except Exception:
            indexed_exists = False
            parsed_status = None

        return {
            "canonical_uri": canonical_uri,
            "filesystem_exists": fs_exists,
            "indexed_exists": indexed_exists,
            "parsed_status": parsed_status,
        }

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
        file_overview_json = json.dumps(self._tables_overview(file=file_path), indent=4)
        system_msg = build_file_manager_ask_about_file_prompt(
            tools=tools,
            table_schemas_json=file_overview_json,
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        # Use filesystem type without exposing absolute paths to LLM
        user_blob = json.dumps(
            {"filesystem": self._fs_type, "file_path": file_path, "question": question},
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
            try:
                ctxs = list(unify.get_contexts(prefix=per_file_prefix))
            except Exception:
                ctxs = []
            for ctx in sorted(ctxs, key=len, reverse=True):
                try:
                    unify.delete_context(ctx)
                except Exception:
                    continue
            # Attempt to drop the root itself as well
            try:
                unify.delete_context(per_file_prefix)
            except Exception:
                pass
        except Exception:
            pass

        # 2) Delete the index context for this filesystem
        try:
            unify.delete_context(self._ctx)
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
