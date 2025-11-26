from __future__ import annotations

import json
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from typing import AsyncIterator

import unify

from unity.common.tool_spec import manager_tool, read_only
from unity.file_manager.base import BaseFileManager
from unity.file_manager.parser.base import BaseParser
from unity.file_manager.parser.docling_parser import DoclingParser
from unity.file_manager.types.file import FileRecord, FileContent
from unity.file_manager.types.config import (
    FilePipelineConfig as _FilePipelineConfig,
    FileEmbeddingSpec,
    TableEmbeddingSpec,
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
    delete_per_file_rows_by_filter as _ops_delete_per_file_rows_by_filter,
    create_file_record as _ops_create_file_record,
    create_file_content as _ops_create_file_content,
    create_file_table as _ops_create_file_table,
    apply_content_ingest_policy as _ops_apply_content_ingest_policy,
)
from .storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
    tables_overview as _storage_tables_overview,
    ctx_for_file as _storage_ctx_for_file,
    ctx_for_file_table as _storage_ctx_for_file_table,
    ensure_file_table_context as _storage_ensure_file_table_context,
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
        self.__parser: Optional[BaseParser] = parser
        # Optional rich progress manager (initialized during parse/parse_async when enabled)
        self._progress_manager = None
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
            # Read-only helpers
            self.parse,
            self.stat,
            self._list_columns,
            self._tables_overview,
            self._filter_files,
            self._search_files,
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
            from .storage import _resolve_file_target as _res_file_target  # type: ignore

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
            from .ops import delete_per_file_rows_by_filter as _ops_del_content

            purged["content_rows"] = int(
                _ops_del_content(self, file_path=(file_path if ident.ingest_mode == "per_file" else (ident.unified_label or "Unified")), filter_expr=None),  # type: ignore[arg-type]
            )
        except Exception:
            pass

        # Purge per-file tables (when present)
        try:
            if bool(getattr(ident, "table_ingest", True)):
                from .ops import delete_per_file_table_rows_by_filter as _ops_del_tbl

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
            self.parse(file_path, config=cfg)
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
            from .storage import _resolve_file_target as _res

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
        from .ops import rename_file as _ops_rename

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
        from .ops import move_file as _ops_move

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
        from .ops import delete_file as _ops_delete

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

    # -------------------- Config + ingestion helpers (private) -------------------- #
    def _resolve_embedding_specs_for_file(
        self,
        file_path: str,
        config: _FilePipelineConfig,
    ) -> List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]]:
        """
        Resolve file_specs for a given file_path.

        This helper function filters file_specs by file_path (exact match or "*" wildcard)
        and returns matching (FileEmbeddingSpec, TableEmbeddingSpec) pairs.

        Parameters
        ----------
        file_path : str
            The file path to resolve specs for
        config : FilePipelineConfig
            Pipeline configuration

        Returns
        -------
        List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]]
            List of (file_spec, table_spec) pairs applicable to this file_path
        """
        resolved_specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]] = []

        if hasattr(config.embed, "file_specs") and config.embed.file_specs:
            for file_spec in config.embed.file_specs:
                # Check if file_path matches (exact match or "*" wildcard)
                if file_spec.file_path == "*" or file_spec.file_path == file_path:
                    # Add all table specs for this file
                    for table_spec in file_spec.tables:
                        resolved_specs.append((file_spec, table_spec))

        return resolved_specs

    def _ingest(
        self,
        *,
        file_path: str,
        document: Any,
        result: Dict[str, Any],
        config: _FilePipelineConfig,
    ) -> List[int]:
        """Index the FileRecord and ingest content + tables according to config.

        Parameters
        ----------
        file_path : str
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
        if bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        ):
            print(f"[Ingest] Indexing file record for: {file_path}")

        ident = self._build_file_identity(file_path)
        _ops_create_file_record(
            self,
            entry=FileRecord.to_file_record_entry(
                file_path=file_path,
                source_uri=getattr(ident, "source_uri", None),
                source_provider=getattr(ident, "source_provider", None),
                result=result,
                ingest_mode=(
                    getattr(getattr(config, "ingest", None), "mode", "per_file")
                    or "per_file"
                ),
                unified_label=(
                    getattr(getattr(config, "ingest", None), "unified_label", None)
                    if getattr(getattr(config, "ingest", None), "mode", "per_file")
                    == "unified"
                    else None
                ),
                table_ingest=bool(
                    getattr(getattr(config, "ingest", None), "table_ingest", True),
                ),
            ),
        )

        # 2) Ingest content rows
        if bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        ):
            total_records = int(
                (result or {}).get("total_records")
                or len(list(result.get("records", []) or [])),
            )
            print(
                f"[Ingest] Content rows → preparing {total_records} parsed rows for insertion",
            )

        _fmt = result.get("file_format")
        file_format = getattr(_fmt, "value", _fmt)
        rows_in = list(result.get("records", []) or [])
        rows = _ops_apply_content_ingest_policy(
            rows_in,
            config=config,
            file_format=(str(file_format).lower().strip() if file_format else None),
        )
        if bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        ):
            print(f"[Ingest] Content rows → inserting {len(rows)} rows after policy")

        inserted_ids = self._ingest_file(
            file_path=file_path,
            records=rows,
            config=config,
        )

        if bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        ):
            print(f"[Ingest] Content rows inserted: {len(inserted_ids)} ids returned")

        # 3) Ingest per-file tables if enabled
        if config.ingest.table_ingest:
            dest_name = (
                file_path
                if config.ingest.mode == "per_file"
                else (config.ingest.unified_label or "Unified")
            )
            if bool(
                getattr(getattr(config, "diagnostics", None), "enable_progress", False),
            ):
                try:
                    tables = (
                        getattr(getattr(document, "metadata", None), "tables", []) or []
                    )
                    print(
                        f"[Ingest] Tables → ingesting {len(tables)} extracted table(s) into per-file contexts",
                    )
                except Exception:
                    print("[Ingest] Tables → ingesting per-file tables")

            self._ingest_tables_for_file(
                file_path=dest_name,
                document=document,
                table_rows_batch_size=config.ingest.table_rows_batch_size,
                config=config,
            )

        return inserted_ids

    def _embed(
        self,
        *,
        file_path: str,
        document: Any,
        result: Dict[str, Any],
        inserted_ids: Optional[List[int]],
        config: _FilePipelineConfig,
    ) -> None:
        """Create embeddings as specified by config for content and/or tables.

        Parameters
        ----------
        file_path : str
            The file path of the file to embed.
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
        # Resolve embedding specs for this file_path
        resolved_specs = self._resolve_embedding_specs_for_file(file_path, config)

        # Embedding is enabled when strategy != "off" and specs are provided.
        if not (
            getattr(getattr(config, "embed", None), "strategy", "auto") != "off"
            and resolved_specs
        ):
            return
        if bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        ):
            print(
                f"[Embed] Starting embeddings for: {file_path} (strategy={getattr(config.embed, 'strategy', 'auto')})",
            )
            print(f"[Embed] Specs: {len(resolved_specs)}")
        # Pre-embed hooks
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

        try:
            ctx_name = (
                file_path
                if config.ingest.mode == "per_file"
                else (config.ingest.unified_label or "Unified")
            )
            for file_spec, table_spec in resolved_specs:
                if file_spec.context == "per_file":
                    ctx = _storage_ctx_for_file(self, file_path=ctx_name)
                    # Iterate over column pairs
                    for source_col, target_col in zip(
                        table_spec.source_columns,
                        table_spec.target_columns,
                    ):
                        if bool(
                            getattr(
                                getattr(config, "diagnostics", None),
                                "enable_progress",
                                False,
                            ),
                        ):
                            print(
                                f"[Embed] Ensuring vector column on Content: ctx={ctx}, target={target_col}, source={source_col}",
                            )
                        ensure_vector_column(
                            ctx,
                            embed_column=target_col,
                            source_column=source_col,
                            from_ids=(inserted_ids or None),
                        )
                elif file_spec.context == "unified":
                    ctx = _storage_ctx_for_file(
                        self,
                        file_path=(config.ingest.unified_label or "Unified"),
                    )
                    # Iterate over column pairs
                    for source_col, target_col in zip(
                        table_spec.source_columns,
                        table_spec.target_columns,
                    ):
                        if bool(
                            getattr(
                                getattr(config, "diagnostics", None),
                                "enable_progress",
                                False,
                            ),
                        ):
                            print(
                                f"[Embed] Ensuring vector column on Unified Content: ctx={ctx}, target={target_col}, source={source_col}",
                            )
                        ensure_vector_column(
                            ctx,
                            embed_column=target_col,
                            source_column=source_col,
                            from_ids=(inserted_ids or None),
                        )
                elif file_spec.context == "per_file_table":
                    # Traverse the file-scoped overview to find per-file Tables contexts.
                    overview = self._tables_overview(file=ctx_name)
                    tables_meta: list[dict] = []
                    for _k, _v in overview.items():
                        if isinstance(_v, dict) and isinstance(_v.get("Tables"), dict):
                            tables_meta.extend(
                                [
                                    tm
                                    for tm in _v["Tables"].values()
                                    if isinstance(tm, dict)
                                ],
                            )
                    # When a specific table label is provided, compare using the manager's safe() mapping
                    # against the actual per-file context suffix (…/Tables/<safe_label>).
                    table_filter = table_spec.table
                    safe_target = None
                    if table_filter not in (None, "*"):
                        try:
                            safe_target = self._safe(str(table_filter))  # type: ignore[attr-defined]
                        except Exception:
                            safe_target = str(table_filter)
                    for meta in tables_meta:
                        ctx_label = meta.get("context")
                        if (
                            not isinstance(ctx_label, str)
                            or "/Tables/" not in ctx_label
                        ):
                            continue
                        if safe_target is not None:
                            tail = ctx_label.split("/Tables/", 1)[-1]
                            if tail != safe_target:
                                continue
                        # Iterate over column pairs
                        for source_col, target_col in zip(
                            table_spec.source_columns,
                            table_spec.target_columns,
                        ):
                            if bool(
                                getattr(
                                    getattr(config, "diagnostics", None),
                                    "enable_progress",
                                    False,
                                ),
                            ):
                                print(
                                    f"[Embed] Ensuring vector column on Table: ctx={ctx_label}, target={target_col}, source={source_col}",
                                )
                            ensure_vector_column(
                                ctx_label,
                                embed_column=target_col,
                                source_column=source_col,
                            )
        except Exception:
            pass

        # Post-embed hooks
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

    def _ingest_file(
        self,
        *,
        file_path: str,
        records: List[Dict[str, Any]],
        config: _FilePipelineConfig,
    ) -> List[int]:
        """Ingest flattened content rows for one file into the configured context.

        Parameters
        ----------
        file_path : str
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
        rows: List[Dict[str, Any]] = list(records or [])
        # Filter to allowed columns if provided
        allowed = (
            set(config.ingest.allowed_columns)
            if config.ingest.allowed_columns
            else None
        )
        if allowed:
            rows = [{k: v for k, v in rec.items() if k in allowed} for rec in rows]

        # Determine destination context name (per-file or unified bucket)
        dest_name = (
            file_path
            if config.ingest.mode == "per_file"
            else (config.ingest.unified_label or "Unified")
        )

        # Lookup file_id from index to set FK on rows
        # Use path as-is for querying
        _rows = unify.get_logs(
            context=self._ctx,
            filter=f"file_path == {file_path!r}",
            limit=1,
            from_fields=["file_id"],
        )
        _fid = _rows[0].entries.get("file_id") if _rows else None

        if _fid is None:
            raise ValueError(f"File ID not found for file_path: {file_path}")

        # Optional: delete existing rows. For map/string layouts, purge-then-insert.
        if config.ingest.replace_existing:
            try:
                if config.ingest.mode == "per_file":
                    _ops_delete_per_file_rows_by_filter(
                        self,
                        file_path=dest_name,
                        filter_expr=None,
                    )
                else:
                    # unified: delete only rows matching this file's file_id
                    filt = f"file_id == {_fid}"
                    _ops_delete_per_file_rows_by_filter(
                        self,
                        file_path=dest_name,
                        filter_expr=filt,
                    )
            except Exception:
                pass

        # Chunk rows according to content_rows_batch_size
        content_rows_batch_size = int(
            getattr(getattr(config, "ingest", None), "content_rows_batch_size", 1000),
        )
        id_layout = getattr(getattr(config, "ingest", None), "id_layout", "map")
        auto_counting_per_file = (
            getattr(config.ingest, "id_hierarchy", None)
            if id_layout == "columns"
            else None
        )

        enable_progress = bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        )

        total_rows = len(rows)
        total_chunks = (
            (total_rows + content_rows_batch_size - 1) // content_rows_batch_size
            if total_rows > 0
            else 0
        )

        # Start content ingestion tracking
        if enable_progress and self._progress_manager is not None:
            self._progress_manager.start_content_ingest(file_path, total_chunks)  # type: ignore[union-attr]

        all_inserted_ids: List[int] = []
        chunk_num = 0

        chunk_iter = range(0, len(rows), content_rows_batch_size)

        for i in chunk_iter:
            chunk = rows[i : i + content_rows_batch_size]

            file_content_entries: List[Dict[str, Any]] = (
                FileContent.to_file_content_entries(
                    file_id=int(_fid),
                    rows=chunk,
                    id_layout=id_layout,
                )
            )

            inserted_ids = _ops_create_file_content(
                self,
                file_path=dest_name,
                auto_counting_per_file=auto_counting_per_file,
                rows=file_content_entries,
            )
            all_inserted_ids.extend(list(inserted_ids or []))

            chunk_num += 1
            # Update progress manager
            if enable_progress and self._progress_manager is not None:
                self._progress_manager.update_content_ingest(file_path, chunk_num)  # type: ignore[union-attr]

        # Complete content ingestion
        if enable_progress and self._progress_manager is not None:
            self._progress_manager.complete_content_ingest(file_path)  # type: ignore[union-attr]

        return all_inserted_ids

        # ---------- Per-table ingestion for spreadsheets (CSV/XLSX/Sheets) ----- #

    def _ingest_tables_for_file(
        self,
        *,
        file_path: str,
        document: Any,
        table_rows_batch_size: int = 100,
        config: Optional[_FilePipelineConfig] = None,
    ) -> None:
        """
        Create one sub-context per extracted table and log its rows.

        Parameters
        ----------
        file_path : str
            Logical file identifier/path.
        document : Any
            Parsed document object exposing ``metadata.tables`` with ``rows`` and
            optional ``columns`` / ``sheet_name``.
        table_rows_batch_size : int, default 100
            Batch size for logging rows to the per-table context.

        Notes
        -----
        - Context naming: Files/<alias>/<safe_file_path>/Tables/<safe_table_label>.
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
                    # Use the first row data as the column headers if the rows are not dict-like
                    columns = [str(val) for val in first]

                # Pop the first row from the rows list to remove the header row
                rows.pop(0)

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
                _storage_ensure_file_table_context(
                    self,
                    file_path=file_path,
                    table=table_label,
                    columns=list(columns) if columns else None,
                    example_row=(
                        rows[0] if (rows and isinstance(rows[0], dict)) else None
                    ),
                )
            except Exception as e:
                print(f"Error ensuring per-file table context: {e}")

            # Batch rows for efficient logging via ops
            enable_progress = (
                bool(
                    getattr(
                        getattr(config, "diagnostics", None),
                        "enable_progress",
                        False,
                    ),
                )
                if config
                else False
            )

            batch: List[Dict[str, Any]] = []
            processed = 0
            total_rows = len(rows)
            total_chunks_for_table = (
                (total_rows + table_rows_batch_size - 1) // table_rows_batch_size
                if total_rows > 0
                else 0
            )

            # Register table with progress manager
            embed_strategy = (
                getattr(getattr(config, "embed", None), "strategy", "off")
                if config
                else "off"
            )
            embed_total = total_chunks_for_table if embed_strategy == "along" else 0

            if enable_progress and self._progress_manager is not None:
                self._progress_manager.register_table(file_path, table_label, total_chunks_for_table, embed_total)  # type: ignore[union-attr]

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
                    try:
                        inserted_ids = _ops_create_file_table(
                            self,
                            file_path=file_path,
                            table=table_label,
                            rows=batch,
                            columns=list(columns) if columns else None,
                            example_row=(
                                rows[0]
                                if (rows and isinstance(rows[0], dict))
                                else None
                            ),
                        )
                        processed += len(batch)
                        chunk_num += 1

                        # Update progress manager
                        if enable_progress and self._progress_manager is not None:
                            self._progress_manager.update_table_ingest(file_path, table_label, chunk_num)  # type: ignore[union-attr]
                    except Exception as e:
                        if enable_progress:
                            print(f"[Ingest] ❌ Error logging table rows batch: {e}")
                        else:
                            print(f"Error logging table rows batch: {e}")
                    batch = []

            # Flush any remaining rows
            if batch:
                try:
                    inserted_ids = _ops_create_file_table(
                        self,
                        file_path=file_path,
                        table=table_label,
                        rows=batch,
                        columns=list(columns) if columns else None,
                        example_row=(
                            rows[0] if (rows and isinstance(rows[0], dict)) else None
                        ),
                    )
                    chunk_num += 1

                    # Update progress manager
                    if enable_progress and self._progress_manager is not None:
                        self._progress_manager.update_table_ingest(file_path, table_label, chunk_num)  # type: ignore[union-attr]
                except Exception as e:
                    if enable_progress:
                        print(f"[Ingest] ❌ Error logging final table rows batch: {e}")
                    else:
                        print(f"Error logging final table rows batch: {e}")

            # Complete table ingestion
            if enable_progress and self._progress_manager is not None:
                self._progress_manager.complete_table_ingest(file_path, table_label)  # type: ignore[union-attr]

    def _ingest_and_embed(
        self,
        *,
        file_path: str,
        document: Any,
        result: Dict[str, Any],
        config: _FilePipelineConfig,
    ) -> None:
        """
        Ingest and embed along (chunk-by-chunk) for a single file until fully processed.

        Ingestion happens sequentially (chunk N+1 starts only after chunk N finishes),
        but embedding jobs run asynchronously without blocking subsequent ingestion.
        All embedding jobs are tracked and awaited at the end.
        """
        # 1) Index the file record (required for embedding)
        if bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        ):
            print(f"[Along] Starting ingest+embed for: {file_path}")
        # Ensure file record exists before embedding (required for file_id lookup)
        ident = self._build_file_identity(file_path)
        _ops_create_file_record(
            self,
            entry=FileRecord.to_file_record_entry(
                file_path=file_path,
                source_uri=getattr(ident, "source_uri", None),
                source_provider=getattr(ident, "source_provider", None),
                result=result,
                ingest_mode=(
                    getattr(getattr(config, "ingest", None), "mode", "per_file")
                    or "per_file"
                ),
                unified_label=(
                    getattr(getattr(config, "ingest", None), "unified_label", None)
                    if getattr(getattr(config, "ingest", None), "mode", "per_file")
                    == "unified"
                    else None
                ),
                table_ingest=bool(
                    getattr(getattr(config, "ingest", None), "table_ingest", True),
                ),
            ),
        )

        # Determine destination display name/context root for this file (content + tables)
        dest_name = (
            file_path
            if config.ingest.mode == "per_file"
            else (config.ingest.unified_label or "Unified")
        )

        # 2) Ingest content rows in chunks, embedding asynchronously
        from .ops import embed_content_chunks_async as _ops_embed_content_chunks_async

        batch_size = int(
            getattr(getattr(config, "ingest", None), "content_rows_batch_size", 1000),
        )
        # Resolve embedding specs for this file_path
        resolved_specs = self._resolve_embedding_specs_for_file(file_path, config)

        # Filter specs for the correct content context
        target_ctx_name = _storage_ctx_for_file(self, file_path=dest_name)
        content_spec_context = (
            "per_file" if config.ingest.mode == "per_file" else "unified"
        )
        content_specs = [
            (file_spec, table_spec)
            for file_spec, table_spec in resolved_specs
            if file_spec.context == content_spec_context
        ]
        enable_progress = bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        )

        _ops_embed_content_chunks_async(
            self,
            file_path=file_path,
            records=list(result.get("records", []) or []),
            target_ctx_name=target_ctx_name,
            content_specs=content_specs,
            result=result,
            document=document,
            config=config,
            batch_size=batch_size,
            enable_progress=enable_progress,
        )

        # 3) Ingest per-file tables in chunks (when enabled), embedding asynchronously
        if bool(getattr(getattr(config, "ingest", None), "table_ingest", True)):
            from .ops import embed_table_chunks_async as _ops_embed_table_chunks_async

            table_specs = [
                (file_spec, table_spec)
                for file_spec, table_spec in resolved_specs
                if file_spec.context == "per_file_table"
            ]

            _ops_embed_table_chunks_async(
                self,
                file_path=file_path,  # Use original file_path for config matching, not dest_name
                document=document,
                target_ctx_name=target_ctx_name,
                table_specs=table_specs,
                result=result,
                config=config,
                table_rows_batch_size=int(
                    getattr(
                        getattr(config, "ingest", None),
                        "table_rows_batch_size",
                        100,
                    ),
                ),
                enable_progress=enable_progress,
            )

        if bool(
            getattr(getattr(config, "diagnostics", None), "enable_progress", False),
        ):
            print(f"[Along] Completed ingest+embed for: {file_path}")

    def parse(self, file_paths: Union[str, List[str]], *, config: Optional[_FilePipelineConfig] = None) -> Dict[str, Any]:  # type: ignore[override]
        """
        Parse one or more files, then ingest their content and tables according to config.

        Return payload depends on cfg.output.return_mode:
        - "compact" (default): format-specific Pydantic model (e.g., ParsedPDF, ParsedXlsx)
        - "full": raw dict from Document.to_parse_result (includes heavy fields)
        - "none": minimal stub with status/format/record counts

        Parameters
        ----------
        file_paths : str | list[str]
            One or more logical file paths to parse.
        config : FilePipelineConfig | None
            Pipeline configuration controlling parser kwargs, ingest layout,
            table ingestion and embedding behavior. When None, defaults are used
            (equivalent to ``FilePipelineConfig()``).

        Returns
        -------
        dict[str, Any]
            Mapping of file path → compact Pydantic model, raw dict, or minimal stub per output mode.
        """
        cfg = config or _FilePipelineConfig()

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        results: Dict[str, Any] = {}
        exported_paths: List[str] = []
        exported_paths_to_original_paths: Dict[str, str] = {}

        temp_dir: Optional[str] = None
        try:
            # Initialize rich progress manager when enabled
            try:
                from .progress_display import FileProgressManager as _FP
            except Exception:
                _FP = None  # type: ignore

            import tempfile as _tempfile

            temp_dir = _tempfile.mkdtemp(prefix="filemanager_parse_")
            print(f"[FileManager] Created temporary directory: {temp_dir}")

            # Resolve parse inputs: prefer in-place local path; otherwise export via adapter
            from pathlib import Path as _P

            for path in file_paths:
                try:
                    p = _P(str(path)).expanduser()
                    if p.is_absolute() and p.exists():
                        exported_path = str(p)
                        print(
                            f"[FileManager] Using local path in-place: {exported_path}",
                        )
                        exported_paths.append(exported_path)
                        exported_paths_to_original_paths[exported_path] = path
                        continue
                except Exception:
                    pass
                try:
                    exported_path = self.export_file(path, temp_dir)
                    print(f"[FileManager] Exported file to: {exported_path}")
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

            # Start progress manager and register files
            if enable_progress and _FP is not None:
                try:
                    self._progress_manager = _FP(enable=True)
                    self._progress_manager.start()
                    # Register files up-front
                    for exp in exported_paths:
                        orig = exported_paths_to_original_paths.get(exp, exp)
                        self._progress_manager.register_file(orig, cfg)  # type: ignore[union-attr]
                except Exception:
                    self._progress_manager = None

            # Parse files using parser's batch method (handles parallelization)
            documents: List[Any] = []
            if len(exported_paths) > 1 and hasattr(self._parser, "parse_batch"):
                try:
                    if enable_progress:
                        print(
                            f"[FileManager] Parsing {len(exported_paths)} files in parallel (batch_size={cfg.parse.batch_size})",
                        )
                    documents = self._parser.parse_batch(
                        exported_paths,
                        batch_size=cfg.parse.batch_size,
                        **cfg.parse.parser_kwargs,
                    )
                    # Mark all files as parsed (parse_batch returns in-order)
                    if enable_progress and self._progress_manager is not None:
                        for exp in exported_paths:
                            orig = exported_paths_to_original_paths.get(exp, exp)
                            self._progress_manager.update_parsing(orig, "complete")  # type: ignore[union-attr]
                except Exception as e:
                    # Batch failure → mark all remaining as errors
                    for fp in exported_paths:
                        original_path = exported_paths_to_original_paths.get(fp)
                        if original_path and original_path not in results:
                            results[original_path] = _Doc.error_result(
                                original_path,
                                f"parse_batch failed: {e}",
                            )
                            if enable_progress and self._progress_manager is not None:
                                try:
                                    self._progress_manager.update_parsing(original_path, "failed")  # type: ignore[union-attr]
                                except Exception:
                                    pass
                    return results
            else:
                try:
                    documents = [
                        self._parser.parse(
                            exported_paths[0],
                            **cfg.parse.parser_kwargs,
                        ),
                    ]
                    if enable_progress and self._progress_manager is not None:
                        orig0 = exported_paths_to_original_paths.get(
                            exported_paths[0],
                            exported_paths[0],
                        )
                        self._progress_manager.update_parsing(orig0, "complete")  # type: ignore[union-attr]
                except Exception as e:
                    original_path = exported_paths_to_original_paths.get(
                        exported_paths[0],
                        exported_paths[0],
                    )
                    results[original_path] = _Doc.error_result(original_path, str(e))
                    if enable_progress and self._progress_manager is not None:
                        try:
                            self._progress_manager.update_parsing(original_path, "failed")  # type: ignore[union-attr]
                        except Exception:
                            pass
                    return results

            # Build results and ingest per-file artifacts
            # Check if we can parallelize (per_file mode only)
            can_parallelize = cfg.ingest.mode == "per_file" and len(documents) > 1

            if can_parallelize:
                # Parallel processing for per_file mode
                if enable_progress:
                    print(
                        f"[FileManager] Parallelizing ingest+embed for {len(documents)} files (per_file mode)",
                    )
                from .ops import process_files_parallel as _process_parallel

                results.update(
                    _process_parallel(
                        self,
                        documents=documents,
                        exported_paths=exported_paths,
                        exported_paths_to_original_paths=exported_paths_to_original_paths,
                        config=cfg,
                        enable_progress=enable_progress,
                    ),
                )
            else:
                # Sequential processing (unified mode or single file)
                from .ops import process_files_sequential as _process_sequential

                results.update(
                    _process_sequential(
                        self,
                        documents=documents,
                        exported_paths=exported_paths,
                        exported_paths_to_original_paths=exported_paths_to_original_paths,
                        config=cfg,
                        enable_progress=enable_progress,
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
            # Stop progress manager if started
            try:
                if self._progress_manager is not None:
                    self._progress_manager.stop()  # type: ignore[union-attr]
                    self._progress_manager = None
            except Exception:
                pass

    async def parse_async(
        self,
        file_paths: Union[str, List[str]],
        *,
        config: Optional[_FilePipelineConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Parse one or more files asynchronously, then ingest their content and tables according to config.

        Yields
        ------
        dict
            Result dict per file. Payload depends on cfg.output.return_mode:
            - "compact" (default): format-specific Pydantic model (e.g., ParsedPDF, ParsedXlsx)
            - "full": raw dict from Document.to_parse_result (includes heavy fields)
            - "none": minimal stub with status/format/record counts

        Parameters
        ----------
        file_paths : str | list[str]
            One or more logical file paths to parse.
        config : FilePipelineConfig | None
            Pipeline configuration controlling parser kwargs, ingest layout,
            table ingestion and embedding behavior. When None, defaults are used
            (equivalent to ``FilePipelineConfig()``).
        """
        import asyncio

        cfg = config or _FilePipelineConfig()

        if isinstance(file_paths, str):
            file_paths = [file_paths]
        # buffer export errors to yield after processing successful files
        export_errors: List[Dict[str, Any]] = []
        exported_paths: List[str] = []
        exported_paths_to_original_paths: Dict[str, str] = {}

        temp_dir: Optional[str] = None
        try:
            # Initialize rich progress manager when enabled
            try:
                from .progress_display import FileProgressManager as _FP
            except Exception:
                _FP = None  # type: ignore

            import tempfile as _tempfile

            temp_dir = _tempfile.mkdtemp(prefix="filemanager_parse_")
            print(f"[FileManager] Created temporary directory: {temp_dir}")

            # Resolve parse inputs: prefer in-place local path; otherwise export via adapter
            from pathlib import Path as _P

            for path in file_paths:
                try:
                    p = _P(str(path)).expanduser()
                    if p.is_absolute() and p.exists():
                        exported_path = str(p)
                        print(
                            f"[FileManager] Using local path in-place: {exported_path}",
                        )
                        exported_paths.append(exported_path)
                        exported_paths_to_original_paths[exported_path] = path
                        continue
                except Exception:
                    pass
                try:
                    exported_path = self.export_file(path, temp_dir)
                    print(f"[FileManager] Exported file to: {exported_path}")
                    exported_paths.append(exported_path)
                    exported_paths_to_original_paths[exported_path] = path
                except Exception as e:
                    # Buffer per-file export failure; do not abort
                    export_errors.append(_Doc.error_result(path, f"export failed: {e}"))

            # Nothing exported successfully
            if not exported_paths:
                # yield any buffered export errors and exit
                for err in export_errors:
                    yield err
                return

            enable_progress = bool(
                getattr(getattr(cfg, "diagnostics", None), "enable_progress", False),
            )

            # Start progress manager and register files
            if enable_progress and _FP is not None:
                try:
                    self._progress_manager = _FP(enable=True)
                    self._progress_manager.start()
                    # Register files up-front
                    for exp in exported_paths:
                        orig = exported_paths_to_original_paths.get(exp, exp)
                        self._progress_manager.register_file(orig, cfg)  # type: ignore[union-attr]
                except Exception:
                    self._progress_manager = None

            # Parse files using parser's async batch method (yields as they complete)
            documents: List[Any] = []
            if len(exported_paths) > 1 and hasattr(self._parser, "parse_batch_async"):
                try:
                    if enable_progress:
                        print(
                            f"[FileManager] Parsing {len(exported_paths)} files in parallel (batch_size={cfg.parse.batch_size})",
                        )
                    # Collect documents as they complete parsing
                    async for idx, document in self._parser.parse_batch_async(
                        exported_paths,
                        batch_size=cfg.parse.batch_size,
                        **cfg.parse.parser_kwargs,
                    ):
                        documents.append((idx, document))
                        # Update parsing progress as each completes
                        if enable_progress and self._progress_manager is not None:
                            try:
                                exp = (
                                    exported_paths[idx]
                                    if idx < len(exported_paths)
                                    else exported_paths[0]
                                )
                                orig = exported_paths_to_original_paths.get(exp, exp)
                                self._progress_manager.update_parsing(orig, "complete")  # type: ignore[union-attr]
                            except Exception:
                                pass
                    # Sort by index to preserve order
                    documents.sort(key=lambda x: x[0])
                    documents = [doc for _, doc in documents]
                except Exception as e:
                    # Batch failure → mark all remaining as errors
                    for fp in exported_paths:
                        original_path = exported_paths_to_original_paths.get(fp)
                        if original_path:
                            yield _Doc.error_result(
                                original_path,
                                f"parse_batch_async failed: {e}",
                            )
                            if enable_progress and self._progress_manager is not None:
                                try:
                                    self._progress_manager.update_parsing(original_path, "failed")  # type: ignore[union-attr]
                                except Exception:
                                    pass
                    # also yield any buffered export errors
                    for err in export_errors:
                        yield err
                    return
            else:
                try:
                    documents = [
                        await asyncio.to_thread(
                            self._parser.parse,
                            exported_paths[0],
                            **cfg.parse.parser_kwargs,
                        ),
                    ]
                    if enable_progress and self._progress_manager is not None:
                        orig0 = exported_paths_to_original_paths.get(
                            exported_paths[0],
                            exported_paths[0],
                        )
                        self._progress_manager.update_parsing(orig0, "complete")  # type: ignore[union-attr]
                except Exception as e:
                    original_path = exported_paths_to_original_paths.get(
                        exported_paths[0],
                        exported_paths[0],
                    )
                    yield _Doc.error_result(original_path, str(e))
                    if enable_progress and self._progress_manager is not None:
                        try:
                            self._progress_manager.update_parsing(original_path, "failed")  # type: ignore[union-attr]
                        except Exception:
                            pass
                    for err in export_errors:
                        yield err
                    return

            # Build results and ingest per-file artifacts
            # Check if we can parallelize (per_file mode only)
            can_parallelize = cfg.ingest.mode == "per_file" and len(documents) > 1

            if can_parallelize:
                # Parallel processing for per_file mode using asyncio.gather
                if enable_progress:
                    print(
                        f"[FileManager] Parallelizing ingest+embed for {len(documents)} files (per_file mode)",
                    )
                from .ops import process_files_parallel_async as _process_parallel_async

                completed_results = await _process_parallel_async(
                    self,
                    documents=documents,
                    exported_paths=exported_paths,
                    exported_paths_to_original_paths=exported_paths_to_original_paths,
                    config=cfg,
                    enable_progress=enable_progress,
                )
                # Yield results in order
                for original_path, result_dict in completed_results:
                    yield result_dict
            else:
                # Sequential processing (unified mode or single file)
                from .ops import (
                    process_files_sequential_async as _process_sequential_async,
                )

                async for result_dict in _process_sequential_async(
                    self,
                    documents=documents,
                    exported_paths=exported_paths,
                    exported_paths_to_original_paths=exported_paths_to_original_paths,
                    config=cfg,
                    enable_progress=enable_progress,
                ):
                    yield result_dict

            # Yield any buffered export errors at the end
            for err in export_errors:
                yield err
        finally:
            # Clean up temporary directory
            if temp_dir:
                try:
                    import shutil as _shutil2

                    _shutil2.rmtree(temp_dir)
                except Exception:
                    pass
            # Stop progress manager if started
            try:
                if self._progress_manager is not None:
                    self._progress_manager.stop()  # type: ignore[union-attr]
                    self._progress_manager = None
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
        from .search import filter_files as _srch_filter_files

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
        from .search import search_files as _srch_search_files

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
        business_context: Optional[str] = None,
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
        business_context : str | None
            Optional domain-specific guidance appended to the system prompt to tailor behaviour
            for a particular use case. Kept separate from general tool instructions.
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
            business_context=business_context,
        )
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
