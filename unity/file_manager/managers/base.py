from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.global_docstrings import CLEAR_METHOD_DOCSTRING
from unity.common.state_managers import BaseStateManager


class BaseFileManager(BaseStateManager):
    """
    Public contract that every concrete file-manager must satisfy.

    Exposes read-only discovery/analysis over a single filesystem and provides
    high-level reorganization capabilities via tools.

    Responsibilities
    ----------------
    • "ask" — answer questions about the entire filesystem (read-only)
    • "ask_about_file" — answer questions about one specific file (read-only)
    • "organize" — plan and optionally execute rename/move operations

    Implementations MUST NOT create or delete files via LLM tools by default.
    Mutations in "organize" are limited to rename/move and are gated by adapter
    capabilities.

    Contexts & Joins
    ----------------
    A concrete FileManager typically manages:
    - a global index context, and
    - per-file contexts, optionally with nested per-table contexts.

    Implementations expose read-only join tools to combine these contexts for
    efficient retrieval:
    - _filter_join / _search_join: join two contexts and then filter or perform
      semantic search over the joined result.
    - _filter_multi_join / _search_multi_join: chain multiple joins (the special
      placeholder '$prev' may be used to refer to the previous step at call-time).

    Reference conventions for join tools are implementation-specific. The
    concrete class must document how callers identify the global index and
    per-file/per-table contexts.
    """

    # ------------------------------------------------------------------ #
    # Basic inventory operations                                          #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Return True if a file with the given file path exists in this filesystem."""

    @abstractmethod
    def list(self) -> List[str]:
        """Return the list of file paths (stable order) for files in this filesystem."""

    @abstractmethod
    def parse(
        self,
        file_paths: Union[str, List[str]],
        **options: Any,
    ) -> Dict[str, Any]:
        """
        Parse one or more files and return structured results per file.

        Parameters
        ----------
        file_paths : str | list[str]
            Single file path or a list of file paths to parse.
        **options : Any
            Parser-specific options (forwarded as-is).

        Returns
        -------
        dict[str, Any]
            Mapping from file path → result. The concrete return depends on the
            configured output mode:

            - compact (default): a typed Pydantic model (per file type) containing
              reference-first pointers (content_ref, tables_ref) and light metadata.
            - full: the raw dict returned by the parser's Document.to_parse_result.
            - none: a minimal stub (file_path, status, error, total_records, file_format).

            The concrete model type is format-specific (e.g., ParsedPDF, ParsedXlsx).

        Options
        -------
        config : FilePipelineConfig | dict | None
            Complete pipeline configuration controlling parsing, ingestion, embeddings,
            plugins, and output return mode. When a dict is provided, it will be coerced
            to `FilePipelineConfig` (unknown keys are ignored).

            Key sub-models and fields:
            - parse.batch_size: int (parse_async parallelism when available)
            - parse.parser_kwargs: dict forwarded to parser.parse/parse_batch
            - ingest.mode: "per_file" | "unified" (destination layout)
            - ingest.allowed_columns: list[str] (column filter)
            - ingest.table_ingest: bool (ingest extracted tables)
            - embed.strategy/large_threshold/hooks_per_chunk/specs: embedding behaviour
            - plugins.pre/post_*: hook lists (dotted names or callables)
            - output.return_mode: "compact" | "full" | "none"

        Notes
        -----
        - Implementations SHOULD accept `config` in **options and default to a sensible
          `FilePipelineConfig()` when omitted.
        - For legacy callers, ad-hoc kwargs MAY be mapped into `parse.parser_kwargs`.
        """

    # ------------------------------------------------------------------ #
    # File export operations (for parsing)                               #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def export_file(self, file_path: str, destination_dir: str) -> str:
        """
        Export a file from the underlying filesystem to a local destination directory.

        This method is used by parse operations to bring files from the adapter's
        filesystem into a local temporary directory with their original file paths preserved.

        Parameters
        ----------
        file_path : str
            The file path of the file to export.
        destination_dir : str
            Local directory path where the file should be exported.

        Returns
        -------
        str
            Full path to the exported file in the destination directory.

        Raises
        ------
        FileNotFoundError
            If the source file doesn't exist.
        """

    @abstractmethod
    def export_directory(self, directory: str, destination_dir: str) -> List[str]:
        """
        Export all files from a directory to a local destination directory.

        This is a batch operation that exports multiple files at once, optimizing
        for the underlying filesystem's capabilities (e.g., zip downloads).

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

    # ------------------------------------------------------------------ #
    # Unify-backed retrieval (private tools)                             #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _list_columns(
        self,
        *,
        include_types: bool = True,
        table: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        """
        Return the schema for a context managed by this FileManager.

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column → logical type. When False,
            return just the list of column names.
        table : str | None, default None
            Logical table name or fully-qualified context. When None, returns
            the FileRecords (index) columns. When provided, resolves logical
            names (e.g., "<root>", "<root>.Tables.<label>") to the correct
            context and returns its columns.

        Returns
        -------
        dict[str, Any] | list[str]
            Column→type mapping when include_types=True, otherwise a list of column names.
        """

    @abstractmethod
    def _filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter the FileRecords index or resolve-and-filter per-file contexts.

        Parameters
        ----------
        filter : str | None
            Row-level predicate evaluated per context (column names in scope).
        offset : int
            Pagination offset per context.
        limit : int
            Maximum rows per context (<= 1000).
        tables : str | list[str] | None
            Logical table names from `tables_overview()` (preferred) or legacy
            refs. When None, only the FileRecords index is scanned.

        Returns
        -------
        list[dict]
            Flat list of rows (index-only when tables=None, concatenated when tables provided).

        Notes
        -----
        - For text-heavy questions, prefer semantic search over joins.
        - When joining or scanning per-file contexts, consider calling
          `tables_overview(file=...)` first and then `list_columns(table=...)`
          to choose the correct columns for filters.
        """

    @abstractmethod
    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        table: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over a resolved context using Unify vector columns and references.

        Notes
        -----
        - The `table` parameter narrows the target context. It accepts logical
          names from `tables_overview()` (e.g., "FileRecords", "<root>",
          "<root>.Tables.<label>") or legacy refs and resolves them identically to
          join tools and `_list_columns(table=...)`.
        - The `filter` parameter is a row-level predicate (evaluated with column
          names as variables) applied before ranking/backfill.
        """

    @abstractmethod
    def _rename_file(self, *, target_id_or_path: str, new_name: str) -> Dict[str, Any]:
        """
        Rename a file in the underlying filesystem.

        Parameters
        ----------
        target_id_or_path : str
            Adapter-native identifier or path for the file.
        new_name : str
            New file name; adapter determines full path semantics.

        Returns
        -------
        dict
            Adapter reference payload or a summary dict of the rename result.
        """

    @abstractmethod
    def _move_file(
        self,
        *,
        target_id_or_path: str,
        new_parent_path: str,
    ) -> Dict[str, Any]:
        """
        Move a file to a new directory in the underlying filesystem.

        Parameters
        ----------
        target_id_or_path : str
            Adapter-native identifier or path for the file.
        new_parent_path : str
            Destination directory path in adapter-native form.

        Returns
        -------
        dict
            Adapter reference payload or a summary dict of the move result.
        """

    @abstractmethod
    def _delete_file(self, *, file_id: int) -> Dict[str, Any]:
        """
        Delete a file record from the Unify table and, if supported by the adapter,
        from the underlying filesystem.

        Parameters
        ----------
        file_id : int
            Unique file ID from the Unify table.

        Returns
        -------
        dict
            Result dictionary with 'outcome' and 'details' keys.

        Raises
        ------
        ValueError
            If no file with the given file_id exists.
        PermissionError
            If the file is protected or the adapter doesn't support deletion.
        """

    # ------------------------------------------------------------------ #
    # Filesystem-level Q&A                                               #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Interrogate the **existing filesystem** (read‑only) and obtain a live
        :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method to locate and inspect files that already exist in the
        store: perform semantic searches over parsed content, aggregate or
        summarise results, compare documents, or shortlist files/folders for a
        subsequent organization pass. This call must never create, modify or
        delete files.

        Clarifications
        --------------
        Do not use this method to ask the human follow‑up questions. If the
        caller needs clarification about what to retrieve (e.g., which folder,
        which file path, which topic), route the question via a dedicated
        ``request_clarification`` tool when available. If no clarification
        channel exists, proceed with sensible defaults/best‑guess values and
        state those assumptions in the outer loop's final reply.

        Do not request how the question should be answered; just ask the
        question in natural language and allow this method to determine the
        best method to answer it (e.g., filter/search/join, parse‑if‑missing
        when explicitly requested).

        Examples
        --------
        • Good: "Which PDFs mention ISO 27001 under /reports?"
          → shortlist files and cite their paths where possible.
        • Bad:  "Open each file and tell me which tool to call." → too
          prescriptive; let the tool loop decide the best approach.

        Parameters
        ----------
        text : str
            Plain‑English question about existing files/folders.
        _return_reasoning_steps : bool, default ``False``
            When ``True`` the handle's :pyfunc:`~SteerableToolHandle.result`
            yields ``(answer, messages)`` – the first element is the
            assistant's reply, the second the hidden chain‑of‑thought (useful
            for debugging).
        _parent_chat_context : list[dict] | None
            Optional read‑only chat history that will be provided to all nested
            tool calls.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Duplex channels enabling interactive clarification questions. If
            supplied the LLM may push a follow‑up question onto
            *_clarification_up_q* and must read the human's answer from
            *_clarification_down_q*.

        Returns
        -------
        SteerableToolHandle
            Handle that eventually yields the answer text (and optionally the
            hidden reasoning steps).
        """

    # ------------------------------------------------------------------ #
    # File-specific Q&A                                                  #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask_about_file(
        self,
        file_path: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        response_format: Optional[Any] = None,
    ) -> SteerableToolHandle:
        """
        Interrogate **one specific file** (read‑only) and obtain a live
        :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method when the caller already knows which file is relevant
        and wants a focused analysis (e.g., summarise this PDF, extract a key
        value from a document).

        Clarifications
        --------------
        Do not use this method to ask the human follow‑up questions. If the
        file_path is ambiguous and a clarification tool is available, route a
        targeted question via ``request_clarification``; if no channel exists,
        proceed with sensible defaults/best‑guess values and state assumptions
        in the outer reply.

        Parameters
        ----------
        file_path : str
            Logical identifier/path of the target file.
        question : str
            Natural‑language question about the specific file.

        Returns
        -------
        SteerableToolHandle
            Handle that eventually yields the answer text (and optionally the
            hidden reasoning steps) for this file‑scoped query.
        """

    # ------------------------------------------------------------------ #
    # Async parse                                                         #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def parse_async(
        self,
        file_paths: Union[str, List[str]],
        **options: Any,
    ) -> Any:
        """
        Asynchronously parse one or more files and yield per-file results.

        Implementations should mirror the return modes of parse and may yield
        compact/full/none results depending on configuration.
        """

    # ------------------------------------------------------------------ #
    # Reorganization                                                     #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def organize(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a **re‑organization** request – rename/move only – and obtain a
        steerable :class:`SteerableToolHandle` for the tool loop.

        Behaviour and constraints
        -------------------------
        - Only rename and move are permitted; do not create or delete files.
        - Mutations are capability‑guarded by the underlying adapter.
        - Use read‑only discovery (``ask``) to identify targets before
          mutating when helpful.

        Clarifications
        --------------
        Do not use the final response to ask the human questions. If the
        request is underspecified (e.g., grouping criteria, destination) and a
        clarification tool is available, route a focused follow‑up; otherwise
        proceed with sensible defaults and state assumptions in the final
        summary.

        Parameters
        ----------
        text : str
            High‑level English description of the desired re‑organization.
        _return_reasoning_steps, _parent_chat_context,
        _clarification_up_q, _clarification_down_q
            Same purpose and semantics as in :py:meth:`ask`.

        Returns
        -------
        SteerableToolHandle
            Handle that eventually yields a natural‑language summary of the
            operations performed (and optionally hidden reasoning steps).
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public sync                                                        #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def sync(self, *, file_path: str) -> Dict[str, Any]:
        """
        Synchronize a previously ingested file with the underlying filesystem.

        Purge existing rows in relevant contexts and re-ingest. Implementations
        must respect ingest layout (per_file vs unified) when purging.
        """


# Attach centralised docstring
BaseFileManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
