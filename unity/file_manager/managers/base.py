from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.global_docstrings import CLEAR_METHOD_DOCSTRING
from unity.singleton_registry import SingletonABCMeta
from unity.common.state_managers import BaseStateManager


class BaseFileManager(BaseStateManager, metaclass=SingletonABCMeta):
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
    def exists(self, filename: str) -> bool:
        """Return True if a file with the given display name exists in this filesystem."""

    @abstractmethod
    def list(self) -> List[str]:
        """Return the list of display names (stable order) for files in this filesystem."""

    @abstractmethod
    def parse(
        self,
        filenames: Union[str, List[str]],
        **options: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse one or more files and return structured results per file.

        Parameters
        ----------
        filenames : str | list[str]
            Single filename or a list of filenames to parse.
        **options : Any
            Parser-specific options (forwarded as-is).

        Returns
        -------
        dict[str, dict]
            Mapping from filename → result dict containing status/records/full_text/metadata.
        """

    # ------------------------------------------------------------------ #
    # File export operations (for parsing)                               #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def export_file(self, filename: str, destination_dir: str) -> str:
        """
        Export a file from the underlying filesystem to a local destination directory.

        This method is used by parse operations to bring files from the adapter's
        filesystem into a local temporary directory with their original filenames preserved.

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
    ) -> Dict[str, Any] | List[str]:
        """
        Return the schema for this manager's primary index context.

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column → logical type. When False,
            return just the list of column names.

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
    ) -> List[Dict[str, Any]]:
        """
        Filter files using a boolean expression evaluated per row.

        Notes
        -----
        For queries that need attributes from per-file content (e.g., content_type,
        title, summary), prefer joining the global index with a per-file context via
        the join tools provided by the concrete manager.
        """

    @abstractmethod
    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over files using Unify vector columns and references.

        Notes
        -----
        For content-aware queries (e.g., search within sections/sentences), prefer
        performing a semantic search after joining the global index with the
        relevant per-file table using the join tools; this focuses embeddings on the
        correct text column (e.g., "summary").
        """

    @abstractmethod
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
            If no file found with the given file_id or no updates provided
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
            New filename or basename; adapter determines full path semantics.

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
        Interrogate the existing filesystem (read-only) and obtain a steerable LLM handle.

        Purpose
        -------
        Use this method to ask semantic questions that may consider multiple files/folders,
        perform semantic search over parsed contents, aggregate/summarise results, or shortlist
        relevant files. This call must never create or delete files.

        Clarifications
        --------------
        Do not ask the human in the final response; when clarification is required and a
        clarification tool is available, push a question to the up-queue and read from the
        down-queue. If no clarification channel exists, proceed with sensible defaults/best guesses
        and state assumptions in the outer response.

        Returns
        -------
        SteerableToolHandle
            A handle controlling the interactive tool-use loop. Call ``await handle.result()``
            to get the final answer.
        """

    # ------------------------------------------------------------------ #
    # File-specific Q&A                                                  #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask_about_file(
        self,
        filename: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Answer a question about a single file (read-only) and obtain a steerable LLM handle.

        Use when the caller already knows which file is relevant and wants a focused analysis
        (e.g., summarise this PDF, extract key data points from this document).

        Parameters
        ----------
        filename : str
            Logical identifier/path of the target file.
        question : str
            Natural-language question about the specific file.

        Returns
        -------
        SteerableToolHandle
            A handle controlling the interactive tool-use loop for file-scoped queries.
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
        Restructure/organize the filesystem (rename/move only) using an LLM-driven tool loop.

        This method runs an async tool loop similar to other managers' `update`/`refactor`, exposing
        safe, capability-gated tools (e.g., rename/move) and read-only discovery tools. The loop
        returns a steerable handle; the final result contains a natural-language summary of the
        reorganization plan and actions executed (if any).

        Returns
        -------
        SteerableToolHandle
            Handle controlling the interactive tool-use loop for reorganization.
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseFileManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
