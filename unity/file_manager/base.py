from __future__ import annotations

# Backwards-compat shim: re-export BaseFileManager from managers.base
from unity.file_manager.managers.base import BaseFileManager  # noqa: F401
from typing import Any, Dict, List, Optional
import asyncio
from abc import abstractmethod
from unity.singleton_registry import SingletonABCMeta
from unity.common.async_tool_loop import SteerableToolHandle
from ..common.state_managers import BaseStateManager


class BaseGlobalFileManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **global file‑manager** must satisfy.

    Purpose
    -------
    A global file‑manager presents a single, unified surface over multiple
    underlying filesystem‑specific file managers (e.g., Local, Interact,
    CodeSandbox). It exposes two conversational entry‑points:

    • ``ask``      — answer read‑only questions about files (aggregated view)
    • ``organize`` — plan/execute safe file rename/move operations

    Retrieval is strictly Unify‑backed: all filtering, listing and semantic
    search MUST be performed via Unify tables, never by walking a filesystem in
    the middle of a tool call. Mutations (rename/move) are delegated to the
    appropriate underlying ``FileManager`` which, in turn, guards adapter
    capabilities.

    High‑level behaviour
    --------------------
    - The global view aggregates results from all configured file managers.
    - Returned filenames SHOULD be namespaced to avoid collisions, e.g.
      ``/google_drive/Quarterly.pdf`` and ``/local/Quarterly.pdf``.
    - Implementations SHOULD add a synthetic ``source_filesystem`` field so
      callers can filter or present results by origin.
    - Organize operations MUST NOT create or delete files; only rename/move is
      allowed and always routed to the filesystem‑specific manager.

    Clarifications
    --------------
    Do not ask the human questions in the final answer. When a clarification
    tool is available, route follow‑ups through it. When no such tool exists,
    proceed with sensible defaults or best‑guess values and explicitly state
    assumptions in the final reply.
    """

    # ------------------------------ Public API ------------------------------ #
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
        Interrogate the **existing files** (read‑only) across all configured
        filesystems and obtain a live :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method to locate and inspect files that already exist in the
        global catalogue: list/compare filenames, perform semantic search over
        parsed content, summarise relevant documents, or shortlist files for an
        organization pass. This method must never create, modify or delete
        files, nor should it manipulate filesystem structure.

        Clarifications
        --------------
        Do not ask the human follow‑ups in the final response. If the caller
        needs clarification (e.g., "which filesystem?", "which folder?",
        "which topic?"), route the question via a dedicated
        ``request_clarification`` tool when available. If no clarification
        channel exists, proceed with sensible defaults/best‑guess values and
        state those assumptions in the outer loop's final reply.

        Parameters
        ----------
        text : str
            The user's plain‑English question (e.g. *"Which PDFs mention ISO
            27001 across Google Drive and the intranet?"*).
        _return_reasoning_steps : bool, default ``False``
            When *True*, the handle's :pyfunc:`~SteerableToolHandle.result`
            yields ``(answer, messages)`` – the reply and the hidden chain‑of‑
            thought (useful for debugging).
        parent_chat_context : list[dict] | None
            Optional read‑only chat history that will be provided to all nested
            tool calls.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Duplex channels enabling interactive clarification questions.

        Returns
        -------
        SteerableToolHandle
            Handle that eventually yields the answer text (and, optionally, the
            hidden reasoning steps). The handle also supports
            ``pause()/resume()/interject()/stop()``.
        """

    @abstractmethod
    async def organize(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        **Restructure** the aggregated filesystem view by renaming or moving
        files and return a steerable handle. This operation delegates to the
        correct underlying ``FileManager`` per file and is strictly
        capability‑guarded by its adapter.

        Behaviour and constraints
        -------------------------
        - Only rename and move are permitted; do **not** create or delete files.
        - Cross‑root moves (e.g., moving a file between two different top‑level
          filesystems) are not allowed; re‑organize within a root.
        - When multiple filesystems are present, the LLM should state which root
          is being organized, or use tools that make the root explicit.

        Clarifications
        --------------
        When the intent or target is ambiguous (e.g., criteria for grouping),
        route follow‑ups through a clarification tool if available. Otherwise
        proceed with sensible defaults and note assumptions in the summary.

        Parameters
        ----------
        text : str
            A high‑level English description of the desired re‑organization
            (e.g., *"Group reports by year/month within their existing
            root."*). The low‑level edits are carried out by the LLM via safe
            tools.
        _return_reasoning_steps, parent_chat_context,
        clarification_up_q, clarification_down_q
            Same semantics as :py:meth:`ask`.

        Returns
        -------
        SteerableToolHandle
            Handle whose :pyfunc:`result` yields a natural‑language summary of
            the operations executed and, optionally, the hidden chain‑of‑thought
            when *_return_reasoning_steps* is *True*.
        """

    # ------------------------------------------------------------------ #
    #  Private retrieval helpers (part of the contract)                  #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        """
        Return the Unify table schema visible through the global view.

        Implementations SHOULD include a synthetic ``source_filesystem`` column
        that identifies the originating filesystem for each row. When
        ``include_types`` is *True*, return a mapping ``{column: type}``; when
        *False*, return a simple list of column names.
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
        Filter files using a boolean expression evaluated per row (Unify‑backed).

        Parameters
        ----------
        filter : str | None, default ``None``
            Python expression applied per row across the global view. When
            *None*, return all files.
        offset : int, default ``0``
            Zero‑based index of the first result to include.
        limit : int, default ``100``
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            Matching rows (aggregated across filesystems). Implementations
            SHOULD prefix/namespace filenames and include ``source_filesystem``.
        """

    @abstractmethod
    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over files using Unify vector columns and reference text(s).

        Parameters
        ----------
        references : dict[str, str] | None, default ``None``
            Mapping of ``source_expr → reference_text`` terms. Each *source_expr*
            is either a plain column (e.g., ``"full_text"``) or a full derived
            expression; reference text is embedded server‑side for ranking.
        k : int, default ``10``
            Maximum number of rows to return (best‑first).

        Returns
        -------
        list[dict]
            Up to *k* rows ordered by similarity, aggregated across all
            filesystems. Implementations SHOULD namespace filenames and include
            the ``source_filesystem`` column.
        """

    @abstractmethod
    def _rename_file(
        self,
        *,
        filesystem: str,
        target_id_or_path: str,
        new_name: str,
    ) -> Dict[str, Any]:
        """Rename a file within a specific filesystem."""

    @abstractmethod
    def _move_file(
        self,
        *,
        filesystem: str,
        target_id_or_path: str,
        new_parent_path: str,
    ) -> Dict[str, Any]:
        """Move a file to a new parent path within a specific filesystem."""

    @abstractmethod
    def _delete_file(self, *, filesystem: str, file_id: int) -> Dict[str, Any]:
        """Delete a file from a specific filesystem."""
