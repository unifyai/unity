from __future__ import annotations

# Backwards-compat shim: re-export BaseFileManager from managers.base
from unity.common.global_docstrings import CLEAR_METHOD_DOCSTRING
from unity.file_manager.managers.base import BaseFileManager  # noqa: F401
from typing import Any, Dict, List, Optional, Type
import asyncio
from abc import abstractmethod
from pydantic import BaseModel
from unity.manager_registry import SingletonABCMeta
from unity.common.async_tool_loop import SteerableToolHandle
from ..common.state_managers import BaseStateManager


class BaseGlobalFileManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **global file‑manager** must satisfy.

    Purpose
    -------
    A global file‑manager presents a unified surface over multiple underlying
    filesystem‑specific file managers (e.g., Local, Interact, CodeSandbox).
    It exposes two conversational entry‑points and delegates the heavy lifting
    to each concrete FileManager via class‑named tools (e.g.
    ``LocalFileManager_ask``):

    • ``ask``      — answer read‑only questions about files (aggregated view)
    • ``organize`` — plan/execute safe file rename/move operations

    The GlobalFileManager itself does not expose low‑level retrieval or
    mutation tools. Instead, its toolsets include:
    - a discovery helper to list configured filesystems, and
    - each underlying manager's class‑named tools (read‑only or organizing),
      so the LLM can directly choose the appropriate manager without aliases.

    High‑level behaviour
    --------------------
    - The global view aggregates answers by delegating to per‑filesystem
      managers. Implementations SHOULD ensure any namespacing needed for paths
      happens within the chosen manager.
    - Organize operations MUST NOT create or delete files; only rename/move is
      allowed and always routed to the filesystem‑specific manager which
      capability‑guards the operation.

    Clarifications
    --------------
    Do not ask the human questions in the final answer. When a clarification
    tool is available, route follow‑ups through it. When no such tool exists,
    proceed with sensible defaults or best‑guess values and explicitly state
    assumptions in the final reply.
    """

    _as_caller_description: str = (
        "the FileManager, managing files on behalf of the end user"
    )

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
        response_format: Optional[Type[BaseModel]] = None,
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
        Do not use the final response to ask the human questions. If the
        request is underspecified (e.g., "which filesystem?", "which folder?",
        "which topic?"), and a clarification tool is available, route a
        follow‑up via ``request_clarification``. When no channel exists,
        proceed with sensible defaults/best‑guess values and explicitly state
        assumptions in the outer reply.

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow this method to determine the
        best method to answer it (e.g., filter/search/join, parse‑if‑missing
        when explicitly requested).

        Parameters
        ----------
        text : str
            The user's plain‑English question (e.g. *"Which PDFs mention ISO
            27001 across all filesystems?"*).
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
        response_format: Optional[Type[BaseModel]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a **re‑organization** request – rename/move only – and obtain a
        steerable :class:`SteerableToolHandle`.

        Behaviour and constraints
        -------------------------
        - Only rename and move are permitted; do **not** create or delete files.
        - Cross‑root moves (e.g., moving a file between two different top‑level
                filesystems) are not allowed; re‑organize within a root.
        - When multiple filesystems are present, the LLM should state which root
          is being organized, or use tools that make the root explicit.

        Clarifications
        --------------
        Do not use the final response to ask the human questions. If the
        request is underspecified (e.g., grouping criteria, destination) and a
        clarification tool is available, route a focused follow‑up; otherwise
        proceed with sensible defaults and state assumptions in the summary.

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

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    # -------------------------- Discovery helper --------------------------- #
    @abstractmethod
    def list_filesystems(self) -> List[str]:
        """
        Return a list of human‑readable identifiers for the configured
        filesystems. Implementations may return class names of the underlying
        managers (recommended) or any other stable labels suitable for prompts.
        """


# Attach centralised docstring
BaseGlobalFileManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
