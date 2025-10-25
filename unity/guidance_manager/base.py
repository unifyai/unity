from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import Any, Dict, List, Optional

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseGuidanceManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete guidance-manager must satisfy.

    Exposes two high-level, English-language operations:
    • ask    — interrogate existing guidance entries (read-only)
    • update — create/edit/delete guidance entries (mutations)
    """

    # ------------------------------- Public API -------------------------------
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
        Interrogate the existing Guidance table (read-only) and obtain a live
        SteerableToolHandle.

        Purpose
        -------
        Use this to search for guidance by title/content and to retrieve
        specific entries that already exist (e.g., by guidance_id).
        This call must never create, modify, or delete entries.

        Clarifications
        --------------
        Do not ask the human questions here. If the caller needs clarification,
        route the question via a dedicated request_clarification tool when
        available. If no clarification channel exists, proceed with sensible
        defaults/best-guess values and state those assumptions in the outer
        loop's final reply.

        Returns
        -------
        SteerableToolHandle
            A live handle that yields the assistant's answer and exposes
            steering operations (pause, resume, interject, stop).
        """

    @abstractmethod
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a mutation request – create, edit, or delete guidance entries –
        expressed in plain English and receive a steerable LLM handle.

        Ask vs Clarification
        --------------------
        • ask is ONLY for inspecting/locating entries that ALREADY EXIST.
        • Do NOT use ask to ask the human for details about NEW entries; call
          request_clarification when a clarification channel is available.
        • When no clarification tool exists, proceed with sensible defaults and
          state those assumptions in the final reply.

        Returns
        -------
        SteerableToolHandle
            Handle whose result yields confirmation of the mutation and (optionally)
            reasoning steps.
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseGuidanceManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
