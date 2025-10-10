from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseGuidanceManager(ABC, metaclass=SingletonABCMeta):
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
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
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
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
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
        """
        Remove all guidance entries and re-initialise the manager's storage.

        Implementations must delete the underlying storage/context for guidance
        and recreate any required schema so that subsequent reads/writes operate
        against a clean slate.

        The method is synchronous to allow safe use inside thread offloads in
        async flows.
        """
        raise NotImplementedError

    # ------------------------------ Private tools -----------------------------
    @abstractmethod
    def _filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List["Guidance"]:
        """
        Return guidance records that satisfy the Python-expression filter.
        """
        raise NotImplementedError

    @abstractmethod
    def _search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List["Guidance"]:
        """
        Semantic search across guidance using title/content (or derived expressions).
        """
        raise NotImplementedError

    @abstractmethod
    def _add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[Dict[str, int]] = None,
    ) -> "ToolOutcome":
        """Create and persist a new guidance entry and return a standard outcome."""
        raise NotImplementedError

    @abstractmethod
    def _update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[Dict[str, int]] = None,
    ) -> "ToolOutcome":
        """Modify one guidance entry identified by guidance_id and return the outcome."""
        raise NotImplementedError

    @abstractmethod
    def _delete_guidance(
        self,
        *,
        guidance_id: int,
    ) -> "ToolOutcome":
        """Delete the specified guidance entry and return the outcome."""
        raise NotImplementedError


if TYPE_CHECKING:
    # Avoid runtime imports to prevent circular dependencies
    from .types.guidance import Guidance
    from ..common.tool_outcome import ToolOutcome
