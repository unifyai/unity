from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.llm_helpers import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseFileManager(ABC, metaclass=SingletonABCMeta):
    """
    Read-only registry for files received or downloaded during a Unity session.

    Lifecycle: Instances are session-scoped and should delete all managed files
    when the session ends.
    """

    # Public interface ----------------------------------------------------- #
    @abstractmethod
    def exists(self, filename: str) -> bool:
        """Return True if a file with the given display name is registered."""

    @abstractmethod
    def list(self) -> List[str]:
        """Return the list of registered display names (stable order)."""

    @abstractmethod
    async def ask(
        self,
        filename: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Start a read-only, steerable tool loop to answer a question about a file.
        The loop exposes at least the `parse` tool.
        """

    @abstractmethod
    def parse(self, filename: str, **options: Any) -> Dict[str, Any]:
        """
        Return a structured representation of the file's contents and metadata.
        Implementations should never mutate storage (read-only).
        """
