from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseSecretManager(ABC, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete secret-manager must satisfy.

    A secret-manager answers questions (`ask`) about stored secrets and handles
    natural-language instructions (`update`) that create or change those secrets.
    All implementations expose exactly these two public methods and return a
    SteerableToolHandle so callers can pause/resume/stop or interject.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
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
        Interrogate existing secrets (read-only) and obtain a live handle.

        Notes
        -----
        - Do not reveal raw secret values to the LLM. Answers must refer to
          secrets by placeholder (e.g., "${name}") or metadata only.
        - This method must never create, modify or delete secrets.
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
        Apply a mutation request – create, edit, or delete secrets – expressed
        in plain English and receive a steerable LLM handle.

        Notes
        -----
        - Raw secret values must never be echoed back to the LLM. Any value
          supplied in this call may be persisted, but tool and user-facing
          messages must use `${name}` placeholders only.
        """

    # ------------------------------------------------------------------ #
    # Private helpers that concrete managers must implement              #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _list_secret_keys(self) -> List[str]:
        """Return all available secret names (keys) stored by the manager.

        Returns
        -------
        list[str]
            Sorted, unique list of secret names.
        """

    @abstractmethod
    def _search_secrets(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List["Secret"]:
        """Semantic search over secrets (typically by description).

        Parameters
        ----------
        references : dict[str, str] | None, default None
            Mapping from a source expression (e.g., "description") to the
            reference text used for similarity ranking. ``None`` or empty
            disables semantic search and should fall back to recency.
        k : int, default 10
            Maximum number of results.

        Returns
        -------
        list[Secret]
            Up to ``k`` redacted Secret models (must not expose raw values).
        """

    @abstractmethod
    def _filter_secrets(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List["Secret"]:
        """Retrieve secret records that satisfy *filter* (read-only)."""

    @abstractmethod
    def _create_secret(
        self,
        *,
        name: str,
        value: str,
        description: Optional[str] = None,
    ) -> "ToolOutcome":
        """Create a new secret with a unique name."""

    @abstractmethod
    def _update_secret(
        self,
        *,
        name: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ToolOutcome":
        """Update an existing secret identified by its unique name."""

    @abstractmethod
    def _delete_secret(
        self,
        *,
        name: str,
    ) -> "ToolOutcome":
        """Permanently remove a secret by name."""


if TYPE_CHECKING:
    from .types import Secret
    from ..common.tool_outcome import ToolOutcome
