from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from ..common.async_tool_loop import SteerableToolHandle
from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseSecretManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete secret-manager must satisfy.

    A secret-manager answers questions (`ask`) about stored secrets and handles
    natural-language instructions (`update`) that create or change those secrets.
    All implementations expose exactly these two public methods and return a
    SteerableToolHandle so callers can pause/resume/stop or interject.
    """

    _as_caller_description: str = (
        "the SecretManager, managing secrets on behalf of the end user"
    )

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Interrogate existing secrets (read-only) and obtain a live handle.

        Parameters
        ----------
        text : str
            The query in plain English (e.g. "Which API keys do we have?").
        response_format : Type[BaseModel] | None, default ``None``
            Optional Pydantic model to request a structured answer.

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
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a mutation request – create, edit, or delete secrets – expressed
        in plain English and receive a steerable LLM handle.

        Parameters
        ----------
        text : str
            The mutation request in plain English
            (e.g. "Store the Stripe API key as 'stripe_key'").
        response_format : Type[BaseModel] | None, default ``None``
            Optional Pydantic model to request a structured outcome.

        Notes
        -----
        - Raw secret values must never be echoed back to the LLM. Any value
          supplied in this call may be persisted, but tool and user-facing
          messages must use `${name}` placeholders only.
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseSecretManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
