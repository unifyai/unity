from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from ..common.async_tool_loop import SteerableToolHandle
from ..manager_registry import SingletonABCMeta
from ..common.state_managers import BaseStateManager


class BaseSkillManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract for a high-level catalogue of assistant "skills".

    A "skill" is a reusable capability backed by an underlying function
    catalogue. The SkillManager provides a natural-language interface for
    discovering and understanding what the assistant can do.

    Implementations must expose one public method `ask`, which answers
    questions about available skills and how they relate to the underlying
    stored functions. Implementations may connect to a real function
    catalogue, a remote service, or a simulated catalogue – but the public
    contract is the same.
    """

    _as_caller_description: str = "the SkillManager, discovering assistant capabilities"

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
        Interrogate the assistant's available skills and obtain a live handle.

        Purpose
        -------
        Use this method to discover what skills exist, what they do, how to
        invoke them, and how skills map to the underlying stored functions –
        including their signatures and docstrings. This call must be read-only
        with respect to the function catalogue.

        Parameters
        ----------
        text : str
            The user's natural-language question about skills, e.g.
            "What can you do with spreadsheets?" or "List your data-cleaning
            skills and their input requirements".
        _return_reasoning_steps : bool, default False
            When True, `SteerableToolHandle.result` yields
            `(answer, messages)` where messages include hidden chain-of-thought.
        _parent_chat_context : list[dict] | None
            Optional read-only context to pass to the inner tool loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels used to ask the human clarifying questions
            when an ambiguous skills question is asked.

        Returns
        -------
        SteerableToolHandle
            Handle for a live tool loop that ultimately yields the assistant's
            answer and exposes steering operations (pause/resume/interject/stop).
        """
