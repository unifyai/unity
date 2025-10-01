from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseSkillManager(ABC, metaclass=SingletonABCMeta):
    """
    Public contract for a high-level catalogue of assistant "skills".

    A "skill" is a reusable capability exposed through stored functions managed
    by the FunctionManager. SkillManager provides a natural-language interface
    for discovering and understanding what the assistant can do.

    Implementations must expose one public method `ask` which answers questions
    about available skills and how they relate to the underlying reusable
    functions. Implementations may connect to a real backing store via
    FunctionManager, or use a simulated catalogue – but the public contract is
    the same.
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
        parent_chat_context : list[dict] | None
            Optional read-only context to pass to the inner tool loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels used to ask the human clarifying questions
            when an ambiguous skills question is asked.

        Returns
        -------
        SteerableToolHandle
            Handle for a live tool loop that ultimately yields the assistant's
            answer and exposes steering operations (pause/resume/interject/stop).
        """
