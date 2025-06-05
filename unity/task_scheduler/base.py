from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.llm_helpers import SteerableToolHandle


class BaseTaskScheduler(ABC):
    """
    *Public* contract that every concrete **task-list-manager** must satisfy.

    Managers expose exactly **two** user-facing methods:

    • `ask`    – answer questions about the current task list
    • `update` – create, modify or delete tasks and queues

    Implementations may use Unify logs, a local DB, a remote API or even a
    purely simulated LLM – but they all obey the signatures & docstrings below.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Answer a natural-language question about the task list.

        Returns
        -------
        SteerableToolHandle
            Await `handle.result()` for the answer (and optionally hidden
            reasoning) or call `handle.interject(...)` / `handle.stop()`.
        """

    @abstractmethod
    def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Execute a plain-English command that modifies the task list
        (create / delete tasks, change status, reorder queue, …).

        All parameters mirror :py:meth:`ask`.  See that method for details.
        """

    @abstractmethod
    def start_task() -> SteerableToolHandle:
        """
        Start execution of *task_id* and return a steerable handle.

        • Fails if another task is already active.
        • Promotes the task's status to **active**.
        • Clears the primed pointer when relevant.
        """
