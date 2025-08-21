from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from unity.common.llm_helpers import SteerableToolHandle

logger = logging.getLogger(__name__)

__all__ = [
    "BaseActiveTask",
    "BasePlanner",
    "PhoneCallHandle",
    "BrowserSessionHandle",
    "ComsManager",
]

# --------------------------------------------------------------------------- #
# BasePlan
# --------------------------------------------------------------------------- #


class BaseActiveTask(SteerableToolHandle, ABC):
    """
    Abstract contract that every concrete *active task* must satisfy.

    An active task represents a long-running task that can be steered at runtime
    (pause / resume / interject / ask / stop) and that ultimately resolves
    to a single result string.

    Sub-classes **must** provide concrete implementations of all abstract
    members below and expose them via ``valid_tools`` so that higher-level
    agents (or the UI) can discover the currently available controls.
    """

    # ───────────────────────────── Public API ───────────────────────────── #

    @abstractmethod
    async def ask(self, question: str) -> str:
        """
        Ask any question about the live (ongoing and active) task being worked on.
        """

    @property
    @abstractmethod
    def valid_tools(self) -> Dict[str, Callable]:
        """
        Map of *public-name* ➜ *callable* for the user-accessible controls
        that are *currently* valid in the plan's lifecycle state.
        """


# --------------------------------------------------------------------------- #
# BasePlanner
# --------------------------------------------------------------------------- #


class BasePlanner(ABC):
    """
    Abstract contract that every concrete *planner* must satisfy.

    A planner is a *factory* that spawns exactly one *active* plan at a time
    (for now).  It keeps a reference to that plan so that external callers
    can query its status or steer it later.
    """

    def __init__(self) -> None:
        self._active_task: Optional[BaseActiveTask] = None

    # ─────────────────────────── Plan management ────────────────────────── #

    async def execute(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> BaseActiveTask:
        """
        Create (and start) a new active task.

        Sub-classes implement the actual creation logic in
        :meth:`_make_plan`.  This thin wrapper only enforces the
        *single-active-plan* rule and stores the reference.
        """
        if self._active_task is not None:
            raise RuntimeError(
                "Another plan is still active. Stop it or wait for "
                "completion before starting a new one.",
            )

        active_task = await self._execute_task_and_return_handle(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        self._active_task = active_task
        return active_task

    @abstractmethod
    async def _execute_task_and_return_handle(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> BaseActiveTask:
        """
        Concrete planner must build **and start** an active task implementation
        (e.g. ``SimulatedActiveTask``) and return it.
        """

    # ────────────────────────── Convenience API ─────────────────────────── #

    @property
    def active_task(self) -> Optional[BaseActiveTask]:
        """Return the currently running task (or *None* if idle)."""
        return self._active_task

    def clear_active_task(self) -> None:
        """Forget the active task (useful once it has completed)."""
        self._active_task = None
