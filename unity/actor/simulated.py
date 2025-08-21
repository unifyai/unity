import asyncio
import os
import json

import unify
from .base import BasePlanner
from typing import Optional
from ..task_scheduler.simulated import SimulatedActiveTask


class SimulatedPlanner(BasePlanner):
    def __init__(
        self,
        *,
        steps: int | None = None,
        timeout: float | None = None,
        _requests_clarification: bool = False,
    ) -> None:
        """
        Initialize a simulated planner.

        Args:
            steps:      *(Optional)* Maximum tool steps each plan should run
                        before auto-completion.
            timeout:    *(Optional)* Maximum wall-clock seconds before plans
                        auto-complete.
        """
        super().__init__()
        self._steps = steps
        self._timeout = timeout
        self._requests_clarification = _requests_clarification

        # One shared, memory-retaining LLM for *all* plans
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* planner and executor. "
            "Invent plausible task progress and remain internally consistent "
            "across multiple plans and calls.",
        )

    async def _execute_task_and_return_handle(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SimulatedActiveTask:
        return SimulatedActiveTask(
            self._llm,
            task_description,
            self._steps,
            timeout=self._timeout,
            parent_chat_context=parent_chat_context,
            _requests_clarification=self._requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
