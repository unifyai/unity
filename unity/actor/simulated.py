import asyncio
import os
import json

import unify
from .base import BaseActor
from typing import Optional

from ..task_scheduler.simulated import SimulatedActiveTask


class SimulatedActor(BaseActor):
    def __init__(
        self,
        *,
        steps: int | None = None,
        timeout: float | None = None,
        _requests_clarification: bool = False,
    ) -> None:
        """
        Initialize a simulated actor.

        Args:
            steps:      *(Optional)* Maximum tool steps each activity should run
                        before auto-completion.
            timeout:    *(Optional)* Maximum wall-clock seconds before an activity
                        auto-completes.
        """
        self._steps = steps
        self._timeout = timeout
        self._requests_clarification = _requests_clarification

        # One shared, memory-retaining LLM for all activities
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a simulated actor and executor. "
            "Invent plausible progress and remain internally consistent "
            "across multiple calls.",
        )

    async def act(
        self,
        description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SimulatedActiveTask:
        return SimulatedActiveTask(
            self._llm,
            description,
            steps=self._steps,
            timeout=self._timeout,
            parent_chat_context=parent_chat_context,
            _requests_clarification=self._requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
