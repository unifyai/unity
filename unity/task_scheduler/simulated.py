# unity/task_scheduler/simulated_task_scheduler.py
import asyncio
import json
import os
import threading
import functools
from typing import List, Optional

import unify

from ..common.llm_helpers import SteerableToolHandle
from .base import BaseTaskScheduler


class _SimulatedTaskScheduleHandle(SteerableToolHandle):
    """A minimal, LLM-backed handle for ask/update interactions."""

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        mode: str,
        _return_reasoning_steps: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> None:
        self._llm = llm
        self._initial_text = initial_text
        self._mode = mode  # "ask" | "update"
        self._ret_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._needs_clar = self._clar_up_q is not None and self._clar_down_q is not None

        # ── fire the clarification request right away ──────────────────
        self._clar_requested = False
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you please clarify exactly what you want?",
                )
                self._clar_requested = True
            except asyncio.QueueFull:
                pass

        self._interjections: List[str] = []

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: Optional[str] = None
        self._messages: List[dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # Public API required by SteerableToolHandle
    # ──────────────────────────────────────────────────────────────────────
    async def result(self):
        """Return the LLM answer (or raise if stopped)."""
        if self._cancelled:
            raise asyncio.CancelledError()

        if not self._done_event.is_set():
            # Wait for clarification answer if required
            if self._needs_clar:
                clar_reply = await self._clar_down_q.get()
                self._interjections.append(f"Clarification: {clar_reply}")

            prompt_parts = [self._initial_text] + self._interjections
            user_block = "\n\n---\n\n".join(prompt_parts)

            answer = await asyncio.to_thread(
                self._llm.generate,
                user_block,
            )

            self._answer = answer
            # very small, synthetic trace of “reasoning”
            self._messages = [
                {"role": "user", "content": user_block},
                {"role": "assistant", "content": answer},
            ]
            self._done_event.set()

        if self._ret_steps:
            return self._answer, self._messages
        return self._answer

    def interject(self, message: str) -> str:
        """Append a follow-up message that will be folded into the prompt."""
        if self._cancelled:
            return "Interaction already stopped."
        self._interjections.append(message)
        return "Noted."

    def stop(self) -> str:
        """Cancel further processing so `.result()` raises."""
        self._cancelled = True
        self._done_event.set()
        return "Stopped."

    # The orchestrator checks this in the stop-test.
    def done(self) -> bool:  # type: ignore[override]  # (property in abc)
        return self._done_event.is_set()

    # Allow outer orchestration to discover zero tools
    @property
    def valid_tools(self):
        return {}


class SimulatedTaskScheduler(BaseTaskScheduler):
    """
    Drop-in replacement for TaskScheduler where the underlying data is
    entirely imaginary – useful for offline demos or unit tests that only
    need the conversational surface.
    """

    def __init__(self, description: str = "You manage an imaginary task list.") -> None:
        self._description = description

        # One shared, stateful LLM client
        self._llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* task-list manager. "
            "There is NO real database – fabricate convincing yet *consistent* "
            "answers and confirmations over time.\n\n"
            f"Back-story: {self._description}",
        )

    # ------------------------------------------------------------------ #
    #  ask                                                               #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.ask, updated=())
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,  # Ignored – we do not expose tools
        parent_chat_context: list[dict] | None = None,  # Unused – synthetic
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        if parent_chat_context:
            self._llm._system_message += (
                f"\nCalling chat context:{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedTaskScheduleHandle(
            self._llm,
            text,
            mode="ask",
            return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # ------------------------------------------------------------------ #
    #  update                                                            #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.update, updated=())
    def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,  # Ignored – no tools here
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        if parent_chat_context:
            self._llm._system_message += (
                f"\nCalling chat context:{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedTaskScheduleHandle(
            self._llm,
            text,
            mode="update",
            return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
