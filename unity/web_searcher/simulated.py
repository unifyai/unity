from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Dict, List, Optional

import unify
from .prompt_builders import build_ask_prompt, build_simulated_method_prompt
from ..common.simulated import mirror_web_searcher_tools
from .base import BaseWebSearcher
from ..common.async_tool_loop import SteerableToolHandle
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
import functools


class _SimulatedWebSearcherHandle(SteerableToolHandle):
    """Minimal LLM-backed handle used by SimulatedWebSearcher.ask."""

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait("Could you clarify your web query?")
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []
        self._done = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False

    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            if self._needs_clar:
                clar = await self._clar_down_q.get()
                self._extra_msgs.append(f"Clarification: {clar}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)

            answer = await self._llm.generate(prompt)
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done.set()

        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    def interject(self, message: str) -> str:
        if self._cancelled:
            return "Interaction stopped."
        self._extra_msgs.append(message)
        return "Acknowledged."

    def stop(self, reason: str | None = None) -> str:
        self._cancelled = True
        self._done.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

    def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._paused = True
        return "Paused."

    def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._paused = False
        return "Resumed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set()

    async def ask(self, question: str) -> "SteerableToolHandle":
        q_msg = (
            f"Your only task is to simulate an answer to the following question: {question}\n\n"
            "However, there is also an ongoing simulated process which had the instructions given below. "
            "Please make your answer realistic and conceivable given the provided context of the simulated task."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial]
            + self._extra_msgs
            + [f"Question to answer (as a reminder!): {question}"],
        )
        return _SimulatedWebSearcherHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        try:
            if self._clar_up_q is not None:
                msg = await self._clar_up_q.get()
                return {"message": msg}
        except Exception:
            pass
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            if self._clar_down_q is not None:
                await self._clar_down_q.put(answer)
        except Exception:
            pass


class SimulatedWebSearcher(BaseWebSearcher):
    """Drop-in simulated WebSearcher with imaginary results and stateful memory."""

    def __init__(
        self,
        description: str = "simulate sensible web research answers",
        *,
        log_events: bool = False,
    ) -> None:
        self._description = description
        self._log_events = log_events
        # Mirror the real manager's tool exposure programmatically for prompts
        self._ask_tools = mirror_web_searcher_tools()

        # Stateful async LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )

        # Reference the real prompt as context (no real tools here)
        ask_msg = build_ask_prompt(tools=self._ask_tools)
        self._llm.set_system_message(
            "You are a simulated web-search assistant. There is no real browser or API – "
            "invent plausible sources and keep your narrative consistent.\n\n"
            "For reference, here is the real system message outline used by the production WebSearcher.ask:"
            f"\n\n{ask_msg}\n\nBack-story: {self._description}",
        )

    @functools.wraps(BaseWebSearcher.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "WebSearcher",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=parent_chat_context,
        )

        handle = _SimulatedWebSearcherHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(handle, call_id, "WebSearcher", "ask")

        return handle

    @functools.wraps(BaseWebSearcher.clear, updated=())
    def clear(self) -> None:
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "simulate sensible web research answers",
            ),
            log_events=getattr(self, "_log_events", False),
        )
