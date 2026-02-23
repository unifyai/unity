from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

import unillm
from .prompt_builders import build_ask_prompt, build_simulated_method_prompt
from ..common.llm_client import new_llm_client
from ..common.simulated import (
    mirror_web_searcher_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    build_followup_prompt,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)
from .base import BaseWebSearcher
from ..common.async_tool_loop import SteerableToolHandle
import functools
from ..logger import LOGGER
from ..common.hierarchical_logger import ICONS


class _SimulatedWebSearcherHandle(SimulatedHandleMixin, SteerableToolHandle):
    """Minimal LLM-backed handle used by SimulatedWebSearcher.ask."""

    def __init__(
        self,
        llm: unillm.Unify,
        initial_text: str,
        *,
        mode: str,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        response_format: Optional[Type[BaseModel]] = None,
        hold_completion: bool = False,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._mode = str(mode or "ask")
        self._response_format = response_format
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

        # Human-friendly log label derived from current lineage:
        # "<outer...>->SimulatedWebSearcher.<mode>(abcd)"
        self._log_label = SimulatedLineage.make_label(
            f"SimulatedWebSearcher.{self._mode}",
        )

        if self._needs_clar:
            try:
                q_text = "Could you clarify your web query?"
                self._clar_up_q.put_nowait(q_text)
                try:
                    SimulatedLog.log_clarification_request(self._log_label, q_text)
                except Exception:
                    pass
                try:
                    LOGGER.info(
                        f"{ICONS['clarification']} [{self._log_label}] Clarification requested",
                    )
                except Exception:
                    pass
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []
        self._done = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False
        # Async cancellation signal to break clarification waits
        self._cancel_event: asyncio.Event = asyncio.Event()

        self._init_completion_gate(hold_completion)

    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            if self._needs_clar:
                try:
                    LOGGER.info(
                        f"{ICONS['pending']} [{self._log_label}] Waiting for clarification answer…",
                    )
                except Exception:
                    pass
                clar: str | None = None
                get_task = asyncio.create_task(self._clar_down_q.get())
                cancel_task = asyncio.create_task(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if cancel_task in done:
                    self._done.set()
                    return "processed stopped early, no result"
                try:
                    clar = get_task.result()
                except Exception:
                    clar = None
                if clar is None:
                    self._done.set()
                    return "processed stopped early, no result"
                self._extra_msgs.append(f"Clarification: {clar}")
                try:
                    SimulatedLog.log_clarification_answer(self._log_label, clar)
                except Exception:
                    pass
                try:
                    LOGGER.info(
                        f"{ICONS['interjection']} [{self._log_label}] Clarification answer received",
                    )
                except Exception:
                    pass

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)

            # Unified simulated LLM roundtrip with lineage-aware logging and gated response preview
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            answer = await simulated_llm_roundtrip(
                self._llm,
                label=self._log_label,
                prompt=prompt,
                response_format=self._response_format,
            )
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            await self._await_completion_gate()
            self._done.set()

        # If cancellation happened after the coroutine started, return a stable post-cancel value.
        if self._cancelled:
            return "processed stopped early, no result"
        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Interject a message into the in-flight handle.

        Args:
            message: The interjection message to inject.
            _parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
        """
        if self._cancelled:
            return
        self._log_interject(message)
        self._extra_msgs.append(message)

    async def stop(
        self,
        reason: str | None = None,
        **kwargs,
    ) -> None:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
        """
        self._log_stop(reason)
        self._open_completion_gate()
        self._cancelled = True
        try:
            self._cancel_event.set()
        except Exception:
            pass
        self._done.set()

    async def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._log_pause()
        self._paused = True
        return "Paused."

    async def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._log_resume()
        self._paused = False
        return "Resumed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set() and self._gate_open

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            _parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
        """
        follow_up_prompt = build_followup_prompt(
            question=question,
            initial_instruction=self._initial,
            extra_messages=list(self._extra_msgs),
        )
        handle = _SimulatedWebSearcherHandle(
            self._llm,
            follow_up_prompt,
            mode=self._mode,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )
        # Align with other simulated components: concise "Question(<parent_label>)" label
        try:
            handle._log_label = SimulatedLineage.question_label(self._log_label)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Emit a human-facing log for the nested ask
        try:
            SimulatedLog.log_request("ask", getattr(handle, "_log_label", ""), question)  # type: ignore[arg-type]
        except Exception:
            pass
        return handle

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        """Block until a clarification arrives, or forever if not requested."""
        if not getattr(self, "_needs_clar", False):
            return await super().next_clarification()
        try:
            if self._clar_up_q is not None:
                msg = await self._clar_up_q.get()
                return {
                    "type": "clarification",
                    "call_id": "unknown",
                    "tool_name": "unknown",
                    "question": msg,
                }
        except Exception:
            pass
        return await super().next_clarification()

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
        hold_completion: bool = False,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._description = description
        self._log_events = log_events
        self._hold_completion = hold_completion
        # Mirror the real manager's tool exposure programmatically for prompts
        self._ask_tools = mirror_web_searcher_tools()

        # Stateful async LLM
        self._llm = new_llm_client(stateful=True, origin="SimulatedWebSearcher")

        # Reference the real prompt as context (no real tools here)
        ask_msg = build_ask_prompt(tools=self._ask_tools).flatten()
        self._llm.set_system_message(
            "You are a simulated web-search assistant. There is no real web client or API – "
            "invent plausible sources and keep your narrative consistent.\n\n"
            "For reference, here is the real system message outline used by the production WebSearcher.ask:"
            f"\n\n{ask_msg}\n\nBack-story: {self._description}",
        )

    @functools.wraps(BaseWebSearcher.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedWebSearcher.ask",
            "ask",
            {
                "text": text if isinstance(text, str) else repr(text),
                "requests_clarification": _requests_clarification,
            },
        )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=_parent_chat_context,
        )

        # When a response_format is requested, use a per-call client configured
        # with the same system message to return structured output.
        llm_for_handle = self._llm
        if response_format is not None:
            schema_llm = new_llm_client(origin="SimulatedWebSearcher")
            # Mirror the stateful system message for continuity
            try:
                schema_llm.set_system_message(getattr(self._llm, "system_message"))
            except Exception:
                # Fallback: rebuild a fresh prompt equivalent
                ask_msg = build_ask_prompt(tools=self._ask_tools).flatten()
                schema_llm.set_system_message(
                    "You are a simulated web-search assistant. There is no real web client or API – "
                    "invent plausible sources and keep your narrative consistent.\n\n"
                    "For reference, here is the real system message outline used by the production WebSearcher.ask:"
                    f"\n\n{ask_msg}\n\nBack-story: {self._description}",
                )
            try:
                schema_llm.set_response_format(response_format)
            except Exception:
                pass
            llm_for_handle = schema_llm

        handle = _SimulatedWebSearcherHandle(
            llm_for_handle,
            instruction,
            mode="ask",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )

        return handle

    @functools.wraps(BaseWebSearcher.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedWebSearcher.clear",
            "clear",
            {},
        )
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "simulate sensible web research answers",
            ),
            log_events=getattr(self, "_log_events", False),
            hold_completion=getattr(self, "_hold_completion", False),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)
