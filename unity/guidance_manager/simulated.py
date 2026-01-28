from __future__ import annotations

import asyncio
import functools
import threading
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel

import unillm
from .base import BaseGuidanceManager
from .types.guidance import Guidance
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.async_tool_loop import SteerableToolHandle
from ..common.simulated import (
    mirror_guidance_manager_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    build_followup_prompt,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)
from ..contact_manager.prompt_builders import build_simulated_method_prompt
from ..common.llm_client import new_llm_client
from ..constants import LOGGER

# ─────────────────────────────────────────────────────────────────────────────
# Internal handle (mirrors contact/knowledge simulated handles)
# ─────────────────────────────────────────────────────────────────────────────


class _SimulatedGuidanceHandle(SteerableToolHandle, SimulatedHandleMixin):
    """
    Minimal LLM-backed handle used by SimulatedGuidanceManager.ask / update.
    """

    def __init__(
        self,
        llm: unillm.AsyncUnify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        mode: str,
        response_format: Optional[Type[BaseModel]] = None,
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
        # "<outer...>->SimulatedGuidanceManager.<mode>(abcd)"
        self._log_label = SimulatedLineage.make_label(
            f"SimulatedGuidanceManager.{self._mode}",
        )

        if self._needs_clar:
            try:
                q_text = "Could you clarify your request about guidance?"
                self._clar_up_q.put_nowait(q_text)
                try:
                    SimulatedLog.log_clarification_request(self._log_label, q_text)
                except Exception:
                    pass
                try:
                    LOGGER.info(f"❓ [{self._log_label}] Clarification requested")
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

    # --------------------------------------------------------------------- #
    # SteerableToolHandle implementation
    # --------------------------------------------------------------------- #
    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            if self._needs_clar:
                try:
                    LOGGER.info(
                        f"⏳ [{self._log_label}] Waiting for clarification answer…",
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
                    LOGGER.info(f"💬 [{self._log_label}] Clarification answer received")
                except Exception:
                    pass

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)
            # Unified simulated roundtrip including optional dumps and gated body preview
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
            self._done.set()

        # If cancellation happened after the coroutine started, return a stable post-cancel value.
        if self._cancelled:
            return "processed stopped early, no result"
        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> str:
        """Interject a message into the in-flight handle.

        Args:
            message: The interjection message to inject.
            parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
        """
        if self._cancelled:
            return "Interaction stopped."
        self._log_interject(message)
        self._extra_msgs.append(message)
        return "Acknowledged."

    def stop(
        self,
        reason: str | None = None,
    ) -> str:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
        """
        self._log_stop(reason)
        self._cancelled = True
        try:
            self._cancel_event.set()
        except Exception:
            pass
        self._done.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

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
        return self._done.is_set()

    async def ask(
        self,
        question: str,
        *,
        parent_chat_context: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
        """
        follow_up_prompt = build_followup_prompt(
            question=question,
            initial_instruction=self._initial,
            extra_messages=list(self._extra_msgs),
        )
        handle = _SimulatedGuidanceHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
            mode=self._mode,
        )
        # Align with other simulated components: concise "Question(<parent>)" label
        try:
            handle._log_label = SimulatedLineage.question_label(self._log_label)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            SimulatedLog.log_request("ask", getattr(handle, "_log_label", ""), question)  # type: ignore[arg-type]
        except Exception:
            pass
        return handle

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        """Retrieve the next clarification request, if any.

        Only surfaces clarification events when this handle explicitly requested
        clarification. This prevents cross-handle consumption of shared clarification
        queues that may be injected by external processes.
        """
        if not getattr(self, "_needs_clar", False):
            return {}
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
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            if self._clar_down_q is not None:
                await self._clar_down_q.put(answer)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public simulated manager
# ─────────────────────────────────────────────────────────────────────────────


class SimulatedGuidanceManager(BaseGuidanceManager):
    """
    Drop-in replacement for GuidanceManager with imaginary data and
    stateful LLM memory.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # Shared, stateful LLM (memory across turns)
        self._llm = new_llm_client(stateful=True)

        # Mirror the real manager's tool exposure programmatically and build
        # the same prompts via shared builders.
        ask_tools = mirror_guidance_manager_tools("ask")
        upd_tools = mirror_guidance_manager_tools("update")

        columns = [{k: str(v.annotation)} for k, v in Guidance.model_fields.items()]

        ask_msg = build_ask_prompt(
            ask_tools,
            10,
            columns,
            include_activity=self._rolling_summary_in_prompts,
        )
        upd_msg = build_update_prompt(
            upd_tools,
            10,
            columns,
            include_activity=self._rolling_summary_in_prompts,
        )

        sys_parts = [
            "You are a *simulated* guidance-manager assistant. ",
            "There is no real database; invent plausible guidance entries and keep your story consistent across turns.\n\n",
            "As a reference, the system messages for the *real* guidance-manager 'ask' and 'update' methods are as follows.",
            "You do not have access to any real tools, so you should just create a final answer to the question/request. ",
            f"\n\n'ask' system message:\n{ask_msg}\n\n",
            f"\n\n'update' system message:\n{upd_msg}\n\n",
            f"Back-story: {self._description}",
        ]
        if (
            isinstance(self._simulation_guidance, str)
            and self._simulation_guidance.strip()
        ):
            sys_parts.append(
                f"\n\nAdditional simulation guidance: {self._simulation_guidance}",
            )
        self._llm.set_system_message("".join(sys_parts))

    # ------------------------------------------------------------------ #
    # ask                                                                #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseGuidanceManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedGuidanceManager.ask",
            "ask",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=_parent_chat_context,
        )
        handle = _SimulatedGuidanceHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            mode="ask",
            response_format=response_format,
        )

        return handle

    # ------------------------------------------------------------------ #
    # update                                                             #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseGuidanceManager.update, updated=())
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedGuidanceManager.update",
            "update",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=_parent_chat_context,
        )
        handle = _SimulatedGuidanceHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            mode="update",
            response_format=response_format,
        )

        return handle

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedGuidanceManager.clear",
            "clear",
            {},
        )
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "nothing fixed, make up some imaginary scenario",
            ),
            log_events=getattr(self, "_log_events", False),
            rolling_summary_in_prompts=getattr(
                self,
                "_rolling_summary_in_prompts",
                True,
            ),
            simulation_guidance=getattr(self, "_simulation_guidance", None),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)


if TYPE_CHECKING:
    from ..common.tool_outcome import ToolOutcome  # noqa: F401
