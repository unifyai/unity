from __future__ import annotations

import asyncio
import functools
import threading
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import unify
from .base import BaseGuidanceManager
from .types.guidance import Guidance
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.async_tool_loop import SteerableToolHandle
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import mirror_guidance_manager_tools
from ..contact_manager.prompt_builders import build_simulated_method_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Internal handle (mirrors contact/knowledge simulated handles)
# ─────────────────────────────────────────────────────────────────────────────


class _SimulatedGuidanceHandle(SteerableToolHandle):
    """
    Minimal LLM-backed handle used by SimulatedGuidanceManager.ask / update.
    """

    def __init__(
        self,
        llm: unify.AsyncUnify,
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
                self._clar_up_q.put_nowait(
                    "Could you clarify your request about guidance?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []

        self._done = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False

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
            "However, there is a also ongoing simulated process which had the instructions given below. "
            "Please make your answer realastic and conceivable given the provided context of the simulated taks."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial]
            + self._extra_msgs
            + [f"Question to answer (as a reminder!): {question}"],
        )
        return _SimulatedGuidanceHandle(
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
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # Shared, stateful LLM (memory across turns)
        from ..common.llm_client import (
            new_llm_client as _new_llm_client,
        )  # local import to avoid cycles

        self._llm = _new_llm_client(stateful=True)

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
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "GuidanceManager",
                "ask",
                phase="incoming",
                question=text,
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
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "GuidanceManager",
                "ask",
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
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "GuidanceManager",
                "update",
                phase="incoming",
                request=text,
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
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "GuidanceManager",
                "update",
            )

        return handle

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
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


if TYPE_CHECKING:
    from ..common.tool_outcome import ToolOutcome  # noqa: F401
