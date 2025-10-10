# unity/task_scheduler/simulated_task_scheduler.py
"""
Simulated task scheduler.

Provides a storage-free interface that returns steerable handles for ask, update,
and execute. All responses are produced by a shared, stateful LLM; no storage
or queue state is read or written.
"""
import asyncio
import json
import os
import threading
import functools
from typing import List, Optional, Callable, Any

import unify

from ..common.async_tool_loop import SteerableToolHandle
from .base import BaseTaskScheduler
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_simulated_method_prompt,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import mirror_task_scheduler_tools


class _SimulatedTaskScheduleHandle(SteerableToolHandle):
    """A minimal, LLM-backed handle for ask/update interactions."""

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        mode: str,
        _return_reasoning_steps: bool = False,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> None:
        self._llm = llm
        self._initial_text = initial_text
        self._mode = mode  # "ask" | "update"
        self._ret_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

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
        self._paused = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API required by SteerableToolHandle
    # ──────────────────────────────────────────────────────────────────────
    async def result(self):
        """Return the LLM answer (or raise if stopped)."""
        if self._cancelled:
            return "processed stopped early, no result"

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done_event.is_set():
            # Wait for clarification answer if required
            if self._needs_clar:
                clar_reply = await self._clar_down_q.get()
                self._interjections.append(f"Clarification: {clar_reply}")

            prompt_parts = [self._initial_text] + self._interjections
            user_block = "\n\n---\n\n".join(prompt_parts)

            answer = await self._llm.generate(user_block)

            self._answer = answer
            # very small, synthetic trace of "reasoning"
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

    def stop(self, *, cancel: bool, reason: Optional[str] = None) -> str:
        """Cancel further processing so `.result()` raises.

        The `cancel` flag is required but ignored; the interaction is always
        cancelled.
        """
        self._cancelled = True
        self._done_event.set()
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

    def done(self) -> bool:
        return self._done_event.is_set()

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

    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":
        q_msg = (
            f"Your only task is to simulate an answer to the following question: {question}\n\n"
            "However, there is a also ongoing simulated process which had the instructions given below. "
            "Please make your answer realastic and conceivable given the provided context of the simulated taks."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial_text]
            + self._interjections
            + [f"Question to answer (as a reminder!): {question}"],
        )

        return _SimulatedTaskScheduleHandle(
            self._llm,
            follow_up_prompt,
            mode=self._mode,
            _return_reasoning_steps=(
                _return_reasoning_steps if _return_reasoning_steps else self._ret_steps
            ),
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )


class SimulatedTaskScheduler(BaseTaskScheduler):
    """
    Simulated scheduler for demos and tests.

    Uses a shared stateful LLM to produce plausible task lists and to run
    ask/update/execute interactions without touching storage.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        # Optional: customise how the SimulatedActor is constructed per execute()
        actor_factory: Optional[Callable[..., Any]] = None,
        actor_steps: Optional[int] = None,
        actor_duration: Optional[float] = None,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance
        # Actor configuration (optional)
        self._actor_factory: Optional[Callable[..., Any]] = actor_factory
        self._actor_steps: Optional[int] = actor_steps
        self._actor_duration: Optional[float] = actor_duration

        # One shared, *stateful* LLM for *everything*
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        # Build tool lists programmatically so prompts match the exposed surface.
        ask_tools = mirror_task_scheduler_tools("ask")
        update_tools = mirror_task_scheduler_tools("update")

        # Provide placeholder counts/columns for the simulated environment
        from .types.task import Task as _Task

        fake_task_columns = [
            {k: str(v.annotation)} for k, v in _Task.model_fields.items()
        ]

        ask_msg = build_ask_prompt(
            ask_tools,
            num_tasks=10,
            columns=fake_task_columns,
            include_activity=self._rolling_summary_in_prompts,
        )
        update_msg = build_update_prompt(
            update_tools,
            num_tasks=10,
            columns=fake_task_columns,
            include_activity=self._rolling_summary_in_prompts,
        )

        self._llm.set_system_message(
            "You are a *simulated* task-list manager. "
            "No real database exists; invent plausible tasks but remain internally "
            "consistent across turns.\n\n"
            "As reference, here are the *real* TaskScheduler prompts:\n\n"
            f"ASK system message:\n{ask_msg}\n\n"
            f"UPDATE system message:\n{update_msg}\n\n"
            f"Back-story: {self._description}",
        )

    @functools.wraps(BaseTaskScheduler.clear, updated=())
    def clear(self) -> None:
        try:
            # Reset the LLM's internal state (best-effort)
            self._llm.reset_state()
        except Exception:
            pass
        # Rebuild and set the system message again to mirror initialisation
        from .types.task import Task as _Task  # local import to avoid cycles

        fake_task_columns = [
            {k: str(v.annotation)} for k, v in _Task.model_fields.items()
        ]
        ask_tools = mirror_task_scheduler_tools("ask")
        update_tools = mirror_task_scheduler_tools("update")
        ask_msg = build_ask_prompt(
            ask_tools,
            num_tasks=10,
            columns=fake_task_columns,
            include_activity=self._rolling_summary_in_prompts,
        )
        update_msg = build_update_prompt(
            update_tools,
            num_tasks=10,
            columns=fake_task_columns,
            include_activity=self._rolling_summary_in_prompts,
        )
        self._llm.set_system_message(
            "You are a *simulated* task-list manager. "
            "No real database exists; invent plausible tasks but remain internally "
            "consistent across turns.\n\n"
            "As reference, here are the *real* TaskScheduler prompts:\n\n"
            f"ASK system message:\n{ask_msg}\n\n"
            f"UPDATE system message:\n{update_msg}\n\n"
            f"Back-story: {self._description}",
        )

    # ------------------------------------------------------------------ #
    #  ask                                                               #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,  # Ignored – we do not expose tools
        parent_chat_context: list[dict] | None = None,  # Unused – synthetic
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
                "TaskScheduler",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=parent_chat_context,
        )
        instruction += (
            "\n\nPlease *always* mention the relevant task id(s) in your response. "
            "If the user asks whether a task already exists in the list, reply 'No' and state it does *not* exist."
        )
        handle = _SimulatedTaskScheduleHandle(
            self._llm,
            instruction,
            mode="ask",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TaskScheduler",
                "ask",
            )

        return handle

    # ------------------------------------------------------------------ #
    #  update                                                            #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.update, updated=())
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,  # Ignored – no tools here
        parent_chat_context: list[dict] | None = None,
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
                "TaskScheduler",
                "update",
                phase="incoming",
                request=text,
            )

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=parent_chat_context,
        )
        instruction += "\n\nIf any tasks were created or updated during the imagined process, include their id(s) in your reply."
        handle = _SimulatedTaskScheduleHandle(
            self._llm,
            instruction,
            mode="update",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TaskScheduler",
                "update",
            )

        return handle

    # ------------------------------------------------------------------ #
    #  execute_task – delegate to SimulatedActor.act                     #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.execute, updated=())
    async def execute(
        self,
        text: str,
        *,
        isolated: Optional[bool] = None,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        """
        Execute a *simulated* task from **free-form** text.

        The implementation pretends that the supplied *text* uniquely
        identifies the task – no attempt is made to reconcile with a real data
        store.  A new :class:`unity.actor.simulated.SimulatedPlan` is spun up
        and its handle returned.
        """
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "TaskScheduler",
                "execute",
                phase="incoming",
                request=text,
            )

        task_description = f"{text} (simulated)"

        # Build actor with configured defaults or via a custom factory
        actor_kwargs = {
            # Respect scheduler-level defaults when provided
            "steps": self._actor_steps,
            "duration": self._actor_duration,
            "_requests_clarification": _requests_clarification,
            "simulation_guidance": self._simulation_guidance,
        }
        # Drop None values so defaults are not forced
        actor_kwargs = {k: v for k, v in actor_kwargs.items() if v is not None}

        if self._actor_factory is not None:
            actor = self._actor_factory(**actor_kwargs)
        else:
            from ..actor.simulated import SimulatedActor

            actor = SimulatedActor(**actor_kwargs)
        handle = await actor.act(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TaskScheduler",
                "execute",
            )

        return handle
