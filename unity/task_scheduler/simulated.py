# unity/task_scheduler/simulated_task_scheduler.py
"""
Simulated task scheduler.

Provides a storage-free interface that returns steerable handles for ask, update,
and execute. All responses are produced by a shared, stateful LLM; no storage
or queue state is read or written.
"""

import asyncio
import threading
import functools
from typing import List, Optional, Callable, Type, Any

import unillm
from pydantic import BaseModel

from ..common.async_tool_loop import SteerableToolHandle
from ..common._async_tool.messages import forward_handle_call
from ..logger import LOGGER
from ..common.hierarchical_logger import ICONS
from .base import BaseTaskScheduler
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_simulated_method_prompt,
)
from ..common.llm_client import new_llm_client
from ..common.simulated import (
    mirror_task_scheduler_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    build_followup_prompt,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
    maybe_tool_log_scheduled_with_label,
)


class _SimulatedTaskScheduleHandle(SimulatedHandleMixin, SteerableToolHandle):
    """A minimal, LLM-backed handle for ask/update interactions."""

    def __init__(
        self,
        llm: unillm.Unify,
        initial_text: str,
        *,
        mode: str,
        _return_reasoning_steps: bool = False,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        response_format: Optional[Type[BaseModel]] = None,
        hold_completion: bool = False,
    ) -> None:
        self._llm = llm
        self._initial_text = initial_text
        self._mode = mode  # "ask" | "update"
        self._ret_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._response_format = response_format
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification
        # Human-friendly log label derived from current lineage, mirroring async loop style:
        # "<outer...>->SimulatedTaskScheduler.<mode>(abcd)"
        self._log_label = SimulatedLineage.make_label(
            f"SimulatedTaskScheduler.{self._mode}",
        )

        # ── fire the clarification request right away ──────────────────
        self._clar_requested = False
        if self._needs_clar:
            try:
                q_text = "Could you please clarify exactly what you want?"
                self._clar_up_q.put_nowait(q_text)
                try:
                    SimulatedLog.log_clarification_request(self._log_label, q_text)
                except Exception:
                    pass
                self._clar_requested = True
                try:
                    LOGGER.info(
                        f"{ICONS['clarification']} [{self._log_label}] Clarification requested",
                    )
                except Exception:
                    pass
            except asyncio.QueueFull:
                pass

        self._interjections: List[str] = []

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: Optional[str] = None
        self._messages: List[dict] = []
        self._paused = False
        # Async cancellation signal to break clarification waits
        self._cancel_event: asyncio.Event = asyncio.Event()

        self._init_completion_gate(hold_completion)

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
                try:
                    LOGGER.info(
                        f"{ICONS['pending']} [{self._log_label}] Waiting for clarification answer…",
                    )
                except Exception:
                    pass
                clar_reply: str | None = None
                get_task = asyncio.create_task(self._clar_down_q.get())
                cancel_task = asyncio.create_task(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if cancel_task in done:
                    self._done_event.set()
                    return "processed stopped early, no result"
                try:
                    clar_reply = get_task.result()
                except Exception:
                    clar_reply = None
                if clar_reply is None:
                    self._done_event.set()
                    return "processed stopped early, no result"
                self._interjections.append(f"Clarification: {clar_reply}")
                try:
                    SimulatedLog.log_clarification_answer(self._log_label, clar_reply)
                except Exception:
                    pass
                try:
                    LOGGER.info(
                        f"{ICONS['interjection']} [{self._log_label}] Clarification answer received",
                    )
                except Exception:
                    pass

            prompt_parts = [self._initial_text] + self._interjections
            user_block = "\n\n---\n\n".join(prompt_parts)

            # LLM roundtrip using shared helper (includes timing, gated reply body, and optional dumps)
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            answer = await simulated_llm_roundtrip(
                self._llm,
                label=self._log_label,
                prompt=user_block,
                response_format=self._response_format,
            )

            self._answer = answer
            # very small, synthetic trace of "reasoning"
            self._messages = [
                {"role": "user", "content": user_block},
                {"role": "assistant", "content": answer},
            ]
            self._done_event.set()

        # If cancellation happened after the coroutine started, return a stable post-cancel value.
        if self._cancelled:
            return "processed stopped early, no result"
        if self._ret_steps:
            return self._answer, self._messages
        return self._answer

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Append a follow-up message that will be folded into the prompt.

        Args:
            message: The interjection message to inject.
            _parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
        """
        if self._cancelled:
            return
        self._log_interject(message)
        self._interjections.append(message)

    async def stop(
        self,
        reason: Optional[str] = None,
        *,
        cancel: bool = False,
        **kwargs,
    ) -> None:
        """Cancel further processing so `.result()` raises.

        The `cancel` flag is accepted for compatibility with TaskScheduler-style
        stop(cancel=..., reason=...), but is ignored here; the interaction is always
        cancelled.

        Args:
            reason: Optional reason for stopping.
            cancel: Ignored; interaction is always cancelled.
        """
        self._log_stop(reason)
        self._cancelled = True
        try:
            self._cancel_event.set()
        except Exception:
            pass
        self._done_event.set()

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

    def done(self) -> bool:
        return self._done_event.is_set()

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

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
            _return_reasoning_steps: Whether to return reasoning steps.
        """
        follow_up_prompt = build_followup_prompt(
            question=question,
            initial_instruction=self._initial_text,
            extra_messages=list(self._interjections),
        )

        # Create the new helper handle first so we can log using its stable label
        handle = _SimulatedTaskScheduleHandle(
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

        # Align with real async tool loop: use a concise "Question(<parent_label>)" log label
        # and avoid lineage chaining arrows here.
        try:
            handle._log_label = SimulatedLineage.question_label(self._log_label)  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            SimulatedLog.log_request("ask", getattr(handle, "_log_label", ""), question)  # type: ignore[arg-type]
        except Exception:
            pass

        return handle


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
        hold_completion: bool = False,
        # Optional: customise how the SimulatedActor is constructed per execute()
        actor_factory: Optional[Callable[..., Any]] = None,
        actor_steps: Optional[int] = None,
        actor_duration: Optional[float] = None,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        self._description = description
        self._hold_completion = hold_completion
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance
        # Actor configuration (optional)
        self._actor_factory: Optional[Callable[..., Any]] = actor_factory
        self._actor_steps: Optional[int] = actor_steps
        self._actor_duration: Optional[float] = actor_duration

        super().__init__()

        # One shared, *stateful* LLM for *everything*
        self._llm = new_llm_client(stateful=True, origin="SimulatedTaskScheduler")
        # Build tool lists programmatically so prompts match the exposed surface.
        # Register them via add_tools so that the inherited update()/ask() methods
        # can retrieve them at runtime via self.get_tools("update"/"ask").
        ask_tools = mirror_task_scheduler_tools("ask")
        update_tools = mirror_task_scheduler_tools("update")
        self.add_tools("ask", ask_tools)
        self.add_tools("update", update_tools)

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
        ).flatten()
        update_msg = build_update_prompt(
            update_tools,
            num_tasks=10,
            columns=fake_task_columns,
            include_activity=self._rolling_summary_in_prompts,
        ).flatten()

        self._llm.set_system_message(
            "You are a *simulated* task-list manager. "
            "No real database exists; invent plausible tasks but remain internally "
            "consistent across turns.\n\n"
            "As reference, here are the *real* TaskScheduler prompts:\n\n"
            f"ASK system message:\n{ask_msg}\n\n"
            f"UPDATE system message:\n{update_msg}\n\n"
            f"Back-story: {self._description}",
        )

    def reduce(
        self,
        *,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Simulated counterpart of the TaskScheduler.reduce tool.

        No real Tasks context exists in simulation; this method returns
        deterministic, shape-correct placeholder values:

        * Single key, no grouping  → scalar.
        * Multiple keys, no grouping → ``dict[key -> scalar]``.
        * With grouping             → nested ``dict[group -> value or dict]``.
        """

        def _scalar(k: str) -> float:
            return float(len(str(k)) or 1)

        key_list: list[str] = [keys] if isinstance(keys, str) else list(keys)

        if group_by is None:
            if isinstance(keys, str):
                return _scalar(keys)
            return {k: _scalar(k) for k in key_list}

        groups: list[str] = (
            [group_by] if isinstance(group_by, str) else [str(g) for g in group_by]
        )
        if isinstance(keys, str):
            return {g: _scalar(keys) for g in groups}
        return {g: {k: _scalar(k) for k in key_list} for g in groups}

    @functools.wraps(BaseTaskScheduler.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedTaskScheduler.clear",
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
            hold_completion=getattr(self, "_hold_completion", False),
            actor_factory=getattr(self, "_actor_factory", None),
            actor_steps=getattr(self, "_actor_steps", None),
            actor_duration=getattr(self, "_actor_duration", None),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)

    # ------------------------------------------------------------------ #
    #  ask                                                               #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,  # Ignored – we do not expose tools
        _parent_chat_context: list[dict] | None = None,  # Unused – synthetic
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # No EventBus publishing for simulated managers

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=_parent_chat_context,
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
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
            hold_completion=self._hold_completion,
        )

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedTaskScheduler.ask",
            "ask",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        # No EventBus publishing for simulated managers

        return handle

    # ------------------------------------------------------------------ #
    #  update                                                            #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.update, updated=())
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,  # Ignored – no tools here
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # No EventBus publishing for simulated managers

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=_parent_chat_context,
        )
        instruction += "\n\nIf any tasks were created or updated during the imagined process, include their id(s) in your reply."
        handle = _SimulatedTaskScheduleHandle(
            self._llm,
            instruction,
            mode="update",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
            hold_completion=self._hold_completion,
        )

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedTaskScheduler.update",
            "update",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        # No EventBus publishing for simulated managers

        return handle

    # ------------------------------------------------------------------ #
    #  execute_task – delegate to SimulatedActor.act                     #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.execute, updated=())
    async def execute(
        self,
        task_id: int | str,
        *,
        trigger_attempt_token: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        isolated: Optional[bool] = None,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # No EventBus publishing for simulated managers

        text = f"Run task {task_id}" if isinstance(task_id, int) else str(task_id)
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

        # Tool-style scheduled log for execute (only when no parent lineage)
        try:
            _exec_label = SimulatedLineage.make_label("SimulatedTaskScheduler.execute")
        except Exception:
            _exec_label = "SimulatedTaskScheduler.execute"
        maybe_tool_log_scheduled_with_label(
            _exec_label,
            "execute",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        if self._actor_factory is not None:
            actor = self._actor_factory(**actor_kwargs)
        else:
            from ..actor.simulated import SimulatedActor

            actor = SimulatedActor(**actor_kwargs)
        # Reuse the scheduler's suffix for the actor session to provide a single session id across logs
        try:
            _suffix = SimulatedLineage.extract_suffix(_exec_label)
        except Exception:
            _suffix = None
        handle = await actor.act(
            task_description,
            response_format=response_format,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
            session_suffix=_suffix,
        )

        # No EventBus publishing for simulated managers

        # Wrap the actor handle to expose TaskScheduler-style stop(cancel=..., reason=...) while
        # delegating all behaviour to the underlying actor. Named to mirror ActiveQueue's surface
        # (single-task, direct-delegation style).
        class SimulatedActiveQueue(SteerableToolHandle, SimulatedHandleMixin):  # type: ignore[abstract-method]
            def __init__(self, inner: SteerableToolHandle, log_label: str) -> None:
                self._inner = inner
                # Provide a stable, scheduler-aligned log label for status lines
                self._log_label = log_label

            # --- steerable surface ---
            async def interject(
                self,
                message: str,
                *,
                _parent_chat_context_cont: list[dict] | None = None,
            ) -> None:  # type: ignore[override]
                self._log_interject(message)
                await forward_handle_call(
                    self._inner,
                    "interject",
                    {
                        "message": message,
                        "_parent_chat_context_cont": _parent_chat_context_cont,
                    },
                    fallback_positional_keys=("message",),
                )

            async def stop(
                self,
                *,
                cancel: bool = False,
                reason: Optional[str] = None,
                **kwargs,
            ) -> None:  # type: ignore[override]
                self._log_stop(reason)
                await forward_handle_call(
                    self._inner,
                    "stop",
                    {"reason": reason, "cancel": cancel},
                    fallback_positional_keys=("reason",),
                )

            async def pause(self) -> Optional[str]:  # type: ignore[override]
                self._log_pause()
                try:
                    return await self._inner.pause()
                except Exception:
                    return "Already completed."

            async def resume(self) -> Optional[str]:  # type: ignore[override]
                self._log_resume()
                try:
                    return await self._inner.resume()
                except Exception:
                    return "Already completed."

            def done(self) -> bool:  # type: ignore[override]
                try:
                    return self._inner.done()
                except Exception:
                    return True

            async def result(self) -> str:  # type: ignore[override]
                try:
                    return await self._inner.result()
                except Exception:
                    return "processed stopped early, no result"

            # --- event APIs (best-effort pass-through) ---
            async def next_clarification(self) -> dict:
                try:
                    return await self._inner.next_clarification()  # type: ignore[attr-defined]
                except Exception:
                    return {}

            async def next_notification(self) -> dict:
                try:
                    return await self._inner.next_notification()  # type: ignore[attr-defined]
                except Exception:
                    return {}

            async def answer_clarification(self, call_id: str, answer: str) -> None:
                try:
                    await self._inner.answer_clarification(call_id, answer)  # type: ignore[attr-defined]
                except Exception:
                    return None

            # --- ask semantics: delegate to inner handle (returns SteerableToolHandle) ---
            async def ask(
                self,
                question: str,
                *,
                _parent_chat_context: list[dict] | None = None,
                _return_reasoning_steps: bool = False,
            ) -> "SteerableToolHandle":
                return await self._inner.ask(question)  # type: ignore[attr-defined]

        return SimulatedActiveQueue(handle, _exec_label)
