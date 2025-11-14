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
from ..common._async_tool.loop_config import TOOL_LOOP_LINEAGE
from ..constants import LOGGER, LLM_IO_DEBUG, SESSION_ID
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
from secrets import token_hex

# Per-run file sink for simulated LLM I/O logs (request/response)
_SIM_TS_LLM_IO_DIR: str | None = None


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
        # Human-friendly log label derived from current lineage, mirroring async loop style:
        # "<outer...>->SimulatedTaskScheduler.<mode>(abcd)"
        try:
            parent_lineage = TOOL_LOOP_LINEAGE.get([])
            parts = list(parent_lineage) if isinstance(parent_lineage, list) else []
        except Exception:
            parts = []
        segment = f"SimulatedTaskScheduler.{self._mode}"
        base = "->".join([*parts, segment]) if parts else segment
        self._log_label = f"{base}({token_hex(2)})"

        # ── fire the clarification request right away ──────────────────
        self._clar_requested = False
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you please clarify exactly what you want?",
                )
                self._clar_requested = True
                try:
                    LOGGER.info(f"❓ [{self._log_label}] Clarification requested")
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
                        f"⏳ [{self._log_label}] Waiting for clarification answer…",
                    )
                except Exception:
                    pass
                clar_reply = await self._clar_down_q.get()
                self._interjections.append(f"Clarification: {clar_reply}")
                try:
                    LOGGER.info(f"💬 [{self._log_label}] Clarification answer received")
                except Exception:
                    pass

            prompt_parts = [self._initial_text] + self._interjections
            user_block = "\n\n---\n\n".join(prompt_parts)

            # LLM step – simulated thinking with timing
            import time as _t
            import os as _os
            import re as _re
            from pathlib import Path as _Path
            import json as _json

            t0 = _t.perf_counter()
            try:
                LOGGER.info(f"🔄 [{self._log_label}] LLM simulating…")
            except Exception:
                pass

            # LLM_IO_DEBUG: write request payload
            _llm_io_debug = bool(LLM_IO_DEBUG)
            _dir: str | None = None
            if _llm_io_debug:
                try:
                    global _SIM_TS_LLM_IO_DIR
                    if _SIM_TS_LLM_IO_DIR is None:
                        root = _Path(_os.getcwd())
                        logs_dir = root / ".llm_io_debug"
                        logs_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            session_safe = _re.sub(r"[^0-9A-Za-z._-]", "-", SESSION_ID)
                        except Exception:
                            session_safe = (
                                SESSION_ID.replace(":", "-")
                                .replace("+", "-")
                                .replace("/", "-")
                            )
                        session_dir = logs_dir / f"{session_safe}"
                        session_dir.mkdir(parents=True, exist_ok=True)
                        _SIM_TS_LLM_IO_DIR = str(session_dir)
                    _dir = _SIM_TS_LLM_IO_DIR
                except Exception:
                    _dir = None

                def _io_write(header: str, body: str) -> None:
                    if not _llm_io_debug or _dir is None:
                        return
                    try:
                        from datetime import datetime as _dt, timezone as _tz
                        import time as _time

                        d = _Path(_dir)
                        now = _dt.now(_tz.utc)
                        hhmmss = now.strftime("%H%M%S")
                        ns = _time.time_ns() % 1_000_000_000
                        base = f"{hhmmss}_{ns:09d}"
                        path = d / f"{base}.txt"
                        if path.exists():
                            i = 1
                            while True:
                                cand = d / f"{base}_{i}.txt"
                                if not cand.exists():
                                    path = cand
                                    break
                                i += 1
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(f"🔄 [{self._log_label}] {header}\n")
                            f.write(body.rstrip())
                            f.write("\n")
                        try:
                            kind = (
                                "request" if "request" in header.lower() else "response"
                            )
                            LOGGER.info(f"📝 LLM {kind} written to {path}")
                        except Exception:
                            pass
                    except Exception:
                        pass

                try:
                    _sys = getattr(self._llm, "system_message", None)
                except Exception:
                    _sys = None
                req_payload = {
                    "model": getattr(self._llm, "model", None),
                    "messages": [{"role": "user", "content": user_block}],
                }
                sys_block = f"System message:\n{_sys}\n\n" if _sys else ""
                _io_write(
                    "LLM request ➡️:",
                    f"{sys_block}{_json.dumps(req_payload, indent=4)}",
                )

            answer = await self._llm.generate(user_block)
            dt_ms = int((_t.perf_counter() - t0) * 1000)
            # Show reply body only when there's no outer async tool loop; otherwise the outer loop
            # will record the tool result and duplication is noisy.
            try:
                try:
                    parent_lineage = TOOL_LOOP_LINEAGE.get([])
                    has_outer = (
                        isinstance(parent_lineage, list) and len(parent_lineage) > 0
                    )
                except Exception:
                    has_outer = False
                if has_outer:
                    LOGGER.info(f"✅ [{self._log_label}] LLM replied in {dt_ms} ms")
                else:
                    _ans_preview = str(answer)
                    if len(_ans_preview) > 800:
                        _ans_preview = _ans_preview[:800] + "…"
                    LOGGER.info(
                        f"✅ [{self._log_label}] LLM replied in {dt_ms} ms:\n{_ans_preview}",
                    )
            except Exception:
                pass

            # LLM_IO_DEBUG: write response payload
            if _llm_io_debug:
                try:
                    _io_write("LLM response ⬅️:", str(answer))
                except Exception:
                    pass

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
        try:
            _preview = message if len(message) <= 120 else f"{message[:120]}…"
            LOGGER.info(f"💬 [{self._log_label}] Interject requested: {_preview}")
        except Exception:
            pass
        self._interjections.append(message)
        return "Noted."

    def stop(self, *, cancel: bool, reason: Optional[str] = None) -> str:
        """Cancel further processing so `.result()` raises.

        The `cancel` flag is required but ignored; the interaction is always
        cancelled.
        """
        try:
            suffix = f" – reason: {reason}" if reason else ""
            LOGGER.info(f"🛑 [{self._log_label}] Stop requested{suffix}")
        except Exception:
            pass
        self._cancelled = True
        self._done_event.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

    def pause(self) -> str:
        if self._paused:
            return "Already paused."
        try:
            LOGGER.info(f"⏸️ [{self._log_label}] Pause requested")
        except Exception:
            pass
        self._paused = True
        return "Paused."

    def resume(self) -> str:
        if not self._paused:
            return "Already running."
        try:
            LOGGER.info(f"▶️ [{self._log_label}] Resume requested")
        except Exception:
            pass
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
            handle._log_label = f"Question({self._log_label})({token_hex(2)})"  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            _preview = question if len(question) <= 120 else f"{question[:120]}…"
            LOGGER.info(f"❓ [{handle._log_label}] Ask requested: {_preview}")  # type: ignore[attr-defined]
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
            "gpt-5@openai",
            reasoning_effort="high",
            service_tier="priority",
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
        _parent_chat_context: list[dict] | None = None,  # Unused – synthetic
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
                "TaskScheduler",
                "ask",
                phase="incoming",
                question=text,
            )

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
        )

        # Emit a human-facing log for the initial ask so tests see immediate feedback
        try:
            _preview = text if len(text) <= 120 else f"{text[:120]}…"
            LOGGER.info(f"❓ [{handle._log_label}] Ask requested: {_preview}")  # type: ignore[attr-defined]
        except Exception:
            pass

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
                "TaskScheduler",
                "update",
                phase="incoming",
                request=text,
            )

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
        )

        # Emit a human-facing log for the initial update so tests see immediate feedback
        try:
            _preview = text if len(text) <= 120 else f"{text[:120]}…"
            LOGGER.info(f"📝 [{handle._log_label}] Update requested: {_preview}")  # type: ignore[attr-defined]
        except Exception:
            pass

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
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
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

        # Emit a scheduler-level execute log with a stable label
        try:
            # Build nested label from current lineage
            parent_lineage = TOOL_LOOP_LINEAGE.get([])
            parts = list(parent_lineage) if isinstance(parent_lineage, list) else []
            segment = "SimulatedTaskScheduler.execute"
            base = "->".join([*parts, segment]) if parts else segment
            _exec_label = f"{base}({token_hex(2)})"
            _preview = text if len(text) <= 120 else f"{text[:120]}…"
            LOGGER.info(f"🎬 [{_exec_label}] Execute requested: {_preview}")
        except Exception:
            pass

        if self._actor_factory is not None:
            actor = self._actor_factory(**actor_kwargs)
        else:
            from ..actor.simulated import SimulatedActor

            actor = SimulatedActor(**actor_kwargs)
        handle = await actor.act(
            task_description,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TaskScheduler",
                "execute",
            )

        return handle
