from __future__ import annotations

import asyncio
import functools
import threading
from typing import List, Dict, Any, Optional, Type

import unillm
from pydantic import BaseModel
from .base import BaseKnowledgeManager
from ..common.async_tool_loop import SteerableToolHandle
from .prompt_builders import (
    build_refactor_prompt,
    build_update_prompt,
    build_ask_prompt,
    build_simulated_method_prompt,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import (
    mirror_knowledge_manager_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    build_followup_prompt,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)
from ..constants import LOGGER
from ..common.llm_client import new_llm_client


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedKnowledgeHandle(SteerableToolHandle, SimulatedHandleMixin):
    """
    Handle returned by SimulatedKnowledgeManager.store / retrieve.
    """

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
    ):
        self._llm = llm
        self._initial = initial_text
        self._mode = str(mode or "ask")
        self._want_steps = _return_reasoning_steps
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

        # Human-friendly log label derived from current lineage:
        # "<outer...>->SimulatedKnowledgeManager.<mode>(abcd)"
        self._log_label = SimulatedLineage.make_label(
            f"SimulatedKnowledgeManager.{self._mode}",
        )

        # fire clarification request immediately if queues supplied
        if self._needs_clar:
            try:
                q_text = "Could you clarify your knowledge request?"
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

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False
        # Async cancellation signal to break clarification waits
        self._cancel_event: asyncio.Event = asyncio.Event()

    # --------------------------------------------------------------------- #
    # SteerableToolHandle API
    # --------------------------------------------------------------------- #
    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        # honour pauses injected by an outer loop
        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done_event.is_set():
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
                    self._done_event.set()
                    return "processed stopped early, no result"
                try:
                    clar = get_task.result()
                except Exception:
                    clar = None
                if clar is None:
                    self._done_event.set()
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
            # Unified LLM roundtrip for consistent simulated logging and optional dumps
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
            self._done_event.set()

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
        self._done_event.set()
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

    def done(self) -> bool:
        return self._done_event.is_set()

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

        handle = _SimulatedKnowledgeHandle(
            self._llm,
            follow_up_prompt,
            mode=self._mode,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )
        # Align with other simulated components: concise "Question(<parent>)" label
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
# Public simulated KnowledgeManager
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedKnowledgeManager(BaseKnowledgeManager):
    """
    A drop-in, side-effect-free replacement for KnowledgeManager that uses a
    single stateful LLM to invent and recall knowledge in-chat.
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

        # One shared, memory-retaining LLM (reuse common client for fast init/clear)
        self._llm = new_llm_client(stateful=True)
        # Mirror the real knowledge manager's tool exposure for prompts
        ref_tools = mirror_knowledge_manager_tools("refactor")
        upd_tools = mirror_knowledge_manager_tools("update")
        ask_tools = mirror_knowledge_manager_tools("ask")

        refactor_ref = build_refactor_prompt(
            ref_tools,
            table_schemas_json="{}",
            include_activity=self._rolling_summary_in_prompts,
        )
        store_ref = build_update_prompt(
            upd_tools,
            table_schemas_json="{}",
            include_activity=self._rolling_summary_in_prompts,
        )
        retrieve_ref = build_ask_prompt(
            ask_tools,
            table_schemas_json="{}",
            include_activity=self._rolling_summary_in_prompts,
        )

        self._llm.set_system_message(
            "You are a *simulated* knowledge-base manager. "
            "There is no real database; invent plausible tables, columns and rows "
            "and keep your story consistent across turns.\n\n"
            "As a reference, the (tool-enabled) system messages for the *real* "
            "knowledge-manager are pasted below. **You do not actually have access "
            "to any tools – just produce the final answer.**\n\n"
            f"'refactor' system message:\n{refactor_ref}\n\n"
            f"'store' system message:\n{store_ref}\n\n"
            f"'retrieve' system message:\n{retrieve_ref}\n\n"
            f"Back-story: {self._description}",
        )

    def reduce(
        self,
        *,
        table: str,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Simulated counterpart of the KnowledgeManager.reduce tool.

        There is no real backing store; this method returns deterministic,
        shape-correct placeholder values so that tests and demos can rely on
        the same return structure as the concrete implementation.
        """

        def _scalar(k: str) -> float:
            # Use both table and key name so different tables get different values
            return float(len(str(table)) + len(str(k)) or 1)

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

    # ------------------------------------------------------------------ #
    #  refactor                                                          #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.refactor, updated=())
    async def refactor(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        """
        Simulated version of KnowledgeManager.refactor – no real DDL is run.
        The LLM simply invents a plausible migration plan and returns it.
        """
        should_log = self._log_events or log_events
        call_id = None

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedKnowledgeManager.refactor",
            "refactor",
            {"text": text if isinstance(text, str) else repr(text)},
        )

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "KnowledgeManager",
                "refactor",
                phase="incoming",
                command=text,
            )

        instruction = build_simulated_method_prompt(
            "refactor",
            text,
            parent_chat_context=_parent_chat_context,
        )
        handle = _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            mode="refactor",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=False,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "KnowledgeManager",
                "refactor",
            )

        return handle

    # ------------------------------------------------------------------ #
    #  store                                                             #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.update, updated=())
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
            "SimulatedKnowledgeManager.update",
            "update",
            {
                "text": text if isinstance(text, str) else repr(text),
                "requests_clarification": _requests_clarification,
            },
        )

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "KnowledgeManager",
                "update",
                phase="incoming",
                request=text,
            )

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=_parent_chat_context,
        )
        # Append additional guidance about tasks, which is domain-specific
        instruction += (
            "\n\nIf the user refers to creating *tasks*, then you should **not** store any tasks. "
            "Tasks should be stored by a separate task manager – explain this in your response if relevant."
        )
        handle = _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            mode="update",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "KnowledgeManager",
                "update",
            )

        return handle

    # ------------------------------------------------------------------ #
    #  retrieve                                                          #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.ask, updated=())
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
            "SimulatedKnowledgeManager.ask",
            "ask",
            {
                "text": text if isinstance(text, str) else repr(text),
                "requests_clarification": _requests_clarification,
            },
        )

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "KnowledgeManager",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "retrieve",
            text,
            parent_chat_context=_parent_chat_context,
        )
        handle = _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            mode="ask",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "KnowledgeManager",
                "ask",
            )

        return handle

    @functools.wraps(BaseKnowledgeManager.clear, updated=())
    def clear(self) -> None:
        # Tool-style scheduled log (only when no parent lineage)
        sched = maybe_tool_log_scheduled(
            "SimulatedKnowledgeManager.clear",
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
