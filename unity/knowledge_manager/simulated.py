from __future__ import annotations

import asyncio
import json
import os
import functools
import threading
from typing import List, Dict, Any, Optional

import unify
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
from ..common.simulated import mirror_knowledge_manager_tools


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedKnowledgeHandle(SteerableToolHandle):
    """
    Handle returned by SimulatedKnowledgeManager.store / retrieve.
    """

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

        # fire clarification question immediately if queues supplied
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you clarify your knowledge request?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False

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
                clar = await self._clar_down_q.get()
                self._extra_msgs.append(f"Clarification: {clar}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)
            answer = await self._llm.generate(prompt)
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done_event.set()

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

        return _SimulatedKnowledgeHandle(
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
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # One shared, memory-retaining LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
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

    # ------------------------------------------------------------------ #
    #  refactor                                                          #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.refactor, updated=())
    async def refactor(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        """
        Simulated version of KnowledgeManager.refactor – no real DDL is run.
        The LLM simply invents a plausible migration plan and returns it.
        """
        should_log = self._log_events or log_events
        call_id = None

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
            parent_chat_context=parent_chat_context,
        )
        handle = _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=False,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "KnowledgeManager",
                "refactor",
            )

        return handle

    # Append guidance for outer orchestrators (mutation idempotence)
    refactor.__doc__ = (refactor.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this mutation with the same arguments multiple times in the same "
        "conversation. Treat this operation as idempotent; if confirmation is needed, perform a single read to verify the outcome."
    )

    # ------------------------------------------------------------------ #
    #  store                                                             #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.update, updated=())
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
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
                "KnowledgeManager",
                "update",
                phase="incoming",
                request=text,
            )

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=parent_chat_context,
        )
        # Append additional guidance about tasks, which is domain-specific
        instruction += (
            "\n\nIf the user refers to creating *tasks*, then you should **not** store any tasks. "
            "Tasks should be stored by a separate task manager – explain this in your response if relevant."
        )
        handle = _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "KnowledgeManager",
                "update",
            )

        return handle

    # Append guidance for outer orchestrators (mutation idempotence)
    update.__doc__ = (update.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this mutation with the same arguments multiple times in the same "
        "conversation. Treat this operation as idempotent; if confirmation is needed, perform a single read to verify the outcome."
    )

    # ------------------------------------------------------------------ #
    #  retrieve                                                          #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
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
                "KnowledgeManager",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "retrieve",
            text,
            parent_chat_context=parent_chat_context,
        )
        handle = _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "KnowledgeManager",
                "ask",
            )

        return handle

    # Provide guidance for outer loops via tool description
    ask.__doc__ = (ask.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this tool repeatedly with the same "
        "arguments within the same conversation. Prefer reusing prior results and "
        "compose the final answer once sufficient information has been gathered."
    )
