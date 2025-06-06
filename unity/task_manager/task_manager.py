# task_manager/task_manager.py
from __future__ import annotations

from typing import Callable, Dict, Optional

import asyncio
import json
import os

import unify

from datetime import datetime, timezone

from ..common.llm_helpers import (
    methods_to_tool_dict,
    start_async_tool_use_loop,
    ToolSpec,
)
from ..contact_manager.base import BaseContactManager
from ..transcript_manager.base import BaseTranscriptManager
from ..knowledge_manager.base import BaseKnowledgeManager
from ..planner.base import BasePlanner
from ..task_scheduler.base import BaseTaskScheduler


class TaskManager:
    """
    Top-level façade that *can* own a maximum of *one* live plan at a time and exposes two
    different tool surfaces which include the knowledge, task list, contacts, and transcript histories:

    • `ask()`     → read-only (passive) tools + passive plan methods
    • `request()` → read-only + *all* active tools + all plan methods
    """

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        simulated: bool = False,
        contact_manager: Optional[BaseContactManager] = None,
        transcript_manager: Optional[BaseTranscriptManager] = None,
        knowledge_manager: Optional[BaseKnowledgeManager] = None,
        task_scheduler: Optional[BaseTaskScheduler] = None,
        planner: Optional[BasePlanner] = None,
    ) -> None:
        """
        Args:
            simulated: When *True* all subordinate managers are replaced by
                      their **simulated** counterparts which keep all state
                      inside an LLM conversation rather than touching real
                      storage back-ends. Defaults to *False* (real managers).
            contact_manager: Optional custom contact manager implementation.
                           If None, will create default based on simulated flag.
            transcript_manager: Optional custom transcript manager implementation.
                              If None, will create default based on simulated flag.
            knowledge_manager: Optional custom knowledge manager implementation.
                             If None, will create default based on simulated flag.
            task_scheduler: Optional custom task scheduler implementation.
                          If None, will create default based on simulated flag.
            planner: Optional custom planner implementation.
                    If None, will create default based on simulated flag.
        """

        if simulated:
            from ..contact_manager.simulated import SimulatedContactManager
            from ..transcript_manager.simulated import SimulatedTranscriptManager
            from ..knowledge_manager.simulated import SimulatedKnowledgeManager
            from ..planner.simulated import SimulatedPlanner
            from ..task_scheduler.simulated import SimulatedTaskScheduler

            # ── Simulated façade (pure-LLM back-ends) ────────────────────
            if contact_manager is not None:
                self._contact_manager = contact_manager
            else:
                self._contact_manager = SimulatedContactManager()

            if transcript_manager is not None:
                self._transcript_manager = transcript_manager
            else:
                self._transcript_manager = SimulatedTranscriptManager()

            if knowledge_manager is not None:
                self._knowledge_manager = knowledge_manager
            else:
                self._knowledge_manager = SimulatedKnowledgeManager()

            if planner is not None:
                self._planner = planner
            else:
                self._planner = SimulatedPlanner()

            if task_scheduler is not None:
                self._task_scheduler = task_scheduler
            else:
                self._task_scheduler = SimulatedTaskScheduler()

        else:
            from ..knowledge_manager.knowledge_manager import KnowledgeManager
            from ..planner.tool_loop_planner import ToolLoopPlanner
            from ..task_scheduler.task_scheduler import TaskScheduler
            from ..transcript_manager.transcript_manager import TranscriptManager
            from ..contact_manager.contact_manager import ContactManager

            # ── Real managers touching Unify back-ends ───────────────────
            if contact_manager is not None:
                self._contact_manager = contact_manager
            else:
                self._contact_manager = ContactManager(self._event_bus)

            if transcript_manager is not None:
                self._transcript_manager = transcript_manager
            else:
                self._transcript_manager = TranscriptManager(
                    self._event_bus,
                    contact_manager=self._contact_manager,
                )

            if knowledge_manager is not None:
                self._knowledge_manager = knowledge_manager
            else:
                self._knowledge_manager = KnowledgeManager()

            if planner is not None:
                self._planner = planner
            else:
                self._planner = ToolLoopPlanner()

            if task_scheduler is not None:
                self._task_scheduler = task_scheduler
            else:
                self._task_scheduler = TaskScheduler(planner=self._planner)

        #  Run-time state & tool-dict helpers
        self._current_plan = None  # type: ignore

        # These two dicts are rebuilt lazily before every ask/request
        self._passive_tools: Dict[str, Callable] = {}
        self._active_tools: Dict[str, Callable] = {}

    # ------------------------------------------------------------------ #
    #  Internal: build tool dictionaries                                 #
    # ------------------------------------------------------------------ #

    def _refresh_tool_dicts(self) -> None:
        """Re-compute passive / active tool maps based on current plan."""

        # -------- base passive helpers -------------------------------- #
        passive = methods_to_tool_dict(
            self._contact_manager.ask,
            self._transcript_manager.ask,
            self._knowledge_manager.retrieve,
            self._task_scheduler.ask,
            include_class_name=True,
        )

        # -------- add plan.ask when a plan is alive ------------------- #
        if self._current_plan is not None and not self._current_plan.done():

            # We expose _ask_plan_call_ (Unify expects this naming)
            async def _plan_ask_proxy(question: str):
                return await self._current_plan.ask(question)  # type: ignore[attr-defined]

            _plan_ask_proxy.__name__ = "_ask_plan_call_"
            passive[_plan_ask_proxy.__name__] = _plan_ask_proxy

        self._passive_tools = passive

        # -------- build active helpers (passive + writers) ------------ #

        # Wrapper to intercept start_task and remember the returned handle
        def _wrapped_start_task(
            task_id: int,
            *,
            parent_chat_context=None,
            clarification_up_q=None,
            clarification_down_q=None,
        ):
            handle = self._task_scheduler.start_task(
                task_id,
                parent_chat_context=parent_chat_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
            )
            # remember the plan so that subsequent questions can use it
            self._current_plan = handle
            return handle

        _wrapped_start_task.__name__ = "_start_task_call_"

        active = {
            **passive,  # read-only tools are also valid here
            **methods_to_tool_dict(
                self._transcript_manager.summarize,
                self._knowledge_manager.store,
                self._task_scheduler.update,
                ToolSpec(_wrapped_start_task, max_concurrent=1),
                include_class_name=True,
            ),
        }

        self._active_tools = active

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ):
        """
        Read-only question: exposes *passive* helpers (+ plan.ask when available).
        """
        self._refresh_tool_dicts()

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(
            "You are the **TaskManager.ask** interface. "
            "You have *read-only* access to tasks, knowledge, contacts & transcripts.\n"
            f"(UTC now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})",
        )

        tools = dict(self._passive_tools)

        # optional clarification helper
        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Asks the user for clarification. Use this if the user's request is ambiguous."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
            log_steps=log_tool_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result

        return handle

    # ------------------------------------------------------------------ #
    #  request  (write-capable)                                          #
    # ------------------------------------------------------------------ #

    def request(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ):
        """
        Full-access entry-point – exposes every passive tool **plus** all
        write-capable helpers and `start_task` (which unlocks plan steering).
        """
        self._refresh_tool_dicts()

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(
            "You are the **TaskManager.request** interface. "
            "You have full read-write access to tasks, knowledge, contacts & transcripts.\n"
            f"(UTC now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})",
        )

        tools = dict(self._active_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Asks the user for clarification. Use this if the user's request is ambiguous."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
            log_steps=log_tool_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result

        return handle

    # ------------------------------------------------------------------ #
    #  Back-compat alias (update → request)                              #
    # ------------------------------------------------------------------ #

    update = request  # many callers still use `.update`
