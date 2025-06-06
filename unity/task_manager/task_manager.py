# task_manager/task_manager.py
from __future__ import annotations

from typing import Callable, Dict, Optional, List, Any

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
from ..events.event_bus import EventBus
from ..contact_manager.base import BaseContactManager
from ..contact_manager.contact_manager import ContactManager
from ..transcript_manager.base import BaseTranscriptManager
from ..transcript_manager.transcript_manager import TranscriptManager
from ..knowledge_manager.base import BaseKnowledgeManager
from ..knowledge_manager.knowledge_manager import KnowledgeManager
from ..planner.base import BasePlanner
from ..planner.tool_loop_planner import ToolLoopPlanner
from ..task_scheduler.base import BaseTaskScheduler
from ..task_scheduler.task_scheduler import TaskScheduler
from .sys_msgs import ASK, REQUEST, START_TASK


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
        event_bus: EventBus,
        *,
        contact_manager: Optional[BaseContactManager] = None,
        transcript_manager: Optional[BaseTranscriptManager] = None,
        knowledge_manager: Optional[BaseKnowledgeManager] = None,
        task_scheduler: Optional[BaseTaskScheduler] = None,
        planner: Optional[BasePlanner] = None,
        traced: bool = True,
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
        # ── Real managers touching Unify back-ends ───────────────────
        # Keep reference to the bus – needed by sub-managers
        self._event_bus = event_bus

        # ── contact manager ────────────────────────────────────────────
        if contact_manager is not None:
            self._contact_manager = contact_manager
        else:
            self._contact_manager = ContactManager(
                event_bus=event_bus,
                traced=traced,
            )

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

        # ── tracing helper – mirrors other managers ────────────────────
        if traced:
            self = unify.traced(self)

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
            ASK.replace(
                "<datetime>",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
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
            REQUEST.replace(
                "<datetime>",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
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
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result

        return handle

    # ------------------------------------------------------------------ #
    #  start_task – new public surface (write-capable but focussed)      #
    # ------------------------------------------------------------------ #
    def start_task(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
        parent_chat_context: Optional[List[dict]] = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ):
        """
        Launch (activate) a task based on a *natural-language* instruction.
        Mirrors the pattern used by `ask` / `request` but exposes a deliberately
        tiny tool-surface: search + nearest + **_start_task_call_**.
        """
        self._refresh_tool_dicts()  # keeps _start_task_call_ in sync

        # ---------------------------------------------------------------- #
        #  1. Build a dedicated tool-dict for this surface                  #
        # ---------------------------------------------------------------- #
        # Re-wrap so that we capture the returned ActiveTask handle and
        # remember it for future plan queries.
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
            self._current_plan = handle
            return handle

        _wrapped_start_task.__name__ = "_start_task_call_"

        tools: Dict[str, Callable[..., Any]] = {
            **methods_to_tool_dict(
                self._task_scheduler._search_tasks,
                self._task_scheduler._nearest_tasks,
                include_class_name=False,
            ),
            _wrapped_start_task.__name__: _wrapped_start_task,
        }

        # optional clarification helper
        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ---------------------------------------------------------------- #
        #  2. Fire up the interactive loop                                 #
        # ---------------------------------------------------------------- #
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(
            START_TASK.replace(
                "<datetime>",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
            log_steps=log_tool_steps,
        )

        if _return_reasoning_steps:
            orig_res = handle.result

            async def _wrapped():
                answer = await orig_res()
                return answer, client.messages

            handle.result = _wrapped

        return handle
