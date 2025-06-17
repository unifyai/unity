# task_manager/task_manager.py
from __future__ import annotations

from typing import Callable, Dict

import asyncio
import json
import os

import unify

from typing import Callable, Dict

from ..common.llm_helpers import (
    methods_to_tool_dict,
    start_async_tool_use_loop,
    ToolSpec,
)
from .prompt_builders import build_ask_prompt, build_request_prompt
from ..contact_manager.simulated import SimulatedContactManager
from ..transcript_manager.simulated import SimulatedTranscriptManager
from ..knowledge_manager.simulated import SimulatedKnowledgeManager
from ..task_scheduler.simulated import SimulatedTaskScheduler


class SimulatedTaskManager:
    """
    Top-level façade that *can* own a maximum of *one* live plan at a time and exposes two
    different tool surfaces which include the knowledge, task list, contacts, and transcript histories:

    • `ask()`     → read-only (passive) tools + passive plan methods
    • `request()` → read-only + *all* active tools + all plan methods
    """

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
    ) -> None:
        """
        Args:
            description: A detailed description of the hypothetical scenario to simulate.
        """

        # ── Simulated façade (pure-LLM back-ends) ────────────────────
        self._contact_manager = SimulatedContactManager(description=description)
        self._transcript_manager = SimulatedTranscriptManager(description=description)
        self._knowledge_manager = SimulatedKnowledgeManager(description=description)
        self._task_scheduler = SimulatedTaskScheduler(description=description)

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
            self._knowledge_manager.ask,
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
                self._contact_manager.update,
                self._transcript_manager.summarize,
                self._knowledge_manager.update,
                self._task_scheduler.update,
                ToolSpec(_wrapped_start_task, max_concurrent=1),
                include_class_name=True,
            ),
        }

        self._active_tools = active

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ):
        """
        Read-only question: exposes *passive* helpers (+ plan.ask when available).
        """
        self._refresh_tool_dicts()

        tools: Dict[str, Callable] = dict(self._passive_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_ask_prompt(tools))

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
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

    async def request(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ):
        """
        Full-access entry-point – exposes every passive tool **plus** all
        write-capable helpers and `start_task` (which unlocks plan steering).
        """
        self._refresh_tool_dicts()

        tools: Dict[str, Callable] = dict(self._active_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_request_prompt(tools))

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.request.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result

        return handle
