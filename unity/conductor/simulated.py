# conductor/conductor.py
from __future__ import annotations

from typing import Callable, Dict, Optional

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
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)


class SimulatedConductor:
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
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """
        Args:
            description: A detailed description of the hypothetical scenario to simulate.
            log_events: Whether to log ManagerMethod events to the EventBus.
        """
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # ── Simulated façade (pure-LLM back-ends) ────────────────────
        self._contact_manager = SimulatedContactManager(
            description=description,
            log_events=log_events,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
        )
        self._transcript_manager = SimulatedTranscriptManager(
            description=description,
            log_events=log_events,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
        )
        self._knowledge_manager = SimulatedKnowledgeManager(
            description=description,
            log_events=log_events,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
        )
        self._task_scheduler = SimulatedTaskScheduler(
            description=description,
            log_events=log_events,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
        )

        #  Run-time state & tool-dict helpers
        self._active_task = None  # type: ignore

        # These two dicts are rebuilt lazily before every ask/request
        self._passive_tools: Dict[str, Callable] = {}
        self._active_tools: Dict[str, Callable] = {}
        """Re-compute passive / active tool maps based on current active task."""

        # -------- base passive helpers -------------------------------- #
        passive = methods_to_tool_dict(
            self._contact_manager.ask,
            self._transcript_manager.ask,
            self._knowledge_manager.ask,
            self._task_scheduler.ask,
            include_class_name=True,
        )

        # -------- add active_task.ask when a plan is alive ------------------- #
        if self._active_task is not None and not self._active_task.done():

            # We expose _ask_plan_call_ (Unify expects this naming)
            async def _plan_ask_proxy(question: str):
                return await self._active_task.ask(question)  # type: ignore[attr-defined]

            _plan_ask_proxy.__name__ = "_ask_plan_call_"
            passive[_plan_ask_proxy.__name__] = _plan_ask_proxy

        self._passive_tools = passive

        # -------- build active helpers (passive + writers) ------------ #

        active = {
            **passive,  # read-only tools are also valid here
            **methods_to_tool_dict(
                self._contact_manager.update,
                self._knowledge_manager.update,
                self._task_scheduler.update,
                ToolSpec(self._task_scheduler.execute_task, max_concurrent=1),
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
        parent_chat_context: list[dict] | None = None,  # Unused – synthetic
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
        rolling_summary_in_prompts: Optional[bool] = None,
    ):
        """
        Read-only question: exposes *passive* helpers (+ active_task.ask when available).
        """
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "Conductor",
                "ask",
                phase="incoming",
                question=text,
            )

        tools: Dict[str, Callable] = dict(self._passive_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client = unify.AsyncUnify(
            "gpt-5->o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_ask_prompt(tools, include_activity=include_activity),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            # Keep behaviour close to the real Conductor: force one tool call on turn 0, then auto
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "Conductor",
                "ask",
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
        log_events: bool = False,
        rolling_summary_in_prompts: Optional[bool] = None,
    ):
        """
        Full-access entry-point – exposes every passive tool **plus** all
        write-capable helpers and `execute_task` (which unlocks plan steering).
        """
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "Conductor",
                "request",
                phase="incoming",
                request=text,
            )

        tools: Dict[str, Callable] = dict(self._active_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client = unify.AsyncUnify(
            "gpt-5->o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_request_prompt(tools, include_activity=include_activity),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.request.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            # Keep behaviour close to the real Conductor: force one tool call on turn 0, then auto
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "Conductor",
                "request",
            )

        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result

        return handle
