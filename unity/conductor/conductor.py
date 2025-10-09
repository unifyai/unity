# conductor/conductor.py
from __future__ import annotations

from typing import Callable, Dict, Optional

import asyncio
import json
import os

import unify
import functools
import inspect

from typing import Callable, Dict

from ..common.llm_helpers import (
    methods_to_tool_dict,
    ToolSpec,
)
from ..common.async_tool_loop import start_async_tool_loop
from .prompt_builders import build_ask_prompt, build_request_prompt
from .base import BaseConductor
from ..contact_manager.base import BaseContactManager
from ..contact_manager.simulated import SimulatedContactManager
from ..transcript_manager.base import BaseTranscriptManager
from ..transcript_manager.simulated import SimulatedTranscriptManager
from ..knowledge_manager.base import BaseKnowledgeManager
from ..knowledge_manager.simulated import SimulatedKnowledgeManager
from ..skill_manager.base import BaseSkillManager
from ..task_scheduler.simulated import SimulatedTaskScheduler
from ..task_scheduler.base import BaseTaskScheduler
from ..web_searcher.base import BaseWebSearcher
from ..web_searcher.simulated import SimulatedWebSearcher
from ..actor.base import BaseActor
from ..actor.simulated import SimulatedActor
from ..skill_manager.simulated import SimulatedSkillManager
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)


class Conductor:
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
        simulation_guidance: Optional[str] = None,
        # Optional manager overrides – fall back to simulated defaults
        contact_manager: Optional[BaseContactManager] = None,
        transcript_manager: Optional[BaseTranscriptManager] = None,
        knowledge_manager: Optional[BaseKnowledgeManager] = None,
        skill_manager: Optional[BaseSkillManager] = None,
        task_scheduler: Optional[BaseTaskScheduler] = None,
        web_searcher: Optional[BaseWebSearcher] = None,
        actor: Optional[BaseActor] = None,
    ) -> None:
        """
        Args:
            description: A detailed description of the hypothetical scenario to simulate.
            log_events: Whether to log ManagerMethod events to the EventBus.
        """
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # ── Managers – use provided instances or default to simulated back-ends ──
        self._contact_manager = (
            contact_manager
            if contact_manager is not None
            else SimulatedContactManager(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        self._transcript_manager = (
            transcript_manager
            if transcript_manager is not None
            else SimulatedTranscriptManager(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        self._knowledge_manager = (
            knowledge_manager
            if knowledge_manager is not None
            else SimulatedKnowledgeManager(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        self._skill_manager = (
            skill_manager
            if skill_manager is not None
            else SimulatedSkillManager(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        self._task_scheduler = (
            task_scheduler
            if task_scheduler is not None
            else SimulatedTaskScheduler(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        self._web_searcher = (
            web_searcher
            if web_searcher is not None
            else SimulatedWebSearcher(
                description=description,
                log_events=log_events,
            )
        )

        # Actor – simulation-only executor for free-form activities
        self._actor = (
            actor
            if actor is not None
            else SimulatedActor(
                steps=0,
                duration=None,
                simulation_guidance=simulation_guidance,
            )
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
            self._skill_manager.ask,
            self._task_scheduler.ask,
            self._web_searcher.ask,
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
                self._actor.act,
                ToolSpec(self._task_scheduler.execute, max_concurrent=1),
                include_class_name=True,
            ),
        }

        # Enforce mutual exclusion between Actor.act and TaskScheduler.execute by
        # tracking a single active handle and masking both tools while one is active.
        def _wrap_and_track(orig_callable):
            @functools.wraps(orig_callable)
            async def _wrapper(*args, **kwargs):
                res = orig_callable(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    res = await res
                try:
                    from unity.common.async_tool_loop import SteerableToolHandle  # type: ignore

                    if isinstance(res, SteerableToolHandle):
                        self._active_task = res  # type: ignore[assignment]

                        async def _clear_when_done(h):
                            try:
                                await h.result()
                            except Exception:
                                pass
                            finally:
                                if getattr(self, "_active_task", None) is h:
                                    self._active_task = None  # type: ignore[assignment]

                        asyncio.create_task(_clear_when_done(res))
                except Exception:
                    # Best-effort tracking only; never break the tool call
                    pass
                return res

            # Preserve original signature/annotations so tool schema stays accurate
            try:
                _wrapper.__signature__ = inspect.signature(orig_callable)
                try:
                    ann = dict(getattr(orig_callable, "__annotations__", {}))
                    ann.pop("self", None)
                    _wrapper.__annotations__ = ann
                except Exception:
                    pass
            except Exception:
                pass
            return _wrapper

        # Locate canonical keys for the two entry-points (names include class prefixes)
        actor_key = next((k for k in active if "actor_act" in k.lower()), None)
        exec_key = next(
            (k for k in active if "taskscheduler_execute" in k.lower()),
            None,
        )

        if actor_key is not None:
            _orig = active[actor_key]
            if isinstance(_orig, ToolSpec):
                active[actor_key] = ToolSpec(
                    fn=_wrap_and_track(_orig.fn),
                    max_concurrent=_orig.max_concurrent,
                    max_total_calls=_orig.max_total_calls,
                )
            else:
                active[actor_key] = _wrap_and_track(_orig)  # type: ignore[arg-type]

        if exec_key is not None:
            _orig = active[exec_key]
            if isinstance(_orig, ToolSpec):
                active[exec_key] = ToolSpec(
                    fn=_wrap_and_track(_orig.fn),
                    max_concurrent=_orig.max_concurrent,
                    max_total_calls=_orig.max_total_calls,
                )
            else:
                active[exec_key] = _wrap_and_track(_orig)  # type: ignore[arg-type]

        self._active_tools = active

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseConductor.ask, updated=())
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
            "gpt-5@openai",
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

        handle = start_async_tool_loop(
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

    @functools.wraps(BaseConductor.request, updated=())
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
        write-capable helpers and `execute` (which unlocks plan steering).
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
            "gpt-5@openai",
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

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.request.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            # Hide Actor.act and TaskScheduler.execute while a session is active
            tool_policy=self._mask_act_execute_policy(),
            # Ensure at most one base tool is scheduled per assistant turn
            max_parallel_tool_calls=1,
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

    # ------------------------------------------------------------------ #
    #  Internal policy – mask Actor.act and TaskScheduler.execute while active
    # ------------------------------------------------------------------ #

    def _mask_act_execute_policy(self):
        def _policy(step_index: int, tools: Dict[str, Callable]):
            mode = "required" if step_index < 1 else "auto"
            filtered = dict(tools)

            try:
                active = getattr(self, "_active_task", None)
                if active is not None and not active.done():
                    # Remove both entry-points from the base toolkit; dynamic helpers remain available
                    actor_key = next(
                        (k for k in list(filtered) if "actor_act" in k.lower()),
                        None,
                    )
                    exec_key = next(
                        (
                            k
                            for k in list(filtered)
                            if "taskscheduler_execute" in k.lower()
                        ),
                        None,
                    )
                    if actor_key:
                        filtered.pop(actor_key, None)
                    if exec_key:
                        filtered.pop(exec_key, None)
            except Exception:
                pass

            return mode, filtered

        return _policy
