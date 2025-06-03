# task_manager/task_manager.py
from __future__ import annotations

from typing import Callable, Dict
import functools

import json
import os

import unify

from ..common.llm_helpers import (
    AsyncToolUseLoopHandle,
    SteerableToolHandle,
    start_async_tool_use_loop,
)
from ..events.event_bus import EventBus

PASSIVE_PLAN_METHODS: set[str] = {"ask"}


class TaskManager:
    """
    Top-level façade that owns *one* live plan at a time and exposes two
    different tool surfaces:

    • `ask()`     → read-only (passive) tools + passive plan methods
    • `request()` → read-only + *all* active tools + all plan methods
    """

    # ------------------------------------------------------------------ #

    def __init__(self, planner_steps: int = 2, *, simulated: bool = False) -> None:
        """
        Args:
            planner_steps:   How many "steps" a simulated plan should take before
                             auto-completing (only relevant for SimulatedPlanner).
            simulated:       When *True* all subordinate managers are replaced by
                             their **simulated** counterparts which keep all state
                             inside an LLM conversation rather than touching real
                             storage back-ends.  Defaults to *False* (real managers).
        """

        self._event_bus = EventBus()

        if simulated:
            from ..knowledge_manager.simulated import SimulatedKnowledgeManager
            from ..planner.simulated import SimulatedPlanner
            from ..task_scheduler.simulated import SimulatedTaskScheduler
            from ..transcript_manager.simulated import SimulatedTranscriptManager
            from ..contact_manager.simulated import SimulatedContactManager

            # ── Simulated façade (pure-LLM back-ends) ────────────────────
            self._contact_manager = SimulatedContactManager()
            self._transcript_manager = SimulatedTranscriptManager()
            self._knowledge_manager = SimulatedKnowledgeManager()
            self._task_scheduler = SimulatedTaskScheduler()
            self._planner = SimulatedPlanner(planner_steps)
        else:
            from ..knowledge_manager.knowledge_manager import KnowledgeManager
            from ..planner.tool_loop_planner import ToolLoopPlanner
            from ..task_scheduler.task_scheduler import TaskScheduler
            from ..transcript_manager.transcript_manager import TranscriptManager
            from ..contact_manager.contact_manager import ContactManager

            # ── Real managers touching Unify back-ends ───────────────────
            self._contact_manager = ContactManager(self._event_bus)
            self._transcript_manager = TranscriptManager(self._event_bus)
            self._knowledge_manager = KnowledgeManager()
            self._task_scheduler = TaskScheduler()
            self._planner = ToolLoopPlanner(planner_steps)

        # static, always-present tools -------------------------------------------------
        self._static_passive_tools: Dict[str, Callable] = {
            # contact
            f"{self._contact_manager.__class__.__name__}_ask": self._contact_manager.ask,
            # transcript
            f"{self._transcript_manager.__class__.__name__}_ask": self._transcript_manager.ask,
            # knowledge
            f"{self._knowledge_manager.__class__.__name__}_retrieve": self._knowledge_manager.retrieve,
            # task-list
            f"{self._task_scheduler.__class__.__name__}_ask": self._task_scheduler.ask,
        }

        self._static_active_tools: Dict[str, Callable] = {
            # transcript
            f"{self._transcript_manager.__class__.__name__}_summarize": self._transcript_manager.summarize,
            # knowledge
            f"{self._knowledge_manager.__class__.__name__}_store": self._knowledge_manager.store,
            # task-list
            f"{self._task_scheduler.__class__.__name__}_update": self._task_scheduler.update,
        }

        # ---------- planner wrappers --------------------------------------------------
        self._wrap_planner_entrypoints()

        # these will be (re)built whenever _current_plan changes
        self._passive_tools: Dict[str, Callable] = {}
        self._active_tools: Dict[str, Callable] = {}
        self._rebuild_tool_sets()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    # 1.  Wrap planner.plan so we notice new plans -------------
    def _wrap_planner_entrypoints(self) -> None:  # noqa: D401
        plan_orig = self._planner.plan

        @functools.wraps(plan_orig)
        def _plan_wrapped(*args, **kw):
            plan_handle = plan_orig(*args, **kw)
            if isinstance(plan_handle, SteerableToolHandle):
                self._rebuild_tool_sets()
            return plan_handle

        self._planner.plan = _plan_wrapped

    # 2.  (Re)build passive & active tool dicts -------------------------
    def _rebuild_tool_sets(self) -> None:
        """Called every time the current plan handle changes."""
        # fresh copies
        passive = dict(self._static_passive_tools)
        active = dict(self._static_active_tools)

        # planner API itself ---------------------------------------------------
        if self._planner.active_plan is None:
            active[f"Planner_plan"] = self._planner.plan

        # live plan tools ------------------------------------------------------
        if self._planner.active_plan:
            for meth_name, bound in self._planner.active_plan.valid_tools.items():
                key = f"Plan_{meth_name}"
                if meth_name in PASSIVE_PLAN_METHODS:
                    passive[key] = bound
                else:
                    active[key] = bound

        # store
        self._passive_tools = passive
        self._active_tools = active
        self._tools = {**passive, **active}

    # 3.  Convenience for new LLM client --------------------------------
    @staticmethod
    def _new_client() -> unify.AsyncUnify:
        return unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    async def ask(
        self,
        question: str,
        *,
        log_tool_steps: bool = False,
    ) -> AsyncToolUseLoopHandle:
        """
        Ask *read-only* questions about the running system or live plan.
        """
        from .sys_msgs import ASK

        client = self._new_client()
        client.set_system_message(ASK)
        return start_async_tool_use_loop(
            client,
            message=question,
            tools=self._passive_tools,
            log_steps=log_tool_steps,
        )

    async def request(
        self,
        text: str,
        *,
        log_tool_steps: bool = False,
    ) -> AsyncToolUseLoopHandle:
        """
        Full-power request: may change the task list, knowledge base or
        live plan (pause/resume/stop/interject/etc.) or even spawn a new one.
        """
        from .sys_msgs import REQUEST

        client = self._new_client()
        client.set_system_message(REQUEST)
        return start_async_tool_use_loop(
            client,
            message=text,
            tools=self._tools,
            log_steps=log_tool_steps,
        )
