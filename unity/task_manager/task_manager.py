# task_manager/task_manager.py
from __future__ import annotations

from typing import Callable, Dict

from ..common.llm_helpers import (
    methods_to_tool_dict,
    ToolSpec,
)


class TaskManager:
    """
    Top-level façade that *can* own a maximum of *one* live plan at a time and exposes two
    different tool surfaces which include the knowledge, task list, contacts, and transcript histories:

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

        if simulated:
            from ..knowledge_manager.simulated import SimulatedKnowledgeManager
            from ..task_scheduler.simulated import SimulatedTaskScheduler
            from ..transcript_manager.simulated import SimulatedTranscriptManager
            from ..contact_manager.simulated import SimulatedContactManager

            # ── Simulated façade (pure-LLM back-ends) ────────────────────
            self._contact_manager = SimulatedContactManager()
            self._transcript_manager = SimulatedTranscriptManager()
            self._knowledge_manager = SimulatedKnowledgeManager()
            self._task_scheduler = SimulatedTaskScheduler()
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
        self._passive_tools: Dict[str, Callable] = methods_to_tool_dict(
            # contact
            self._contact_manager.ask,
            # transcript
            self._transcript_manager.ask,
            # knowledge
            self._knowledge_manager.retrieve,
            # task-list
            self._task_scheduler.ask,
            include_class_name=True,
        )

        self._active_tools: Dict[str, Callable] = methods_to_tool_dict(
            # transcript
            self._transcript_manager.summarize,
            # knowledge
            self._knowledge_manager.store,
            # task-list
            self._task_scheduler.update,
            # live task
            ToolSpec(self._task_scheduler.start_task, max_concurrent=1),
            include_class_name=True,
        )

        # these will be (re)built whenever _current_plan changes
        self._passive_tools: Dict[str, Callable] = {}
        self._active_tools: Dict[str, Callable] = {}
