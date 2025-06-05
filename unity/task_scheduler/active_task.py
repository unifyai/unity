import asyncio
import functools
from typing import Optional, Dict, Callable, TYPE_CHECKING

from ..planner.base import BasePlanner, BasePlan
from ..planner.simulated import SimulatedPlanner

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler


class ActiveTask(BasePlan):

    def __init__(
        self,
        description: str,
        planner: Optional[BasePlanner] = SimulatedPlanner(1),
        *,
        task_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
    ) -> None:
        """
        Thin wrapper that:
        • exposes the underlying plan's steer-controls and\
        • **optionally** keeps the task table in sync when a *scheduler* is supplied.

        Parameters
        ----------
        description
            Human-readable task description (passed straight to the planner).
        planner
            The concrete planner implementation responsible for spawning a plan.
        task_id, scheduler
            When provided, every lifecycle transition (pause/resume/stop/finish)
            is mirrored back into the task list via ``scheduler._update_task_status``.
        """

        self._plan = planner.plan(description)
        self._scheduler: Optional["TaskScheduler"] = scheduler
        self._task_id: Optional[int] = task_id

    @functools.wraps(BasePlan.ask, updated=())
    async def ask(self, message: str) -> str:
        return await asyncio.to_thread(self._plan.ask, message)

    @functools.wraps(BasePlan.interject, updated=())
    async def interject(self, message: str) -> None:
        await self._plan.interject(message)

    @functools.wraps(BasePlan.stop, updated=())
    def stop(self) -> Optional[str]:
        ret = self._plan.stop()
        self._mirror_status("cancelled")
        self._clear_active_pointer()
        return ret

    @functools.wraps(BasePlan.pause, updated=())
    def pause(self) -> Optional[str]:
        ret = self._plan.pause()
        self._mirror_status("paused")
        return ret

    @functools.wraps(BasePlan.resume, updated=())
    def resume(self) -> Optional[str]:
        return self._plan.resume()

    @functools.wraps(BasePlan.done, updated=())
    def done(self) -> bool:
        ret = self._plan.done()
        self._mirror_status("active")
        return ret

    @functools.wraps(BasePlan.result, updated=())
    async def result(self) -> str:
        ret = await self._plan.result()
        # If the task wasn't explicitly cancelled/failed, mark as completed.
        if self._scheduler and self._task_id is not None:
            row = self._scheduler._search_tasks(  # type: ignore[attr-defined]
                filter=f"task_id == {self._task_id}",
                limit=1,
            )[0]
            if row["status"] not in ("cancelled", "failed"):
                self._mirror_status("completed")
        self._clear_active_pointer()
        return ret

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _mirror_status(self, new_status: str) -> None:
        """Update the task-row status if we were instantiated by a scheduler."""
        if self._scheduler and self._task_id is not None:
            self._scheduler._update_task_status(  # type: ignore[attr-defined]
                task_ids=self._task_id,
                new_status=new_status,
            )

    def _clear_active_pointer(self) -> None:
        """Free the scheduler's active-task slot, if any."""
        if self._scheduler and getattr(self._scheduler, "_active_task", None):
            if self._scheduler._active_task["task_id"] == self._task_id:  # type: ignore[attr-defined]
                self._scheduler._active_task = None  # type: ignore[attr-defined]

    # ── handy passthrough also exposed to the LLM ──────────────────────────
    def ask(self, question: str) -> str:  # type: ignore[override]
        """Ask the running plan a question (simply forwards the call)."""
        return self._plan.ask(question)

    @property
    @functools.wraps(BasePlan.valid_tools, updated=())
    def valid_tools(self) -> Dict[str, Callable]:
        tools = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        if self._paused:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools
