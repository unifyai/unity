import functools
from typing import Optional, Dict, Callable, TYPE_CHECKING

from ..actor.base import BaseActiveTask

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler


class ActiveTask(BaseActiveTask):
    def __init__(
        self,
        active_task: BaseActiveTask,
        *,
        task_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
    ):
        """
        Thin wrapper that:
        • exposes the underlying plan's steer-controls and\
        • **optionally** keeps the task table in sync when a *scheduler* is supplied.

        Parameters
        ----------
        description
            Human-readable task description (passed straight to the planner).
        planner
            The concrete planner implementation responsible for spawning an active task.
        task_id, instance_id, scheduler
            When provided, every lifecycle transition (pause/resume/stop/finish)
            is mirrored back into the task list via ``scheduler._update_task_status``.
        """
        self._active_task = active_task
        self._scheduler: Optional["TaskScheduler"] = scheduler
        self._task_id: Optional[int] = task_id
        self._instance_id: Optional[int] = instance_id
        self._was_stopped: bool = False

    @functools.wraps(BaseActiveTask.ask, updated=())
    async def ask(self, message: str) -> str:
        return await self._active_task.ask(message)

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(self, message: str) -> None:
        await self._active_task.interject(message)

    @functools.wraps(BaseActiveTask.stop, updated=())
    def stop(self, reason: Optional[str] = None) -> Optional[str]:
        ret = self._active_task.stop(reason)  # type: ignore[call-arg]
        self._was_stopped = True
        self._mirror_status("cancelled")
        # Optionally reinstate the task back into its prior queue position
        try:
            if self._scheduler and self._task_id is not None:
                # Lightweight, synchronous decision (heuristic) – if guidance suggests
                # resuming later or returning to the original schedule, reinsert now.
                self._scheduler._maybe_reinstate_after_stop(  # type: ignore[attr-defined]
                    task_id=self._task_id,
                    reason=reason,
                )
        except Exception:
            # Reinsertion is best-effort and must not interfere with stop semantics
            pass
        self._clear_active_pointer()
        return ret

    @functools.wraps(BaseActiveTask.pause, updated=())
    def pause(self) -> Optional[str]:
        ret = self._active_task.pause()
        self._mirror_status("paused")
        return ret

    @functools.wraps(BaseActiveTask.resume, updated=())
    def resume(self) -> Optional[str]:
        return self._active_task.resume()

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        ret = self._active_task.done()
        self._mirror_status("active")
        return ret

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        ret = await self._active_task.result()
        # If the task wasn't explicitly cancelled/failed, mark as completed.
        if self._scheduler and self._task_id is not None and not self._was_stopped:
            row = self._scheduler._filter_tasks(  # type: ignore[attr-defined]
                filter=f"task_id == {self._task_id} and instance_id == {self._instance_id}",
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
        if (
            self._scheduler
            and self._task_id is not None
            and self._instance_id is not None
        ):
            self._scheduler._update_task_status_instance(  # type: ignore[attr-defined]
                task_id=self._task_id,
                instance_id=self._instance_id,
                new_status=new_status,
            )

    def _clear_active_pointer(self) -> None:
        """Free the scheduler's active-task slot, if any."""
        if self._scheduler and getattr(self._scheduler, "_active_task", None):
            active = self._scheduler._active_task  # type: ignore[attr-defined]
            if (
                active["task_id"] == self._task_id
                and active["instance_id"] == self._instance_id
            ):
                self._scheduler._active_task = None  # type: ignore[attr-defined]

    @property
    @functools.wraps(BaseActiveTask.valid_tools, updated=())
    def valid_tools(self) -> Dict[str, Callable]:
        tools = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        # Reflect paused state from the underlying task handle when available.
        paused_flag = getattr(self._active_task, "_paused", False)
        if paused_flag:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools
