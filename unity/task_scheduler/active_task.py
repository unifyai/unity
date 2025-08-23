import functools
import asyncio
from typing import Optional, Dict, Callable, TYPE_CHECKING

from .base import BaseActiveTask
from ..actor.base import BaseActor
from unity.common.llm_helpers import SteerableToolHandle

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler


class ActiveTask(BaseActiveTask):
    def __init__(
        self,
        actor_handle: SteerableToolHandle,
        *,
        task_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
    ):
        """
        Thin wrapper around an actor-backed active plan handle, keeping the
        corresponding Tasks row in sync when a scheduler is provided.

        Use ``ActiveTask.create(...)`` to construct an instance from a
        ``BaseActor`` and a task description.
        """
        self._actor_handle = actor_handle
        self._scheduler: Optional["TaskScheduler"] = scheduler
        self._task_id: Optional[int] = task_id
        self._instance_id: Optional[int] = instance_id
        self._was_stopped: bool = False

    @classmethod
    async def create(
        cls,
        actor: BaseActor,
        *,
        task_description: str,
        parent_chat_context: Optional[list[dict]] = None,
        clarification_up_q: Optional["asyncio.Queue[str]"] = None,
        clarification_down_q: Optional["asyncio.Queue[str]"] = None,
        task_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        scheduler: Optional["TaskScheduler"] = None,
    ) -> "ActiveTask":
        """
        Create an ActiveTask by starting work on the provided ``actor``.

        This is the preferred constructor: it ensures the underlying active
        handle is running before returning an instance.
        """
        actor_steerable_handle = await actor.act(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        return cls(
            actor_steerable_handle,  # type: ignore[arg-type]
            task_id=task_id,
            instance_id=instance_id,
            scheduler=scheduler,
        )

    @functools.wraps(BaseActiveTask.ask, updated=())
    async def ask(
        self,
        message: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> SteerableToolHandle:
        """Answer a read-only question about the live activity and return a handle."""
        answer: str = await self._actor_handle.ask(message)

        # Lightweight static handle that simply returns the captured answer
        class _AnswerHandle(SteerableToolHandle):  # type: ignore[abstract-method]
            def __init__(self) -> None:
                pass

            async def interject(self, message: str): ...

            def stop(self, reason: Optional[str] = None): ...

            def pause(self): ...

            def resume(self): ...

            def done(self) -> bool:
                return True

            async def result(self) -> str:
                # Ignoring _return_reasoning_steps for ActiveTask.ask; only answer string is returned.
                return answer

            async def ask(self, question: str) -> "SteerableToolHandle":  # type: ignore[override]
                return self

        return _AnswerHandle()

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(self, message: str) -> None:
        await self._actor_handle.interject(message)

    @functools.wraps(BaseActiveTask.stop, updated=())
    def stop(self, *, cancel: bool, reason: Optional[str] = None) -> Optional[str]:
        """Stop the running activity with explicit intent.

        When ``cancel`` is True the task instance is marked cancelled. When False, the
        task is deferred and we attempt to reinstate it to its previous queue/schedule
        position using the stored reintegration plan (when available).
        """
        ret = self._actor_handle.stop(reason)  # type: ignore[call-arg]
        self._was_stopped = True

        # Cancel → mark cancelled; Defer → try reinstatement
        if cancel:
            self._mirror_status("cancelled")
        else:
            try:
                if self._scheduler and self._task_id is not None:
                    # Prefer strict reinstatement using the stored plan when present.
                    try:
                        self._scheduler._reinstate_task_to_previous_queue(  # type: ignore[attr-defined]
                            task_id=self._task_id,
                        )
                    except Exception:
                        # If no plan exists, fall back to heuristic best-effort reinsertion.
                        self._scheduler._maybe_reinstate_after_stop(  # type: ignore[attr-defined]
                            task_id=self._task_id,
                            reason=reason,
                        )
            except Exception:
                # Best-effort – failure to reinstate must not break stop semantics
                pass

        self._clear_active_pointer()
        return ret

    @functools.wraps(BaseActiveTask.pause, updated=())
    def pause(self) -> Optional[str]:
        ret = self._actor_handle.pause()
        self._mirror_status("paused")
        return ret

    @functools.wraps(BaseActiveTask.resume, updated=())
    def resume(self) -> Optional[str]:
        ret = self._actor_handle.resume()
        self._mirror_status("active")
        return ret

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        ret = self._actor_handle.done()
        self._mirror_status("active")
        return ret

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        ret = await self._actor_handle.result()
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
        paused_flag = getattr(self._actor_handle, "_paused", False)
        if paused_flag:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools
