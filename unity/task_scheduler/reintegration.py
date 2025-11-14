"""
Reinstatement utilities for the task scheduler.

Restores a deferred task to its previous queue or schedule position using a
stored ReintegrationPlan. Selects viable neighbours, reconstructs head
timestamps when applicable, derives the correct lifecycle status, validates
invariants, writes symmetric links and status updates via the scheduler, and
reconciles adjacent task state when the head changes.
"""

from __future__ import annotations

from typing import Any, Optional, Callable

import unify

from unity.common.tool_outcome import ToolOutcome

from .types.status import Status
from .types.schedule import Schedule
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler
from .types.reintegration_plan import ReintegrationPlan
from .queue_engine import derive_status_after_queue_edit


class ReintegrationManager:
    """
    Restores a task to its previous queue or schedule position using a
    ReintegrationPlan. Chooses viable neighbours, sets head timestamps when
    appropriate, derives the target status, validates invariants, attaches
    links, updates status, and clears the consumed plan.
    """

    def __init__(self, scheduler: "TaskScheduler") -> None:
        self._s = scheduler

    def _get_log_obj(self, tid_int: int) -> Optional[unify.Log]:
        rows = self._s._get_logs_by_task_ids(task_ids=tid_int, return_ids_only=False)
        if not rows:
            return None
        assert len(rows) == 1, "Task IDs should be unique"
        return rows[0]  # type: ignore[return-value]

    def _is_viable(self, neighbour_tid: Optional[int]) -> bool:
        if neighbour_tid is None:
            return False
        rows = self._s._filter_tasks(filter=f"task_id == {neighbour_tid}", limit=1)
        if not rows:
            return False
        return rows[0].status not in self._s._TERMINAL_STATUSES

    def _select_final_neighbours(
        self,
        *,
        task_id: int,
        was_head: bool,
        original_prev: Optional[int],
        original_next: Optional[int],
        queue_ids: list[int],
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Decide (final_prev, final_next) for reinstatement using a minimal, deterministic
        policy and no I/O.

        Policy:
        - If was_head: final_prev=None; final_next is original_next if viable, otherwise the
        current head (first in queue_ids) if different from task_id, otherwise None.
        - If middle: final_prev is original_prev if viable, else None; final_next is
        original_next if viable and distinct from final_prev, else None.
        - Avoid self-loops and identical prev/next; prefer keeping prev and dropping next.
        """
        current_head_id = queue_ids[0] if queue_ids else None

        def _clean(tid: Optional[int]) -> Optional[int]:
            return None if tid == task_id else tid

        if was_head:
            final_prev = None
            if self._is_viable(original_next):
                final_next = original_next
            else:
                final_next = (
                    current_head_id
                    if (current_head_id is not None and current_head_id != task_id)
                    else None
                )
        else:
            final_prev = original_prev if self._is_viable(original_prev) else None
            final_next = (
                original_next
                if (self._is_viable(original_next) and original_next != final_prev)
                else None
            )

        final_prev = _clean(final_prev)
        final_next = _clean(final_next)
        if (
            final_prev is not None
            and final_next is not None
            and final_prev == final_next
        ):
            final_next = None

        return final_prev, final_next

    def apply(self, *, task_id: int, allow_active: bool = False) -> ToolOutcome:
        # Locate plan (prefer non-terminal instance)
        tasks = self._s._filter_tasks(filter=f"task_id == {task_id}", limit=10)
        live = [t for t in tasks if t.status not in self._s._TERMINAL_STATUSES]
        instance_id = None
        plan: Optional[ReintegrationPlan] = None
        if live:
            instance_id = sorted(live, key=lambda r: r.instance_id)[0].instance_id
            plan = self._s._reintegration_plans.get((task_id, instance_id))
        else:
            for (_task_id, _instance_id), p in self._s._reintegration_plans.items():
                if _task_id == task_id:
                    plan = p
                    instance_id = _instance_id
                    break

        if not plan:
            raise ValueError("No reintegration plan available.")
        if plan.task_id != task_id:
            raise ValueError(
                f"Reintegration plan exists for task_id={plan.task_id}, not {task_id}",
            )

        tid = plan.task_id
        prev_task_id = plan.prev_task
        next_task_id = plan.next_task
        was_head = bool(plan.was_head)
        original_start_at = plan.start_at
        original_status = plan.original_status

        tasks = self._s._filter_tasks(filter=f"task_id == {tid}", limit=1)
        task = tasks[0] if tasks else None

        if task is not None and (task.status == Status.active and not allow_active):
            raise RuntimeError(
                "Cannot reinstate while the task is active. Stop/defer first.",
            )

        if task is not None and task.trigger is not None:
            raise ValueError(
                "Task currently has a trigger; remove the trigger before restoring its schedule/queue position.",
            )

        # Use the plan's queue_id when present, otherwise derive from the current task
        qid = plan.queue_id
        if qid is not None:
            queue_list = self._s._get_queue(queue_id=qid)
        else:
            queue_list = self._s._get_queue_for_task(task_id=tid)
        queue_ids = [t.task_id for t in queue_list]

        final_prev, final_next = self._select_final_neighbours(
            task_id=tid,
            was_head=was_head,
            original_prev=prev_task_id,
            original_next=next_task_id,
            queue_ids=queue_ids,
        )

        cur_sched = Schedule(
            prev_task=final_prev,
            next_task=final_next,
        )

        plan_head_start = plan.head_start_at
        if final_prev is None:
            _head_ts = (
                plan_head_start if plan_head_start is not None else original_start_at
            )
            if _head_ts is not None:
                cur_sched.start_at = _head_ts

        # Determine the desired lifecycle using the central helper
        existing_status = (
            original_status if original_status is not None else Status.queued
        )
        desired_status = derive_status_after_queue_edit(
            existing_status=existing_status,
            is_head=(final_prev is None),
            head_has_start_at=(cur_sched.start_at is not None),
        )
        # Avoid conflicting primed states when another task is already primed
        if (
            desired_status == Status.primed
            and self._s._primed_task is not None
            and self._s._primed_task.task_id != tid
        ):
            desired_status = Status.queued

        self._s._validate_scheduled_invariants(
            status=desired_status,
            schedule=cur_sched,
            err_prefix=f"While reinstating task {tid}:",
        )

        # Use queue_utils to attach with symmetric linkage and invariant enforcement
        from .queue_utils import attach_with_links as _attach_with_links

        _attach_with_links(
            self._s,
            task_id=tid,
            prev_task=final_prev,
            next_task=final_next,
            head_start_at=(
                cur_sched.start_at.isoformat() if final_prev is None else None
            ),
            err_prefix=f"While reinstating task {tid}:",
        )

        if desired_status != Status.active:
            self._s._update_task_status_instance(
                task_id=tid,
                instance_id=plan.instance_id,
                new_status=desired_status,
            )

        if was_head and final_next is not None:

            def _fix_next_status():
                next_rows = self._s._filter_tasks(
                    filter=f"task_id == {final_next}",
                    limit=1,
                )
                if next_rows:
                    next_row = next_rows[0]
                    if (
                        next_row.schedule_prev is not None
                        and (next_row.schedule_start_at is None)
                        and next_row.status in (Status.scheduled, Status.primed)
                    ):
                        self._s._update_task_status_instance(
                            task_id=final_next,
                            instance_id=next_row.instance_id,
                            new_status=Status.queued,
                        )

            _best_effort(_fix_next_status)

        if desired_status == Status.primed:
            self._s._refresh_primed_cache(tid)

        try:
            self._s._reintegration_plans.pop((tid, instance_id), None)
        except Exception:
            pass

        return {
            "outcome": "task reinstated to previous queue position",
            "details": {"task_id": tid},
        }


def _best_effort(func: Callable[[], Any]) -> None:
    try:
        func()
    except Exception:
        pass
