from __future__ import annotations

from typing import Any, Dict, Optional

import unify

from .types.status import Status
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler
from .types.reintegration_plan import ReintegrationPlan


class ReintegrationManager:
    """
    Encapsulates logic for restoring a task to its previous queue/schedule
    position using a stored ReintegrationPlan. Behaviour mirrors the existing
    TaskScheduler._reinstate_task_to_previous_queue contract.
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
        return (
            self._s._to_status(rows[0].get("status")) not in self._s._TERMINAL_STATUSES
        )

    def apply(self, *, task_id: int, allow_active: bool = False) -> Dict[str, str]:
        # Locate plan (prefer non-terminal instance)
        rows = self._s._filter_tasks(filter=f"task_id == {task_id}", limit=10)
        live = [
            r
            for r in rows
            if self._s._to_status(r.get("status")) not in self._s._TERMINAL_STATUSES
        ]
        instance_id = None
        plan: Optional[ReintegrationPlan] = None
        if live:
            instance_id = sorted(live, key=lambda r: r.get("instance_id", 0))[0].get(
                "instance_id",
            )
            plan = self._s._reintegration_plans.get((task_id, instance_id))
        else:
            for (tid, iid), p in getattr(self._s, "_reintegration_plans", {}).items():
                if tid == task_id:
                    plan = p
                    instance_id = iid
                    break

        if not plan:
            raise ValueError("No reintegration plan available.")
        if plan.task_id != task_id:
            raise ValueError(
                f"Reintegration plan exists for task_id={plan.task_id}, not {task_id}",
            )

        tid = plan.task_id
        prev_tid = plan.prev_task
        next_tid = plan.next_task
        was_head = bool(plan.was_head)
        original_start_at = plan.start_at
        original_status = plan.original_status

        cur_rows = self._s._filter_tasks(filter=f"task_id == {tid}", limit=1)
        cur_row = cur_rows[0] if cur_rows else {}

        if (
            self._s._to_status(cur_row.get("status")) == Status.active
            and not allow_active
        ):
            raise RuntimeError(
                "Cannot reinstate while the task is active. Stop/defer first.",
            )

        if cur_row.get("trigger") is not None:
            raise ValueError(
                "Task currently has a trigger; remove the trigger before restoring its schedule/queue position.",
            )

        queue_list = self._s._get_task_queue()
        queue_ids = [t.task_id for t in queue_list]

        final_prev, final_next = self._s._select_final_neighbours(
            task_id=tid,
            was_head=was_head,
            original_prev=prev_tid,
            original_next=next_tid,
            queue_ids=queue_ids,
            is_viable=self._is_viable,
        )

        cur_sched: Dict[str, Any] = {
            "prev_task": final_prev,
            "next_task": final_next,
        }
        plan_head_start = getattr(plan, "head_start_at", None)
        if final_prev is None:
            _head_ts = (
                plan_head_start if plan_head_start is not None else original_start_at
            )
            if _head_ts is not None:
                cur_sched["start_at"] = _head_ts

        desired_status = (
            self._s._to_status(str(original_status))
            if original_status is not None
            else Status.queued
        )
        if (
            desired_status == Status.primed
            and self._s._primed_task is not None
            and self._s._primed_task.get("task_id") != tid
        ):
            desired_status = Status.queued

        if final_prev is None and cur_sched.get("start_at") is not None:
            desired_status = Status.scheduled

        self._s._validate_scheduled_invariants(
            status=desired_status,
            schedule=cur_sched,
            err_prefix=f"While reinstating task {tid}:",
        )

        self._s._attach_with_links(
            task_id=tid,
            prev_task=final_prev,
            next_task=final_next,
            head_start_at=(cur_sched.get("start_at") if final_prev is None else None),
            err_prefix=f"While reinstating task {tid}:",
        )

        if desired_status != Status.active:
            self._s._update_task_status_instance(
                task_id=tid,
                instance_id=plan.instance_id,
                new_status=str(desired_status),
            )

        if was_head and final_next is not None:

            def _fix_next_status():
                next_rows = self._s._filter_tasks(
                    filter=f"task_id == {final_next}",
                    limit=1,
                )
                if next_rows:
                    next_row = next_rows[0]
                    next_sched = next_row.get("schedule") or {}
                    if (
                        self._s._sched_prev(next_sched) is not None
                        and (next_sched.get("start_at") is None)
                        and self._s._to_status(next_row.get("status"))
                        in {Status.scheduled, Status.primed}
                    ):
                        self._s._update_task_status_instance(
                            task_id=final_next,
                            instance_id=next_row["instance_id"],
                            new_status="queued",
                        )

            self._s._best_effort(_fix_next_status)

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
