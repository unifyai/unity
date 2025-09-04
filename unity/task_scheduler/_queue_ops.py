from __future__ import annotations

from typing import Optional, Dict, Any, List, TYPE_CHECKING

import unify

from ._queue_utils import (
    sched_prev as _q_prev,
    sched_next as _q_next,
    attach_with_links as _q_attach_with_links,
)
from .types.task import Task
from .types.status import Status
from .types.reintegration_plan import ReintegrationPlan

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler


def get_task_queue(
    scheduler: "TaskScheduler",
    *,
    task_id: Optional[int] = None,
) -> List[Task]:
    """Return the runnable task queue from head to tail.

    Mirrors TaskScheduler._get_task_queue but lives in a separate module to
    keep the main scheduler lean.
    """

    def _get_task_by_task_id(tid: int) -> Optional[dict]:
        rows = scheduler._filter_tasks(filter=f"task_id == {tid}", limit=1)
        return rows[0] if rows else None

    def _choose_start_node(tid: Optional[int]) -> Optional[dict]:
        if tid is not None:
            row = _get_task_by_task_id(tid)
            if row is not None:
                return row
        else:
            if scheduler._primed_task:
                # Validate that the cached primed task still exists in storage.
                primed_id = scheduler._primed_task.get("task_id")
                primed_row = (
                    _get_task_by_task_id(primed_id) if primed_id is not None else None
                )
                if primed_row is not None:
                    return primed_row
                # Stale cache – clear it so we fall back to detecting the head from storage.
                try:
                    scheduler._refresh_primed_cache()
                except Exception:
                    scheduler._primed_task = None

        head_candidates = scheduler._filter_tasks(
            filter=scheduler._HEAD_FILTER,
            limit=2,
        )
        if not head_candidates:
            return None
        assert len(head_candidates) == 1, f"Multiple heads detected: {head_candidates}"
        return head_candidates[0]

    def _walk_to_head(row: dict) -> dict:
        cur = row
        while True:
            prev_id = _q_prev(cur.get("schedule"))
            if prev_id is None:
                break
            prev_row = _get_task_by_task_id(prev_id)
            if prev_row is None:
                break
            cur = prev_row
        return cur

    def _walk_forward(head_row: dict) -> List[Task]:
        ordered: List[Task] = []
        cur = head_row
        while cur:
            if scheduler._to_status(cur.get("status")) not in {
                Status.completed,
                Status.cancelled,
                Status.failed,
            }:
                # Defensive read: drop stale activation metadata on non-active rows
                _row = dict(cur)
                try:
                    if scheduler._to_status(_row.get("status")) != Status.active:  # type: ignore[arg-type]
                        _row.pop("activated_by", None)
                except Exception:
                    if str(_row.get("status")) != str(Status.active):
                        _row.pop("activated_by", None)
                ordered.append(Task(**_row))

            nxt_id = _q_next(cur.get("schedule"))
            if nxt_id is None:
                break
            cur = _get_task_by_task_id(nxt_id)
            if cur is None:
                break
        return ordered

    start_row = _choose_start_node(task_id)
    if start_row is None:
        return []

    if start_row.get("schedule") is None:
        _row = dict(start_row)
        try:
            if scheduler._to_status(_row.get("status")) != Status.active:  # type: ignore[arg-type]
                _row.pop("activated_by", None)
        except Exception:
            if str(_row.get("status")) != str(Status.active):
                _row.pop("activated_by", None)
        return [Task(**_row)]

    head_row = _walk_to_head(start_row)
    return _walk_forward(head_row)


def detach_from_queue_for_activation(
    scheduler: "TaskScheduler",
    *,
    task_id: int,
) -> None:
    """Detach a task from the runnable queue ahead of activation.

    Behavioural contract (from tests):
    - General: Always record a ``ReintegrationPlan`` capturing ``prev_task``,
      ``next_task``, whether the task was the head (``was_head``), the task's
      original ``start_at`` (if any), the original status, and the queue head's
      ``start_at`` at the time of detachment (``head_start_at``). This plan is
      the only source used later by reinstatement.
    - Isolated activation (default):
      * If detaching the head: promote the next task to head, copy the head-level
        ``start_at`` to it, and set its status to ``scheduled``. The detached task
        loses its schedule.
      * If detaching a middle task: unlink it from neighbours (``prev.next = next``
        and ``next.prev = prev``) and ensure the successor does not carry a
        ``start_at`` timestamp (only heads may carry it). The detached task loses
        its schedule.
    - Chained activation (opt-in via env in tests): keep the queue behind the
      activated task attached to it. When promoting the activated task to head,
      place any head-level ``start_at`` on it and remove ``start_at`` from the
      immediate successor.

    These semantics are intentionally minimal and exist solely to make
    reinstatement deterministic and easy to reason about.
    """

    candidate_rows = scheduler._filter_tasks(
        filter=(
            f"task_id == {task_id} and status not in "
            "('completed','cancelled','failed','active')"
        ),
    )
    if not candidate_rows:
        raise ValueError(f"No runnable task found with id={task_id}")
    task_row = sorted(candidate_rows, key=lambda r: r.get("instance_id", 0))[0]

    sched = task_row.get("schedule") or {}
    prev_tid = _q_prev(sched)
    next_tid = _q_next(sched)
    start_at = sched.get("start_at") if isinstance(sched, dict) else None

    # Derive the current head's start_at so downstream tasks can be reinstated as
    # head-scheduled later if their original predecessor becomes terminal.
    def _get_row(tid: int) -> Optional[dict]:
        rows = scheduler._filter_tasks(filter=f"task_id == {tid}", limit=1)
        return rows[0] if rows else None

    head_start_at: Optional[str] = None
    if prev_tid is not None:
        # Walk up to the head for the current queue
        cur = _get_row(task_id)
        while cur is not None and _q_prev(cur.get("schedule")) is not None:
            cur = _get_row(_q_prev(cur.get("schedule")))
        if cur is not None:
            _sched = cur.get("schedule") or {}
            if isinstance(_sched, dict):
                head_start_at = _sched.get("start_at")

    def _get_log_obj(tid: int) -> Optional[unify.Log]:
        try:
            logs = scheduler._get_logs_by_task_ids(
                task_ids=tid,
                return_ids_only=False,
            )
        except ValueError:
            return None
        assert len(logs) == 1, "Task IDs should be unique"
        return logs[0]  # type: ignore[return-value]

    # Small local helpers to reduce repetition and keep behaviour identical
    def _log_id(log_or_id: Any) -> Any:
        return log_or_id.id if hasattr(log_or_id, "id") else log_or_id

    def _load_sched(log_obj: Optional[unify.Log]) -> Dict[str, Any]:
        return {
            **(((getattr(log_obj, "entries", {}) or {}).get("schedule")) or {}),
        }

    def _update_schedule(
        log_or_id: Any,
        new_sched: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entries: Dict[str, Any] = {"schedule": new_sched}
        if extra:
            entries.update(extra)
        scheduler._store.update(
            logs=_log_id(log_or_id),
            entries=entries,
            overwrite=True,
        )

    # Always record a reintegration plan for precise restore on defer stop,
    # regardless of execution scope. This enables queue execution with later
    # reinstatement to the original position when requested.
    plan = ReintegrationPlan(
        task_id=task_id,
        instance_id=task_row.get("instance_id"),
        prev_task=prev_tid,
        next_task=next_tid,
        start_at=start_at,
        was_head=prev_tid is None,
        original_status=task_row.get("status"),
        head_start_at=head_start_at,
    )
    # Store per-instance plan (single source of truth)
    key = (
        task_id,
        (
            task_row.get("instance_id")
            if task_row.get("instance_id") is not None
            else -1
        ),
    )
    scheduler._reintegration_plans[key] = plan  # type: ignore[attr-defined]

    # Disconnect previous neighbour's next pointer when promoting current task to head
    # for queue-based execution. We always keep followers attached.
    if prev_tid is not None:
        prev_log = _get_log_obj(prev_tid)
        if prev_log is not None:
            prev_sched = _load_sched(prev_log)
            if prev_sched.get("next_task") == task_id:
                prev_sched["next_task"] = None
                _update_schedule(prev_log, prev_sched)

    # Apply rewiring for queue-based execution (always)
    if sched is not None:
        # Promote current task to head and keep followers attached
        cur_log = _get_log_obj(task_id)
        new_sched: Dict[str, Any] = {"prev_task": None, "next_task": next_tid}
        if start_at is not None:
            new_sched["start_at"] = start_at
        _update_schedule(cur_log, new_sched)

        if next_tid is not None:
            next_log = _get_log_obj(next_tid)
            if next_log is not None:
                next_sched = _load_sched(next_log)
                # CAS-like guard: only set when still pointing to our task
                if next_sched.get("prev_task") == task_id:
                    next_sched.pop("start_at", None)
                    _update_schedule(next_log, next_sched)


def attach_with_links(
    scheduler: "TaskScheduler",
    *,
    task_id: int,
    prev_task: Optional[int],
    next_task: Optional[int],
    head_start_at: Optional[str],
    err_prefix: str,
) -> None:
    """Attach a task into the runnable queue and update neighbours symmetrically."""
    _q_attach_with_links(
        scheduler,
        task_id=task_id,
        prev_task=prev_task,
        next_task=next_task,
        head_start_at=head_start_at,
        err_prefix=err_prefix,
    )
