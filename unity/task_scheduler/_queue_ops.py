from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING

import unify

from ._queue_utils import (
    sched_prev as _q_prev,
    sched_next as _q_next,
    attach_with_links as _q_attach_with_links,
)
from .types.reintegration_plan import ReintegrationPlan

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler


"""
This module previously exposed a `get_task_queue` helper used by
`TaskScheduler._get_task_queue`. The scheduler now provides explicit helpers:
`_get_queue(queue_id=...)`, `_walk_queue_from_task(task_id=...)`, and
`_get_queue_for_task(task_id=...)`. The generic traversal is kept within the
scheduler to ensure a single invariant funnel for reads and writes.
"""


def detach_from_queue_for_activation(
    scheduler: "TaskScheduler",
    *,
    task_id: int,
    detach: bool = True,
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
    - Chained activation: keep the queue behind the
      activated task attached to it. When promoting the activated task to head,
      place any head-level ``start_at`` on it (use the previous head's timestamp
      when the current task did not have one) and remove ``start_at`` from the
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

    # Compute head_start_at with at most one backend read.
    # Fast path: when current task is the head, reuse its own start_at.
    head_start_at: Optional[str] = None
    if prev_tid is None:
        head_start_at = start_at
    else:
        # Prefer LocalTaskView for a single-step head start_at resolution.
        try:
            _qid = task_row.get("queue_id")
        except Exception:
            _qid = None
        if isinstance(_qid, int):
            try:
                head_start_at = scheduler._view.get_head_start_at(int(_qid))  # type: ignore[attr-defined]
            except Exception:
                head_start_at = None
        # Fallbacks when queue_id is missing or the local view returned nothing
        if head_start_at is None:
            # Fallback: walk prev pointers (rare case when queue_id is absent)
            cur_head = _get_row(task_id)
            while (
                cur_head is not None and _q_prev(cur_head.get("schedule")) is not None
            ):
                cur_head = _get_row(_q_prev(cur_head.get("schedule")))
            if cur_head is not None:
                _sched_head = cur_head.get("schedule") or {}
                if isinstance(_sched_head, dict):
                    head_start_at = _sched_head.get("start_at")

    # Batch-fetch log objects for all relevant task_ids in one backend call
    def _build_log_cache(ids: list[int]) -> Dict[int, unify.Log]:
        cache: Dict[int, unify.Log] = {}
        if not ids:
            return cache
        try:
            logs = scheduler._view.get_log_ids_by_task_ids(  # type: ignore[attr-defined]
                task_ids=ids,
                return_ids_only=False,
            )
        except Exception:
            logs = []
        for lg in logs or []:
            try:
                entries = getattr(lg, "entries", {}) or {}
                tid = entries.get("task_id")
                if isinstance(tid, int):
                    cache[int(tid)] = lg
            except Exception:
                continue
        return cache

    needed_ids: list[int] = []
    for _tid in (task_id, prev_tid, next_tid):
        try:
            if isinstance(_tid, int):
                needed_ids.append(int(_tid))
        except Exception:
            continue
    _log_cache = _build_log_cache(needed_ids)

    def _get_log_obj(tid: int) -> Optional[unify.Log]:
        if not isinstance(tid, int):
            return None
        lg = _log_cache.get(int(tid))
        if lg is not None:
            return lg
        # Defensive fallback (should be rare within this call)
        try:
            logs = scheduler._view.get_log_ids_by_task_ids(  # type: ignore[attr-defined]
                task_ids=int(tid),
                return_ids_only=False,
            )
        except Exception:
            return None
        return logs[0] if logs else None  # type: ignore[return-value]

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
        scheduler._view.write_entries(  # type: ignore[attr-defined]
            logs=_log_id(log_or_id),
            entries=entries,
            overwrite=True,
        )

    # Always record a reintegration plan for precise restore on defer stop,
    # regardless of execution scope. This enables queue execution with later
    # reinstatement to the original position when requested.
    # Capture current queue_id from the row (top-level), never from schedule
    try:
        queue_id = task_row.get("queue_id")
    except Exception:
        queue_id = None

    plan = ReintegrationPlan(
        task_id=task_id,
        instance_id=task_row.get("instance_id"),
        prev_task=prev_tid,
        next_task=next_tid,
        start_at=start_at,
        was_head=prev_tid is None,
        original_status=task_row.get("status"),
        head_start_at=head_start_at,
        queue_id=queue_id,
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

    if detach:
        # ----- Isolation semantics -----
        if prev_tid is None:
            # Detaching the head: successor becomes head and inherits head-level start_at
            if next_tid is not None:
                next_log = _get_log_obj(next_tid)
                if next_log is not None:
                    next_sched = _load_sched(next_log)
                    next_sched["prev_task"] = None
                    # Preserve existing next linkage; remove any stale start_at first
                    next_sched.pop("start_at", None)
                    if head_start_at is not None:
                        next_sched["start_at"] = head_start_at
                    # Merge status promotion into the same write to avoid an extra backend call
                    _update_schedule(
                        next_log,
                        next_sched,
                        extra=(
                            {"status": "scheduled"}
                            if head_start_at is not None
                            else None
                        ),
                    )
            # Clear schedule on the detached task entirely (isolated)
            cur_log = _get_log_obj(task_id)
            _update_schedule(cur_log, {}, extra={"schedule": None})
        else:
            # Middle task: unlink from neighbours
            if prev_tid is not None:
                prev_log = _get_log_obj(prev_tid)
                if prev_log is not None:
                    prev_sched = _load_sched(prev_log)
                    if prev_sched.get("next_task") == task_id:
                        prev_sched["next_task"] = next_tid
                        _update_schedule(prev_log, prev_sched)
            if next_tid is not None:
                next_log = _get_log_obj(next_tid)
                if next_log is not None:
                    next_sched = _load_sched(next_log)
                    if next_sched.get("prev_task") == task_id:
                        next_sched["prev_task"] = prev_tid
                        # Non-head must not carry start_at
                        next_sched.pop("start_at", None)
                        _update_schedule(next_log, next_sched)
            # Clear schedule on the detached task
            cur_log = _get_log_obj(task_id)
            _update_schedule(cur_log, {}, extra={"schedule": None})
    else:
        # ----- Chained queue execution semantics -----
        # Disconnect previous neighbour's next pointer when promoting current task to head
        if prev_tid is not None:
            prev_log = _get_log_obj(prev_tid)
            if prev_log is not None:
                prev_sched = _load_sched(prev_log)
                if prev_sched.get("next_task") == task_id:
                    prev_sched["next_task"] = None
                    _update_schedule(prev_log, prev_sched)

        if sched is not None:
            # Promote current task to head and keep followers attached
            cur_log = _get_log_obj(task_id)
            new_sched: Dict[str, Any] = {"prev_task": None, "next_task": next_tid}
            # Move queue-level start_at to the new head: prefer own start_at, else head's
            eff_start_at = start_at if start_at is not None else head_start_at
            if eff_start_at is not None:
                new_sched["start_at"] = eff_start_at
            _update_schedule(cur_log, new_sched)

            if next_tid is not None:
                next_log = _get_log_obj(next_tid)
                if next_log is not None:
                    next_sched = _load_sched(next_log)
                    # CAS-like guard: only set when still pointing to our task
                    if next_sched.get("prev_task") == task_id:
                        next_sched.pop("start_at", None)
                        _update_schedule(next_log, next_sched)

    # Signal linkage barrier: create or set an event for this task_id
    try:
        import asyncio as _aio  # local import to avoid global dependency

        ev = getattr(scheduler, "_linkage_barriers", {}).get(task_id)
        if ev is None:
            # Create and store a new event
            ev = _aio.Event()
            try:
                scheduler._linkage_barriers[task_id] = ev  # type: ignore[attr-defined]
            except Exception:
                pass
        # Set the event to signal completion
        try:
            ev.set()
        except Exception:
            pass
    except Exception:
        # Never let signalling break detachment
        pass


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
