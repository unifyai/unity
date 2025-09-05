from __future__ import annotations

from typing import Any, Dict, List, Optional

from .types.status import Status


def _to_status(value: Any) -> Status:
    if isinstance(value, Status):
        return value
    try:
        return Status(value)
    except Exception:
        # Fallback: treat unknown as queued to avoid raising inside planning
        return Status.queued


def _sched_prev(sched: Any) -> Optional[int]:
    if sched is None:
        return None
    if isinstance(sched, dict):
        return sched.get("prev_task")
    return getattr(sched, "prev_task", None)


def _sched_start_at(sched: Any) -> Optional[str]:
    if sched is None:
        return None
    if isinstance(sched, dict):
        return sched.get("start_at")
    # Pydantic model: accept datetime/str and leave conversion to caller
    try:
        return getattr(sched, "start_at", None)
    except Exception:
        return None


def derive_status_after_queue_edit(
    *,
    existing_status: Any,
    is_head: bool,
    head_has_start_at: bool,
) -> Status:
    """
    Determine the desired status after a queue reorder for a single task.

    Behaviour mirrors the existing TaskScheduler implementation:
    - Preserve "active" as-is.
    - Head with a start_at timestamp becomes "scheduled".
    - Otherwise, keep the existing status except that a previously "scheduled"
      non-head is downgraded to "queued".
    """
    current = _to_status(existing_status)
    if current == Status.active:
        return Status.active
    if is_head and head_has_start_at:
        return Status.scheduled
    # Non-heads must not remain scheduled; downgrade to queued
    if not is_head and current == Status.scheduled:
        return Status.queued
    return current


def plan_reorder_queue(
    *,
    new_order: List[int],
    rows_by_id: Dict[int, Dict[str, Any]],
    queue_id: Optional[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Compute the set of invariant-preserving updates required to reorder a queue
    to exactly match ``new_order``.

    Inputs are read-only. The function returns a mapping of ``task_id → entries``
    where each entries dict contains a ``schedule`` payload and (when required)
    a ``status`` value. Callers should apply these via the scheduler's single
    write funnel to guarantee validation and neighbour symmetry.
    """
    # Determine the queue-level timestamp from the current head (if any)
    queue_start_ts: Optional[str] = None
    try:
        for r in rows_by_id.values():
            sched = r.get("schedule") or {}
            if _sched_prev(sched) is None:
                ts = _sched_start_at(sched)
                if ts is not None:
                    queue_start_ts = ts
                    break
    except Exception:
        queue_start_ts = None

    updates: Dict[int, Dict[str, Any]] = {}

    for idx, tid in enumerate(new_order):
        prev_tid = None if idx == 0 else new_order[idx - 1]
        next_tid = None if idx == len(new_order) - 1 else new_order[idx + 1]
        start_ts = queue_start_ts if idx == 0 else None

        sched_payload: Dict[str, Any] = {
            "prev_task": prev_tid,
            "next_task": next_tid,
            "queue_id": queue_id,
        }
        if start_ts is not None:
            sched_payload["start_at"] = start_ts

        existing_status = rows_by_id.get(tid, {}).get("status", Status.queued)
        desired_status = derive_status_after_queue_edit(
            existing_status=existing_status,
            is_head=(idx == 0),
            head_has_start_at=(start_ts is not None),
        )
        # Non-head tasks can never remain primed – downgrade to queued
        if idx != 0 and desired_status == Status.primed:
            desired_status = Status.queued

        payload: Dict[str, Any] = {"schedule": sched_payload}
        if _to_status(existing_status) != desired_status:
            payload["status"] = desired_status

        updates[tid] = payload

    return updates
