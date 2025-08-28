from __future__ import annotations

from typing import Optional, Dict, Any, Union, TYPE_CHECKING

import unify

from .types.schedule import Schedule

if TYPE_CHECKING:
    from .task_scheduler import TaskScheduler


# ---------------------------------------------------------------------------- #
#  Queue/linkage helpers (private module)                                      #
# ---------------------------------------------------------------------------- #


def sched_prev(sched: Union[Schedule, dict, None]) -> Optional[int]:
    """Return prev_task from a Schedule-like value."""
    if sched is None:
        return None
    if isinstance(sched, dict):
        return sched.get("prev_task")
    # assume pydantic Schedule
    return getattr(sched, "prev_task", None)


def sched_next(sched: Union[Schedule, dict, None]) -> Optional[int]:
    """Return next_task from a Schedule-like value."""
    if sched is None:
        return None
    if isinstance(sched, dict):
        return sched.get("next_task")
    return getattr(sched, "next_task", None)


def sync_adjacent_links(
    scheduler: "TaskScheduler",
    *,
    task_id: int,
    schedule: Optional[Union[Schedule, dict]],
) -> None:
    """
    Guarantee link symmetry when (re)linking a task in the runnable queue.
    """
    if schedule is None:
        return

    if isinstance(schedule, Schedule):
        schedule = schedule.model_dump()

    neighbours: list[tuple[str, str, int]] = []
    if schedule.get("prev_task") is not None:
        neighbours.append(("next_task", "prev_task", schedule["prev_task"]))
    if schedule.get("next_task") is not None:
        neighbours.append(("prev_task", "next_task", schedule["next_task"]))

    for field_to_set, _, neighbour_id in neighbours:
        rows = scheduler._filter_tasks(filter=f"task_id == {neighbour_id}", limit=1)
        if not rows:
            # Neighbour went missing – skip symmetric update instead of failing
            continue

        row = rows[0]
        n_sched = {**(row.get("schedule") or {})}
        if n_sched.get(field_to_set) == task_id:
            continue  # already correct

        # Strip start_at if the neighbour ceases to be queue head
        if field_to_set == "prev_task":
            n_sched.pop("start_at", None)

        n_sched[field_to_set] = task_id
        try:
            log_id = scheduler._get_logs_by_task_ids(task_ids=row["task_id"])
        except ValueError:
            # Neighbour was deleted after we fetched rows – skip
            continue
        unify.update_logs(
            logs=log_id,
            context=scheduler._ctx,
            entries={"schedule": n_sched},
            overwrite=True,
        )

        # Was the neighbour the *primed* task?  Keep cache in lock-step.
        if (
            scheduler._primed_task is not None
            and scheduler._primed_task["task_id"] == neighbour_id
        ):
            scheduler._refresh_primed_cache(neighbour_id)


def attach_with_links(
    scheduler: "TaskScheduler",
    *,
    task_id: int,
    prev_task: Optional[int],
    next_task: Optional[int],
    head_start_at: Optional[str],
    err_prefix: str,
) -> None:
    """
    Attach a task into the runnable queue between prev_task and next_task and
    enforce start_at placement on the head only. Updates neighbour pointers
    symmetrically and validates invariants for the resulting schedule.
    """

    def _get_log_obj(tid_int: int) -> Optional[unify.Log]:
        """
        Best-effort fetch of a neighbour's log object.

        Returns None when the referenced task no longer exists, instead of
        raising, so callers can gracefully skip neighbour updates.
        """
        try:
            rows = scheduler._get_logs_by_task_ids(
                task_ids=tid_int,
                return_ids_only=False,
            )
        except ValueError:
            # Neighbour was deleted or does not exist in the current context
            return None
        if not rows:
            return None
        assert len(rows) == 1, "Task IDs should be unique"
        return rows[0]  # type: ignore[return-value]

    # Update neighbours first
    if prev_task is not None:
        prev_log = _get_log_obj(prev_task)
        if prev_log is not None:
            prev_sched = {
                **((getattr(prev_log, "entries", {}) or {}).get("schedule") or {}),
            }
            prev_sched["next_task"] = task_id
            unify.update_logs(
                logs=prev_log.id if hasattr(prev_log, "id") else prev_log,
                context=scheduler._ctx,
                entries={"schedule": prev_sched},
                overwrite=True,
            )

    if next_task is not None:
        next_log = _get_log_obj(next_task)
        if next_log is not None:
            next_sched = {
                **((getattr(next_log, "entries", {}) or {}).get("schedule") or {}),
            }
            next_sched["prev_task"] = task_id
            # If we are restoring head-level start_at back to current task, strip from next
            if head_start_at is not None:
                next_sched.pop("start_at", None)
            unify.update_logs(
                logs=next_log.id if hasattr(next_log, "id") else next_log,
                context=scheduler._ctx,
                entries={"schedule": next_sched},
                overwrite=True,
            )

    # Build current task schedule and write via validated funnel
    cur_sched: Dict[str, Any] = {"prev_task": prev_task, "next_task": next_task}
    # Carry through queue_id onto the head when known (non-breaking additive field)
    try:
        from .types.reintegration_plan import ReintegrationPlan as _RP  # local import

        # We cannot access the exact plan here reliably; rely on `head_start_at` signal.
        # If reinstating at head with a known queue id on neighbour, propagate it.
        if prev_task is None and head_start_at is not None:
            # Probe next_task for queue_id if present
            if next_task is not None:
                nxt_rows = scheduler._filter_tasks(
                    filter=f"task_id == {next_task}",
                    limit=1,
                )
                if nxt_rows:
                    qid = (nxt_rows[0].get("schedule") or {}).get("queue_id")
                    if qid is not None:
                        cur_sched["queue_id"] = qid
    except Exception:
        pass
    if prev_task is None and head_start_at is not None:
        cur_sched["start_at"] = head_start_at
    scheduler._validated_write(
        task_id=task_id,
        entries={"schedule": cur_sched},
        err_prefix=err_prefix,
    )
