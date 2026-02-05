"""
Queue execution handle for running a chain of tasks head→tail.

ActiveQueue sequences tasks using the live queue order, adopting each task's
steerable handle in turn. It:
- Routes interjections to current and future tasks with an LLM-based router,
  queuing messages for later delivery when needed.
- Provides queue-aware ask() by prepending a compact chain status and task list.
- Emits per-task completion events and a final chain summary.
- Uses direct delegation when the queue is a singleton to preserve the inner handle's
  behavior and timing characteristics.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, List, Any, TYPE_CHECKING
import json

from ..settings import SETTINGS
from ..common.llm_client import new_llm_client
from ..common.async_tool_loop import SteerableToolHandle
from ..common.handle_wrappers import HandleWrapperMixin
from .types.activated_by import ActivatedBy
from .types.status import to_status

if TYPE_CHECKING:  # avoid import cycles at runtime
    from .task_scheduler import TaskScheduler


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers (extracted to reduce ActiveQueue size, same-file & private)
# ─────────────────────────────────────────────────────────────────────────────


class _InterjectionRouter:
    @staticmethod
    async def route(
        *,
        queue_rows: list[dict],
        message: str,
        current_task_id: int,
    ) -> tuple[list[dict], bool]:
        """Return (routes, uncovered_flag) using a dedicated LLM call with timeout.

        Mirrors the previous `_route_interjection_llm` logic verbatim so that
        behaviour remains unchanged while keeping `ActiveQueue` concise.
        """
        try:

            def _safe_dump(value):
                try:
                    return json.dumps(value, default=str)
                except Exception:
                    return str(value)

            client = new_llm_client()
            schema_hint = '{\n  "type": "object",\n  "properties": {\n    "routes": {\n      "type": "array",\n      "items": {\n        "type": "object",\n        "properties": {\n          "task_ids": {"type": "array", "items": {"type": "integer"}},\n          "instruction": {"type": "string"}\n        },\n        "required": ["task_ids", "instruction"]\n      }\n    },\n    "directives": {\n      "type": "array",\n      "items": {\n        "type": "object",\n        "properties": {\n          "kind": {"type": "string", "enum": ["all", "first", "last", "by_description"]},\n          "description_match": {"type": "string"}\n        },\n        "required": ["kind"]\n      }\n    },\n    "uncovered_directives": {"type": "array", "items": {"type": "string"}}\n  },\n  "required": ["routes"]\n}'
            sys = (
                "You route user interjections to one or more tasks in a queue.\n"
                "Return ONLY JSON matching the schema below. Never include commentary.\n"
                f"Schema:\n{schema_hint}\n"
                "Guidelines: Select task_ids explicitly from the provided queue.\n"
                "- If the instruction applies to all tasks, include all task_ids.\n"
                "- If it targets the last task, include ONLY the last task_id.\n"
                "- If it mentions a task by name/description, choose the best matching ids.\n"
                "- If nothing special is implied, target ONLY the current task.\n"
                "- You MUST include a separate route for each distinct directive present in the user's message; list these under 'directives' and set 'uncovered_directives' to [] when all are mapped.\n"
                "Ambiguity & clarification policy:\n"
                "- Do NOT guess. When the instruction is ambiguous or underspecified (e.g., phrases like 'the rest', 'later', 'soon',\n"
                "  conflicting directives, or missing explicit task_ids/clear directives), mark those items under 'uncovered_directives'.\n"
                "- Only include unambiguous routes in 'routes'. If nothing can be routed unambiguously, return routes: [].\n"
                "- Examples of ambiguity that MUST produce non-empty 'uncovered_directives':\n"
                "  'do the rest later', 'maybe the last one unless it's urgent', 'whichever is best',\n"
                "  or any directive that cannot be mapped deterministically to concrete task_ids.\n"
            )
            client.set_system_message(sys)

            try:
                first_task_id: int | None = None
                last_task_id: int | None = None
                if queue_rows:
                    first_task_id = (
                        int(queue_rows[0].get("task_id"))
                        if queue_rows[0].get("task_id") is not None
                        else None
                    )
                    last_task_id = (
                        int(queue_rows[-1].get("task_id"))
                        if queue_rows[-1].get("task_id") is not None
                        else None
                    )
            except Exception:
                first_task_id = None
                last_task_id = None

            user = (
                "Chain (head→tail):\n"
                + _safe_dump(queue_rows)
                + "\nMetadata:\n"
                + f"first_task_id: {first_task_id}\n"
                + f"last_task_id: {last_task_id}\n"
                + "current_task_id: "
                + str(current_task_id)
                + "\nInterjection:"
                + f"\n{(message or '').strip()}"
            )

            timeout_s = SETTINGS.task.ROUTER_TIMEOUT_SECONDS

            try:
                raw = await asyncio.wait_for(client.generate(user), timeout=timeout_s)
            except asyncio.TimeoutError:
                raw = ""

            try:
                data = json.loads(raw)
            except Exception:
                return [], True

            routes = data.get("routes") if isinstance(data, dict) else None
            if not isinstance(routes, list):
                return [], True
            uncovered = data.get("uncovered_directives") or []
            uncovered_flag = bool(isinstance(uncovered, list) and uncovered)
            # Normalise routes: cast ids to int, drop unknown ids and empty instructions
            known_ids = {
                int(r.get("task_id"))
                for r in queue_rows
                if r.get("task_id") is not None
            }
            norm_routes: list[dict] = []
            try:
                for r in routes:
                    instr = str(r.get("instruction", "")).strip()
                    if not instr:
                        continue
                    ids_raw = r.get("task_ids", [])
                    ids_int: list[int] = []
                    for t in ids_raw:
                        try:
                            tid = int(t)
                            if tid in known_ids:
                                ids_int.append(tid)
                        except Exception:
                            continue
                    if ids_int:
                        norm_routes.append({"task_ids": ids_int, "instruction": instr})
            except Exception:
                norm_routes = []
            return norm_routes, uncovered_flag
        except Exception:
            return [], True


class _QueueSnapshot:
    @staticmethod
    def build_rows(scheduler: "TaskScheduler", current_task_id: int) -> list[dict]:
        """Build a compact queue snapshot (head→tail) using the scheduler's live view."""
        try:
            queue = scheduler._get_queue_for_task(task_id=current_task_id)
        except Exception:
            queue = []
        out: list[dict] = []
        for t in queue:
            out.append(
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "description": t.description,
                    "status": t.status,
                    "schedule": t.schedule,
                },
            )
        return out

    @staticmethod
    def build_preamble(
        scheduler: "TaskScheduler",
        current_task_id: int,
    ) -> Optional[str]:
        """Return a concise queue preamble (headline + task list) or None on failure."""
        try:
            queue_rows: list[dict] = _QueueSnapshot.build_rows(
                scheduler,
                current_task_id,
            )

            # Prefer a full chain snapshot (including terminal predecessors) for
            # progress math, while keeping the details list runnable-only.
            try:
                _full_chain = (
                    scheduler._walk_queue_from_task(  # type: ignore[attr-defined]
                        task_id=current_task_id,
                    )
                )
            except Exception:
                _full_chain = []

            full_rows: list[dict] = []
            for t in _full_chain:
                full_rows.append(
                    {
                        "task_id": t.task_id,
                        "name": t.name,
                        "description": t.description,
                        "status": t.status,
                        "schedule": t.schedule,
                    },
                )

            # Compute indices against both views
            total_count_runnable = len(queue_rows)
            total_count_full = len(full_rows) if full_rows else total_count_runnable

            # Identify current index in each view
            current_index_runnable = -1
            for idx, r in enumerate(queue_rows):
                if r.get("task_id") == current_task_id:
                    current_index_runnable = idx
                    break
            current_index_full = -1
            if full_rows:
                for idx, r in enumerate(full_rows):
                    if r.get("task_id") == current_task_id:
                        current_index_full = idx
                        break

            # Count statuses
            def _to_status_str(row: dict) -> str:
                try:
                    return str(to_status(row.get("status")))
                except Exception:
                    return str(row.get("status"))

            # Use full chain (when available) to count completed predecessors and
            # compute remaining/position; fall back to runnable-only otherwise.
            if full_rows:
                completed_count = sum(
                    1 for r in full_rows if _to_status_str(r) == "completed"
                )
                total_count = total_count_full
                remaining_count = max(0, total_count - completed_count)
                current_index = current_index_full
            else:
                completed_count = sum(
                    1 for r in queue_rows if _to_status_str(r) == "completed"
                )
                total_count = total_count_runnable
                remaining_count = max(0, total_count - completed_count)
                current_index = current_index_runnable

            # Next tasks preview (up to 3) – derive from runnable view to avoid index drift
            next_names: list[str] = []
            if current_index_runnable >= 0 and total_count_runnable > 0:
                for j in range(
                    current_index_runnable + 1,
                    min(current_index_runnable + 4, total_count_runnable),
                ):
                    try:
                        nm = queue_rows[j].get("name")
                    except Exception:
                        nm = None
                    if nm:
                        next_names.append(str(nm))

            # High-level summary line
            if current_index >= 0 and current_index < total_count:
                # Use runnable view for the display name to ensure the active task is referenced
                if 0 <= current_index_runnable < len(queue_rows):
                    current_name = (
                        queue_rows[current_index_runnable].get("name")
                        or "(unnamed task)"
                    )
                elif full_rows and 0 <= current_index_full < len(full_rows):
                    current_name = (
                        full_rows[current_index_full].get("name") or "(unnamed task)"
                    )
                else:
                    current_name = "(unnamed task)"
                # Position uses full-chain index when available (e.g., 2/4), else runnable
                current_pos = (
                    (current_index_full + 1)
                    if (full_rows and current_index_full >= 0)
                    else (current_index_runnable + 1)
                )
                headline = (
                    f"Chain status: {completed_count}/{total_count} completed; "
                    f"{remaining_count} remaining; executing {current_pos}/{total_count}: "
                    f"{current_name}."
                )
            else:
                headline = (
                    f"Chain status: {completed_count}/{total_count} completed; "
                    f"{remaining_count} remaining; executing: (unknown current index)."
                )
            if next_names:
                headline += f" Next: {', '.join(next_names)}."

            # Essential rows: only id and name to keep preamble concise
            details_lines: list[str] = ["Chain tasks (head→tail):"]
            for r in queue_rows:
                tid = r.get("task_id")
                name = r.get("name") or ""
                details_lines.append(f"- Task {tid}: {name}")

            return "CHAIN CONTEXT\n" + headline + "\n" + "\n".join(details_lines)
        except Exception:
            return None


class ActiveQueue(SteerableToolHandle, HandleWrapperMixin):  # type: ignore[abstract-method]
    def __init__(
        self,
        scheduler: "TaskScheduler",
        *,
        first_task_id: int,
        first_handle: SteerableToolHandle,
        parent_chat_context: Optional[List[Dict[str, Any]]],
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
    ) -> None:
        self._s = scheduler
        self._current_task_id = first_task_id
        self._current_handle: SteerableToolHandle = first_handle
        self._parent_ctx = parent_chat_context
        self._clar_up = clarification_up_q
        self._clar_down = clarification_down_q
        self._done_evt: asyncio.Event = asyncio.Event()
        self._final_result: Optional[str] = None
        # Track tasks that completed successfully within this queue run
        self._completed_tasks: list[tuple[int, str]] = []
        # Stream of per-task completion events (name → result text)
        self._completions: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        # Queue-level universal notification stream (dict events)
        self._notif_q: asyncio.Queue[dict] = asyncio.Queue()
        # Sticky direct-delegation flag: enabled only when the queue truly contains a
        # single task at creation time; once disabled it never re-enables for the
        # lifetime of this ActiveQueue instance.
        try:
            initial_q = self._s._get_queue_for_task(task_id=self._current_task_id)
            size = len(initial_q) if initial_q is not None else 0
            # Treat isolated/detached (no queue membership) or true singleton as direct delegation
            self._direct_delegation_enabled: bool = size <= 1
        except Exception:
            # Fallback to direct delegation to preserve inner handle semantics in ambiguous cases
            self._direct_delegation_enabled = True

        # Background driver
        self._driver = asyncio.create_task(self._drive())

    # Standardized wrapper registration: always expose the current inner handle
    def _get_wrapped_handles(self):  # type: ignore[override]
        try:
            return {"current": self._current_handle}
        except Exception:
            return []

    # ----------------------------
    # Small summary helper
    # ----------------------------
    def _summarise_completions(self) -> str:
        if self._completed_tasks:
            summary_items = [
                f"Task {tid}: {name}" for tid, name in self._completed_tasks
            ]
            return "Completed the following tasks: " + ", ".join(summary_items) + "."
        return "Chain completed."

    # ----------------------------
    # Notification helper
    # ----------------------------
    def _emit_notification(self, event: dict) -> None:
        try:
            self._notif_q.put_nowait(event)
        except Exception:
            try:
                asyncio.create_task(self._notif_q.put(event))
            except Exception:
                pass

    # ----------------------------
    # Snapshot helper (head→tail)
    # ----------------------------
    def _build_queue_rows_snapshot(self) -> list[dict]:
        # Backwards-compat wrapper while callers migrate; delegates to helper
        return _QueueSnapshot.build_rows(self._s, self._current_task_id)

    # ----------------------------
    # Internal clarification tool
    # ----------------------------
    async def _request_clarification(
        self,
        question: str,
    ) -> Optional[str]:
        """
        Queue-level clarification for internal use by ActiveQueue.

        Uses the same clarification queues that inner tasks receive. When queues
        are not provided, this becomes a no-op and returns None, allowing the
        queue orchestrator to proceed with conservative defaults.
        """
        try:
            # Only operate when both channels are available
            if self._clar_up is None or self._clar_down is None:
                return None
            # Enqueue question; await answer (blocking semantics by design)
            try:
                self._clar_up.put_nowait(question)
            except Exception:
                await self._clar_up.put(question)
            try:
                ans = await self._clar_down.get()
            except Exception:
                ans = None
            return ans
        except Exception:
            # Defensive: clarification should never break the queue
            return None

    # ----------------------------
    # Direct delegation helper methods
    # ----------------------------
    def _current_queue_size(self) -> int:
        try:
            q = self._s._get_queue_for_task(task_id=self._current_task_id)
            # When the current task is no longer a member of any queue (isolated/detached),
            # treat the queue as a singleton for direct delegation purposes.
            try:
                contains_current = any(t.task_id == self._current_task_id for t in q)
            except Exception:
                contains_current = True
            if not contains_current:
                return 1
            return len(q)
        except Exception:
            return 0

    def _should_delegate_directly(self) -> bool:
        """Return True when we should directly delegate to the inner handle.

        This is allowed only while the queue remains a true singleton. The
        moment an additional task appears in the queue (size > 1), we
        permanently disable direct delegation for the lifetime of this instance.
        """
        if not getattr(self, "_direct_delegation_enabled", False):
            return False
        # If at any point the queue grows beyond a single task, flip sticky off.
        size = self._current_queue_size()
        if size > 1:
            self._direct_delegation_enabled = False
            return False
        return True

    def _next_runnable_follower(self) -> Optional[int]:
        """Return the next runnable task id after the current one based on the live queue.

        Fallback behaviour: if the current task is no longer part of the runnable
        queue (e.g., it just completed and runnable views exclude it), consult the
        current task's stored ``schedule.next_task`` to identify the follower.
        """
        try:
            live_queue = self._s._get_queue_for_task(task_id=self._current_task_id)
        except Exception:
            live_queue = []

        ids: list[int] = [t.task_id for t in live_queue]

        if not ids:
            return None

        cur_id = None
        try:
            cur_id = int(self._current_task_id)
        except Exception:
            cur_id = None

        # If current id is not found (e.g., task just completed and is excluded from
        # the live runnable view), fall back to the stored next pointer on the row.
        try:
            if cur_id is None:
                return None
            idx = ids.index(cur_id)
        except ValueError:
            # Fallback: read the current row and follow its schedule.next_task
            try:
                rows = self._s._filter_tasks(
                    filter=f"task_id == {int(self._current_task_id)}",
                    limit=1,
                )
                if not rows:
                    return None
                nxt = rows[0].schedule_next
                try:
                    nxt_int = int(nxt) if nxt is not None else None
                except Exception:
                    nxt_int = None  # type: ignore[assignment]
                # Prefer returning a follower that is present in the runnable view; otherwise
                # return the pointer as-is and let the callee handle missing rows defensively.
                if nxt_int is None:
                    return None
                return nxt_int if (nxt_int in ids or ids == []) else nxt_int
            except Exception:
                return None

        # Return the first follower, if any
        return ids[idx + 1] if (idx + 1) < len(ids) else None

    async def _drive(self) -> None:
        try:
            while True:
                try:
                    result = await self._current_handle.result()
                except asyncio.CancelledError:
                    self._final_result = "Stopped."
                    break

                text = str(result or "")
                # If ActiveTask flagged an explicit stop/defer, end the queue immediately
                try:
                    was_stopped = bool(
                        getattr(self._current_handle, "_was_stopped", False),
                    )
                except Exception:
                    was_stopped = False
                # Record successful completion of the just-finished task when not stopped/deferred
                if not was_stopped and "stopped" not in text.lower():
                    try:
                        rows = self._s._filter_tasks(
                            filter=f"task_id == {int(self._current_task_id)}",
                            limit=1,
                        )
                        if rows:
                            name = (
                                rows[0].name or rows[0].description or "(unnamed task)"
                            )
                            # Emit a standardized completion notification
                            try:
                                evt_completed = {
                                    "type": "queue.task.completed",
                                    "task_id": int(self._current_task_id),
                                    "name": str(name),
                                    "instance_id": rows[0].instance_id,
                                    "queue_id": rows[0].queue_id,
                                    "result": text,
                                }
                                self._emit_notification(evt_completed)
                            except Exception:
                                pass
                            self._completed_tasks.append(
                                (int(self._current_task_id), str(name)),
                            )
                            # Emit completion event non-blockingly for active_task_done()
                            evt = {"name": str(name), "result": text}
                            try:
                                self._completions.put_nowait(evt)
                            except Exception:
                                asyncio.create_task(self._completions.put(evt))
                    except Exception:
                        pass
                if was_stopped:
                    # Emit a stop/defer notification for the current task
                    try:
                        intent = getattr(self._current_handle, "_last_intent", None)
                        reason = getattr(
                            self._current_handle,
                            "_last_intent_reason",
                            None,
                        )
                    except Exception:
                        intent, reason = None, None
                    try:
                        self._emit_notification(
                            {
                                "type": "queue.task.stopped",
                                "task_id": int(self._current_task_id),
                                "intent": intent,
                                "reason": reason,
                            },
                        )
                    except Exception:
                        pass
                    self._final_result = text or "Stopped."
                    break
                # If the stop/defer path was taken, end the queue here
                if "stopped" in text.lower():
                    try:
                        self._emit_notification(
                            {
                                "type": "queue.task.stopped",
                                "task_id": int(self._current_task_id),
                                "intent": None,
                                "reason": text,
                            },
                        )
                    except Exception:
                        pass
                    self._final_result = text
                    break
                # Determine the next task to run using the live queue only
                next_tid: Optional[int] = self._next_runnable_follower()

                if next_tid is None:
                    # Queue exhausted – compose a completion summary across all tasks
                    self._final_result = self._summarise_completions()
                    try:
                        self._emit_notification(
                            {
                                "type": "queue.completed",
                                "summary": self._final_result,
                                "completed": [
                                    {"task_id": tid, "name": name}
                                    for tid, name in self._completed_tasks
                                ],
                            },
                        )
                    except Exception:
                        pass
                    break

                # Start next task using CHAIN linkage semantics
                self._current_task_id = next_tid
                # Emit a standardized started notification
                try:
                    # Best-effort fetch for name/metadata
                    _rows = self._s._filter_tasks(
                        filter=f"task_id == {int(next_tid)}",
                        limit=1,
                    )
                    _nm = _rows[0].name if _rows and _rows[0].name is not None else None
                    self._emit_notification(
                        {
                            "type": "queue.task.started",
                            "task_id": int(next_tid),
                            "name": _nm,
                            "queue_id": (_rows[0].queue_id if _rows else None),
                            "instance_id": (_rows[0].instance_id if _rows else None),
                        },
                    )
                except Exception:
                    pass
                self._current_handle = await self._s._execute_internal(
                    task_id=next_tid,
                    parent_chat_context=self._parent_ctx,
                    clarification_up_q=self._clar_up,
                    clarification_down_q=self._clar_down,
                    activated_by=ActivatedBy.explicit,
                    # Do NOT detach followers from each other; keep queue links intact
                    detach=False,
                )
                # Deliver any queued interjections for the newly active task
                try:
                    pending_msgs = getattr(self, "_queued_interjections", {}).pop(
                        self._current_task_id,
                        [],
                    )
                    for _item in pending_msgs:
                        try:
                            if isinstance(_item, dict):
                                _m = _item.get("message", "")
                            else:
                                _m = _item
                            await self._current_handle.interject(_m)
                        except Exception:
                            pass
                except Exception:
                    pass
        finally:
            self._done_evt.set()
            # active_task_done() awaits on the completions queue

    # ----- Steerable surface proxies -----
    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:  # type: ignore[override]
        """Route interjections to specific tasks in the queue using an LLM router.

        The router receives the full queue snapshot and the user's instruction and
        returns structured routes of the form:

            {"routes": [{"task_ids": [<int>, ...], "instruction": "..."}, ...]}

        Interjections for the currently active task are delivered immediately.
        Interjections for future tasks are queued and delivered when those
        tasks become active.
        """

        # Direct delegation when this is a true singleton queue (sticky while true)
        if self._should_delegate_directly():
            if not (message or "").strip():
                return
            await self._current_handle.interject(
                message,
                _parent_chat_context_cont=_parent_chat_context_cont,
            )
            return

        # Fast path: empty/whitespace → no-op
        if not (message or "").strip():
            return

        # Always use the LLM router for multi-task routing

        # Perform routing via helper; keep main method small
        queue_rows: list[dict] = _QueueSnapshot.build_rows(
            self._s,
            self._current_task_id,
        )

        routes, uncovered = await _InterjectionRouter.route(
            queue_rows=queue_rows,
            message=message,
            current_task_id=self._current_task_id,
        )

        if uncovered:
            if self._clar_up is not None and self._clar_down is not None:

                async def _clar_flow():
                    try:
                        await self._request_clarification(
                            "Your instruction could not be routed to all intended tasks without guessing. "
                            "Please specify exact task_ids, or use clear directives such as 'all', 'first', 'last', "
                            "or name the tasks explicitly, and provide the instruction for each group.",
                        )
                    except Exception:
                        pass

                try:
                    asyncio.create_task(_clar_flow())
                except Exception:
                    pass
                return
            await self._current_handle.interject(message)
            return

        if not hasattr(self, "_queued_interjections"):
            self._queued_interjections = {}

        for route in routes:
            task_ids = route.get("task_ids", [])
            instr = str(route.get("instruction", "")).strip()
            if not instr:
                continue
            for tid in task_ids:
                if tid == self._current_task_id:
                    try:
                        await self._current_handle.interject(instr)
                    except Exception:
                        pass
                else:
                    self._queued_interjections.setdefault(tid, []).append(instr)
        return

    async def stop(self, *, cancel: bool = False, reason: Optional[str] = None, **kwargs) -> None:  # type: ignore[override]
        try:
            await self._current_handle.stop(cancel=cancel, reason=reason)
        except Exception:
            pass

    async def pause(self) -> Optional[str]:  # type: ignore[override]
        try:
            if hasattr(self._current_handle, "done") and self._current_handle.done():  # type: ignore[attr-defined]
                return "Already completed."
        except Exception:
            pass
        try:
            ret = await self._current_handle.pause()
            return ret
        except Exception:
            return "Already completed."

    async def resume(self) -> Optional[str]:  # type: ignore[override]
        try:
            if hasattr(self._current_handle, "done") and self._current_handle.done():  # type: ignore[attr-defined]
                return "Already completed."
        except Exception:
            pass
        try:
            ret = await self._current_handle.resume()
            return ret
        except Exception:
            return "Already completed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done_evt.is_set()

    async def result(self):  # type: ignore[override]
        # Direct delegation when the queue remains a true singleton at call time
        if self._should_delegate_directly():
            ret = await self._current_handle.result()
            # Ensure the queue-level handle reflects completion immediately after
            # the inner handle resolves in direct delegation mode.
            try:
                if not self._done_evt.is_set():
                    # Prefer any final result the driver may have assembled; otherwise use ret
                    if not self._final_result:
                        self._final_result = ret
                    self._done_evt.set()
            except Exception:
                # Defensive: never fail caller's result() due to bookkeeping
                pass
            return ret

        await self._done_evt.wait()
        # If the driver did not assemble a summary (e.g. early stop without any
        # completions recorded), build a best-effort one now.
        if self._final_result:
            return self._final_result
        if self._completed_tasks:
            return self._summarise_completions()
        return ""

    async def _active_task_done(self) -> str:
        """
        Await until the next task in the queue completes (or return immediately
        if tasks have already completed since the last call) and return a JSON
        string mapping task names to their individual result strings for all
        tasks completed since the previous call to this method.

        Behaviour
        ---------
        - If called repeatedly, each call returns only the completions that
          occurred since the last call (cumulative cursor semantics).
        - If called after multiple tasks have already completed, the call
          returns immediately with all completions since the prior call.
        - If the queue has already finished and no new completions happened
          since the last call, returns an empty JSON object "{}".
        """

        # Drain any immediately available completion events first
        collected: list[Dict[str, Any]] = []
        try:
            while True:
                collected.append(self._completions.get_nowait())
        except asyncio.QueueEmpty:
            pass

        if not collected:
            # If queue already finished and nothing pending, return empty
            if self._done_evt.is_set():
                return "{}"
            # Otherwise wait for the next event, then drain the rest
            try:
                first = await self._completions.get()
                collected.append(first)
                try:
                    while True:
                        collected.append(self._completions.get_nowait())
                except asyncio.QueueEmpty:
                    pass
            except Exception:
                # Defensive: if awaiting failed unexpectedly, return empty or best-effort
                return "{}" if self._done_evt.is_set() else "{}"

        payload = {e.get("name", ""): e.get("result", "") for e in collected if e}
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:  # pass-through when supported
        try:
            if hasattr(self._current_handle, "next_clarification"):
                return await self._current_handle.next_clarification()  # type: ignore[attr-defined]
        except Exception:
            pass
        return {}

    async def next_notification(self) -> dict:  # pass-through when supported
        # If we have pending queue-level events, deliver them immediately
        try:
            return self._notif_q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        except Exception:
            pass

        # Create awaitables for both sources: queue-level and inner-handle
        notif_task = asyncio.create_task(self._notif_q.get())
        inner_task = None
        try:
            if hasattr(self._current_handle, "next_notification"):
                inner_task = asyncio.create_task(self._current_handle.next_notification())  # type: ignore[attr-defined]
        except Exception:
            inner_task = None

        # If no inner source, just await queue-level
        if inner_task is None:
            try:
                return await notif_task
            except Exception:
                return {}

        done, pending = await asyncio.wait(
            {notif_task, inner_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Retrieve result from first completed task
        result: dict = {}
        for t in done:
            try:
                res = await t
                if isinstance(res, dict):
                    result = res
                else:
                    # Wrap non-dict as a generic event
                    result = {"type": "inner.notification", "data": res}
            except Exception:
                result = {}
            break
        # Cancel the other task to avoid leaks
        for t in pending:
            try:
                t.cancel()
            except Exception:
                pass
        return result

    async def answer_clarification(
        self,
        call_id: str,
        answer: str,
    ) -> None:  # pass-through or use queue channel
        try:
            if hasattr(self._current_handle, "answer_clarification"):
                await self._current_handle.answer_clarification(call_id, answer)  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        # Fallback: if we have a queue-level clarification channel, push there
        try:
            if self._clar_down is not None:
                await self._clar_down.put(answer)
        except Exception:
            pass

    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":  # type: ignore[override]
        """Answer questions using a queue-level LLM that decides granularity.

        Policy
        ------
        - If and only if this queue has always remained a singleton since creation
          (sticky direct delegation), bypass the queue-level LLM and delegate directly to
          the inner task's ask() with the user question unchanged.
        - Otherwise: construct a compact chain snapshot (head→tail) and headline;
          forward the user's question verbatim to the current task and capture its
          response; then provide the LLM with snapshot JSON, headline, completions
          since start, and the inner task's answer. The LLM decides how much
          task‑level detail is appropriate for the user’s question (high‑level vs
          granular).
        """

        # Sticky singleton direct delegation for ask(): delegate directly when this queue
        # has only ever contained a single task. If the queue ever grew (size > 1),
        # _should_delegate_directly() permanently disables delegation for this instance.
        if self._should_delegate_directly():
            try:
                return await self._current_handle.ask(question)  # type: ignore[arg-type]
            except Exception:
                # Defensive fallback: proceed with the LLM synthesis path below
                pass

        # Build queue context
        queue_preamble: str | None = _QueueSnapshot.build_preamble(
            self._s,
            self._current_task_id,
        )
        queue_rows: list[dict] = _QueueSnapshot.build_rows(
            self._s,
            self._current_task_id,
        )

        # Determine chain size for copy policy decisions inside the LLM
        try:
            chain_size = len(queue_rows)
        except Exception:
            chain_size = 0

        # Ask inner task the user's question verbatim (no progress prompt)
        inner_task_response: str = ""
        try:
            ih = await self._current_handle.ask(question)  # type: ignore[arg-type]
            try:
                inner_task_response = await ih.result()
            except Exception:
                inner_task_response = ""
        except Exception:
            inner_task_response = ""

        # Compose completions snapshot since this queue started
        try:
            completed_pairs = list(self._completed_tasks)
        except Exception:
            completed_pairs = []
        completions_summary = (
            ", ".join([f"Task {tid}: {name}" for tid, name in completed_pairs])
            if completed_pairs
            else ""
        )

        # Use an LLM to decide the appropriate granularity and compose the answer
        client = new_llm_client()

        sys = (
            "You answer questions about a running chain of tasks. Decide the appropriate level of detail.\n"
            "Guidance:\n"
            "- If the user’s question is high‑level (overall progress), respond concisely with totals, current position, and what’s next.\n"
            "- Only include detailed, task‑specific progress when the question itself is granular.\n"
            "- Prefer paraphrasing and aggregation over verbatim dumps.\n"
            "- Always reference task ids when useful; avoid unnecessary minutiae.\n"
            "- Keep answers skimmable: brief paragraphs or short bullet points.\n"
            "\n"
            "COPY POLICY (critical):\n"
            "- If CHAIN_SIZE == 1 and the user’s question is about the current task (not the overall queue/chain),\n"
            "  you MUST return exactly the INNER_TASK_RESPONSE verbatim — no extra words, headings, or formatting.\n"
            "- Only when the question is explicitly about the queue/chain (e.g., overall/next/remaining/how many/comparisons)\n"
            "  should you synthesise a queue‑aware answer using the provided snapshot and context.\n"
        )
        client.set_system_message(sys)

        def _safe_dump(value: Any) -> str:
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                return str(value)

        user_msg = (
            (queue_preamble or "")
            + "\n\nCHAIN_ROWS_JSON:\n"
            + _safe_dump(queue_rows)
            + "\n\nCHAIN_SIZE: "
            + str(chain_size)
            + "\nCOMPLETIONS_SINCE_START:\n"
            + (completions_summary or "(none)")
            + "\n\nCURRENT_TASK_ID: "
            + str(self._current_task_id)
            + "\nINNER_TASK_RESPONSE (verbatim-ready):\n"
            + (inner_task_response or "")
            + "\n\nUSER QUESTION:\n"
            + question
        )

        answer = await client.generate(user_msg)

        # Return a lightweight static handle that yields the synthesized answer
        class _AnswerHandle(SteerableToolHandle):  # type: ignore[abstract-method]
            def __init__(self, text: str) -> None:
                self._text = text

            async def interject(self, message: str, **kwargs): ...

            async def stop(self, reason: Optional[str] = None, **kwargs): ...

            async def pause(self): ...

            async def resume(self): ...

            def done(self) -> bool:
                return True

            async def result(self) -> str:
                return self._text

            async def ask(self, question: str, **kwargs) -> "SteerableToolHandle":  # type: ignore[override]
                return self

            # New abstract event APIs – provide harmless stubs for the static handle
            async def next_clarification(self) -> dict:
                return {}

            async def next_notification(self) -> dict:
                return {}

            async def answer_clarification(self, call_id: str, answer: str) -> None:
                return None

        return _AnswerHandle(answer)

    # ----------------------------
    # Queue steering: append tail
    # ----------------------------
    def append_to_queue(self, task_id: int) -> Optional[str]:
        """Append an existing runnable task to the back of the current chain.

        Behaviour
        ---------
        - If the active chain belongs to a numeric ``queue_id``, move the
          given task to that queue's tail using the scheduler's move helper.
        - If the active chain has no numeric ``queue_id`` (isolated/linked),
          first materialize the current chain into a new queue (preserving
          order) and then move the given task to the new queue's tail.
        - When the task is already a member of the current chain, this is a
          no-op and a notification is emitted with reason ``already_member``.

        Returns a short human-readable summary string on success or no-op.
        Raises assertion/value errors from scheduler helpers for invalid ids
        or terminal/trigger-based tasks.
        """

        # Resolve the live chain view (head→tail)
        try:
            queue_rows: list[dict] | list[Any] = (
                self._s._get_queue_for_task(  # type: ignore[attr-defined]
                    task_id=int(self._current_task_id),
                )
                or []
            )
        except Exception:
            queue_rows = []

        # Derive current order and queue_id (if any)
        current_order: list[int] = []
        current_qid: Optional[int] = None
        try:
            for r in queue_rows:
                try:
                    tid_val = getattr(r, "task_id", None)
                    if tid_val is not None:
                        current_order.append(int(tid_val))
                except Exception:
                    continue
            if queue_rows:
                qid_val = getattr(queue_rows[0], "queue_id", None)
                if qid_val is not None:
                    current_qid = int(qid_val)
        except Exception:
            current_qid = None

        append_tid = int(task_id)

        # No-op when already present
        if append_tid in current_order:
            try:
                self._emit_notification(
                    {
                        "type": "queue.appended.skipped",
                        "task_id": append_tid,
                        "reason": "already_member",
                        "queue_id": current_qid,
                    },
                )
            except Exception:
                pass
            return f"Task {append_tid} is already a member of the current chain."

        # Branch on presence of a numeric queue_id for the current chain
        if isinstance(current_qid, int):
            # Use move helper to detach from any source queue and append to tail
            res = self._s._move_tasks_to_queue(  # type: ignore[attr-defined]
                task_ids=[append_tid],
                queue_id=int(current_qid),
                position="back",
            )
            # Disable singleton direct delegation permanently after growth
            try:
                self._direct_delegation_enabled = False
            except Exception:
                pass
            try:
                self._emit_notification(
                    {
                        "type": "queue.appended",
                        "task_id": append_tid,
                        "position": "tail",
                        "queue_id": res.get("details", {}).get("queue_id", current_qid),
                    },
                )
            except Exception:
                pass
            return f"Appended task {append_tid} to queue {int(current_qid)}."

        # No numeric queue_id → materialize current chain, then append via move
        base_order = list(current_order)
        if not base_order:
            try:
                base_order = [int(self._current_task_id)]
            except Exception:
                base_order = []

        set_res = self._s._set_queue(  # type: ignore[attr-defined]
            queue_id=None,
            order=base_order,
        )
        try:
            new_qid = set_res.get("details", {}).get("queue_id")
        except Exception:
            new_qid = None

        move_res = self._s._move_tasks_to_queue(  # type: ignore[attr-defined]
            task_ids=[append_tid],
            queue_id=new_qid,
            position="back",
        )

        try:
            self._direct_delegation_enabled = False
        except Exception:
            pass

        try:
            q_emit = move_res.get("details", {}).get("queue_id", new_qid)
        except Exception:
            q_emit = new_qid
        try:
            self._emit_notification(
                {
                    "type": "queue.appended",
                    "task_id": append_tid,
                    "position": "tail",
                    "queue_id": q_emit,
                },
            )
        except Exception:
            pass

        return f"Appended task {append_tid} to queue {q_emit}."
