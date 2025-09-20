from __future__ import annotations

import asyncio
from typing import Dict, Callable, Optional, List, Any, TYPE_CHECKING
import json
import os

import unify

from ..common.llm_helpers import SteerableToolHandle
from .types.activated_by import ActivatedBy

if TYPE_CHECKING:  # avoid import cycles at runtime
    from .task_scheduler import TaskScheduler


class ActiveQueue(SteerableToolHandle):  # type: ignore[abstract-method]
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
        # Sticky pass-through flag: enabled only when the queue truly contains a
        # single task at creation time; once disabled it never re-enables for the
        # lifetime of this ActiveQueue instance.
        try:
            initial_q = self._s._get_queue_for_task(task_id=self._current_task_id)
            self._passthrough_enabled: bool = len(initial_q) == 1
        except Exception:
            self._passthrough_enabled = False

        # Background driver
        self._driver = asyncio.create_task(self._drive())

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
    # Snapshot helper (head→tail)
    # ----------------------------
    def _build_queue_rows_snapshot(self) -> list[dict]:
        """
        Build a compact queue snapshot (head→tail) using the scheduler's live
        queue view. Returns a list of simple dict rows.
        """
        try:
            queue = self._s._get_queue_for_task(task_id=self._current_task_id) or []
        except Exception:
            queue = []
        out: list[dict] = []
        for t in queue:
            try:
                out.append(
                    {
                        "task_id": getattr(t, "task_id", None),
                        "name": getattr(t, "name", None),
                        "description": getattr(t, "description", None),
                        "status": getattr(t, "status", None),
                        "schedule": getattr(t, "schedule", None),
                    },
                )
            except Exception:
                continue
        return out

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
            # Enqueue question; await answer
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
    # Pass-through helper methods
    # ----------------------------
    def _current_queue_size(self) -> int:
        try:
            q = self._s._get_queue_for_task(task_id=self._current_task_id)
            return len(q)
        except Exception:
            return 0

    def _should_passthrough(self) -> bool:
        """Return True when we should directly delegate to the inner handle.

        This is allowed only while the queue remains a true singleton. The
        moment an additional task appears in the queue (size > 1), we
        permanently disable pass-through for the lifetime of this instance.
        """
        if not getattr(self, "_passthrough_enabled", False):
            return False
        # If at any point the queue grows beyond a single task, flip sticky off.
        size = self._current_queue_size()
        if size > 1:
            self._passthrough_enabled = False
            return False
        return True

    def _next_runnable_follower(self) -> Optional[int]:
        """Return the next runnable task id after the current one, from the live queue.

        Relies solely on the scheduler's queue view (non-terminal membership),
        avoiding local schedule.next hints and repair scans.
        """
        try:
            live_queue = (
                self._s._get_queue_for_task(task_id=self._current_task_id) or []
            )
        except Exception:
            live_queue = []

        ids: list[int] = []
        for t in live_queue:
            try:
                tid_val = int(getattr(t, "task_id", -1))
            except Exception:
                continue
            ids.append(tid_val)

        if not ids:
            return None

        cur_id = None
        try:
            cur_id = int(self._current_task_id)
        except Exception:
            cur_id = None

        # If current id is not found (e.g., was detached), start at the head
        try:
            if cur_id is None:
                return ids[0]
            idx = ids.index(cur_id)
        except ValueError:
            return ids[0]

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
                                rows[0].get("name")
                                or rows[0].get("description")
                                or "(unnamed task)"
                            )
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
                    self._final_result = text or "Stopped."
                    break
                # If the stop/defer path was taken, end the queue here
                if "stopped" in text.lower():
                    self._final_result = text
                    break

                # Await linkage barrier from the scheduler to ensure neighbour
                # writes are visible before advancing.
                try:
                    barrier = self._s._get_linkage_barrier(
                        task_id=self._current_task_id,
                    )
                except Exception:
                    barrier = None
                if barrier is not None:
                    try:
                        # Wait briefly; if already set this returns immediately
                        await asyncio.wait_for(barrier.wait(), timeout=1.0)
                    except Exception:
                        pass

                # Determine the next task to run using the live queue only
                next_tid: Optional[int] = self._next_runnable_follower()

                if next_tid is None:
                    # Queue exhausted – compose a completion summary across all tasks
                    self._final_result = self._summarise_completions()
                    break

                # Start next task using CHAIN linkage semantics
                self._current_task_id = next_tid
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
                    for _msg in pending_msgs:
                        try:
                            await self._current_handle.interject(_msg)
                        except Exception:
                            pass
                except Exception:
                    pass
        finally:
            self._done_evt.set()
            # No waiter/cursor machinery – active_task_done() awaits on the completions queue

    # ----- Steerable surface proxies -----
    async def interject(self, message: str) -> None:  # type: ignore[override]
        """Route interjections to specific tasks in the queue using an LLM router.

        The router receives the full queue snapshot and the user's instruction and
        returns structured routes of the form:

            {"routes": [{"task_ids": [<int>, ...], "instruction": "..."}, ...]}

        Interjections for the currently active task are delivered immediately.
        Interjections for future tasks are queued and delivered when those
        tasks become active.
        """

        # Passthrough when this is a true singleton queue (sticky while true)
        if self._should_passthrough():
            if not (message or "").strip():
                return
            await self._current_handle.interject(message)
            return

        # Fast path: empty/whitespace → no-op
        if not (message or "").strip():
            return

        # Optional bypass: disable LLM router via env flag and route to current task
        try:
            if str(os.getenv("UNITY_TS_DISABLE_LLM_ROUTER", "")).lower() in {
                "1",
                "true",
                "yes",
            }:
                await self._current_handle.interject(message)
                return
        except Exception:
            pass

        # Perform routing via helper; keep main method small
        queue_rows: list[dict] = self._build_queue_rows_snapshot()

        routes, uncovered = await self._route_interjection_llm(
            queue_rows=queue_rows,
            message=message,
        )

        if uncovered:
            if self._clar_up is not None and self._clar_down is not None:
                await self._request_clarification(
                    "Your instruction could not be routed to all intended tasks without guessing. "
                    "Please specify exact task_ids, or use clear directives such as 'all', 'first', 'last', "
                    "or name the tasks explicitly, and provide the instruction for each group.",
                )
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

    async def _route_interjection_llm(
        self,
        *,
        queue_rows: list[dict],
        message: str,
    ) -> tuple[list[dict], bool]:
        """Return (routes, uncovered_flag) using a dedicated LLM call with timeout.

        On any error, returns ([], True) so callers can apply the clarification
        or fallback policy without duplicating error handling.
        """
        try:

            def _safe_dump(value):
                try:
                    import json as _json  # local import

                    return _json.dumps(value, default=str)
                except Exception:
                    return str(value)

            client = unify.AsyncUnify(
                "gpt-5@openai",
                cache=True,
                traced=True,
                reasoning_effort="high",
                service_tier="priority",
            )
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
                + str(self._current_task_id)
                + "\nInterjection:"
                + f"\n{(message or '').strip()}"
            )

            try:
                timeout_s = float(os.getenv("UNITY_TS_ROUTER_TIMEOUT_SECONDS", "60.0"))
            except Exception:
                timeout_s = 60.0

            try:
                raw = await asyncio.wait_for(client.generate(user), timeout=timeout_s)
            except asyncio.TimeoutError:
                raw = ""

            try:
                import json as _json

                data = _json.loads(raw)
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

    def stop(self, *, cancel: bool, reason: Optional[str] = None) -> Optional[str]:  # type: ignore[override]
        try:
            return self._current_handle.stop(cancel=cancel, reason=reason)
        except Exception:
            return "Stopped."

    def pause(self) -> Optional[str]:  # type: ignore[override]
        try:
            if hasattr(self._current_handle, "done") and self._current_handle.done():  # type: ignore[attr-defined]
                return "Already completed."
        except Exception:
            pass
        try:
            ret = self._current_handle.pause()
            return ret
        except Exception:
            return "Already completed."

    def resume(self) -> Optional[str]:  # type: ignore[override]
        try:
            if hasattr(self._current_handle, "done") and self._current_handle.done():  # type: ignore[attr-defined]
                return "Already completed."
        except Exception:
            pass
        try:
            ret = self._current_handle.resume()
            return ret
        except Exception:
            return "Already completed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done_evt.is_set()

    async def result(self):  # type: ignore[override]
        # Passthrough when the queue remains a true singleton at call time
        if self._should_passthrough():
            ret = await self._current_handle.result()
            # Ensure the queue-level handle reflects completion immediately after
            # the inner handle resolves in passthrough mode.
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

    async def active_task_done(self) -> str:
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

    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":  # type: ignore[override]
        """Answer questions with queue-aware context and delegate to inner handle.

        Builds a compact queue snapshot (head→tail) and a concise progress
        headline, then prepends it to the forwarded question. If snapshot
        construction fails, falls back to the raw question.
        """

        # Fast-path: when queue remains a true singleton, delegate directly
        if self._should_passthrough():
            try:
                return await self._current_handle.ask(
                    question,
                    _return_reasoning_steps=_return_reasoning_steps,
                )
            except TypeError:
                return await self._current_handle.ask(question)

        queue_preamble: str | None = None
        try:
            # Build queue snapshot (includes terminal statuses when available)
            queue_rows: list[dict] = self._build_queue_rows_snapshot()

            total_count = len(queue_rows)
            # Identify current index
            current_index = -1
            for idx, r in enumerate(queue_rows):
                if r.get("task_id") == self._current_task_id:
                    current_index = idx
                    break

            # Count statuses
            def _to_status_str(row: dict) -> str:
                try:
                    return str(self._s._to_status(row.get("status")))
                except Exception:
                    return str(row.get("status"))

            completed_count = sum(
                1 for r in queue_rows if _to_status_str(r) == "completed"
            )
            remaining_count = max(0, total_count - completed_count)

            # Next tasks preview (up to 3)
            next_names: list[str] = []
            if current_index >= 0:
                for j in range(current_index + 1, min(current_index + 4, total_count)):
                    nm = queue_rows[j].get("name")
                    if nm:
                        next_names.append(str(nm))

            # High-level summary line
            if current_index >= 0 and current_index < total_count:
                current_name = queue_rows[current_index].get("name") or "(unnamed task)"
                headline = (
                    f"Chain status: {completed_count}/{total_count} completed; "
                    f"{remaining_count} remaining; executing {current_index + 1}/{total_count}: "
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

            queue_preamble = (
                "CHAIN CONTEXT\n" + headline + "\n" + "\n".join(details_lines)
            )
        except Exception:
            queue_preamble = None

        composed_question = (
            f"{queue_preamble}\n\nUSER QUESTION:\n{question}"
            if queue_preamble
            else question
        )

        try:
            return await self._current_handle.ask(  # type: ignore[arg-type]
                composed_question,
                _return_reasoning_steps=_return_reasoning_steps,
            )
        except TypeError:
            # Older handles may not accept the kwarg – retry without it.
            return await self._current_handle.ask(composed_question)  # type: ignore[arg-type]

    @property
    def valid_tools(self) -> Dict[str, Callable]:  # type: ignore[override]
        tools = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        paused_flag = getattr(self._current_handle, "_paused", False)
        if paused_flag:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools
