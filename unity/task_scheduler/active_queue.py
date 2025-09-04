from __future__ import annotations

import asyncio
from typing import Dict, Callable, Optional, List, Any, TYPE_CHECKING

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
        # Sticky pass-through flag: enabled only when the queue truly contains a
        # single task at creation time; once disabled it never re-enables for the
        # lifetime of this ActiveQueue instance.
        try:
            initial_q = self._s._get_task_queue(task_id=self._current_task_id)
            self._passthrough_enabled: bool = len(initial_q) == 1
        except Exception:
            self._passthrough_enabled = False
        # Background driver
        self._driver = asyncio.create_task(self._drive())

    # ----------------------------
    # Internal clarification tool
    # ----------------------------
    async def _request_clarification(
        self,
        question: str,
        *,
        on_request: Callable[[str], Any] | None = None,
        on_answer: Callable[[str], Any] | None = None,
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

            # Prefer the scheduler's wrapper so behaviour matches other managers
            tool = self._s._make_request_clarification_tool(
                self._clar_up,
                self._clar_down,
            )

            # Best-effort notify hooks (non-blocking if they are sync)
            try:
                if on_request is not None:
                    maybe = on_request(question)
                    if asyncio.iscoroutine(maybe):
                        await maybe
            except Exception:
                pass

            answer = await tool(question)

            try:
                if on_answer is not None:
                    maybe2 = on_answer(answer)
                    if asyncio.iscoroutine(maybe2):
                        await maybe2
            except Exception:
                pass

            return answer
        except Exception:
            # Defensive: clarification should never break the queue
            return None

    # ----------------------------
    # Pass-through helper methods
    # ----------------------------
    def _current_queue_size(self) -> int:
        try:
            q = self._s._get_task_queue(task_id=self._current_task_id)
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
        if self._current_queue_size() > 1:
            self._passthrough_enabled = False
            return False
        return True

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
                    except Exception:
                        pass
                if was_stopped:
                    self._final_result = text or "Stopped."
                    break
                # If the stop/defer path was taken, end the queue here
                if "stopped" in text.lower():
                    self._final_result = text
                    break

                # Find next runnable in the same queue (head->tail from current)
                queue = self._s._get_task_queue(task_id=self._current_task_id)
                next_tid = None
                for t in queue:
                    if t.task_id != self._current_task_id:
                        next_tid = t.task_id
                        break
                if next_tid is None:
                    # Queue exhausted – compose a completion summary across all tasks
                    if self._completed_tasks:
                        summary_items = [
                            f"Task {tid}: {name}" for tid, name in self._completed_tasks
                        ]
                        summary = (
                            "Completed the following tasks: "
                            + ", ".join(summary_items)
                            + "."
                        )
                    else:
                        summary = "Chain completed."
                    self._final_result = summary
                    break

                # Start next task using CHAIN linkage semantics
                self._current_task_id = next_tid
                self._current_handle = await self._s._execute_internal(
                    task_id=next_tid,
                    parent_chat_context=self._parent_ctx,
                    clarification_up_q=self._clar_up,
                    clarification_down_q=self._clar_down,
                    activated_by=ActivatedBy.explicit,
                    execution_scope="queue",
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

        # Build a compact queue snapshot (head→tail) including ids and labels
        def _safe_dump(value):
            try:
                import json as _json  # local import

                return _json.dumps(value, default=str)
            except Exception:
                return str(value)

        def _get_row(tid: int):
            try:
                rows = self._s._filter_tasks(
                    filter=f"task_id == {int(tid)}",
                    limit=1,
                )
                return rows[0] if rows else None
            except Exception:
                return None

        try:
            cur_row = _get_row(self._current_task_id)
            head_row = cur_row
            while head_row is not None:
                prev_id = self._s._sched_prev((head_row.get("schedule") or {}))
                if prev_id is None:
                    break
                prev_row = _get_row(prev_id)
                if prev_row is None:
                    break
                head_row = prev_row

            queue_rows: list[dict] = []
            seen: set[int] = set()
            node = head_row
            while node is not None:
                tid = node.get("task_id")
                try:
                    tid_int = int(tid)
                except Exception:
                    tid_int = None  # type: ignore[assignment]
                if tid_int is not None and tid_int in seen:
                    break
                if tid_int is not None:
                    seen.add(tid_int)
                queue_rows.append(
                    {
                        k: v
                        for k, v in node.items()
                        if v is not None and not str(k).startswith("_")
                    },
                )
                nxt_id = self._s._sched_next((node.get("schedule") or {}))
                if nxt_id is None:
                    break
                node = _get_row(nxt_id)

            if not queue_rows:
                # Fallback to best-effort non-terminal queue snapshot
                queue = self._s._get_task_queue(task_id=self._current_task_id)
                queue_rows = [
                    {
                        "task_id": getattr(t, "task_id", None),
                        "name": getattr(t, "name", None),
                        "description": getattr(t, "description", None),
                        "status": getattr(t, "status", None),
                        "schedule": getattr(t, "schedule", None),
                    }
                    for t in queue
                ]
        except Exception:
            queue_rows = []

        # (debug logging removed)

        # Create a dedicated router client with high reasoning and priority tier
        try:
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
            )
            client.set_system_message(sys)
            # Compute first/last ids for explicit metadata to aid deterministic mapping
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
            raw = await client.generate(user)
        except Exception:
            raw = ""

        # Parse structured routes; fall back to current-only on failure
        try:
            import json as _json

            data = _json.loads(raw)
            routes = data.get("routes") if isinstance(data, dict) else None
            if not isinstance(routes, list):
                raise ValueError("no routes")

            # If the model indicates missing coverage, request a corrected set in a second pass.
            try:
                uncovered = data.get("uncovered_directives") or []
            except Exception:
                uncovered = []
            if isinstance(uncovered, list) and uncovered:
                try:
                    client2 = unify.AsyncUnify(
                        "gpt-5@openai",
                        cache=True,
                        traced=True,
                        reasoning_effort="high",
                        service_tier="priority",
                    )
                    client2.set_system_message(
                        "You are correcting a routing result to ensure ALL directives are covered.\n"
                        "Return ONLY JSON with the 'routes' field per the same schema as before; do not include commentary.\n",
                    )
                    user2 = (
                        "Original queue (head→tail):\n"
                        + _safe_dump(queue_rows)
                        + "\nMetadata:\n"
                        + f"first_task_id: {first_task_id}\n"
                        + f"last_task_id: {last_task_id}\n"
                        + f"current_task_id: {self._current_task_id}\n"
                        + "Original interjection:\n"
                        + (message or "").strip()
                        + "\nPreviously returned routes:\n"
                        + _safe_dump(routes)
                        + "\nUncovered directives to cover now:\n"
                        + _safe_dump(uncovered)
                    )
                    raw2 = await client2.generate(user2)
                    data2 = _json.loads(raw2)
                    routes2 = data2.get("routes") if isinstance(data2, dict) else None
                    if isinstance(routes2, list) and routes2:
                        routes = routes2
                    else:
                        # Still ambiguous – proactively request clarification if channels exist
                        await self._request_clarification(
                            "Your instruction could not be routed to all intended tasks. "
                            "Please specify exact task_ids, or use clear directives such as 'all', 'first', 'last', "
                            "or name the tasks explicitly, and provide the instruction for each group.",
                        )
                except Exception:
                    pass

            # Ensure pending registry exists
            if not hasattr(self, "_queued_interjections"):
                self._queued_interjections = {}

            # Build a set of known ids for safety
            known_ids = {
                int(r.get("task_id"))
                for r in queue_rows
                if r.get("task_id") is not None
            }

            for route in routes:
                try:
                    task_ids = [int(t) for t in route.get("task_ids", [])]
                    instr = str(route.get("instruction", "")).strip()
                except Exception:
                    continue
                if not instr:
                    continue
                for tid in task_ids:
                    if tid not in known_ids:
                        continue
                    if tid == self._current_task_id:
                        try:
                            await self._current_handle.interject(instr)
                        except Exception:
                            pass
                    else:
                        self._queued_interjections.setdefault(tid, []).append(instr)
            return
        except Exception:
            # Fallback: deliver to current task only
            # If clarification channels are available and the message looks ambiguous,
            # try to disambiguate before defaulting to current-only delivery.
            try:
                ambiguous_tokens = ("all", "rest", "remaining", "first", "last")
                looks_ambiguous = any(
                    tok in (message or "").lower() for tok in ambiguous_tokens
                )
                if looks_ambiguous:
                    await self._request_clarification(
                        "Your interjection could refer to multiple tasks. "
                        "Please specify which tasks it applies to (by id or directive such as 'all', 'first', 'last'), "
                        "and provide the instruction text for each group.",
                    )
            except Exception:
                pass
            await self._current_handle.interject(message)

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
            return await self._current_handle.result()

        await self._done_evt.wait()
        # If the driver did not assemble a summary (e.g. early stop without any
        # completions recorded), build a best-effort one now.
        if self._final_result:
            return self._final_result
        if self._completed_tasks:
            summary_items = [
                f"Task {tid}: {name}" for tid, name in self._completed_tasks
            ]
            return "Completed the following tasks: " + ", ".join(summary_items) + "."
        return ""

    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":  # type: ignore[override]
        """Answer questions with queue-aware context and delegate to inner handle.

        Builds a compact queue snapshot (head→tail) including all non-None
        task fields for each task (completed and non-completed) and a
        high-level progress summary, then prepends it to the forwarded
        question. If snapshot construction fails, falls back to the raw
        question.
        """

        def _safe_dump(value):
            try:
                import json as _json  # local import

                return _json.dumps(value, default=str)
            except Exception:
                return str(value)

        def _get_row(tid: int):
            try:
                rows = self._s._filter_tasks(
                    filter=f"task_id == {int(tid)}",
                    limit=1,
                )
                return rows[0] if rows else None
            except Exception:
                return None

        queue_preamble: str | None = None
        try:
            # 1) Locate current row and walk to head (using schedule.prev_task)
            cur_row = _get_row(self._current_task_id)
            head_row = cur_row
            while head_row is not None:
                prev_id = self._s._sched_prev((head_row.get("schedule") or {}))
                if prev_id is None:
                    break
                prev_row = _get_row(prev_id)
                if prev_row is None:
                    break
                head_row = prev_row

            # 2) Walk forward to collect the entire queue, including terminal statuses
            queue_rows: list[dict] = []
            seen: set[int] = set()
            node = head_row
            while node is not None:
                tid = node.get("task_id")
                try:
                    tid_int = int(tid)
                except Exception:
                    tid_int = None  # type: ignore[assignment]
                if tid_int is not None and tid_int in seen:
                    break  # safety loop-breaker
                if tid_int is not None:
                    seen.add(tid_int)
                queue_rows.append(node)
                nxt_id = self._s._sched_next((node.get("schedule") or {}))
                if nxt_id is None:
                    break
                node = _get_row(nxt_id)

            # Fallback: if queue_rows is empty, try non-terminal queue as a best-effort snapshot
            if not queue_rows:
                queue = self._s._get_task_queue(task_id=self._current_task_id)
                queue_rows = [
                    {
                        # best-effort row-like dict shape
                        "task_id": getattr(t, "task_id", None),
                        "instance_id": getattr(t, "instance_id", None),
                        "name": getattr(t, "name", None),
                        "description": getattr(t, "description", None),
                        "status": getattr(t, "status", None),
                        "schedule": getattr(t, "schedule", None),
                        "trigger": getattr(t, "trigger", None),
                        "deadline": getattr(t, "deadline", None),
                        "repeat": getattr(t, "repeat", None),
                        "priority": getattr(t, "priority", None),
                        "response_policy": getattr(t, "response_policy", None),
                        "activated_by": getattr(t, "activated_by", None),
                    }
                    for t in queue
                ]

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

            # Detailed rows with all non-None fields
            details_lines: list[str] = ["Chain tasks (head→tail):"]
            for r in queue_rows:
                tid = r.get("task_id")
                name = r.get("name") or ""
                details_lines.append(f"- Task {tid}: {name}")
                # Print all non-None, non-_internal keys
                for k, v in r.items():
                    if v is None or k in {"name"}:
                        continue
                    if str(k).startswith("_"):
                        continue
                    details_lines.append(f"    {k}: {_safe_dump(v)}")

            queue_preamble = (
                "CHAIN CONTEXT\n" + headline + "\n" + "\n".join(details_lines)
            )
        except Exception:
            queue_preamble = None

        # Passthrough when the queue remains a true singleton at call time
        if self._should_passthrough():
            try:
                return await self._current_handle.ask(
                    question,
                    _return_reasoning_steps=_return_reasoning_steps,
                )
            except TypeError:
                return await self._current_handle.ask(question)

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
