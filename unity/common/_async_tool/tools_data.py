import asyncio
import inspect
import json
import traceback
import time
import dataclasses


from typing import (
    Dict,
    Set,
    Tuple,
    Any,
    Optional,
    TYPE_CHECKING,
)
from .tools_utils import ToolCallMetadata, create_tool_call_message
from .messages import (
    insert_tool_message_after_assistant,
    chat_context_repr,
    _normalise_kwargs_for_bound_method,
    forward_handle_call,
)
from .message_dispatcher import LoopMessageDispatcher
from ..tool_spec import normalise_tools
from ..llm_helpers import method_to_schema
from .formatting import serialize_tool_content, sanitize_tool_msg_for_logging
from contextlib import suppress

if TYPE_CHECKING:  # TODO: remove once dependencies are fixed
    from .loop import LoopLogger, _LoopToolFailureTracker


class ToolsData:
    def __init__(self, tools, *, client, logger: "LoopLogger"):
        self._client = client
        self._logger = logger
        self.normalized = normalise_tools(tools)
        self.pending: Set[asyncio.Task] = set()
        self.info: Dict[asyncio.Task, ToolCallMetadata] = {}
        # Per-tool hidden total-call quotas (counted per loop instance)
        self.call_counts: Dict[str, int] = {}
        self.clarification_channels: Dict[
            str,
            Tuple[asyncio.Queue[str], asyncio.Queue[str]],
        ] = {}
        self.completed_results: Dict[str, str] = {}

    # Local helper: pretty-print tool payloads consistently
    @staticmethod
    def _pretty_tool_payload(tool_name: str, payload: Any) -> str:
        # Centralized serialization for progress/notification placeholders
        return serialize_tool_content(
            tool_name=tool_name,
            payload=payload,
            is_final=False,
        )

    def _quota_count(self, task_name: str) -> int:
        return self.call_counts.get(task_name, 0)

    def _can_offer_tool(self, task_name: str) -> bool:
        limit = self.normalized[task_name].max_concurrent
        return limit is None or self.active_count(task_name) < limit

    # ── small helper: add completion tool message pair ──────────────
    @staticmethod
    async def _emit_completion_pair(
        result: str,
        call_id: str,
        msg_dispatcher: LoopMessageDispatcher,
    ) -> dict:
        """
        Append a synthetic assistant→tool pair that carries the *final*
        outcome for `call_id`.  Returns the tool-message so callers can
        reuse it for logging / event-bus.
        """
        dummy_id = f"{call_id}_status"

        assistant_stub = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": dummy_id,
                    "type": "function",
                    "function": {
                        "name": f"check_status_{call_id}",
                        "arguments": "{}",
                    },
                },
            ],
            "content": "",
        }
        tool_msg = create_tool_call_message(
            name=f"check_status_{call_id}",
            call_id=dummy_id,
            content=result,
        )

        await msg_dispatcher.append_msgs([assistant_stub, tool_msg])
        return tool_msg

    def has_exceeded_quota_for_tool(self, task_name: str) -> bool:
        if task_name not in self.normalized:
            return False

        limit = self.normalized[task_name].max_total_calls
        return limit is not None and self._quota_count(task_name) >= limit

    def has_exceeded_concurrent_limit_for_tool(self, task_name: str) -> bool:
        if task_name not in self.normalized:
            return False

        limit = self.normalized[task_name].max_concurrent
        return limit is not None and self.active_count(task_name) >= limit

    def save_task(self, coro, metadata: ToolCallMetadata):
        self.pending.add(coro)
        self.info[coro] = metadata

    def pop_task(self, coro: asyncio.Task) -> ToolCallMetadata:
        self.pending.discard(coro)
        return self.info.pop(coro, None)

    def active_count(self, task_name: str) -> int:
        return sum(1 for _t, _inf in self.info.items() if _inf.name == task_name)

    def quota_ok(self, task_name: str) -> bool:
        limit = self.normalized[task_name].max_total_calls
        return limit is None or self._quota_count(task_name) < limit

    def concurrency_ok(self, task_name: str) -> bool:
        return task_name not in self.normalized or self._can_offer_tool(task_name)

    async def cancel_pending_tasks(self):
        for task in self.pending:
            # Explicitly stop active handles because task.cancel() doesn't
            # propagate to underlying threads (e.g. asyncio.to_thread).
            info = self.info.get(task)
            if info and info.handle and hasattr(info.handle, "stop"):
                try:
                    res = info.handle.stop("loop cancelled")
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass
            task.cancel()
        await asyncio.gather(*self.pending, return_exceptions=True)
        self.pending.clear()

    # Remove any tool_calls in an assistant message that would exceed the
    # hidden per-tool total-call quota. Operates in-place on asst_msg.
    def prune_over_quota_tool_calls(self, asst_msg: dict) -> None:
        with suppress(Exception):
            tool_calls = asst_msg.get("tool_calls") or []
            if not isinstance(tool_calls, list) or not tool_calls:
                return

            # Compute remaining budget per base tool (in this loop instance)
            remaining: Dict[str, int] = {}
            for name, spec in self.normalized.items():
                lim = spec.max_total_calls
                if lim is None:
                    continue
                remaining[name] = max(0, lim - self._quota_count(name))

            kept: list = []
            for call in tool_calls:
                with suppress(Exception):
                    fn_name = call.get("function", {}).get("name")
                if "fn_name" not in locals():
                    fn_name = None

                # Only enforce quota on base tools that define a limit
                if fn_name in remaining:
                    if remaining[fn_name] > 0:
                        kept.append(call)
                        remaining[fn_name] -= 1
                    else:
                        # drop this over-quota call silently
                        continue
                else:
                    kept.append(call)

            # In-place update only if changed
            if len(kept) != len(tool_calls):
                asst_msg["tool_calls"] = kept

    # Helper: schedule a base tool call (shared by main path and backfill)
    async def schedule_base_tool_call(
        self,
        asst_msg: dict,
        *,
        name: str,
        args_json: Any,
        call_id: str,
        call_idx: int,
        parent_chat_context,
        propagate_chat_context,
        assistant_meta,
        initial_paused: bool = False,
    ) -> None:
        # Base tool must exist
        if name not in self.normalized:
            return

        fn = self.normalized[name].fn

        # Enforce hidden per-tool total call quota: should be pre-pruned from
        # the assistant message, but guard here as well and simply skip.
        with suppress(Exception):
            lim = self.normalized[name].max_total_calls
            if lim is not None and self.call_counts.get(name, 0) >= lim:
                return

        # Build extra kwargs (chat context, interject/clarification/pause)
        extra_kwargs: dict = {}
        if propagate_chat_context:
            cur_msgs = [m for m in self._client.messages if not m.get("_ctx_header")]
            ctx_repr = chat_context_repr(parent_chat_context, cur_msgs)
            extra_kwargs["_parent_chat_context"] = ctx_repr

        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        sig_accepts_interject_q = "_interject_queue" in params or has_varkw
        sig_accepts_pause_event = "_pause_event" in params or has_varkw
        sig_accepts_clar_qs = (
            "_clarification_up_q" in params and "_clarification_down_q" in params
        ) or has_varkw
        sig_accepts_progress = "_notification_up_q" in params or has_varkw

        pause_ev: Optional[asyncio.Event] = None
        if sig_accepts_pause_event:
            pause_ev = asyncio.Event()
            if initial_paused:
                pause_ev.clear()  # start paused
            else:
                pause_ev.set()  # start running
            extra_kwargs["_pause_event"] = pause_ev

        clar_up_q: Optional[asyncio.Queue[str]] = None
        clar_down_q: Optional[asyncio.Queue[str]] = None
        if sig_accepts_clar_qs:
            clar_up_q = asyncio.Queue()
            clar_down_q = asyncio.Queue()
            extra_kwargs["_clarification_up_q"] = clar_up_q
            extra_kwargs["_clarification_down_q"] = clar_down_q

        progress_q: Optional[asyncio.Queue[dict]] = None
        if sig_accepts_progress:
            progress_q = asyncio.Queue()
            extra_kwargs["_notification_up_q"] = progress_q

        sub_q: Optional[asyncio.Queue[str]] = None
        if sig_accepts_interject_q:
            sub_q = asyncio.Queue()
            extra_kwargs["_interject_queue"] = sub_q

        # Parse args
        with suppress(Exception):
            call_args = (
                json.loads(args_json)
                if isinstance(args_json, str)
                else (args_json or {})
            )
        if "call_args" not in locals():
            call_args = {}

        # Filter extras to match fn signature, and normalise base call args via shared helper
        filtered_extras = {
            k: v for k, v in extra_kwargs.items() if k in params or has_varkw
        }
        allowed_call_args = _normalise_kwargs_for_bound_method(fn, call_args)
        merged_kwargs = {**allowed_call_args, **filtered_extras}

        # Legacy arg-scoped image normalization removed; inner tools should accept ImageRefs explicitly.

        # (Argument pretty-printing now handled in assistant message logs only)

        # Build coroutine
        if asyncio.iscoroutinefunction(fn):
            coro = fn(**merged_kwargs)
        else:
            coro = asyncio.to_thread(fn, **merged_kwargs)

        call_dict = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args_json},
        }

        t = asyncio.create_task(coro, name=f"ToolCall_{name}")
        metadata = ToolCallMetadata(
            name=name,
            call_id=call_id,
            assistant_msg=asst_msg,
            call_dict=call_dict,
            call_idx=call_idx,
            is_interjectable=sig_accepts_interject_q,
            interject_queue=sub_q,
            chat_context=extra_kwargs.get("_parent_chat_context"),
            clar_up_queue=clar_up_q,
            clar_down_queue=clar_down_q,
            notification_queue=progress_q,
            pause_event=pause_ev,
            # Debug helpers for failure logging
            tool_schema=method_to_schema(fn, name),
            llm_arguments=allowed_call_args,
            raw_arguments_json=args_json,
        )
        self.save_task(t, metadata)

        if self._logger.log_steps:
            self._logger.info(
                f"{name} - {call_id}",
                prefix=f"🛠️  ToolCall Scheduled",
            )

        # Increment hidden quota counter only once scheduling succeeds
        with suppress(Exception):
            self.call_counts[name] = self.call_counts.get(name, 0) + 1

        if clar_up_q is not None:
            self.clarification_channels[call_id] = (
                clar_up_q,
                clar_down_q,
            )

        # Ensure assistant meta exists for deterministic insertion ordering
        assistant_meta.setdefault(id(asst_msg), {"results_count": 0})

    # ── *single* authoritative implementation of "task finished" handling ──
    async def process_completed_task(
        self,
        task: asyncio.Task,
        consecutive_failures: "_LoopToolFailureTracker",
        outer_handle_container,
        assistant_meta,
        msg_dispatcher,
    ) -> bool:
        """
        Deal with a finished tool *task* exactly once:

        1.  Pop bookkeeping (``pending`` / ``task_info``).
        2.  Serialise *success* or *exception* into ``result``.
        3.  Patch or insert the correct **tool** message so the transcript
            stays perfectly chronological.
        4.  Emit the event-bus hook (if configured).
        5.  Record the payload in ``completed_results`` for potential post-hoc lookups.
        6.  Enforce the *max_consecutive_failures* safety valve.
        """

        def _at_tail(msg: dict) -> bool:
            """True when *msg* is the very last entry in client.messages."""
            return bool(self._client.messages) and self._client.messages[-1] is msg

        info: ToolCallMetadata = self.pop_task(task)
        name = info.name
        call_id = info.call_id

        # 1️⃣-a. Drain any pending notifications that arrived just before completion
        #      (prevents missing progress events when the tool finishes quickly).
        try:
            q = info.notification_queue
        except Exception:
            q = None
        if q is not None:
            while True:
                try:
                    payload = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    break

                # Pretty-print content for transcript placeholder
                pretty = self._pretty_tool_payload(name, payload)

                # Create/update a single tool-reply placeholder for this call_id
                placeholder = info.tool_reply_msg
                if placeholder is None:
                    placeholder = create_tool_call_message(
                        name=name,
                        call_id=call_id,
                        content=pretty,
                    )
                    await insert_tool_message_after_assistant(
                        assistant_meta,
                        info.assistant_msg,
                        placeholder,
                        self._client,
                        msg_dispatcher,
                    )
                    info.tool_reply_msg = placeholder
                else:
                    placeholder["content"] = pretty

                # Forward a programmatic notification event to the outer handle
                with suppress(Exception):
                    outer = (
                        outer_handle_container[0] if outer_handle_container else None
                    )
                    if outer is not None and hasattr(outer, "_notification_q"):
                        event_payload = (
                            payload
                            if isinstance(payload, dict)
                            else {"message": str(payload)}
                        )
                        await outer._notification_q.put(
                            {
                                "type": "notification",
                                "call_id": call_id,
                                "tool_name": name,
                                **event_payload,
                            },
                        )

        # 2️⃣  obtain result -------------------------------------------------
        try:
            raw = task.result()

            # ───────────────────────────────────────────────────────────────
            #  NEW:  the tool *did not really finish* – it returned *another*
            #        AsyncToolLoopHandle.  We:
            #        (1) schedule `handle.result()` as a *new* task,
            #        (2) keep the **same** `call_id` so the continue/-cancel
            #            helpers keep working,
            #        (3) create / patch one placeholder "still running…"
            #            tool-message in the transcript.
            # ───────────────────────────────────────────────────────────────
            # treat ANY AsyncToolLoopHandle (or subclass) as a nested loop
            from unity.common.async_tool_loop import SteerableToolHandle

            if isinstance(raw, SteerableToolHandle):
                await self.adopt_nested(
                    info,
                    raw,
                    msg_dispatcher=msg_dispatcher,
                    assistant_meta=assistant_meta,
                    outer_handle_container=outer_handle_container,
                )
                return False  # ⬅️  no LLM turn required

            # ───────────────────────────────────────────────────────────────
            #  Normal (non-handle) result – unchanged path
            # ───────────────────────────────────────────────────────────────
            # ── finished successfully – promote any embedded images ─────────
            # Centralized serialization for final tool results
            result = serialize_tool_content(tool_name=name, payload=raw, is_final=True)

            consecutive_failures.reset_failures()
        except Exception:
            consecutive_failures.increment_failures()
            result = traceback.format_exc()
            if self._logger.log_steps:
                self._logger.error(
                    f"Error: {name} failed "
                    f"(attempt {consecutive_failures.current_failures}/{consecutive_failures.max_failures}):\n{result}",
                    prefix="❌",
                )
                # Additional debug context: show the exact tool schema and arguments
                # that were presented to the LLM for this failed call. This helps
                # diagnose docstrings/argspec mismatches that cause tool misuse.
                with suppress(Exception):
                    debug_payload = {
                        "tool_name": name,
                        "call_id": call_id,
                        "llm_function_schema": info.tool_schema,
                        "llm_arguments": info.llm_arguments,
                        "raw_arguments_json": info.raw_arguments_json,
                    }
                    self._logger.error(
                        f"FAILED TOOL SCHEMA (as given to LLM):\n{json.dumps(debug_payload, indent=2)}",
                        prefix="🧩",
                    )

        # 3️⃣  remember so later lookups can answer instantly
        self.completed_results[call_id] = result

        # 4️⃣  update / insert tool-result message --------------------------
        asst_msg = info.assistant_msg
        clarify_ph = info.clarify_placeholder
        tool_reply_msg = info.tool_reply_msg

        if clarify_ph is not None:
            if _at_tail(clarify_ph):
                clarify_ph["content"] = result
                tool_msg = clarify_ph
            else:
                tool_msg = await self._emit_completion_pair(
                    result,
                    call_id,
                    msg_dispatcher,
                )

        elif tool_reply_msg is not None:
            if _at_tail(tool_reply_msg):
                # If the current tail tool message is a placeholder, choose the strategy:
                # - For streaming progress placeholders, append a synthetic assistant→tool pair
                #   (preserves progress history as append-only).
                # - For simple 'pending' or 'nested_start' placeholders, update in-place so the
                #   final result lives under the original tool message (restores pre-change UX).
                placeholder_kind: Optional[str] = None
                with suppress(Exception):
                    _content_str = tool_reply_msg.get("content") or ""
                    if isinstance(_content_str, str):
                        parsed = json.loads(_content_str)
                        if isinstance(parsed, dict):
                            pk = parsed.get("_placeholder")
                            if isinstance(pk, str) and pk:
                                placeholder_kind = pk
                if placeholder_kind == "progress":
                    tool_msg = await self._emit_completion_pair(
                        result,
                        call_id,
                        msg_dispatcher,
                    )
                else:
                    tool_reply_msg["content"] = result
                    tool_msg = tool_reply_msg
            else:
                # Not at tail: emit a synthetic assistant→tool pair to carry the result
                tool_msg = await self._emit_completion_pair(
                    result,
                    call_id,
                    msg_dispatcher,
                )

        else:
            tool_msg = create_tool_call_message(name, call_id, result)
            await insert_tool_message_after_assistant(
                assistant_meta,
                asst_msg,
                tool_msg,
                self._client,
                msg_dispatcher,
            )

        # ── optional console logging for every finished tool call ────────────
        #     (mirrors the assistant-message logging above)
        if self._logger.log_steps:
            # Log EXACLY what was inserted, but redact base64 data URLs for readability
            try:
                safe_for_logs = sanitize_tool_msg_for_logging(tool_msg)
                self._logger.info(
                    f"{json.dumps(safe_for_logs, indent=4)}",
                    prefix=f"✅  ToolCall Completed [{time.perf_counter() - info.scheduled_time:.2f}s]",
                )
            except Exception:
                pass

        # 6️⃣  failure guard -------------------------------------------------
        if consecutive_failures.has_exceeded_failures():
            if self._logger.log_steps:
                self._logger.error(f"Aborting: too many tool failures.", prefix="🚨")
            raise RuntimeError(
                "Aborted after too many consecutive tool failures.",
            )

        # successful (or failed) *final* result → LLM may need to react
        return True

    # ── Helper: adopt a nested SteerableToolHandle into the current loop -----
    async def adopt_nested(
        self,
        info: "ToolCallMetadata",
        child_handle,
        *,
        msg_dispatcher,
        assistant_meta,
        outer_handle_container,
    ) -> None:
        """Adopt a child SteerableToolHandle returned by a tool into this loop.

        Creates/updates a single placeholder tool message, schedules the child's
        result as a nested task with inherited metadata, wires clarification
        channels, and synchronises passthrough interjections/pause/stop with the
        outer handle when applicable.
        """
        # Passthrough wiring: replay steering commands that were issued AFTER this
        # tool was scheduled and BEFORE adoption (per-child, no duplication), then
        # sync pause/stop state minimally.
        try:
            if (
                getattr(child_handle, "__passthrough__", False)
                and outer_handle_container
                and outer_handle_container[0] is not None
            ):
                _outer = outer_handle_container[0]
                adopt_now = time.perf_counter()
                POST_ADOPT_EPSILON = (
                    0.05  # small cushion to exclude post-adoption events
                )
                _log = list(getattr(_outer, "_steer_log", []) or [])
                for rec in _log:
                    # Skip replay for this child if the event was already forwarded to it
                    try:
                        fwd_list = rec.get("forwarded_to", [])
                        if isinstance(fwd_list, (list, tuple)) and str(
                            info.call_id,
                        ) in set(str(x) for x in fwd_list):
                            continue
                    except Exception:
                        pass
                    # Lower bound: event must have been recorded when this call_id
                    # was already scheduled (state-based, robust to timing races).
                    try:
                        sched_ids = rec.get("scheduled_call_ids") or []
                        if str(info.call_id) not in set(str(x) for x in sched_ids):
                            continue
                    except Exception:
                        continue
                    # Upper bound: exclude events that clearly arrived after adoption.
                    try:
                        t = rec.get("t", 0.0)
                        if (
                            isinstance(t, (int, float))
                            and (t - POST_ADOPT_EPSILON) > adopt_now
                        ):
                            continue
                    except Exception:
                        pass
                    method = rec.get("method") or ""
                    if not isinstance(method, str) or not method:
                        continue
                    # Do NOT replay pause/resume; adoption will sync current state below
                    _m_base = method.lower().strip()
                    if _m_base in ("pause", "resume"):
                        continue
                    args = rec.get("args") or ()
                    kwargs = rec.get("kwargs") or {}
                    fb = rec.get("fallback") or ()
                    # For custom methods (non built-ins), only replay when the child supports it
                    is_builtin = _m_base in (
                        "interject",
                        "ask",
                        "pause",
                        "resume",
                        "stop",
                        "clarify",
                    )
                    if not is_builtin:
                        try:
                            has_exact = callable(getattr(child_handle, method, None))
                        except Exception:
                            has_exact = False
                        try:
                            has_base = callable(getattr(child_handle, _m_base, None))
                        except Exception:
                            has_base = False
                        if not (has_exact or has_base):
                            # Skip replay and do not synthesize mirrors when unsupported
                            continue
                        # Prefer exact name if present
                        method_to_call = method if has_exact else _m_base
                    else:
                        method_to_call = method
                    await forward_handle_call(  # type: ignore[name-defined]
                        child_handle,
                        method_to_call,
                        kwargs,
                        call_args=args if isinstance(args, (list, tuple)) else (),
                        fallback_positional_keys=(
                            fb if isinstance(fb, (list, tuple)) else ()
                        ),
                    )
                    # Also mirror as a synthetic helper tool_call and acknowledgement (no LLM step)
                    try:
                        base = _m_base
                        helper_name = f"{base}_{info.name}_{str(info.call_id)[-6:]}"
                        if base in ("pause", "resume"):
                            # Already skipped replay; do not synthesize mirrors either
                            continue
                        # Build assistant message with a single tool_call
                        call_id = f"mirror_{int(time.perf_counter()*1000)}"
                        args_json = {}
                        if base == "interject":
                            msg = (kwargs or {}).get("message") or (kwargs or {}).get(
                                "content",
                            )
                            if msg is not None:
                                args_json["content"] = msg
                        elif base == "ask":
                            q = (kwargs or {}).get("question")
                            if q is not None:
                                args_json["question"] = q
                        elif base == "stop":
                            if "reason" in (kwargs or {}):
                                args_json["reason"] = kwargs.get("reason")
                        elif base == "clarify":
                            if "answer" in (kwargs or {}):
                                args_json["answer"] = kwargs.get("answer")
                        asst_msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": helper_name,
                                        "arguments": json.dumps(args_json or {}),
                                    },
                                },
                            ],
                        }
                        await msg_dispatcher.append_msgs([asst_msg])
                        # Ensure assistant_meta bookkeeping before inserting ack
                        assistant_meta[id(asst_msg)] = {"results_count": 0}
                        from .messages import acknowledge_helper_call  # local import

                        await acknowledge_helper_call(
                            asst_msg,
                            call_id,
                            helper_name,
                            json.dumps(args_json or {}),
                            assistant_meta=assistant_meta,
                            client=self._client,
                            msg_dispatcher=msg_dispatcher,
                        )
                    except Exception:
                        pass
                try:
                    if not getattr(_outer, "_pause_event", None).is_set() and hasattr(
                        child_handle,
                        "pause",
                    ):
                        child_handle.pause()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    if getattr(_outer, "_cancel_event", None).is_set() and hasattr(
                        child_handle,
                        "stop",
                    ):
                        maybe = child_handle.stop()  # type: ignore[attr-defined]
                        if asyncio.iscoroutine(maybe):
                            asyncio.create_task(maybe)
                except Exception:
                    pass
        except Exception:
            pass

        # Upgrade interject flag based on child capability
        if hasattr(child_handle, "interject"):
            info.is_interjectable = True

        h_up_q = getattr(child_handle, "clarification_up_q", info.clar_up_queue)
        h_down_q = getattr(child_handle, "clarification_down_q", info.clar_down_queue)
        if (h_up_q is not None) ^ (h_down_q is not None):
            raise AttributeError(
                f"Handle returned by tool {info.name!r} exposes only one of "
                "'clarification_up_q' / 'clarification_down_q'. Both are required (or neither).",
            )

        # Schedule child's result as nested task
        if inspect.iscoroutinefunction(child_handle.result):
            nested_coro = child_handle.result()
        else:
            nested_coro = asyncio.to_thread(child_handle.result)
        nested_task = asyncio.create_task(nested_coro)

        # Insert/update single placeholder for this call_id
        ph = info.tool_reply_msg
        if ph is None:
            ph = create_tool_call_message(
                name=info.name,
                call_id=info.call_id,
                content=json.dumps({"_placeholder": "nested_start"}, indent=4),
            )
            await insert_tool_message_after_assistant(
                assistant_meta,
                info.assistant_msg,
                ph,
                self._client,
                msg_dispatcher,
            )
            info.tool_reply_msg = ph
        else:
            ph["content"] = json.dumps({"_placeholder": "nested_start"}, indent=4)

        # Book-keeping for the new task (inherit, share placeholder)
        metadata = dataclasses.replace(
            info,
            handle=child_handle,
            is_interjectable=hasattr(child_handle, "interject"),
            tool_reply_msg=ph,
            clar_up_queue=h_up_q,
            clar_down_queue=h_down_q,
            notification_queue=info.notification_queue,
            is_passthrough=getattr(child_handle, "__passthrough__", False),
        )
        self.save_task(nested_task, metadata)
        if h_up_q is not None:
            self.clarification_channels[info.call_id] = (h_up_q, h_down_q)
