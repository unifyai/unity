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
)
from .message_dispatcher import LoopMessageDispatcher
from ..tool_spec import normalise_tools
from ..llm_helpers import method_to_schema, _collect_images, _strip_image_keys, _dumps
from contextlib import suppress
from .loop_config import LIVE_IMAGES_REGISTRY
from .tools_utils import parse_arg_scoped_span, extract_alignment_text_from_value
from unity.image_manager.utils import substring_from_span

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
        with suppress(Exception):
            content_payload = (
                payload if isinstance(payload, dict) else {"message": str(payload)}
            )
            return _dumps({"tool": tool_name, **content_payload}, indent=4)
        return _dumps({"tool": tool_name, "message": str(payload)}, indent=4)

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
            extra_kwargs["parent_chat_context"] = ctx_repr

        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        sig_accepts_interject_q = "interject_queue" in params or has_varkw
        sig_accepts_pause_event = "pause_event" in params or has_varkw
        sig_accepts_clar_qs = (
            "clarification_up_q" in params and "clarification_down_q" in params
        ) or has_varkw
        sig_accepts_progress = "notification_up_q" in params or has_varkw

        pause_ev: Optional[asyncio.Event] = None
        if sig_accepts_pause_event:
            pause_ev = asyncio.Event()
            pause_ev.set()  # start running
            extra_kwargs["pause_event"] = pause_ev

        clar_up_q: Optional[asyncio.Queue[str]] = None
        clar_down_q: Optional[asyncio.Queue[str]] = None
        if sig_accepts_clar_qs:
            clar_up_q = asyncio.Queue()
            clar_down_q = asyncio.Queue()
            extra_kwargs["clarification_up_q"] = clar_up_q
            extra_kwargs["clarification_down_q"] = clar_down_q

        progress_q: Optional[asyncio.Queue[dict]] = None
        if sig_accepts_progress:
            progress_q = asyncio.Queue()
            extra_kwargs["notification_up_q"] = progress_q

        sub_q: Optional[asyncio.Queue[str]] = None
        if sig_accepts_interject_q:
            sub_q = asyncio.Queue()
            extra_kwargs["interject_queue"] = sub_q

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

        # ── Normalise arg-scoped image mapping for inner tools, but skip
        #     source-scoped helpers like `ask_image` which expect `<source>[x:y]`.
        if "images" in params and isinstance(merged_kwargs.get("images"), dict):
            if name == "ask_image":
                # Keep source-scoped mapping verbatim for the helper to process.
                pass
            else:
                try:
                    raw_images = dict(merged_kwargs.get("images") or {})
                    registry = LIVE_IMAGES_REGISTRY.get()
                    norm_images: dict[str, Any] = {}
                    for key, val in raw_images.items():
                        parsed = parse_arg_scoped_span(str(key))
                        if not parsed:
                            continue
                        arg_name, span = parsed
                        # Only accept if referenced arg is available in the call
                        if arg_name not in params and arg_name not in merged_kwargs:
                            continue
                        # Resolve id → handle or accept provided handle
                        handle = None
                        try:
                            if isinstance(val, int):
                                handle = (
                                    registry.get(int(val))
                                    if isinstance(registry, dict)
                                    else None
                                )
                            elif hasattr(val, "image_id"):
                                handle = val
                            elif isinstance(val, dict):
                                # Accept explicit id fields inside the dict
                                _id_field = None
                                for _k in ("image_id", "imageId", "id"):
                                    if _k in val:
                                        _id_field = val[_k]
                                        break
                                if _id_field is not None:
                                    try:
                                        handle = (
                                            registry.get(int(_id_field))
                                            if isinstance(registry, dict)
                                            else None
                                        )
                                    except Exception:
                                        handle = None
                                elif bool(val.get("__handle__")):
                                    # Fallback: when a single live image exists, use it
                                    if (
                                        isinstance(registry, dict)
                                        and len(registry) == 1
                                    ):
                                        try:
                                            handle = next(iter(registry.values()))
                                        except Exception:
                                            handle = None
                        except Exception:
                            handle = None
                        if handle is None:
                            continue
                        # Validate the span against the referenced argument's text; drop if invalid/empty.
                        try:
                            align_txt = extract_alignment_text_from_value(
                                merged_kwargs.get(arg_name),
                            )
                            # Only keep non-empty matches
                            if align_txt is not None:
                                matched = substring_from_span(str(align_txt), span)
                                if isinstance(matched, str) and matched != "":
                                    norm_images[str(key)] = handle
                        except Exception:
                            # If validation fails, skip this entry
                            continue
                    merged_kwargs["images"] = norm_images
                except Exception:
                    # If anything goes wrong, leave images as-is
                    pass

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
            chat_context=extra_kwargs.get("parent_chat_context"),
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
                # Passthrough: do NOT hand over control to the nested handle.
                # Keep the outer loop alive. We still want to forward early
                # interjections and synchronise pause/stop state with the
                # newly created handle, but without adopting it as a delegate.
                if (
                    getattr(raw, "__passthrough__", False)
                    and outer_handle_container
                    and outer_handle_container[0] is not None
                ):
                    try:
                        _outer = outer_handle_container[0]
                        # Wire pause/stop state with the new handle; outer loop keeps running.
                        # Forward any early interjections that were queued before
                        # this passthrough handle existed. Do NOT consume the outer
                        # buffer so that subsequent passthrough handles also receive them.
                        _early = list(getattr(_outer, "_early_interjects", []))
                        for _msg in _early:
                            try:
                                if isinstance(_msg, dict):
                                    maybe_coro = raw.interject(  # type: ignore[attr-defined]
                                        _msg.get("message", ""),
                                        parent_chat_context_cont=_msg.get(
                                            "parent_chat_context_continuted",
                                        ),
                                    )
                                else:
                                    maybe_coro = raw.interject(_msg)  # type: ignore[attr-defined]
                                if asyncio.iscoroutine(maybe_coro):
                                    await maybe_coro
                            except Exception as _exc:
                                pass
                        # Synchronise pause/cancel signals with the new handle
                        try:
                            if not getattr(
                                _outer,
                                "_pause_event",
                                None,
                            ).is_set() and hasattr(raw, "pause"):
                                raw.pause()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        try:
                            if getattr(
                                _outer,
                                "_cancel_event",
                                None,
                            ).is_set() and hasattr(raw, "stop"):
                                maybe = raw.stop()  # type: ignore[attr-defined]
                                if asyncio.iscoroutine(maybe):
                                    asyncio.create_task(maybe)
                        except Exception:
                            pass
                    except Exception:
                        pass

                # ── upgrade interject / clarification flags from handle ─────
                if hasattr(raw, "interject"):
                    info.is_interjectable = True

                h_up_q = getattr(raw, "clarification_up_q", info.clar_up_queue)
                h_down_q = getattr(raw, "clarification_down_q", info.clar_down_queue)

                if (h_up_q is not None) ^ (h_down_q is not None):
                    raise AttributeError(
                        f"Handle returned by tool {info.name!r} exposes only "
                        "one of 'clarification_up_q' / 'clarification_down_q'. "
                        "Both queues are required (or neither).",
                    )

                # 1️⃣ spawn the nested waiter (passthrough/non-passthrough nested handle)
                if inspect.iscoroutinefunction(raw.result):
                    nested_coro = raw.result()  # already a coroutine
                else:
                    nested_coro = asyncio.to_thread(raw.result)  # turn sync → coroutine

                nested_task = asyncio.create_task(nested_coro)

                # 2️⃣ insert / update a single placeholder
                ph = info.tool_reply_msg
                if ph is None:
                    ph = create_tool_call_message(
                        name=info.name,
                        call_id=call_id,
                        content="Nested async tool loop started… waiting for result.",
                    )
                    await insert_tool_message_after_assistant(
                        assistant_meta,
                        info.assistant_msg,
                        ph,
                        self._client,
                        msg_dispatcher,
                    )
                    info.tool_reply_msg = ph  # remember on *parent*
                else:
                    ph["content"] = (
                        "Nested async tool loop started… waiting for result."
                    )

                # 3️⃣ book-keeping for the *new* task (inherit + share placeholder)
                metadata = dataclasses.replace(
                    info,
                    handle=raw,
                    is_interjectable=hasattr(raw, "interject"),
                    tool_reply_msg=ph,
                    clar_up_queue=h_up_q,
                    clar_down_queue=h_down_q,
                    notification_queue=info.notification_queue,
                    is_passthrough=getattr(raw, "__passthrough__", False),
                )
                self.save_task(nested_task, metadata)
                if h_up_q is not None:
                    self.clarification_channels[call_id] = (h_up_q, h_down_q)
                return False  # ⬅️  no LLM turn required

            # ───────────────────────────────────────────────────────────────
            #  Normal (non-handle) result – unchanged path
            # ───────────────────────────────────────────────────────────────
            # ── finished successfully – promote any embedded images ─────────
            images: list[str] = []
            _collect_images(raw, images)

            text_repr = _dumps(_strip_image_keys(raw), indent=4)

            if images:
                content_blocks: list = []
                if text_repr and text_repr != "{}":
                    content_blocks.append({"type": "text", "text": text_repr})
                content_blocks.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                    for b64 in images
                )
                result = content_blocks
            else:
                result = text_repr

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
                # If the current tail tool message looks like a progress payload,
                # do NOT emit another tool reply for the same call_id – instead
                # create a synthetic assistant→tool pair to carry the final result.
                with suppress(Exception):
                    _content_str = tool_reply_msg.get("content") or ""
                if "_content_str" not in locals():
                    _content_str = ""
                if isinstance(_content_str, str) and '"tool"' in _content_str:
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
            # Create a clean version of tool_msg for logging (strip image data)
            tool_msg_for_logging = tool_msg.copy()
            if isinstance(tool_msg_for_logging.get("content"), list):
                # Filter out image_url items and keep only text content
                tool_msg_for_logging["content"] = [
                    item
                    for item in tool_msg_for_logging["content"]
                    if item.get("type") != "image_url"
                ]
            self._logger.info(
                f"{json.dumps(tool_msg_for_logging, indent=4)}\n",
                prefix=f"✅  ToolCall Completed [{time.perf_counter() - info.scheduled_time:.2f}s]",
            )

        # 6️⃣  failure guard -------------------------------------------------
        if consecutive_failures.has_exceeded_failures():
            if self._logger.log_steps:
                self._logger.error(f"Aborting: too many tool failures.", prefix="🚨")
            raise RuntimeError(
                "Aborted after too many consecutive tool failures.",
            )

        # successful (or failed) *final* result → LLM may need to react
        return True
