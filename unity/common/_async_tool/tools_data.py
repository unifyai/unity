import asyncio
import inspect
import json
import traceback
import dataclasses
import time


from typing import (
    Callable,
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
    _normalise_kwargs_for_bound_method,
)
from ..tool_spec import normalise_tools
from ..llm_helpers import method_to_schema
from .formatting import serialize_tool_content, sanitize_tool_msg_for_logging
from contextlib import suppress
from .propagation_mode import ChatContextPropagation
from .context_tracker import LoopContextState

if TYPE_CHECKING:  # TODO: remove once dependencies are fixed
    from .loop import LoopLogger, _LoopToolFailureTracker
    from .message_dispatcher import LoopMessageDispatcher
    from .time_context import TimeContext


# Sentinel for bare top-level handles (no label needed).
_HANDLE_SENTINEL = "<steerable handle — now in-flight>"


def _handle_label_sentinel(label: str) -> str:
    """Labeled sentinel for a handle inside a composite return."""
    return f"[{label}: steerable]"


@dataclasses.dataclass
class _MultiHandleState:
    """Shared state for multiple handles spawned from one tool return."""

    parent_call_id: str
    parent_name: str
    placeholder_msg: dict
    template: Any  # cleaned structure with labeled sentinels
    results: dict  # label -> raw result (None while pending)

    def update_placeholder(self) -> None:
        """Rebuild the shared placeholder with any completed results merged in."""
        updated = _rebuild_multi_handle_content(self.template, self.results)
        all_done = all(v is not None for v in self.results.values())
        self.placeholder_msg["content"] = serialize_tool_content(
            tool_name=self.parent_name,
            payload=updated,
            is_final=all_done,
        )


def _rebuild_multi_handle_content(template, results):
    """Replace labeled sentinels in *template* with completed results."""

    def _walk(node):
        if isinstance(node, str):
            for label, raw in results.items():
                if node == _handle_label_sentinel(label) and raw is not None:
                    return raw
            return node
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(_walk(v) for v in node)
        return node

    return _walk(template)


def _extract_nested_handle(obj):
    """Walk *obj* (dict / list / tuple / Pydantic model) looking for ``SteerableToolHandle`` instances.

    Returns ``(result, cleaned_obj)`` where:

    - **Bare top-level handle**: ``result`` is the handle itself, ``cleaned_obj``
      is :data:`_HANDLE_SENTINEL`.
    - **Handles nested in containers** (one or more): ``result`` is a list of
      ``(handle, label)`` tuples (``"h0"``, ``"h1"``, …), ``cleaned_obj`` is a
      copy of *obj* with handles replaced by labeled sentinels.
    - **No handles**: ``(None, obj)`` unchanged.
    """
    from unity.common.async_tool_loop import SteerableToolHandle

    if isinstance(obj, SteerableToolHandle):
        return obj, _HANDLE_SENTINEL

    found: list = []
    counter = [0]

    def _walk(node):
        if isinstance(node, SteerableToolHandle):
            label = f"h{counter[0]}"
            counter[0] += 1
            found.append((node, label))
            return _handle_label_sentinel(label)

        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}

        if isinstance(node, list):
            return [_walk(v) for v in node]

        if isinstance(node, tuple):
            return tuple(_walk(v) for v in node)

        # Pydantic BaseModel: walk public field values so handles inside
        # e.g. ExecutionResult.result are detected.
        try:
            from pydantic import BaseModel

            if isinstance(node, BaseModel):
                changed = False
                updates = {}
                for field_name in node.model_fields:
                    val = getattr(node, field_name)
                    cleaned_val = _walk(val)
                    if cleaned_val is not val:
                        updates[field_name] = cleaned_val
                        changed = True
                if changed:
                    return node.model_copy(update=updates)
                return node
        except ImportError:
            pass

        return node

    cleaned = _walk(obj)

    if found:
        return found, cleaned

    return None, obj


def compute_context_injection(
    *,
    args: dict,
    propagate_chat_context: ChatContextPropagation,
    context_state: LoopContextState,
    client_messages: list,
    call_id: str,
    accepts_parent_ctx: bool,
    accepts_parent_ctx_cont: bool,
    target_context_opted_in: Optional[bool] = None,
    is_continuation_only: bool = False,
) -> Tuple[dict, bool]:
    """
    Shared helper for computing context injection kwargs.

    This is used by both base tool dispatch and dynamic tool dispatch to ensure
    consistent handling of include_parent_chat_context and include_parent_chat_context_cont.

    Parameters
    ----------
    args : dict
        The tool call arguments (will be mutated to pop context control params).
    propagate_chat_context : ChatContextPropagation
        The loop's propagation mode (ALWAYS, NEVER, or LLM_DECIDES).
    context_state : LoopContextState
        The loop's context tracking state.
    client_messages : list
        The current conversation messages (filtered for _ctx_header if needed).
    call_id : str
        Unique identifier for this tool call (used for context tracking).
    accepts_parent_ctx : bool
        Whether the target function accepts _parent_chat_context.
    accepts_parent_ctx_cont : bool
        Whether the target function accepts _parent_chat_context_cont.
    target_context_opted_in : Optional[bool]
        For steering tools: whether the target tool initially opted into context.
        If None, this is treated as a new tool call (not steering).
    is_continuation_only : bool
        If True, only compute continuation context (for interject_*).
        If False, compute full initial context (for base tools and ask_*).

    Returns
    -------
    Tuple[dict, bool]
        (extra_kwargs, context_opted_in) where extra_kwargs contains the context
        params to inject and context_opted_in indicates the opt-in decision.
    """
    extra_kwargs: dict = {}

    # Pop the LLM control parameters from args
    llm_include_ctx = args.pop("include_parent_chat_context", True)
    llm_include_ctx_cont = args.pop("include_parent_chat_context_cont", True)

    # Determine whether to inject context based on propagation mode
    should_inject_ctx = False

    if is_continuation_only:
        # For steering tools like interject_*, check if the target tool opted in
        if target_context_opted_in:
            if propagate_chat_context == ChatContextPropagation.ALWAYS:
                should_inject_ctx = True
            elif propagate_chat_context == ChatContextPropagation.LLM_DECIDES:
                should_inject_ctx = llm_include_ctx_cont
            # NEVER mode: should_inject_ctx stays False
    else:
        # For base tools and ask_*, use the standard logic
        if accepts_parent_ctx or accepts_parent_ctx_cont:
            if propagate_chat_context == ChatContextPropagation.ALWAYS:
                should_inject_ctx = True
            elif propagate_chat_context == ChatContextPropagation.NEVER:
                should_inject_ctx = False
            elif propagate_chat_context == ChatContextPropagation.LLM_DECIDES:
                should_inject_ctx = llm_include_ctx

    # Compute and inject context if needed
    if should_inject_ctx:
        cur_msgs = [m for m in client_messages if not m.get("_ctx_header")]

        if is_continuation_only:
            # For steering tools, only compute continuation
            _, ctx_cont = context_state.compute_context_for_inner_tool(
                call_id,
                cur_msgs,
            )
            if ctx_cont and accepts_parent_ctx_cont:
                extra_kwargs["_parent_chat_context_cont"] = ctx_cont
        else:
            # For base tools / ask_*, compute full context
            parent_ctx, parent_ctx_cont = context_state.compute_context_for_inner_tool(
                call_id,
                cur_msgs,
            )
            if parent_ctx is not None and accepts_parent_ctx:
                extra_kwargs["_parent_chat_context"] = parent_ctx
            if parent_ctx_cont is not None and accepts_parent_ctx_cont:
                extra_kwargs["_parent_chat_context_cont"] = parent_ctx_cont

    return extra_kwargs, should_inject_ctx


class ToolsData:
    def __init__(
        self,
        tools,
        *,
        client,
        logger: "LoopLogger",
        time_ctx: "Optional[TimeContext]" = None,
        extra_ask_tools: "Optional[Dict[str, Callable]]" = None,
    ):
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
        # Tool name for every completed tool (steerable or not), keyed by call_id.
        self._completed_tool_names: Dict[str, str] = {}
        # Callback for refreshing dynamic helpers when a handle is adopted
        self._on_handle_adopted: Optional[Callable[[asyncio.Task], None]] = None
        # Time context for inline timing annotations on tool results
        self._time_ctx: Optional["TimeContext"] = time_ctx
        # Reference to the live dynamic_tools dict managed by DynamicToolFactory.
        # Set by the loop after the factory is initialised each turn.
        self._dynamic_tools_ref: Optional[Dict[str, Callable]] = None
        self._completed_ask_handles: Dict[str, Callable] = {}
        self._task_ask_keys: Dict[asyncio.Task, str] = {}
        # Metadata for completed steerable tools, keyed by call_id.
        # Each entry: {"name": str, "call_id": str, "arg_repr": str, "ask_fn": Callable, "handle": Any}
        self._completed_askable_tools: Dict[str, dict] = {}
        # Caller-supplied ask tools injected at construction time (e.g.
        # domain-specific read-only tools for handle.ask() inspection loops).
        self._extra_ask_tools: Dict[str, Callable] = (
            dict(extra_ask_tools) if extra_ask_tools else {}
        )

    def get_ask_tools(self) -> Dict[str, Callable]:
        """Return a snapshot of currently available ``ask_*`` dynamic tools.

        Merges three sources with increasing precedence:
        completed ask handles < extra_ask_tools < live dynamic tools.

        The ``ask_about_completed_tool`` meta-dispatcher is excluded because
        it is a routing tool, not a per-tool ask function.
        """
        result = dict(self._completed_ask_handles)
        result.update(self._extra_ask_tools)
        dt = self._dynamic_tools_ref
        if dt and isinstance(dt, dict):
            result.update(
                {
                    k: v
                    for k, v in dt.items()
                    if k.startswith("ask_") and k != "ask_about_completed_tool"
                },
            )
        return result

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

    def _at_tail(self, msg: dict) -> bool:
        """True when *msg* is the very last entry in client.messages."""
        return bool(self._client.messages) and self._client.messages[-1] is msg

    async def _emit_completion_pair(
        self,
        result: str,
        call_id: str,
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> dict:
        """
        Append a synthetic assistant→tool pair carrying the final result
        at the chronologically correct position (end of messages).
        """
        status_call_id = f"{call_id}_completed"
        status_tool_name = f"check_status_{call_id}"

        assistant_stub = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": status_call_id,
                    "type": "function",
                    "function": {
                        "name": status_tool_name,
                        "arguments": "{}",
                    },
                },
            ],
        }
        tool_msg = create_tool_call_message(
            name=status_tool_name,
            call_id=status_call_id,
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
        # Before removing, retain the ask_* dynamic tool handle for this task
        # so handle.ask() can propagate post-completion.
        info = self.info.get(coro)
        ask_name = self._task_ask_keys.pop(coro, None)
        if ask_name is not None:
            dt = self._dynamic_tools_ref
            if dt and isinstance(dt, dict) and ask_name in dt:
                ask_fn = dt[ask_name]
                self._completed_ask_handles[ask_name] = ask_fn
                # Store metadata for the ask_about_completed_tool dispatcher.
                if info is not None:
                    call_id = info.call_id
                    arg_json = info.call_dict["function"].get("arguments", "{}")
                    try:
                        arg_dict = json.loads(arg_json)
                        arg_repr = ", ".join(f"{k}={v!r}" for k, v in arg_dict.items())
                    except Exception:
                        arg_repr = arg_json
                    self._completed_askable_tools[call_id] = {
                        "name": info.name,
                        "call_id": call_id,
                        "arg_repr": arg_repr,
                        "ask_fn": ask_fn,
                        "handle": info.handle,
                    }
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

    def prune_over_quota_tool_calls(self, asst_msg: dict) -> None:
        """
        In-place remove tool_calls from asst_msg if they would exceed the per-tool quota.
        This ensures strict provider compliance: calls that are not executed
        must not remain in the history without a response.
        """
        tcs = asst_msg.get("tool_calls")
        if not tcs:
            return

        # Track counts locally to handle multiple calls in this single batch
        # without permanently modifying self.call_counts yet (that happens on schedule).
        temp_counts = self.call_counts.copy()

        valid_tcs = []
        for tc in tcs:
            try:
                name = tc.get("function", {}).get("name")

                if name not in self.normalized:
                    # Unknown tools are kept (handled by execution/error logic)
                    valid_tcs.append(tc)
                    continue

                spec = self.normalized[name]
                limit = spec.max_total_calls
                current = temp_counts.get(name, 0)

                if limit is not None and current >= limit:
                    # Prune this call - do not add to valid_tcs
                    continue

                # Keep it, and increment temp counter
                temp_counts[name] = current + 1
                valid_tcs.append(tc)
            except Exception:
                # Malformed tool call, keep it
                valid_tcs.append(tc)

        # In-place mutation of the assistant message
        asst_msg["tool_calls"] = valid_tcs

        # If the message becomes empty (no content, no tools), inject a placeholder content
        # to satisfy API constraints and inform the model.
        has_content = bool(asst_msg.get("content"))
        if not valid_tcs and not has_content:
            asst_msg["content"] = "(Tool calls were removed due to quota limits)"

    # Helper: schedule a base tool call (shared by main path and backfill)
    async def schedule_base_tool_call(
        self,
        asst_msg: dict,
        *,
        name: str,
        args_json: Any,
        call_id: str,
        call_idx: int,
        context_state: LoopContextState,
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

        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        # Parse args early so we can check include_parent_chat_context
        with suppress(Exception):
            call_args = (
                json.loads(args_json)
                if isinstance(args_json, str)
                else (args_json or {})
            )
        if "call_args" not in locals():
            call_args = {}

        sig_accepts_parent_ctx = "_parent_chat_context" in params or has_varkw
        sig_accepts_parent_ctx_cont = "_parent_chat_context_cont" in params or has_varkw

        # Use shared helper for context injection logic
        ctx_extra_kwargs, context_opted_in = compute_context_injection(
            args=call_args,
            propagate_chat_context=propagate_chat_context,
            context_state=context_state,
            client_messages=self._client.messages,
            call_id=call_id,
            accepts_parent_ctx=sig_accepts_parent_ctx,
            accepts_parent_ctx_cont=sig_accepts_parent_ctx_cont,
            is_continuation_only=False,
        )

        # Build extra kwargs (chat context, interject/clarification/pause)
        extra_kwargs: dict = dict(ctx_extra_kwargs)

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
            # Track context opt-in for steering method context propagation
            context_opted_in=context_opted_in,
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
        3.  Patch or insert the correct **tool** message.
        4.  Emit the event-bus hook (if configured).
        5.  Record the payload in ``completed_results`` for potential post-hoc lookups.
        6.  Enforce the *max_consecutive_failures* safety valve.
        """
        import time as _pct_time

        _pct_t0 = _pct_time.perf_counter()

        def _pct_ms():
            return f"{(_pct_time.perf_counter() - _pct_t0) * 1000:.0f}ms"

        info: ToolCallMetadata = self.pop_task(task)
        name = info.name
        call_id = info.call_id

        _pickup_delay = _pct_time.perf_counter() - info.scheduled_time
        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} ({call_id}) "
            f"total_elapsed={_pickup_delay:.2f}s",
        )

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
                        skip_event_bus=True,  # Progress placeholder; final result published later
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

        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} notification drain done",
        )

        # 2️⃣  obtain result -------------------------------------------------
        try:
            raw = task.result()

            # ───────────────────────────────────────────────────────────────
            #  Multi-handle child: progressive placeholder update.
            #  Each child completes independently; the shared placeholder
            #  is rebuilt with the newly resolved result.
            # ───────────────────────────────────────────────────────────────
            mh_state = getattr(info, "_multi_handle_state", None)
            if mh_state is not None:
                label = info._multi_handle_label
                mh_state.results[label] = raw
                mh_state.update_placeholder()
                self.completed_results[call_id] = serialize_tool_content(
                    tool_name=name,
                    payload=raw,
                    is_final=True,
                )
                self._completed_tool_names[call_id] = name
                consecutive_failures.reset_failures()
                if self._logger.log_steps:
                    self._logger.info(
                        f"{name} [{label}] - {call_id}",
                        prefix="✅  MultiHandle Child Completed",
                    )
                return True

            # ───────────────────────────────────────────────────────────────
            #  Bare handle: the tool returned a SteerableToolHandle directly.
            # ───────────────────────────────────────────────────────────────
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
            #  Composite return: one or more handles nested inside a
            #  dict/list/tuple alongside intermediate data surfaced to the
            #  LLM as progress while each handle is steered independently.
            # ───────────────────────────────────────────────────────────────
            nested_handles, cleaned = _extract_nested_handle(raw)
            if nested_handles is not None:
                await self.adopt_multi_nested(
                    info,
                    nested_handles,
                    cleaned,
                    msg_dispatcher=msg_dispatcher,
                    assistant_meta=assistant_meta,
                    outer_handle_container=outer_handle_container,
                )
                return True  # ⬅️  LLM turn required — intermediate content to process

            # ───────────────────────────────────────────────────────────────
            #  Normal (non-handle) result – unchanged path
            # ───────────────────────────────────────────────────────────────
            # ── finished successfully – promote any embedded images ─────────
            # Centralized serialization for final tool results
            result = serialize_tool_content(tool_name=name, payload=raw, is_final=True)

            # Wrap with inline timing metadata for non-dynamic (base) tools
            if self._time_ctx is not None and not info.is_dynamic:
                result = self._time_ctx.wrap_result(result, info.scheduled_time)

            consecutive_failures.reset_failures()
        except Exception:
            # Multi-handle child error: update shared placeholder and return early
            mh_state = getattr(info, "_multi_handle_state", None)
            if mh_state is not None:
                label = info._multi_handle_label
                error_tb = traceback.format_exc()
                mh_state.results[label] = f"[{label}: error]\n{error_tb}"
                mh_state.update_placeholder()
                self.completed_results[call_id] = error_tb
                self._completed_tool_names[call_id] = name
                consecutive_failures.increment_failures()
                if self._logger.log_steps:
                    self._logger.error(
                        f"{name} [{label}] - {call_id}\n{error_tb}",
                        prefix="❌  MultiHandle Child Failed",
                    )
                if consecutive_failures.has_exceeded_failures():
                    raise RuntimeError(
                        "Aborted after too many consecutive tool failures.",
                    )
                return True

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
        self._completed_tool_names[call_id] = name

        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} result obtained",
        )

        # 4️⃣  update / insert tool-result message --------------------------
        asst_msg = info.assistant_msg
        clarify_ph = info.clarify_placeholder
        tool_reply_msg = info.tool_reply_msg

        # Placeholder handling with chronological ordering:
        # - At tail: update in-place
        # - Not at tail: mark as completed, emit check_status pair at end
        placeholder = clarify_ph or tool_reply_msg

        if placeholder is not None:
            if self._at_tail(placeholder):
                placeholder["content"] = result
                tool_msg = placeholder
                # Publish the now-complete tool message to EventBus
                # (placeholder insertion skipped EventBus; now we have final content)
                await msg_dispatcher.publish_to_event_bus([tool_msg])
            else:
                placeholder["content"] = json.dumps(
                    {
                        "_placeholder": "completed",
                        "status": "Tool completed. See check_status result below.",
                        "result_call_id": f"{call_id}_completed",
                    },
                )
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

        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} tool message emitted",
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

        # 5️⃣  failure guard -------------------------------------------------
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
        intermediate_content: Any = None,
    ) -> None:
        """Adopt a child SteerableToolHandle returned by a tool into this loop.

        Creates/updates a single placeholder tool message, schedules the child's
        result as a nested task with inherited metadata, and wires clarification
        channels.

        When *intermediate_content* is provided (from a composite return where
        the handle was nested inside a data structure), the placeholder is
        populated with the intermediate data formatted as a progress notification
        so the LLM can react to it while steering continues.
        """
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

        # Insert/update single placeholder for this call_id.
        # When intermediate_content is provided the placeholder carries
        # the partial data formatted as progress so the LLM can react.
        if intermediate_content is not None:
            placeholder_content = serialize_tool_content(
                tool_name=info.name,
                payload=intermediate_content,
                is_final=False,
            )
        else:
            placeholder_content = json.dumps(
                {"_placeholder": "nested_start"},
                indent=4,
            )

        ph = info.tool_reply_msg
        if ph is None:
            ph = create_tool_call_message(
                name=info.name,
                call_id=info.call_id,
                content=placeholder_content,
            )
            await insert_tool_message_after_assistant(
                assistant_meta,
                info.assistant_msg,
                ph,
                self._client,
                msg_dispatcher,
                skip_event_bus=True,  # Nested placeholder; final result published when nested task completes
            )
            info.tool_reply_msg = ph
        else:
            ph["content"] = placeholder_content

        # Book-keeping for the new task (inherit, share placeholder)
        metadata = dataclasses.replace(
            info,
            handle=child_handle,
            is_interjectable=hasattr(child_handle, "interject"),
            tool_reply_msg=ph,
            clar_up_queue=h_up_q,
            clar_down_queue=h_down_q,
            notification_queue=info.notification_queue,
        )
        self.save_task(nested_task, metadata)
        if h_up_q is not None:
            self.clarification_channels[info.call_id] = (h_up_q, h_down_q)
        # Refresh dynamic helpers immediately now that handle is available
        if self._on_handle_adopted is not None:
            with suppress(Exception):
                self._on_handle_adopted(nested_task)

    # ── Helper: adopt multiple handles from a single composite return --------
    async def adopt_multi_nested(
        self,
        info: "ToolCallMetadata",
        handles: list,
        intermediate_content: Any,
        *,
        msg_dispatcher,
        assistant_meta,
        outer_handle_container,
    ) -> None:
        """Adopt multiple handles from a single tool's composite return.

        Creates one placeholder for the parent call_id with intermediate content
        showing labeled sentinels (``[h0: steerable]``, etc.), then schedules
        each handle as an independent child task with a synthesized call_id.

        Each child completes independently, progressively updating the shared
        placeholder via :class:`_MultiHandleState`.
        """
        parent_call_id = info.call_id

        # Create / update placeholder with intermediate content
        placeholder_content = serialize_tool_content(
            tool_name=info.name,
            payload=intermediate_content,
            is_final=False,
        )
        ph = info.tool_reply_msg
        if ph is None:
            ph = create_tool_call_message(
                name=info.name,
                call_id=parent_call_id,
                content=placeholder_content,
            )
            await insert_tool_message_after_assistant(
                assistant_meta,
                info.assistant_msg,
                ph,
                self._client,
                msg_dispatcher,
                skip_event_bus=True,
            )
        else:
            ph["content"] = placeholder_content

        # Shared state that all children reference
        state = _MultiHandleState(
            parent_call_id=parent_call_id,
            parent_name=info.name,
            placeholder_msg=ph,
            template=intermediate_content,
            results={label: None for _, label in handles},
        )

        for handle, label in handles:
            # Append label directly (no underscore) so the 8-char safe_call_id
            # includes both parent uniqueness and the handle label, avoiding
            # collisions when multiple parents each have an h0.
            synth_call_id = f"{parent_call_id}{label}"

            # Schedule the handle's result coroutine
            if inspect.iscoroutinefunction(handle.result):
                nested_coro = handle.result()
            else:
                nested_coro = asyncio.to_thread(handle.result)
            nested_task = asyncio.create_task(nested_coro)

            # Wire clarification channels from the handle
            h_up_q = getattr(handle, "clarification_up_q", None)
            h_down_q = getattr(handle, "clarification_down_q", None)

            metadata = dataclasses.replace(
                info,
                call_id=synth_call_id,
                handle=handle,
                is_interjectable=hasattr(handle, "interject"),
                tool_reply_msg=ph,
                clar_up_queue=h_up_q,
                clar_down_queue=h_down_q,
                notification_queue=None,
                _multi_handle_state=state,
                _multi_handle_label=label,
            )
            self.save_task(nested_task, metadata)

            if h_up_q is not None:
                self.clarification_channels[synth_call_id] = (h_up_q, h_down_q)

            if self._on_handle_adopted is not None:
                with suppress(Exception):
                    self._on_handle_adopted(nested_task)
