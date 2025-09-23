import asyncio
import time
import unify
import json
import inspect
import traceback
import dataclasses

from typing import Dict, Union, Callable, Tuple, Any, Set, Optional
from dataclasses import dataclass
from contextlib import suppress
from pydantic import BaseModel

from ...constants import LOGGER
from ..tool_spec import ToolSpec, normalise_tools
from .utils import maybe_await
from .event_bus_util import to_event_bus
from .messages import (
    find_unreplied_assistant_entries,
    chat_context_repr,
    generate_with_preprocess,
)
from .message_dispatcher import LoopMessageDispatcher
from .tools_utils import ToolCallMetadata, create_tool_call_message
from ..llm_helpers import method_to_schema, _dumps, _strip_image_keys, _collect_images
from .loop_config import LoopConfig, TOOL_LOOP_LINEAGE
from .tools_utils import ToolCallMessage, ToolCallMetadata, create_tool_call_message
from .timeout_timer import TimeoutTimer

# Dynamic-handle helpers ––––––––––––––––––––––––––––––––––––––––––––––––––––––
#  Public methods we *do not* expose again (already wrapped by dedicated helpers
#  or meaningless to the LLM).
_MANAGEMENT_METHOD_NAMES: set[str] = {
    "interject",
    "pause",
    "resume",
    "stop",
    "done",
    "result",
}


class LoopLogger:
    def __init__(self, cfg: LoopConfig, log_steps: bool | str) -> None:
        self._label = cfg.label
        self._log_steps = log_steps

    @property
    def log_steps(self):
        return self._log_steps

    @property
    def log_label(self):
        return self._label

    def info(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.info(txt)

    def error(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.error(txt)


# TODO this is not really required, but this just simplifies the extraction of the logic from the loop.
class _LoopToolFailureTracker:
    def __init__(self, max_consecutive_failures: int):
        self._consecutive_failures = 0
        self._max_consecutive_failures = max_consecutive_failures

    @property
    def current_failures(self):
        return self._consecutive_failures

    @property
    def max_failures(self):
        return self._max_consecutive_failures

    def has_exceeded_failures(self) -> bool:
        return self._consecutive_failures >= self._max_consecutive_failures

    def increment_failures(self):
        self._consecutive_failures += 1

    def reset_failures(self):
        self._consecutive_failures = 0


class _ToolsData:
    def __init__(self, tools, *, client, logger: LoopLogger):
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
                try:
                    fn_name = call.get("function", {}).get("name")
                except Exception:
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

        sub_q: Optional[asyncio.Queue[str]] = None
        if sig_accepts_interject_q:
            sub_q = asyncio.Queue()
            extra_kwargs["interject_queue"] = sub_q

        # Parse args
        try:
            call_args = (
                json.loads(args_json)
                if isinstance(args_json, str)
                else (args_json or {})
            )
        except Exception:
            call_args = {}

        # Filter extras to match fn signature
        filtered_extras = {
            k: v for k, v in extra_kwargs.items() if k in params or has_varkw
        }

        # Forward ALL call args verbatim. Let the callee raise if unsupported.
        allowed_call_args = call_args
        merged_kwargs = {**allowed_call_args, **filtered_extras}

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
        consecutive_failures: _LoopToolFailureTracker,
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
        5.  Record the payload in ``completed_results`` for later
            `_continue_<id>` helpers.
        6.  Enforce the *max_consecutive_failures* safety valve.
        """

        def _at_tail(msg: dict) -> bool:
            """True when *msg* is the very last entry in client.messages."""
            return bool(self._client.messages) and self._client.messages[-1] is msg

        info: ToolCallMetadata = self.pop_task(task)
        name = info.name
        call_id = info.call_id
        fn = info.call_dict["function"]["name"]
        arg = info.call_dict["function"]["arguments"]

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
                # If the nested handle explicitly requests pass-through behaviour
                # expose it directly to the outer caller *immediately*.
                if (
                    getattr(raw, "__passthrough__", False)
                    and outer_handle_container
                    and outer_handle_container[0] is not None
                ):
                    outer_handle_container[0]._adopt(raw)
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

                # 1️⃣ spawn the nested waiter
                #
                # ⤷ `handle.result` can now be **sync OR async**:
                #    • async ⇒ use the coroutine directly,
                #    • sync  ⇒ run it in a worker-thread so the event-loop never blocks.
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
                    await _insert_tool_message_after_assistant(
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
                try:
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
                except Exception:
                    pass

        # 3️⃣  remember so later `_continue_*` helpers can answer instantly
        self.completed_results[call_id] = result

        # 4️⃣  update / insert tool-result message --------------------------
        asst_msg = info.assistant_msg
        continue_msg = info.continue_msg
        clarify_ph = info.clarify_placeholder
        tool_reply_msg = info.tool_reply_msg

        if continue_msg is not None:
            if _at_tail(continue_msg):  # ✅ safe to overwrite
                continue_msg["content"] = result
                continue_msg["name"] = (
                    f"{fn}({arg}) completed successfully, "
                    "the return values are in the `content` field below."
                )
                tool_msg = continue_msg
            else:  # 🆕 keep history stable
                tool_msg = await self._emit_completion_pair(
                    result,
                    call_id,
                    msg_dispatcher,
                )

        elif clarify_ph is not None:
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
                tool_reply_msg["content"] = result
                tool_msg = tool_reply_msg
            else:
                tool_msg = await self._emit_completion_pair(
                    result,
                    call_id,
                    msg_dispatcher,
                )

        else:
            tool_msg = create_tool_call_message(name, call_id, result)
            await _insert_tool_message_after_assistant(
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
                prefix=f"🛠️  ToolCall Completed [{time.perf_counter() - info.scheduled_time:.2f}s]",
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


class DynamicToolFactory:
    @dataclass
    class _ToolContext:
        fn_name: str
        arg_repr: str
        call_id: str
        safe_call_id: str

    def __init__(self, tools_data: _ToolsData):
        self.dynamic_tools = {}
        self.tools_data = tools_data

    # Shared steering helpers – reduce duplication across dynamic helper tools
    @staticmethod
    def _adopt_signature_and_annotations(from_callable, to_wrapper) -> None:
        """Copy signature and annotations (excluding 'self') from from_callable to to_wrapper."""
        try:
            to_wrapper.__signature__ = inspect.signature(from_callable)
            try:
                ann = dict(getattr(from_callable, "__annotations__", {}))
                ann.pop("self", None)
                to_wrapper.__annotations__ = ann
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _discover_custom_public_methods(handle) -> dict[str, Callable]:
        """
        Return a mapping ``name → bound_method`` of *public* callables on *handle*:
            • name does **not** start with ``_``  _and_
            • name is not one of the management helpers above.
        """
        methods: dict[str, Callable] = {}
        for name, attr in inspect.getmembers(handle):
            if (
                name.startswith("_")
                or name in _MANAGEMENT_METHOD_NAMES
                or not callable(attr)
            ):
                continue
            # Bind the method to *handle* (important for late-added attributes).
            try:
                bound = handle.__getattribute__(name)
            except Exception:
                # Attribute access raised – treat as non-callable.
                continue

            methods[name] = bound
        return methods

    # helper: register a freshly-minted coroutine as a *temporary* tool
    def _register_tool(
        self,
        func_name: str,
        fallback_doc: str,
        fn: Callable,
    ) -> None:
        # prefer the function's own docstring if it exists, else fall back
        existing = inspect.getdoc(fn)
        fn.__doc__ = existing.strip() if existing else fallback_doc
        fn.__name__ = func_name[:64]
        fn.__qualname__ = func_name[:64]
        self.dynamic_tools[func_name.lstrip("_")] = fn

    def _create_continue_tool(
        self,
        tool_context: _ToolContext,
    ) -> None:
        async def _continue() -> Dict[str, str]:
            return {"status": "continue", "call_id": tool_context.call_id}

        self._register_tool(
            func_name=f"continue_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=f"Continue waiting for {tool_context.fn_name}({tool_context.arg_repr}).",
            fn=_continue,
        )

    def _create_stop_tool(
        self,
        tool_context: _ToolContext,
        task: asyncio.Task,
        handle: Any,
    ) -> None:
        doc = (
            f"Stop pending call {tool_context.fn_name}({tool_context.arg_repr}). "
            "Accepts any arguments supported by the underlying handle's `stop` method (e.g. `reason`)."
        )

        async def _stop(
            **_kw,
        ) -> Dict[str, str]:
            # Forward stop intent to the running handle with any extra kwargs
            if handle is not None and hasattr(handle, "stop"):
                await _forward_handle_call(
                    handle,
                    "stop",
                    _kw,
                    fallback_positional_keys=["reason"],
                )
            if not task.done():
                task.cancel()  # kill the waiter coroutine
            self.tools_data.pop_task(task)
            return {"status": "stopped", "call_id": tool_context.call_id, **_kw}

        self._register_tool(
            func_name=f"stop_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_stop,
        )
        # Expose full argspec of handle.stop in the helper schema
        with suppress(Exception):
            if handle is not None and hasattr(handle, "stop"):
                self._adopt_signature_and_annotations(getattr(handle, "stop"), _stop)

    def _create_interject_tool(
        self,
        tool_context: _ToolContext,
        task_info: ToolCallMetadata,
        handle: Any,
    ) -> None:
        doc = (
            f"Inject additional instructions for {tool_context.fn_name}({tool_context.arg_repr}). "
            "Accepts any arguments supported by the underlying handle's `interject` method (e.g. `content`)."
        )

        if handle is not None:

            async def _interject(**_kw) -> Dict[str, str]:
                # nested async-tool loop: delegate to its public API with full argspec
                with suppress(Exception):
                    await _forward_handle_call(
                        handle,
                        "interject",
                        _kw,
                        fallback_positional_keys=["content", "message"],
                    )
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    **{k: v for k, v in _kw.items()},
                }

            # Expose the downstream handle's signature to the LLM
            with suppress(Exception):
                self._adopt_signature_and_annotations(
                    getattr(handle, "interject"),
                    _interject,
                )

        else:

            async def _interject(content: str) -> Dict[str, str]:
                # regular tool: push onto its private queue
                await task_info.interject_queue.put(content)
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    "content": content,
                }

        self._register_tool(
            func_name=f"interject_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_interject,
        )

    def _create_clarify_tool(
        self,
        tool_context: _ToolContext,
    ) -> None:
        doc = (
            f"Provide an answer to the clarification which was requested by the (currently pending) tool "
            f"{tool_context.fn_name}({tool_context.arg_repr}). Takes a single argument `answer`."
        )

        async def _clarify(answer: str) -> Dict[str, str]:  # type: ignore[valid-type]
            return {
                "status": "clar_answer",
                "call_id": tool_context.call_id,
                "answer": answer,
            }

        self._register_tool(
            func_name=f"clarify_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_clarify,
        )

    def _create_pause_tool(
        self,
        tool_context: _ToolContext,
        handle: Any,
        pause_event: Optional[asyncio.Event],
    ) -> None:
        handle_available = handle is not None

        if handle_available and hasattr(handle, "pause"):

            async def _pause(**_kw) -> Dict[str, str]:
                with suppress(Exception):
                    await _forward_handle_call(handle, "pause", _kw)
                return {"status": "paused", "call_id": tool_context.call_id, **_kw}

            # Reflect downstream signature/annotations
            with suppress(Exception):
                self._adopt_signature_and_annotations(
                    getattr(handle, "pause"),
                    _pause,
                )

        else:

            async def _pause() -> Dict[str, str]:
                if handle_available and hasattr(handle, "pause"):
                    await maybe_await(handle.pause())
                elif pause_event is not None:
                    pause_event.clear()
                return {"status": "paused", "call_id": tool_context.call_id}

        self._register_tool(
            func_name=f"pause_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=f"Pause the pending call {tool_context.fn_name}({tool_context.arg_repr}).",
            fn=_pause,
        )

    def _create_resume_tool(
        self,
        tool_context: _ToolContext,
        handle: Any,
        pause_event: Optional[asyncio.Event],
    ) -> None:
        doc = f"Resume the previously paused call {tool_context.fn_name}({tool_context.arg_repr})."

        handle_available = handle is not None

        if handle_available and hasattr(handle, "resume"):

            async def _resume(**_kw) -> Dict[str, str]:
                with suppress(Exception):
                    await _forward_handle_call(handle, "resume", _kw)
                return {"status": "resumed", "call_id": tool_context.call_id, **_kw}

            with suppress(Exception):
                self._adopt_signature_and_annotations(
                    getattr(handle, "resume"),
                    _resume,
                )

        else:

            async def _resume() -> Dict[str, str]:
                if handle_available and hasattr(handle, "resume"):
                    await maybe_await(handle.resume())
                elif pause_event is not None:
                    pause_event.set()
                return {"status": "resumed", "call_id": tool_context.call_id}

        self._register_tool(
            func_name=f"resume_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_resume,
        )

    def _expose_public_methods(self, tool_context: _ToolContext, handle: Any):
        public_methods = self._discover_custom_public_methods(handle)

        # ── honour handle.valid_tools, if present ──────────────
        if hasattr(handle, "valid_tools"):
            allowed: set[str] = set(getattr(handle, "valid_tools", []))
            public_methods = {
                name: bound for name, bound in public_methods.items() if name in allowed
            }

        # Identify write-only helpers declared by the handle
        write_only_set: set[str] = set()
        with suppress(Exception):
            wo = getattr(handle, "write_only_methods", None)
            if wo is not None:
                write_only_set |= set(wo)

        with suppress(Exception):
            wo2 = getattr(handle, "write_only_tools", None)
            if wo2 is not None:
                write_only_set |= set(wo2)

        for meth_name, bound in public_methods.items():
            # use the same name we're about to give fn.__name__
            func_name = (
                f"{meth_name}_{tool_context.fn_name}_{tool_context.safe_call_id}"
            )
            helper_key = func_name

            # Skip if we already generated one this turn (possible when
            # the loop revisits the same pending task).
            if helper_key in self.dynamic_tools:
                continue

            # Write-only helpers: fire-and-forget operations
            if meth_name in write_only_set:

                async def _invoke_handle_method(
                    _method_name=meth_name,
                    **_kw,
                ):
                    # Robust forwarding incl. kwargs normalisation and fallbacks
                    with suppress(Exception):
                        await _forward_handle_call(
                            handle,
                            _method_name,
                            _kw,
                        )
                    # Write-only: no result propagation
                    return {"call_id": tool_context.call_id, "status": "ack"}

            else:

                async def _invoke_handle_method(
                    _method_name=meth_name,
                    **_kw,
                ):  # default args → capture current method name
                    """
                    Auto-generated wrapper that calls the corresponding
                    method on the live handle and **waits** for the return
                    value (sync or async).
                    """
                    # Use shared forwarding to support flexible args and fallbacks
                    res = await _forward_handle_call(
                        handle,
                        _method_name,
                        _kw,
                    )
                    return {"call_id": tool_context.call_id, "result": res}

            # override the wrapper's signature to match the real method
            _invoke_handle_method.__signature__ = inspect.signature(bound)

            self._register_tool(
                func_name=func_name,
                fallback_doc=(
                    (
                        f"Perform `{meth_name}` on the running handle (id={tool_context.call_id}). "
                        "Fire-and-forget write-only operation; returns immediately."
                    )
                    if meth_name in write_only_set
                    else (
                        f"Invoke `{meth_name}` on the running handle (id={tool_context.call_id}). "
                        "Returns when that method finishes."
                    )
                ),
                fn=_invoke_handle_method,
            )
            # Mark write-only helpers so scheduling can acknowledge and avoid tracking
            if meth_name in write_only_set:
                with suppress(Exception):
                    self.dynamic_tools[helper_key].__write_only__ = True  # type: ignore[attr-defined]

    def _process_task(self, task: asyncio.Task):
        info = self.tools_data.info[task]
        handle = info.handle
        task_pause_event = info.pause_event
        handle_available = handle is not None

        # ── DYNAMIC capability refresh (handle may change) ─────
        if handle_available:
            # 1. interjection
            info.is_interjectable = hasattr(handle, "interject")

            # 2. clarification queues
            h_up_q = getattr(
                handle,
                "clarification_up_q",
                info.clar_up_queue,
            )
            h_dn_q = getattr(
                handle,
                "clarification_down_q",
                info.clar_down_queue,
            )

            if (h_up_q is not None) ^ (h_dn_q is not None):
                raise AttributeError(
                    f"Handle of call {info.call_id} now exposes only one "
                    "of clarification queues; both or neither required.",
                )

            # update bookkeeping & channel map
            prev_up_q = info.clar_up_queue
            if h_up_q is not prev_up_q:
                # remove old mapping if any
                self.tools_data.clarification_channels.pop(info.call_id, None)
                if h_up_q is not None:
                    self.tools_data.clarification_channels[info.call_id] = (
                        h_up_q,
                        h_dn_q,
                    )
            info.clar_up_queue = h_up_q
            info.clar_down_queue = h_dn_q

        _call_id: str = info.call_id
        # Create a sanitized version of the call_id for use in function names.
        _safe_call_id: str = _call_id.replace("-", "_").split("_")[-1]
        _fn_name: str = info.name
        _arg_json: str = info.call_dict["function"]["arguments"]
        try:
            _arg_dict = json.loads(_arg_json)
            _arg_repr = ", ".join(f"{k}={v!r}" for k, v in _arg_dict.items())
        except Exception:
            _arg_repr = _arg_json  # fallback: raw JSON string

        create_tool_ctx = self._ToolContext(
            fn_name=_fn_name,
            arg_repr=_arg_repr,
            call_id=_call_id,
            safe_call_id=_safe_call_id,
        )

        if not info.waiting_for_clarification:
            self._create_continue_tool(create_tool_ctx)

        self._create_stop_tool(
            create_tool_ctx,
            task,
            handle,
        )

        if info.is_interjectable:
            self._create_interject_tool(
                create_tool_ctx,
                info,
                handle,
            )

        if info.clar_up_queue is not None:
            self._create_clarify_tool(create_tool_ctx)

        can_pause = (handle_available and hasattr(handle, "pause")) or task_pause_event
        if can_pause:
            self._create_pause_tool(
                create_tool_ctx,
                handle,
                task_pause_event,
            )

        can_resume = (
            handle_available and hasattr(handle, "resume")
        ) or task_pause_event
        if can_resume:
            self._create_resume_tool(
                create_tool_ctx,
                handle,
                task_pause_event,
            )

        # 7.  expose *all* other public methods of the handle
        if handle_available:
            self._expose_public_methods(create_tool_ctx, handle)

    def generate(self):
        for task in list(self.tools_data.pending):
            self._process_task(task)


# Helper Functions
def _normalise_kwargs_for_bound_method(bound_method, incoming_kw: dict) -> dict:
    """Normalise kwargs for a bound method: expand nested kwargs, drop noise keys,
    map common aliases when there is a single public param, and filter unknown keys
    unless **kwargs is accepted."""
    try:
        import inspect as _inspect

        sig = _inspect.signature(bound_method)
        params = sig.parameters
        has_varkw = any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        kw = dict(incoming_kw or {})

        # 1) Expand nested {"kwargs": {...}}
        if "kwargs" in kw and isinstance(kw["kwargs"], dict):
            nested_kw = kw.pop("kwargs")
            for k, v in nested_kw.items():
                kw.setdefault(k, v)

        # 2) Drop common placeholder noise keys when empty
        for _noise in ("a", "kw"):
            if _noise in kw and (kw[_noise] is None or kw[_noise] == ""):
                kw.pop(_noise, None)

        # 3) If exactly one public param, accept common aliases
        public_params = [n for n in params if n != "self"]
        if len(public_params) == 1 and public_params[0] not in kw:
            for alias in (
                "content",
                "message",
                "text",
                "prompt",
                "guidance",
                "instruction",
                "question",
                "query",
            ):
                if alias in kw:
                    kw[public_params[0]] = kw.pop(alias)
                    break

        # 4) Filter unknown keys unless **kwargs is accepted
        if not has_varkw:
            kw = {k: v for k, v in kw.items() if k in params}
        return kw
    except Exception:
        # Best-effort; return original
        return dict(incoming_kw or {})


async def _forward_handle_call(
    handle: Any,
    method_name: str,
    kwargs: dict | None,
    *,
    fallback_positional_keys: list[str] | tuple[str, ...] = (),
):
    """Invoke a steering method on a handle with robust kwargs handling.

    - Filters/normalises kwargs against the bound method's signature.
    - If the method rejects kwargs, tries positional fallback with the first
      available key from fallback_positional_keys (e.g., reason/content).
    - Finally falls back to calling without arguments.
    """
    try:
        bound = getattr(handle, method_name)
    except Exception:
        return None

    try:
        normalised = _normalise_kwargs_for_bound_method(bound, kwargs or {})
        return await maybe_await(bound(**normalised))
    except TypeError:
        # Fallbacks for legacy signatures
        for k in fallback_positional_keys:
            if kwargs and k in kwargs:
                try:
                    return await maybe_await(bound(kwargs.get(k)))  # type: ignore[misc]
                except Exception:
                    pass
        try:
            return await maybe_await(bound())  # type: ignore[misc]
        except Exception:
            return None
    except Exception:
        # Defensive: never let steering failures crash the loop
        return None


# ASYNC TOOL USE INNER HELPERS ────────────────────────────────────────────────


# Helper: detect helper-tool names (continue_/stop_/pause_/resume_/clarify_/interject_)
def _is_helper_tool(name: str) -> bool:
    return (
        name.startswith("continue_")
        or name.startswith("stop_")
        or name.startswith("pause_")
        or name.startswith("resume_")
        or name.startswith("clarify_")
        or name.startswith("interject_")
    )


# Helper: build human-readable acknowledgement content for helper tools
def _build_helper_ack_content(name: str, args_json: Any) -> str:
    ack_content = "Acknowledged."
    try:
        payload = (
            json.loads(args_json or "{}")
            if isinstance(args_json, str)
            else (args_json or {})
        )
    except Exception:
        payload = {}

    if name.startswith("continue_"):
        ack_content = "Continue request acknowledged. Still waiting for the original tool call to finish."
    elif name.startswith("stop_"):
        ack_content = "Stop request acknowledged. If the underlying call is still running, it will be stopped."
    elif name.startswith("pause_"):
        ack_content = "Pause request acknowledged. If the underlying call is still running, it will be paused."
    elif name.startswith("resume_"):
        ack_content = "Resume request acknowledged. If the underlying call was paused, it will be resumed."
    elif name.startswith("clarify_"):
        ans = payload.get("answer")
        ack_content = (
            f"Clarification answer received: {ans!r}. Waiting for the original tool to proceed."
            if ans is not None
            else "Clarification helper acknowledged. Waiting for the original tool to proceed."
        )
    elif name.startswith("interject_"):
        guidance = payload.get("content")
        ack_content = (
            f"Guidance forwarded to the running tool: {guidance!r}."
            if guidance
            else "Interjection acknowledged and forwarded to the running tool."
        )
    else:
        # Default acknowledgement for custom write-only helpers
        ack_content = (
            f"Operation {name!r} acknowledged and forwarded to the running tool."
        )
    return ack_content


# ── small helper: keep assistant→tool chronology DRY ────────────────────
async def _insert_tool_message_after_assistant(
    assistant_meta: dict,
    parent_msg: dict,
    tool_msg: ToolCallMessage,
    client,
    msg_dispatcher: LoopMessageDispatcher,
) -> None:
    """
    Append *tool_msg* and move it directly after *parent_msg*, while
    updating the per-assistant `results_count` bookkeeping.
    """
    meta = assistant_meta.setdefault(
        id(parent_msg),
        {"results_count": 0},
    )
    await msg_dispatcher.append_msgs([tool_msg])
    insert_pos = client.messages.index(parent_msg) + 1 + meta["results_count"]
    client.messages.insert(insert_pos, client.messages.pop())
    meta["results_count"] += 1


# Helper: propagate a stop request to any nested SteerableToolHandle returned
# by base tools. This ensures outer stop/cancel signals reach inner loops.
async def _propagate_stop_to_nested_handles(
    task_info,
    reason: Optional[str] = None,
) -> None:
    try:
        for _t, _inf in list(task_info.items()):
            h = _inf.get("handle")
            if h is not None and hasattr(h, "stop"):
                try:
                    await _forward_handle_call(
                        h,
                        "stop",
                        {"reason": reason} if reason is not None else {},
                        fallback_positional_keys=["reason"],
                    )
                except Exception:
                    # Best effort – never let propagation failure crash the loop
                    pass
    except Exception:
        pass


async def _propagate_stop_once(
    task_info,
    stop_forward_once,
    reason: Optional[str],
) -> bool:
    if stop_forward_once:
        return stop_forward_once
    await _propagate_stop_to_nested_handles(task_info, reason)
    return True


# Helper: insert a tool-acknowledgement message for helper tools
async def _acknowledge_helper_call(
    asst_msg: dict,
    call_id: str,
    name: str,
    args_json: Any,
    *,
    assistant_meta,
    client,
    msg_dispatcher,
) -> None:
    tool_msg = create_tool_call_message(
        name=name,
        call_id=call_id,
        content=_build_helper_ack_content(name, args_json),
    )
    await _insert_tool_message_after_assistant(
        assistant_meta,
        asst_msg,
        tool_msg,
        client,
        msg_dispatcher,
    )


# Ensure placeholder tool messages exist for pending tasks. If assistant_msg
# is provided, only affects tasks spawned by that assistant turn; otherwise
# applies to all pending tasks. Returns the list of call_ids for which a
# placeholder was created.
async def _ensure_placeholders_for_pending(
    assistant_msg: Optional[dict] = None,
    *,
    content: Optional[str] = None,
    tools_data: _ToolsData,
    assistant_meta,
    client,
    msg_dispatcher,
) -> list[str]:
    created: list[str] = []
    placeholder_content = (
        content
        if content is not None
        else "Pending… tool call accepted. Working on it."
    )
    for task in list(tools_data.pending):
        _inf = tools_data.info.get(task)
        if not _inf:
            continue
        if assistant_msg is not None and _inf.assistant_msg is not assistant_msg:
            continue
        if _inf.tool_reply_msg or _inf.continue_msg or _inf.clarify_placeholder:
            continue

        placeholder = create_tool_call_message(
            name=_inf.name,
            call_id=_inf.call_id,
            content=placeholder_content,
        )
        await _insert_tool_message_after_assistant(
            assistant_meta,
            _inf.assistant_msg,
            placeholder,
            client,
            msg_dispatcher,
        )
        _inf.tool_reply_msg = placeholder
        created.append(_inf.call_id)

    return created


# Helper: schedule a subset of tool_calls on a past assistant message and
# insert placeholders immediately. Skips already-scheduled/finished ids.
async def _schedule_missing_for_message(
    asst_msg: dict,
    only_ids: set[str],
    *,
    tools_data: _ToolsData,
    parent_chat_context,
    propagate_chat_context,
    assistant_meta,
    client,
    msg_dispatcher,
) -> list[str]:
    scheduled: list[str] = []
    try:
        tool_calls = asst_msg.get("tool_calls") or []
        for idx, call in enumerate(tool_calls):
            cid = call.get("id")
            if cid not in only_ids:
                continue

            # Skip if already pending or completed
            if any(task_info.call_id == cid for task_info in tools_data.info.values()):
                continue
            if cid in tools_data.completed_results:
                continue

            name = call["function"]["name"]
            args_json = call["function"].get("arguments", "{}")

            # Handle dynamic helpers similarly to main path
            if _is_helper_tool(name):
                # Do not execute helpers during backfill, only acknowledge
                try:
                    await _acknowledge_helper_call(
                        asst_msg,
                        cid,
                        name,
                        args_json,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=msg_dispatcher,
                    )
                except Exception:
                    pass
                scheduled.append(cid)
                continue

            # Base tool: locate function
            if name not in tools_data.normalized:
                scheduled.append(cid)
                continue

            await tools_data.schedule_base_tool_call(
                asst_msg,
                name=name,
                args_json=args_json,
                call_id=cid,
                call_idx=idx,
                parent_chat_context=parent_chat_context,
                propagate_chat_context=propagate_chat_context,
                assistant_meta=assistant_meta,
            )
            scheduled.append(cid)
    except Exception:
        pass
    # Ensure placeholders are present for backfilled items
    with suppress(Exception):
        await _ensure_placeholders_for_pending(
            assistant_msg=asst_msg,
            tools_data=tools_data,
            assistant_meta=assistant_meta,
            client=client,
            msg_dispatcher=msg_dispatcher,
        )
    return scheduled


def _check_valid_response_format(response_format: Any):
    # Require a Pydantic model class – anything else is a configuration error.
    if not (
        isinstance(response_format, type) and issubclass(response_format, BaseModel)
    ):
        raise TypeError(
            "response_format must be a Pydantic BaseModel subclass (e.g. MySchema).",
        )

    return response_format.model_json_schema()


async def async_tool_use_loop_inner(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    loop_id: Optional[str] = None,
    lineage: Optional[list[str]] = None,
    interject_queue: asyncio.Queue[str],
    cancel_event: asyncio.Event,
    stop_event: asyncio.Event | None = None,
    pause_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates: bool = True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    outer_handle_container: Optional[list] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    persist: bool = False,
) -> str:
    r"""
    Orchestrate an *interactive* "function-calling" dialogue between an LLM
    and a set of Python callables until the model yields a **final** plain-
    text answer.

    Key design points
    -----------------
    • **Concurrency** – every tool suggested by the model is wrapped in its
      own ``asyncio.Task`` so multiple long-running calls may advance in
      parallel; the loop always waits only for the *first* one to finish.

    • **Interruptibility** – the outer caller may:
        – set ``cancel_event`` → graceful shutdown (all tasks cancelled &
          awaited, then ``asyncio.CancelledError`` is re-raised);
        – queue ``interject_queue.put(text)`` → a new *user* turn injected
          just before the *next* LLM step without disturbing already running
          tools.

    • **Robustness** – exceptions inside tools are caught, serialised, and
      shown to the model; after ``max_consecutive_failures`` consecutive
      crashes the whole loop aborts with ``RuntimeError`` (prevents infinite
      failure ping-pong).

    • **Low coupling** – all transport (e.g. websockets, HTTP) can live
      outside; an optional ``event_bus`` lets a UI or logger subscribe to
      every message without the loop having to know who is listening.

    Parameters
    ----------
    client : ``unify.AsyncUnify``
        Pre-initialised Unify client that provides ``append_messages`` and
        ``generate``.  All tokens sent to / received from the LLM flow
        through this object.

    message : ``str``
        The very first user prompt that kicks-off the whole interactive
        session.

    tools : ``dict[str, Callable]``
        A mapping ``name → function`` describing every callable the LLM may
        invoke.  Each function must be fully type-hinted and have a concise
        docstring – these are automatically converted to an OpenAI *tool
        schema* via :pyfunc:`method_to_schema`.

    interject_queue : ``asyncio.Queue[str]``
        Thread-safe channel through which the *outer* application can push
        additional user turns at any time (e.g. the human changes their
        mind mid-generation).

    cancel_event : ``asyncio.Event``
        Flips to *set* when the outer caller wants graceful shutdown.  The
        loop then cancels every running task and propagates
        ``asyncio.CancelledError`` upstream.

    max_consecutive_failures : ``int``, default ``3``
        Hard safety valve: after this many back-to-back exceptions coming
        from tools the loop bails out with ``RuntimeError`` to avoid an
        infinite crash-and-retry ping-pong.

    ignore_tool_duplicates : ``bool``, default ``True``
        Deduplicates model-requested tool calls that have *identical*
        ``function.name`` **and** argument JSON.  Duplicates are pruned
        **in-place** before ever touching chat history or being scheduled.

    interrupt_llm_with_interjection : ``bool``, default ``True``
        Controls latency to fresh user input.  When *True* any in-flight
        ``client.generate`` is cancelled the moment a new user turn arrives
        so the assistant can pivot instantly.  When *False* the loop waits
        for the model to finish (legacy behaviour).

    propagate_chat_context : ``bool``, default ``True``
        If *True*, the entire conversation state of **this** loop is
        threaded into any child tool that accepts a
        ``parent_chat_context`` keyword argument.
        If *True*, the entire conversation state of **this** loop is threaded
        into any child tool via the *internal-only* ``parent_chat_context``
        argument.  This parameter is added automatically and is **not**
        exposed to the LLM.

     tool_policy : ``Callable | None``, default ``None``
         Optional callable that *dynamically* controls tool exposure **and**
         whether a tool call is **required** on a given turn.  Receives the
         current turn index (starting at ``0``) and the full mapping
         ``{name → callable}``.  It must return a tuple ``(policy, tools)``
         where ``policy`` is either ``"auto"`` or ``"required"`` (fed straight
         into ``tool_choice``) and ``tools`` is the possibly-filtered mapping
         of base tools visible on that turn.

    parent_chat_context : ``list[dict] | None``
        Nested chat structure passed from an **outer** loop.  When
        ``propagate_chat_context`` is on, the helper
        :pyfunc:`_chat_context_repr` merges this with the current
        ``client.messages`` and forwards the result downward.

    log_steps : ``bool | str``, default ``True``
        Controls verbosity of step logging to ``LOGGER``:
          • ``False`` – no logging
          • ``True``  – log everything except system messages
          • ``"full"`` – log everything including system messages

    Returns
    -------
    str
        The assistant's final plain-text reply *after* every tool result has
        been fed back into the conversation.
    """
    # unique id / lineage
    cfg = LoopConfig(loop_id, lineage, TOOL_LOOP_LINEAGE.get([]))
    logger = LoopLogger(cfg, log_steps)
    _token = TOOL_LOOP_LINEAGE.set(cfg.lineage)

    # normalise optional graceful stop event
    stop_event = stop_event or asyncio.Event()

    # If structured output is expected, inform the model up-front so it can
    # plan its reasoning with the final JSON shape in mind.  Enforcement via
    # `set_response_format` still happens at the end of the loop.
    if response_format is not None:
        try:
            _schema = _check_valid_response_format(response_format)
            _hint = (
                "\n\nNOTE: After completing all tool calls, your **final** assistant reply must be valid JSON that conforms to the following schema. Do NOT include any extra keys or commentary.\n"
                + json.dumps(_schema, indent=2)
            )

            client.set_system_message((client.system_message or "") + _hint)
        except Exception as _exc:  # noqa: BLE001
            logger.error(f"response_format hint failed: {_exc!r}")

    # ── runtime guards ────────────────────────────────────────────────────
    # rolling timeout ----------------------------------------------------
    timer: TimeoutTimer = TimeoutTimer(
        timeout=timeout,
        max_steps=max_steps,
        raise_on_limit=raise_on_limit,
        client=client,
    )
    _msg_dispatcher = LoopMessageDispatcher(client, cfg, timer)

    if log_steps:
        if log_steps == "full":
            if parent_chat_context:
                logger.info(
                    f"Parent Context: {json.dumps(parent_chat_context, indent=4)}\n",
                    prefix="⬇️",
                )
            logger.info(f"System Message: {client.system_message}\n", prefix="📋")
        logger.info(f"User Message: {message}\n", prefix="🧑‍💻")

    # ── 0-a. Inject **system** header with broader context ───────────────────
    #
    # When a parent context is supplied we prepend a single synthetic system
    # message that *summarises* it.  This offers the LLM immediate awareness
    # of the wider conversation without having to scroll the nested JSON.
    # The special marker ``_ctx_header=True`` lets us later strip it when
    # propagating context further down (avoids duplication).
    # -----------------------------------------------------------------------

    if parent_chat_context:
        sys_msg = {
            "role": "system",
            "_ctx_header": True,
            "content": (
                "Broader context (read-only):\n"
                f"{json.dumps(parent_chat_context, indent=2)}\n\n"
                "Resolve the *next* user request in light of this."
            ),
        }
        await _msg_dispatcher.append_msgs([sys_msg])

    # ── initial prompt ───────────────────────────────────────────────────────
    # ── 0-b. Coerce tools → ToolSpec & helper lambdas ───────────────────────
    #
    # • «tools_data.normalized» holds the *canonical* mapping name → ToolSpec
    # • helper for the active-count of one tool (cheap O(#pending))
    # • helper that answers "may we launch / advertise *this* tool right now?"
    #   by comparing the live count with max_concurrent.
    # -----------------------------------------------------------------------

    # Initialise loop state early so preflight backfill can schedule tasks
    tools_data: _ToolsData = _ToolsData(tools, client=client, logger=logger)
    consecutive_failures = _LoopToolFailureTracker(max_consecutive_failures)
    assistant_meta: Dict[int, Dict[str, Any]] = {}
    step_index: int = 0  # per assistant turn
    # Expose live task_info mapping on the current Task so outer handles/tests
    # can introspect currently running nested handles (used by ask/stop helpers).
    with suppress(Exception):
        _self_task = asyncio.current_task()
        if _self_task is not None:
            setattr(_self_task, "task_info", tools_data.info)  # type: ignore[attr-defined]

    # Ensure we forward stop to nested handles at most once, even if multiple
    # branches detect cancellation/stop around the same time.
    _stop_forwarded_once: bool = False

    # Preflight repair: backfill any pre-existing assistant tool_calls without replies
    with suppress(Exception):
        unreplied = find_unreplied_assistant_entries(client)
        if unreplied:
            # backfill for all such assistant messages (oldest → newest)
            for entry in unreplied:
                amsg = entry["assistant_msg"]
                # Before scheduling, drop any over-quota tool calls in this message
                tools_data.prune_over_quota_tool_calls(amsg)
                missing_ids = set(entry["missing"])
                await _schedule_missing_for_message(
                    amsg,
                    missing_ids,
                    tools_data=tools_data,
                    parent_chat_context=parent_chat_context,
                    propagate_chat_context=propagate_chat_context,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                )

    # ── initial **user** message
    if isinstance(message, dict):
        initial_user_msg = message
    else:
        initial_user_msg = {"role": "user", "content": message}

    await _msg_dispatcher.append_msgs([initial_user_msg])

    # ── helper: graceful early-exit when limits are hit ────────────────────
    async def _handle_limit_reached(reason: str) -> str:
        """
        Gracefully terminate the loop when *timeout* or *max_steps* are
        exceeded and `raise_on_limit` is *False*:
          • stop every pending tool (via handle.stop() if available)
          • cancel waiter coroutines
          • append a short assistant notice
        """
        for task in list(tools_data.pending):
            with suppress(Exception):
                inf = tools_data.info.get(task)
                if inf is not None and inf.handle is not None and hasattr(inf.handle, "stop"):  # type: ignore[attr-defined]
                    await maybe_await(inf.handle.stop())
            if not task.done():
                task.cancel()
        await asyncio.gather(*tools_data.pending, return_exceptions=True)
        tools_data.pending.clear()

        notice = {
            "role": "assistant",
            "content": f"🔚 Terminating early: {reason}",
        }
        await _msg_dispatcher.append_msgs([notice])
        if log_steps:
            logger.info(f"Early exit – {reason}", prefix="⏹️")
        return notice["content"]

    # Set to *True* whenever the loop must grant the LLM an immediate turn
    # before waiting again (user interjection, clarification answer, etc.).
    llm_turn_required = False

    # Last known assistant answer when the model produced a final tool-less reply.
    # Used when `persist=True` to return a stable result upon explicit stop.
    last_final_answer: Optional[str] = None

    try:
        while True:

            # ── 0-α-P. Global *pause* gate  ────────────────────────────
            # Keep handling tool completions & cancellation, but *never*
            # let the LLM speak while we're paused.
            if not pause_event.is_set():
                # Give any pending tool tasks a chance to finish OR wait until the
                # loop is resumed / cancelled.  Every coroutine is wrapped in an
                # asyncio.Task so `asyncio.wait()` is happy.
                if tools_data.pending:
                    pause_waiter = asyncio.create_task(
                        pause_event.wait(),
                        name="PauseEventWait",
                    )
                    cancel_waiter = asyncio.create_task(
                        cancel_event.wait(),
                        name="CancelEventWait",
                    )
                    graceful_stop_waiter = asyncio.create_task(
                        stop_event.wait(),
                        name="StopEventWait",
                    )
                    waiters = tools_data.pending | {
                        pause_waiter,
                        cancel_waiter,
                        graceful_stop_waiter,
                    }

                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # helper-task cleanup so they don't dangle
                    for w in (pause_waiter, cancel_waiter, graceful_stop_waiter):
                        if w not in done and not w.done():
                            w.cancel()
                            await asyncio.gather(w, return_exceptions=True)

                    # tool finished?
                    for t in done & tools_data.pending:
                        await tools_data.process_completed_task(
                            task=t,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        )
                    if cancel_event.is_set():
                        # Forward stop to any nested handles before aborting
                        with suppress(Exception):
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop cancelled",
                            )
                        raise asyncio.CancelledError
                    if stop_event.is_set() and persist:
                        with suppress(Exception):
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop stopped",
                            )
                        # Graceful stop requested during pause
                        return last_final_answer or ""
                    continue  # remain paused: do not allow the LLM to speak while paused
                else:
                    # nothing running – just idle until resumed or cancelled
                    done, _ = await asyncio.wait(
                        {
                            asyncio.create_task(
                                pause_event.wait(),
                                name="PauseEventWait",
                            ),
                            asyncio.create_task(
                                cancel_event.wait(),
                                name="CancelEventWait",
                            ),
                            asyncio.create_task(
                                stop_event.wait(),
                                name="StopEventWait",
                            ),
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # resumed?
                    if pause_event.is_set():
                        continue  # back to main loop, un-paused

                    # cancelled?
                    if cancel_event.is_set():
                        with suppress(Exception):
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop cancelled",
                            )
                        raise asyncio.CancelledError
                    if stop_event.is_set() and persist:
                        with suppress(Exception):
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop stopped",
                            )
                        return last_final_answer or ""
                        continue  # top-of-loop, still paused

            # 0-α. **Global timeout**
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # 0-β. **Chat history length**
            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            # 0-γ. Repair any outstanding assistant tool_calls missing replies
            #      before we allow new user interjections to be appended.
            with suppress(Exception):
                # Only consider the very latest assistant with missing replies first
                if unreplied := find_unreplied_assistant_entries(client):
                    last_problem = unreplied[-1]
                    amsg = last_problem["assistant_msg"]
                    missing_ids = set(last_problem["missing"])
                    # Skip if we already scheduled for this assistant turn
                    if id(amsg) not in assistant_meta:
                        backfilled = await _schedule_missing_for_message(
                            amsg,
                            missing_ids,
                            tools_data=tools_data,
                            parent_chat_context=parent_chat_context,
                            propagate_chat_context=propagate_chat_context,
                            assistant_meta=assistant_meta,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                        )

            # ── 0. Drain *all* queued interjections, allowed at any time ──
            # NOTE: We must do this *before* waiting on tool completion so a
            # fast typist can still sneak in a question while long-running
            # tools are in flight.  Doing it here keeps latency <1π loop.
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                llm_turn_required = True
                # Build system message based on the user-visible history stored on the outer handle.
                history_lines: list[str] = []
                try:
                    outer_handle = (
                        outer_handle_container[0] if outer_handle_container else None
                    )
                    uvh = (
                        getattr(outer_handle, "_user_visible_history", [])
                        if outer_handle
                        else []
                    )
                    for _m in uvh:
                        role = _m.get("role")
                        content = (_m.get("content") or "").strip()
                        if role in ("user", "assistant") and content:
                            history_lines.append(f"{role}: {content}")
                except Exception:
                    # Fallback to just the original user prompt if available
                    try:
                        first_user = next(
                            (
                                m.get("content", "")
                                for m in client.messages
                                if m.get("role") == "user"
                            ),
                            "",
                        )
                        if first_user:
                            history_lines = [f"user: {first_user}"]
                    except Exception:
                        history_lines = []

                sys_content = (
                    "The user *cannot* see *any* the contents of this ongoing tool use chat context. "
                    "They have just interjected with the following message (in bold at the bottom). "
                    "From their perspective, the conversation thus far is as follows:\n"
                    "--\n" + ("\n".join(history_lines)) + f"\nuser: **{extra}**\n"
                    "--\n"
                    "Please consider and incorporate *all* interjections in your final response to the user. "
                    "Later interjections should always override earlier interjections if there are "
                    "any conflicting comments/requests across the different interjections."
                )
                interjection_msg = {"role": "system", "content": sys_content}
                await _msg_dispatcher.append_msgs([interjection_msg])

                # Append this interjection to the user-visible history for future context
                with suppress(Exception):
                    if outer_handle:
                        outer_handle._user_visible_history.append(
                            {"role": "user", "content": extra},
                        )

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            # If a child just asked for clarification we also want to give
            # the LLM a chance to react immediately.
            # Skip this whole block if the model already needs to speak.
            # NOTE: ``asyncio.wait`` lets us race three conditions:
            #       • any tool task finishes
            #       • ``cancel_event`` flips
            #       • a *new* interjection appears
            if tools_data.pending and not llm_turn_required:
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                graceful_stop_waiter = asyncio.create_task(
                    stop_event.wait(),
                    name="StopEventWait",
                )
                clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in tools_data.pending:
                    # Only listen for *new* clarification questions.
                    # If the task is already awaiting an answer,
                    # `waiting_for_clarification` will be True.
                    if tools_data.info[_t].waiting_for_clarification:
                        continue

                    cuq = tools_data.info[_t].clar_up_queue
                    if cuq is not None:
                        w = asyncio.create_task(cuq.get(), name="ClarificationQueueGet")
                        clar_waiters[w] = _t
                waiters = (
                    tools_data.pending
                    | set(clar_waiters)
                    | {cancel_waiter, interject_w, graceful_stop_waiter}
                )

                # ── honour global *timeout* while we wait for tools ───────────
                if timer.has_exceeded_time():
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )

                done, _ = await asyncio.wait(
                    waiters,
                    timeout=timer.remaining_time(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # ── hit the timeout while waiting? ────────────────────────────
                if not done:
                    # nothing completed → the wait *timed out*
                    if raise_on_limit:
                        raise asyncio.TimeoutError(
                            f"Loop exceeded {timeout}s wall-clock limit",
                        )
                    else:
                        return await _handle_limit_reached(
                            f"timeout ({timeout}s) exceeded",
                        )

                # ── ensure *unused* auxiliary waiters don't linger ──────────
                # If one helper won the race we *must* cancel/await the other
                # so that it cannot consume the next interjection invisibly.
                for aux in (
                    interject_w,
                    cancel_waiter,
                    graceful_stop_waiter,
                    *clar_waiters.keys(),
                ):
                    if aux not in done and not aux.done():
                        aux.cancel()
                        await asyncio.gather(aux, return_exceptions=True)

                if interject_w in done:
                    # re-queue so branch 0 will handle user turn immediately
                    await interject_queue.put(interject_w.result())
                    continue  # → loop, will be processed in 0.

                if cancel_waiter in done:
                    with suppress(Exception):
                        _stop_forwarded_once = await _propagate_stop_once(
                            tools_data.info,
                            _stop_forwarded_once,
                            "outer-loop cancelled",
                        )
                    raise asyncio.CancelledError  # cancellation wins
                if graceful_stop_waiter in done and persist:
                    with suppress(Exception):
                        _stop_forwarded_once = await _propagate_stop_once(
                            tools_data.info,
                            _stop_forwarded_once,
                            "outer-loop stopped",
                        )
                    return last_final_answer or ""

                # ── clarification request bubbled up from a child tool ──────────────
                if done & clar_waiters.keys():
                    for cw in done & clar_waiters.keys():
                        question = cw.result()  # the text from the child
                        src_task = clar_waiters[cw]
                        call_id = tools_data.info[src_task].call_id

                        # 1️⃣ mark the task as waiting
                        tools_data.info[src_task].waiting_for_clarification = True

                        # 2️⃣ REUSE the existing placeholder if we already inserted one
                        ph = tools_data.info[src_task].tool_reply_msg
                        if ph is None:
                            # no placeholder yet → create one exactly once
                            ph = create_tool_call_message(
                                name=f"clarification_request_{call_id}",
                                call_id=call_id,
                                content="",  # will fill below
                            )
                            await _insert_tool_message_after_assistant(
                                assistant_meta,
                                tools_data.info[src_task].assistant_msg,
                                ph,
                                client,
                                _msg_dispatcher,
                            )
                            tools_data.info[src_task].tool_reply_msg = ph

                        # 3️⃣ turn (or update) the placeholder into the request
                        ph["name"] = f"clarification_request_{call_id}"
                        ph["content"] = (
                            "Tool incomplete, please answer the following to continue "
                            f"tool execution:\n{question}"
                        )
                        tool_msg = ph  # for event_bus

                    # let the assistant answer immediately
                    llm_turn_required = True
                    continue

                needs_turn = False
                for task in done:  # finished tool(s)
                    if await tools_data.process_completed_task(
                        task=task,
                        consecutive_failures=consecutive_failures,
                        outer_handle_container=outer_handle_container,
                        assistant_meta=assistant_meta,
                        msg_dispatcher=_msg_dispatcher,
                    ):
                        needs_turn = True

                # Other tools may still be running.
                if tools_data.pending:
                    if needs_turn:  # only when something new
                        llm_turn_required = True
                    continue  # jump to top-of-loop

            # ── B: wait for remaining tools before asking the LLM again,
            # unless the model already deserves a turn
            if tools_data.pending and not llm_turn_required:
                # Ensure placeholders exist for any pending calls before the next assistant turn
                await _ensure_placeholders_for_pending(
                    content=(
                        "Still running… you can use any of the available helper tools "
                        "to interact with this tool call while it is in progress."
                    ),
                    tools_data=tools_data,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                )
                continue  # still waiting for other tool tasks

            # ── C.  Add temporary tools so the LLM can **continue** or **cancel**
            #       any still‑running tool calls ────────────────────────────────
            #
            # For each pending ``asyncio.Task`` we synthesise two VERY small helper
            # tools and expose them to the model on the *next* LLM step.  Each
            # helper's docstring is a single line that embeds **both** the name of
            # the original function **and** the concrete arguments it was invoked
            # with – this gives the agent just enough context without overwhelming
            # the token budget.
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # 1.  Build the *static* part of the toolkit **fresh on every turn**
            #     so that concurrency changes (tasks finishing, stopping, …)
            #     are immediately reflected in what the LLM can see.
            # ------------------------------------------------------------------

            # 0.  Decide policy & tool-subset for this turn  ───────────────
            if tool_policy is not None:
                try:
                    tool_choice_mode, filtered = tool_policy(
                        step_index,
                        {n: s.fn for n, s in tools_data.normalized.items()},
                    )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    logger.error(
                        f"tool_policy raised on turn {step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", {
                        n: s.fn for n, s in tools_data.normalized.items()
                    }
                policy_tools_norm = normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = tools_data.normalized

            visible_base_tools_schema = [
                method_to_schema(spec.fn, name)
                for name, spec in policy_tools_norm.items()
                if tools_data.concurrency_ok(name) and tools_data.quota_ok(name)
            ]

            # Inject `final_answer` tool automatically whenever a `response_format` is
            # supplied. The tool accepts a single `answer` argument whose schema matches
            # the provided Pydantic model.
            if response_format is not None:
                try:
                    _answer_schema = _check_valid_response_format(response_format)

                    visible_base_tools_schema.append(
                        {
                            "type": "function",
                            "strict": True,
                            "function": {
                                "name": "final_answer",
                                "description": (
                                    "Submit your final answer in the required JSON format. "
                                    "Calling this tool marks the conversation as complete."
                                ),
                                "parameters": {
                                    "type": "object",
                                    "properties": {"answer": _answer_schema},
                                    "required": ["answer"],
                                },
                            },
                        },
                    )
                except Exception as _injection_exc:  # noqa: BLE001
                    logger.error(
                        f"Failed to inject final_answer tool: {_injection_exc!r}",
                    )

            dynamic_tool_factory = DynamicToolFactory(tools_data)
            dynamic_tool_factory.generate()
            dynamic_tools = dynamic_tool_factory.dynamic_tools

            # make sure every pending call already has a *tool* reply ──
            #  (a placeholder) before we let the assistant speak again.
            await _ensure_placeholders_for_pending(
                content=(
                    "Still running… you can use any of the available helper tools "
                    "to interact with this tool call while it is in progress."
                ),
                tools_data=tools_data,
                assistant_meta=assistant_meta,
                client=client,
                msg_dispatcher=_msg_dispatcher,
            )

            # Merge helpers into the visible toolkit for the upcoming LLM step
            tmp_tools = visible_base_tools_schema + [
                method_to_schema(
                    fn,
                    include_class_name=include_class_in_dynamic_tool_names,
                )
                for fn in dynamic_tools.values()
            ]

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                logger.info(f"LLM thinking…", prefix="🔄")

            if interrupt_llm_with_interjections:
                # ––––– new *pre-emptive* mode ––––––––––––––––––––––––––––
                # ➊ start the LLM step …
                _gen_kwargs = {
                    "return_full_completion": True,
                    "tools": tmp_tools,
                    "tool_choice": tool_choice_mode,
                    "stateful": True,
                }
                if max_parallel_tool_calls is not None:
                    _gen_kwargs["max_tool_calls"] = max_parallel_tool_calls

                llm_task = asyncio.create_task(
                    generate_with_preprocess(client, preprocess_msgs, **_gen_kwargs),
                    name="LLMGenerate",
                )
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )

                # ➋ …but ALSO watch the tool tasks that were still pending
                pending_snapshot = set(tools_data.pending)

                done, _ = await asyncio.wait(
                    pending_snapshot | {llm_task, interject_w, cancel_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # helper cleanup
                for tsk in (llm_task, interject_w, cancel_waiter):
                    if tsk not in done and not tsk.done():
                        tsk.cancel()
                await asyncio.gather(interject_w, cancel_waiter, return_exceptions=True)

                # 0️⃣ A *different* tool finished before the LLM answered -----
                if done & pending_snapshot:  # ← NEW
                    # — cancel the half-finished reasoning step
                    if not llm_task.done():
                        llm_task.cancel()
                    for aux in (interject_w, cancel_waiter):
                        if aux not in done and not aux.done():
                            aux.cancel()
                    await asyncio.gather(
                        llm_task,
                        interject_w,
                        cancel_waiter,
                        return_exceptions=True,
                    )
                    # — handle each newly-finished task exactly as branch A does
                    needs_turn = False
                    for task in done & pending_snapshot:
                        if await tools_data.process_completed_task(
                            task=task,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        ):
                            needs_turn = True

                    # …then restart the main loop so the model sees the new info
                    if needs_turn:  # assistant speaks only if needed
                        llm_turn_required = True
                    continue

                # 1️⃣ user interjected → restart immediately
                if interject_w in done:
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    await interject_queue.put(interject_w.result())
                    continue  # top of loop

                # 2️⃣ cancellation requested
                if cancel_waiter in done:
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    raise asyncio.CancelledError

                # 3️⃣ LLM finished normally
                if llm_task.exception():
                    try:
                        llm_task.result()
                    except Exception as e:
                        raise Exception(
                            f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
                        ) from e

            else:
                # ––––– legacy *blocking* mode ––––––––––––––––––––––––––––
                try:
                    _gen_kwargs = {
                        "return_full_completion": True,
                        "tools": tmp_tools,
                        "tool_choice": tool_choice_mode,
                        "stateful": True,
                    }
                    if max_parallel_tool_calls is not None:
                        _gen_kwargs["max_tool_calls"] = max_parallel_tool_calls

                    await generate_with_preprocess(
                        client,
                        preprocess_msgs,
                        **_gen_kwargs,
                    )
                except Exception:
                    raise Exception(
                        f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
                    )

            msg = client.messages[-1]
            await to_event_bus(msg, cfg)

            if log_steps:
                try:
                    logger.info(f"{json.dumps(msg, indent=4)}\n", prefix="🤖")
                except Exception:
                    logger.info(
                        f"Assistant message appended (unserializable)",
                        prefix="🤖",
                    )

            # ── timeout guard (post-LLM) ───────────────────────────────
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # LLM has just spoken – reset the flag
            llm_turn_required = False
            # one full assistant turn completed
            step_index += 1

            # ── E.  Launch any new tool calls  ──────────────────────────────
            # NOTE: The model returned `tool_calls`.  For *each* call we:
            #   1. JSON-parse the arguments once (costly in Python – do it
            #      outside the worker thread).
            #   2. Wrap sync functions in `asyncio.to_thread` so the event
            #      loop is never blocked by CPU / I/O.
            #   3. Create an `asyncio.Task` and remember contextual metadata
            #      in `task_info` so we can later insert the result in the
            #      exact chronological position.
            #   4. Keep a pristine copy of the original `tool_calls` list;
            #      step A temporarily hides it to avoid "naked" unresolved
            #      calls flashing in the UI, and restores it once *any*
            #      result for that assistant turn is ready.
            # Finally we `continue` so control jumps back to *branch A*
            # where we wait for the **first** task / cancel / interjection.
            if msg["tool_calls"]:
                # ── De-duplicate tool calls (optional) ────────────────────────
                if prune_tool_duplicates:
                    seen: Set[tuple[str, str]] = set()
                    unique_calls: list = []
                    for call in msg["tool_calls"]:
                        sig = (call["function"]["name"], call["function"]["arguments"])
                        if sig not in seen:
                            seen.add(sig)
                            unique_calls.append(call)
                    if len(unique_calls) != len(msg["tool_calls"]):
                        # mutate in-place so history never contains duplicates
                        msg["tool_calls"] = unique_calls

                # Always ensure over-quota tool calls are removed regardless of
                # deduplication settings, before any scheduling occurs.
                tools_data.prune_over_quota_tool_calls(msg)
                for idx, call in enumerate(msg["tool_calls"]):  # capture index
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])

                    # Special-case: handle synthetic `final_answer` tool
                    if name == "final_answer" and response_format is not None:
                        try:
                            payload = (
                                args.get("answer") if isinstance(args, dict) else None
                            )
                            if payload is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            # Validate payload with the provided Pydantic model.
                            response_format.model_validate(payload)

                            tool_msg = create_tool_call_message(
                                name="final_answer",
                                call_id=call["id"],
                                content=_dumps(payload, indent=4),
                            )

                            await _insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )

                            return json.dumps(payload)
                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name="final_answer",
                                call_id=call["id"],
                                content=(
                                    "⚠️ Validation failed – proceeding with standard formatting step.\n"
                                    + str(_exc)
                                ),
                            )
                            await _insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    # ── Special-case dynamic helpers ──────────────────────
                    # • continue_* → acknowledge, no scheduling
                    # • cancel_*   → cancel underlying task & purge metadata
                    if name.startswith("continue_"):
                        # Helper names are of the form: continue_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        tgt_task = next(
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        orig_fn = (
                            tools_data.info[tgt_task].name if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task].call_dict["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"continue {orig_fn}({arg_json})"

                        if tgt_task:  # still running → insert generated placeholder now
                            info = tools_data.info[tgt_task]
                            name = info.name
                            arg_json = info.call_dict["function"]["arguments"]
                            tool_reply_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    "The following tool calls are still running. If any of them are no longer "
                                    "relevant to the sequence of user requests, then you can call their "
                                    f"`_cancel_*` helper, otherwise feel free to call the corresponding "
                                    f"`_continue_*` helper to keep waiting:\n"
                                    f" • {name}({arg_json}) → cancel_{call['id']} / continue_{call['id']}"
                                ),
                            )
                            await _insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_reply_msg,
                                client,
                                _msg_dispatcher,
                            )
                            info.continue_msg = tool_reply_msg
                        else:  # the original tool already finished
                            # Lookup finished result by matching call-id suffix
                            _full_id = next(
                                (
                                    k
                                    for k in tools_data.completed_results.keys()
                                    if k.endswith(call_id_suffix)
                                ),
                                None,
                            )
                            finished = tools_data.completed_results.get(
                                _full_id,
                                _dumps(
                                    {"status": "not-found", "call_id": call_id_suffix},
                                    indent=4,
                                ),
                            )
                            tool_msg = create_tool_call_message(
                                name=pretty_name,
                                call_id=call["id"],
                                content=finished,
                            )
                            await _insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                        continue  # completed handling of this _continue

                    if name.startswith("stop_") and not name.startswith(
                        "_stop_tasks",
                    ):
                        # Helper names are of the form: stop_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        # ── locate & cancel the underlying coroutine ──────
                        task_to_cancel = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if str(info.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        orig_fn = (
                            tools_data.info[task_to_cancel].name
                            if task_to_cancel
                            else "unknown"
                        )
                        arg_json = (
                            tools_data.info[task_to_cancel].call_dict["function"][
                                "arguments"
                            ]
                            if task_to_cancel
                            else "{}"
                        )
                        pretty_name = f"stop   {orig_fn}({arg_json})"

                        # Parse payload to forward extras to handle.stop if available
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                        except Exception:
                            payload = {}

                        # ── gracefully shut down any *nested* async-tool loop first ──────
                        if task_to_cancel:
                            nested_handle = tools_data.info[task_to_cancel].handle
                            if nested_handle is not None:
                                # public API call – propagates cancellation downwards
                                await _forward_handle_call(
                                    nested_handle,
                                    "stop",
                                    payload,
                                    fallback_positional_keys=["reason"],
                                )

                        # ── then cancel the waiter coroutine itself ───────────────────────────
                        if task_to_cancel and not task_to_cancel.done():
                            task_to_cancel.cancel()
                        if task_to_cancel:
                            tools_data.pop_task(task_to_cancel)

                        tool_msg = create_tool_call_message(
                            name=pretty_name,
                            call_id=call["id"],
                            content=f"The tool call [{call_id_suffix}] has been stopped successfully.",
                        )
                        await _insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )

                        continue  # nothing else to schedule

                    # ── _pause helper ────────────────────────────────────────────────
                    if name.startswith("pause_") and not name.startswith(
                        "_pause_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if call_id_suffix in info.call_id
                            ),
                            None,
                        )
                        orig_fn = (
                            tools_data.info[tgt_task].name if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task].call_dict["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"pause {orig_fn}({arg_json})"

                        # Forward any extra kwargs to handle.pause if available
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                        except Exception:
                            payload = {}

                        if tgt_task:
                            h = tools_data.info[tgt_task].handle
                            ev = tools_data.info[tgt_task].pause_event
                            if h is not None and hasattr(h, "pause"):
                                await _forward_handle_call(h, "pause", payload)
                            elif ev is not None:
                                ev.clear()

                        tool_msg = create_tool_call_message(
                            name=pretty_name,
                            call_id=call["id"],
                            content=f"The tool call [{call_id_suffix}] has been paused successfully.",
                        )
                        await _insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue  # helper handled, move on

                    # ── _resume helper ───────────────────────────────────────────────
                    if name.startswith("resume_") and not name.startswith(
                        "_resume_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if call_id_suffix in info.call_id
                            ),
                            None,
                        )
                        orig_fn = (
                            tools_data.info[tgt_task].name if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task].call_dict["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"resume {orig_fn}({arg_json})"

                        # Forward any extra kwargs to handle.resume if available
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                        except Exception:
                            payload = {}

                        if tgt_task:
                            h = tools_data.info[tgt_task].handle
                            ev = tools_data.info[tgt_task].pause_event
                            if h is not None and hasattr(h, "resume"):
                                await _forward_handle_call(h, "resume", payload)
                            elif ev is not None:
                                ev.set()

                        tool_msg = create_tool_call_message(
                            name=pretty_name,
                            call_id=call["id"],
                            content=f"The tool call [{call_id_suffix}] has been resumed successfully.",
                        )
                        await _insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue  # helper handled

                    if name.startswith("clarify_"):
                        # Helper names are of the form: clarify_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]
                        ans = args["answer"]

                        # ── find the underlying pending task (if still alive) ───────────────
                        tgt_task = next(  # ← NEW
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        # Find clarification channel by matching call-id suffix
                        _clar_key = next(
                            (
                                k
                                for k in tools_data.clarification_channels.keys()
                                if k.endswith(call_id_suffix)
                            ),
                            None,
                        )
                        if _clar_key is not None:
                            await tools_data.clarification_channels[_clar_key][1].put(
                                ans,
                            )  # down-queue
                            # ✔️ the tool is un-blocked – start watching it again
                            for _t, _inf in tools_data.info.items():
                                if str(_inf.call_id).endswith(
                                    call_id_suffix,
                                ):
                                    _inf.waiting_for_clarification = False
                                    break
                        tool_reply_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=(
                                f"Clarification answer sent upstream: {ans!r}\n"
                                "⏳ Waiting for the original tool to finish…"
                            ),
                        )
                        await _insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_reply_msg,
                            client,
                            _msg_dispatcher,
                        )
                        if tgt_task is not None:
                            tools_data.info[tgt_task].clarify_placeholder = (
                                tool_reply_msg
                            )
                        continue

                    if name.startswith("interject_"):
                        # helper signature mirrors downstream handle.interject (content plus any extras)
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                            new_text = payload.get("content") or payload.get("message")
                            if new_text is None:
                                new_text = ""
                        except Exception:
                            payload = {}
                            new_text = "<unparsable>"

                        # Helper names are of the form: interject_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        # locate the underlying long-running task
                        tgt_task = next(
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        pretty_name = (
                            f"interject {tools_data.info[tgt_task].name}({new_text})"
                            if tgt_task
                            else name
                        )

                        # ― push guidance onto the private queue or forward to handle with full kwargs -------------
                        if tgt_task:
                            iq = tools_data.info[tgt_task].interject_queue
                            h = tools_data.info[tgt_task].handle

                            if iq is not None:
                                await iq.put(new_text)
                            elif h is not None and hasattr(h, "interject"):
                                await _forward_handle_call(
                                    h,
                                    "interject",
                                    payload,
                                    fallback_positional_keys=["content", "message"],
                                )

                        # ― emit a tool message so the chat log stays tidy ---
                        tool_msg = create_tool_call_message(
                            name=pretty_name,
                            call_id=call["id"],
                            content=f'Guidance "{new_text}" forwarded to the running tool.',
                        )
                        await _insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )

                        continue  # nothing else to schedule

                    # Respect hidden per-tool total-call quotas (pre-pruned); guard
                    if tools_data.has_exceeded_quota_for_tool(name):
                        continue

                    # Respect *per-tool* concurrency limits  ────────────────
                    if tools_data.has_exceeded_concurrent_limit_for_tool(name):
                        # Concurrency cap reached → immediately insert a
                        # *tool-error* message and **do not** schedule.
                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=(
                                f"⚠️ Cannot start '{name}': "
                                f"max_concurrent={tools_data.normalized[name].max_concurrent} "
                                "already reached. Wait for an existing call to "
                                "finish or stop one before retrying."
                            ),
                        )
                        await _insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    # first check any dynamic helpers we generated for long-running handles
                    if name in dynamic_tools:
                        fn = dynamic_tools[name]

                        # ── build **extra** kwargs (chat context + queue) for dynamic helper ──
                        extra_kwargs: dict = {}
                        if propagate_chat_context:
                            cur_msgs = [
                                m for m in client.messages if not m.get("_ctx_header")
                            ]
                            ctx_repr = chat_context_repr(parent_chat_context, cur_msgs)
                            extra_kwargs["parent_chat_context"] = ctx_repr

                        sig = inspect.signature(fn)
                        params = sig.parameters
                        has_varkw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params.values()
                        )
                        filtered_extras = {
                            k: v
                            for k, v in extra_kwargs.items()
                            if k in params or has_varkw
                        }
                        # Forward ALL call args verbatim. Let the callee raise if unsupported.
                        allowed_call_args = args
                        merged_kwargs = {**allowed_call_args, **filtered_extras}

                        if asyncio.iscoroutinefunction(fn):
                            coro = fn(**merged_kwargs)
                        else:
                            coro = asyncio.to_thread(fn, **merged_kwargs)

                        call_dict = {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        # If this dynamic helper is marked as write-only, acknowledge immediately
                        # and run fire-and-forget without tracking in pending/task_info.
                        if getattr(fn, "__write_only__", False):
                            with suppress(Exception):
                                tool_msg = create_tool_call_message(
                                    name=name,
                                    call_id=call["id"],
                                    content=_build_helper_ack_content(
                                        name,
                                        call["function"]["arguments"],
                                    ),
                                )
                                await _insert_tool_message_after_assistant(
                                    assistant_meta,
                                    msg,
                                    tool_msg,
                                    client,
                                    _msg_dispatcher,
                                )
                            with suppress(Exception):
                                asyncio.create_task(coro, name=f"ToolCall_{name}")
                            continue

                        # Scheduling dynamic helper call
                        t = asyncio.create_task(coro, name=f"ToolCall_{name}")
                        metadata = ToolCallMetadata(
                            name=name,
                            call_id=call["id"],
                            assistant_msg=msg,
                            call_dict=call_dict,
                            call_idx=idx,
                            is_interjectable=False,
                            chat_context=extra_kwargs.get("parent_chat_context"),
                            pause_event=None,
                            # Debug helpers for failure logging
                            tool_schema=method_to_schema(
                                fn,
                                include_class_name=include_class_in_dynamic_tool_names,
                            ),
                            llm_arguments=allowed_call_args,
                            raw_arguments_json=call["function"]["arguments"],
                        )
                        tools_data.save_task(
                            coro=t,
                            metadata=metadata,
                        )
                    else:
                        # Use shared helper for base tools
                        await tools_data.schedule_base_tool_call(
                            msg,
                            name=name,
                            args_json=call["function"]["arguments"],
                            call_id=call["id"],
                            call_idx=idx,
                            parent_chat_context=parent_chat_context,
                            propagate_chat_context=propagate_chat_context,
                            assistant_meta=assistant_meta,
                        )

                # metadata for orderly insertion
                assistant_meta[id(msg)] = {
                    "results_count": 0,
                }

                # Immediately insert placeholder tool replies for every newly scheduled call
                #  to satisfy API ordering even if a user interjection arrives instantly.
                try:
                    await _ensure_placeholders_for_pending(
                        assistant_msg=msg,
                        content="Pending… tool call accepted. Working on it.",
                        tools_data=tools_data,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
                except Exception as _ph_exc:
                    logger.error(
                        f"Failed to insert immediate placeholders: {_ph_exc!r}",
                    )

                continue  # finished scheduling tools, back to the very top

            # ── F.  No new tool calls  ──────────────────────────────────────
            # NOTE: Two scenarios reach this block:
            #   • `pending` **non-empty** → older tool tasks are still in
            #     flight; loop back to wait for them.
            #   • `pending` empty        → the model just produced a plain
            #     assistant message; nothing more to do – return it.
            if tools_data.pending:  # still running – stop them proactively, then finish
                try:
                    for t in list(tools_data.pending):
                        with suppress(Exception):
                            info_t = tools_data.info.get(t)
                            if (
                                info_t is not None
                                and (nested_handle := info_t.handle) is not None
                                and hasattr(
                                    nested_handle,
                                    "stop",
                                )
                            ):
                                await maybe_await(nested_handle.stop())
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tools_data.pending, return_exceptions=True)
                except Exception:
                    pass
                finally:
                    tools_data.pending.clear()

            # ── timeout guard (final turn) ──────────────────────────────────
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            final_answer = msg["content"]

            if not persist:
                return final_answer  # DONE!

            # Persist mode: remember latest final answer and enter a lingering state
            last_final_answer = final_answer

            # Wait for either a new interjection (to extend the loop),
            # a graceful stop (to return the last answer), or a hard cancel.
            while True:
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                graceful_stop_waiter = asyncio.create_task(
                    stop_event.wait(),
                    name="StopEventWait",
                )

                done, _ = await asyncio.wait(
                    {interject_w, cancel_waiter, graceful_stop_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # cleanup unused helpers
                for tsk in (interject_w, cancel_waiter, graceful_stop_waiter):
                    if tsk not in done and not tsk.done():
                        tsk.cancel()
                        await asyncio.gather(tsk, return_exceptions=True)

                if interject_w in done:
                    # push back so the standard interjection drain builds system guidance
                    await interject_queue.put(interject_w.result())
                    llm_turn_required = True
                    break  # resume main loop to handle new turn

                if cancel_waiter in done:
                    raise asyncio.CancelledError

                if graceful_stop_waiter in done:
                    return last_final_answer or ""

    except asyncio.CancelledError:  # graceful shutdown
        # NOTE: Caller (or parent task) requested cancellation.  We propagate
        # the signal to *all* running tool tasks first so each can release
        # resources cleanly.  Only after every task has finished/aborted do
        # we re-raise the same `CancelledError`, preserving expected asyncio
        # semantics for upstream callers.
        with suppress(Exception):
            _stop_forwarded_once = await _propagate_stop_once(
                tools_data.info,
                _stop_forwarded_once,
                "outer-loop cancelled",
            )
        await tools_data.cancel_pending_tasks()
        raise
    finally:
        with suppress(Exception):
            TOOL_LOOP_LINEAGE.reset(_token)
