from __future__ import annotations

import asyncio
from enum import Enum
import os
import json
from typing import Optional, Dict, Callable, Tuple, Any, Union, TypedDict, Literal

import unify

# Delegate to the existing inner loop for now (skeleton orchestrator).
from .loop import async_tool_loop_inner as _legacy_tool_loop_inner
from .loop import _check_valid_response_format as _resp_schema
from .messages import generate_with_preprocess as _gwp
from .messages import _is_helper_tool as _is_helper_tool
from .messages import build_helper_ack_content as _helper_ack
from .messages import (
    ensure_placeholders_for_pending as _ensure_placeholders_for_pending,
)
from .messages import (
    insert_tool_message_after_assistant as _insert_tool_message_after_assistant,
)
from .utils import maybe_await
from ..llm_helpers import method_to_schema as _method_to_schema
from ..llm_helpers import _dumps as _json_pretty
from ..tool_spec import normalise_tools
from ...constants import LOGGER
from .tools_utils import create_tool_call_message
from .tools_data import ToolsData as _ToolsData
from .message_dispatcher import LoopMessageDispatcher as _Dispatcher
from .loop_config import LoopConfig as _LoopConfig
from .timeout_timer import TimeoutTimer as _Timer
from .loop import LoopLogger as _LoopLogger
from .loop import _LoopToolFailureTracker as _FailureTracker
import inspect


# ─────────────────────────────────────────────────────────────────────────────
# Event and State definitions (skeleton)
# ─────────────────────────────────────────────────────────────────────────────


class State(Enum):
    IDLE = "idle"
    WAITING_LLM = "waiting_llm"
    RUNNING_TOOLS = "running_tools"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"


class _BaseEvent(TypedDict):
    type: str


class LLMCompletedEvent(_BaseEvent, total=False):
    type: Literal["llm_completed"]
    message: dict


class LLMFailedEvent(_BaseEvent, total=False):
    type: Literal["llm_failed"]
    error: str


class LLMPreemptedEvent(_BaseEvent, total=False):
    type: Literal["llm_preempted"]


class ToolCompletedEvent(_BaseEvent, total=False):
    type: Literal["tool_completed"]
    call_id: str
    name: str
    result: Any


class ToolFailedEvent(_BaseEvent, total=False):
    type: Literal["tool_failed"]
    call_id: str
    name: str
    error: str


class ClarificationRequestedEvent(_BaseEvent, total=False):
    type: Literal["clarification_requested"]
    call_id: str
    tool_name: str
    question: str


class NotificationReceivedEvent(_BaseEvent, total=False):
    type: Literal["notification_received"]
    call_id: str
    tool_name: str
    message: str


class InterjectedEvent(_BaseEvent, total=False):
    type: Literal["interjected"]
    content: Any


class PauseRequestedEvent(_BaseEvent, total=False):
    type: Literal["pause_requested"]


class ResumeRequestedEvent(_BaseEvent, total=False):
    type: Literal["resume_requested"]


class CancelRequestedEvent(_BaseEvent, total=False):
    type: Literal["cancel_requested"]
    reason: Optional[str]


class TimeoutEvent(_BaseEvent, total=False):
    type: Literal["timeout"]


Event = Union[
    LLMCompletedEvent,
    LLMFailedEvent,
    LLMPreemptedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ClarificationRequestedEvent,
    NotificationReceivedEvent,
    InterjectedEvent,
    PauseRequestedEvent,
    ResumeRequestedEvent,
    CancelRequestedEvent,
    TimeoutEvent,
]


class Orchestrator:
    """
    Event-driven orchestrator skeleton. For now, delegates to the legacy loop.
    """

    def __init__(
        self,
        *,
        client: unify.AsyncUnify,
        message: str | dict | list[str | dict],
        tools: Dict[str, Callable],
        loop_id: Optional[str],
        lineage: Optional[list[str]],
        interject_queue: asyncio.Queue[dict | str],
        cancel_event: asyncio.Event,
        stop_event: asyncio.Event | None,
        pause_event: asyncio.Event,
        max_consecutive_failures: int,
        prune_tool_duplicates: bool,
        interrupt_llm_with_interjections: bool,
        propagate_chat_context: bool,
        parent_chat_context: Optional[list[dict]],
        log_steps: Union[bool, str],
        max_steps: Optional[int],
        timeout: Optional[int],
        raise_on_limit: bool,
        include_class_in_dynamic_tool_names: bool,
        tool_policy: Optional[
            Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]
        ],
        preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]],
        outer_handle_container: Optional[list],
        response_format: Optional[Any],
        max_parallel_tool_calls: Optional[int],
        semantic_cache: Optional[bool],
        images: Optional[dict[str, Any]],
    ) -> None:
        self.client = client
        self.message = message
        self.tools = tools
        self.loop_id = loop_id
        self.lineage = lineage
        self.interject_queue = interject_queue
        self.cancel_event = cancel_event
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.max_consecutive_failures = max_consecutive_failures
        self.prune_tool_duplicates = prune_tool_duplicates
        self.interrupt_llm_with_interjections = interrupt_llm_with_interjections
        self.propagate_chat_context = propagate_chat_context
        self.parent_chat_context = parent_chat_context
        self.log_steps = log_steps
        self.max_steps = max_steps
        self.timeout = timeout
        self.raise_on_limit = raise_on_limit
        self.include_class_in_dynamic_tool_names = include_class_in_dynamic_tool_names
        self.tool_policy = tool_policy
        self.preprocess_msgs = preprocess_msgs
        self.outer_handle_container = outer_handle_container
        self.response_format = response_format
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.semantic_cache = semantic_cache
        self.images = images

        self.state: State = State.IDLE
        self.events: "asyncio.Queue[Event]" = asyncio.Queue()
        self._tg: Optional[asyncio.TaskGroup] = None
        # Map tool call_id to its clarification down queue for answer routing
        self._clar_down: dict[str, asyncio.Queue[str]] = {}
        # Map tool call_id to a nested handle for clarify routing
        self._clar_nested: dict[str, Any] = {}
        # Track child tasks we spawn so we can cancel them explicitly on control events
        self._children: set[asyncio.Task] = set()
        # Normalized tools and quota counters for hidden per-loop limits
        try:
            self._normalized_tools = normalise_tools(self.tools or {})
        except Exception:
            self._normalized_tools = {}
        self._call_counts: dict[str, int] = {}

    def _build_interjection_system_content(self, payload: Any) -> str:
        # Mirror legacy wording and consolidation semantics
        try:
            outer = (
                self.outer_handle_container[0] if self.outer_handle_container else None
            )
        except Exception:
            outer = None
        history_lines: list[str] = []
        try:
            uvh = getattr(outer, "_user_visible_history", []) if outer else []
            for _m in uvh:
                role = _m.get("role")
                _content = _m.get("content")
                if isinstance(_content, dict):
                    _text = str(_content.get("message", "")).strip()
                else:
                    _text = str(_content or "").strip()
                if role in ("user", "assistant") and _text:
                    history_lines.append(f"{role}: {_text}")
        except Exception:
            try:
                first_user = next(
                    (
                        m.get("content", "")
                        for m in self.client.messages
                        if m.get("role") == "user"
                    ),
                    "",
                )
                if first_user:
                    history_lines = [f"user: {first_user}"]
            except Exception:
                history_lines = []

        # Extract interjection text and continued context
        try:
            if isinstance(payload, dict):
                _msg_text = str(payload.get("message", "")).strip()
                _ctx_cont = payload.get("parent_chat_context_continuted")
                _ctx_str = None
                if _ctx_cont is not None:
                    try:
                        _ctx_str = json.dumps(_ctx_cont, indent=2)
                    except Exception:
                        _ctx_str = None
            else:
                _msg_text = str(payload)
                _ctx_cont = None
                _ctx_str = None
        except Exception:
            _msg_text, _ctx_cont, _ctx_str = "", None, None

        sys_content = (
            "The user *cannot* see *any* the contents of this ongoing tool use chat context. "
            "They have just interjected with the following message (in bold at the bottom). "
            "From their perspective, the conversation thus far is as follows:\n"
            "--\n" + ("\n".join(history_lines)) + f"\nuser: **{_msg_text}**\n"
            "--\n"
        )
        if _ctx_cont is not None:
            sys_content += (
                "A continued parent chat context has been provided for this interjection.\n"
                + (_ctx_str or "(unserializable)")
                + "\n"
            )
        sys_content += (
            "Please consider and incorporate *all* interjections in your final response to the user. "
            "Later interjections should always override earlier interjections if there are "
            "any conflicting comments/requests across the different interjections."
        )
        return sys_content

    async def run(self) -> str:
        # Experimental: single LLM slice before legacy, gated by UNITY_EVENTED_LLM_SLICE
        try:
            do_slice = json.loads(os.environ.get("UNITY_EVENTED_LLM_SLICE", "false"))
        except Exception:
            do_slice = False
        try:
            do_first_turn = json.loads(
                os.environ.get("UNITY_EVENTED_FIRST_TURN", "true"),
            )
        except Exception:
            do_first_turn = True

        # If any tools define a hidden per-loop quota, skip the evented first-turn
        # path to avoid drift between preconsumed quotas and the legacy loop's
        # internal ToolsData counters.
        try:
            has_hidden_quotas = any(
                getattr(spec, "max_total_calls", None) is not None
                for spec in (self._normalized_tools or {}).values()
            )
        except Exception:
            has_hidden_quotas = False
        # If a finite timeout is set, rely on the legacy loop's TimeoutTimer semantics for now.
        has_finite_timeout = self.timeout is not None and float(self.timeout) > 0
        if has_hidden_quotas or has_finite_timeout:
            do_first_turn = False
            do_slice = False
            try:
                reason = (
                    "quotas_detected"
                    if has_hidden_quotas
                    else f"finite_timeout={self.timeout}"
                )
                LOGGER.info(f"orchestrator: skip evented first turn due to {reason}")
            except Exception:
                pass

        if do_first_turn:
            # First-turn: run a single LLM step with optional tool schemas and
            # honour interjection preemption. Do not mutate the transcript beyond
            # what the LLM produces; let the legacy loop handle scheduling,
            # placeholders, helper-acks, and all ordering semantics for parity.
            # Determine tool exposure and policy for the first LLM turn
            try:
                expose_tools = json.loads(
                    os.environ.get("UNITY_EVENTED_LLM_TOOLS", "true"),
                )
            except Exception:
                expose_tools = True

            tool_choice_mode = "auto"
            filtered_map: Dict[str, Callable] = dict(self.tools or {})
            if self.tool_policy is not None:
                try:
                    tool_choice_mode, filtered_map = self.tool_policy(
                        0,
                        dict(self.tools or {}),
                    )
                except Exception:
                    tool_choice_mode = "auto"
                    filtered_map = dict(self.tools or {})

            tools_param: list[dict] = []
            if expose_tools:
                # Build schemas from the filtered mapping
                tools_param = [
                    _method_to_schema(fn, tool_name=name)
                    for name, fn in filtered_map.items()
                ]
                # Inject final_answer tool when a response_format is provided
                if self.response_format is not None:
                    try:
                        _answer_schema = _resp_schema(self.response_format)
                        tools_param.append(
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
                    except Exception:
                        pass

            # Parity: provide structured-output hint via system_message (property, not a chat message)
            if self.response_format is not None:
                try:
                    _schema = _resp_schema(self.response_format)
                    _hint = (
                        "\n\nNOTE: After completing all tool calls, your **final** assistant reply must be valid JSON that conforms to the following schema. Do NOT include any extra keys or commentary.\n"
                        + json.dumps(_schema, indent=2)
                    )
                    base_sys = getattr(self.client, "system_message", "") or ""
                    if hasattr(self.client, "set_system_message"):
                        try:
                            self.client.set_system_message(base_sys + _hint)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Kick off the LLM call
            gen_kwargs = {
                "return_full_completion": True,
                "tools": tools_param,
                "tool_choice": tool_choice_mode,
                "stateful": True,
            }
            if self.max_parallel_tool_calls is not None:
                try:
                    gen_kwargs["max_tool_calls"] = self.max_parallel_tool_calls
                except Exception:
                    pass

            # Launch adapters under a TaskGroup for future full evented operation
            # (Interject adapter gated to avoid competing with legacy preemption logic)
            # Interject adapter enabled by default for full evented operation
            try:
                enable_interject_adapter = json.loads(
                    os.environ.get("UNITY_EVENTED_INTERJECT_ADAPTER", "true"),
                )
            except Exception:
                enable_interject_adapter = True

            async with asyncio.TaskGroup() as tg:
                self._tg = tg
                # Always wire control adapter (safe, posts cancel events without interference)
                try:
                    ControlAdapter(self).schedule()
                except Exception:
                    pass
                if enable_interject_adapter:
                    try:
                        InterjectAdapter(self).schedule()
                    except Exception:
                        pass

                # Time/step guard for first-turn LLM
                timer0 = _Timer(
                    timeout=self.timeout,
                    max_steps=self.max_steps,
                    raise_on_limit=self.raise_on_limit,
                    client=self.client,
                )
                try:
                    if timer0.has_exceeded_time() or timer0.has_exceeded_msgs():
                        raise asyncio.TimeoutError("pre-LLM first-turn limits reached")
                except Exception:
                    pass

                llm_task = asyncio.create_task(
                    _gwp(self.client, self.preprocess_msgs, **gen_kwargs),
                    name="EventedFirstTurnLLM",
                )
                interject_w = (
                    asyncio.create_task(
                        self.interject_queue.get(),
                        name="EventedFirstTurnInterject",
                    )
                    if not enable_interject_adapter
                    else None
                )
                cancel_w = asyncio.create_task(
                    self.cancel_event.wait(),
                    name="EventedFirstTurnCancel",
                )

                waitset0 = {llm_task, cancel_w} | (
                    {interject_w} if interject_w is not None else set()
                )
                done, _ = await asyncio.wait(
                    waitset0,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timer0.remaining_time(),
                )
                # Timeout handling: cancel llm and mark stop
                if not done:
                    try:
                        if not llm_task.done():
                            llm_task.cancel()
                            await asyncio.gather(llm_task, return_exceptions=True)
                        if self.stop_event is not None:
                            self.stop_event.set()
                    except Exception:
                        pass

            # If cancelled or preempted by interjection, cancel the LLM and fall back
            if cancel_w in done:
                try:
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    if self.stop_event is not None:
                        self.stop_event.set()
                except Exception:
                    pass
                # Ensure auxiliary waiter is cleaned up
                try:
                    if interject_w not in done and not interject_w.done():
                        interject_w.cancel()
                        await asyncio.gather(interject_w, return_exceptions=True)
                except Exception:
                    pass
            elif (
                (interject_w in done) if interject_w is not None else False
            ) and self.interrupt_llm_with_interjections:
                try:
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                except Exception:
                    pass
                # Re-queue the interjection so the legacy loop processes it identically
                try:
                    if interject_w is not None:
                        payload = interject_w.result()
                        await self.interject_queue.put(payload)
                except Exception:
                    pass
                # Cleanup the cancel waiter
                try:
                    if cancel_w not in done and not cancel_w.done():
                        cancel_w.cancel()
                        await asyncio.gather(cancel_w, return_exceptions=True)
                except Exception:
                    pass
            else:
                # LLM finished; no further action here – legacy handles scheduling
                try:
                    # Ensure any exception is surfaced for logging parity
                    _ = llm_task.exception()
                except Exception:
                    pass
                # Insert placeholders for base tools immediately after the assistant turn
                try:
                    msg_ref = (
                        self.client.messages[-1]
                        if getattr(self.client, "messages", None)
                        else None
                    )
                    if isinstance(msg_ref, dict):
                        tcs = list(msg_ref.get("tool_calls") or [])
                        if tcs:
                            self._insert_placeholders_for_calls(msg_ref, tcs)
                except Exception:
                    pass
                # Minimal hygiene: prune `wait` and acknowledge other helper tools
                try:
                    msg = (
                        self.client.messages[-1]
                        if getattr(self.client, "messages", None)
                        else None
                    )
                except Exception:
                    msg = None
                if isinstance(msg, dict):
                    try:
                        tool_calls = list(msg.get("tool_calls") or [])
                    except Exception:
                        tool_calls = []
                    if tool_calls:
                        try:
                            remaining_calls = []
                            for call in tool_calls:
                                fn_meta = call.get("function", {}) or {}
                                name = fn_meta.get("name")
                                args_json = fn_meta.get("arguments", "{}")
                                call_id = call.get("id") or "call"
                                if isinstance(name, str) and _is_helper_tool(name):
                                    if name == "wait":
                                        # Drop `wait` from assistant tool_calls to avoid clutter
                                        continue
                                    # Insert acknowledgement for other helpers
                                    try:
                                        self._insert_helper_ack(
                                            msg,
                                            name,
                                            args_json,
                                            str(call_id),
                                        )
                                    except Exception:
                                        pass
                                else:
                                    remaining_calls.append(call)
                            if len(remaining_calls) != len(tool_calls):
                                if remaining_calls:
                                    msg["tool_calls"] = remaining_calls
                                else:
                                    # If nothing remains and no content, drop the assistant message
                                    try:
                                        content_present = bool(
                                            (msg.get("content") or "").strip(),
                                        )
                                        if (
                                            not content_present
                                            and self.client.messages
                                            and self.client.messages[-1] is msg
                                        ):
                                            self.client.messages.pop()
                                        else:
                                            msg.pop("tool_calls", None)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                # Schedule and await first completion/clarification/notification using ToolsData
                try:
                    msg0 = (
                        self.client.messages[-1]
                        if getattr(self.client, "messages", None)
                        else None
                    )
                except Exception:
                    msg0 = None
                if isinstance(msg0, dict):
                    # Setup dispatcher and tools data
                    cfg = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
                    timer = _Timer(
                        timeout=self.timeout,
                        max_steps=self.max_steps,
                        raise_on_limit=self.raise_on_limit,
                        client=self.client,
                    )
                    dispatcher = _Dispatcher(self.client, cfg, timer)
                    logger = _LoopLogger(cfg, self.log_steps)
                    tools_data = _ToolsData(
                        self.tools,
                        client=self.client,
                        logger=logger,
                    )
                    assistant_meta: Dict[int, Dict[str, Any]] = {}
                    # Schedule base tool calls (enforce max_parallel_tool_calls)
                    scheduled_count = 0
                    max_calls = (
                        int(self.max_parallel_tool_calls)
                        if self.max_parallel_tool_calls is not None
                        else None
                    )
                    for idx, call in enumerate(list(msg0.get("tool_calls") or [])):
                        try:
                            name = (call.get("function", {}) or {}).get("name")
                            if not isinstance(name, str) or _is_helper_tool(name):
                                continue
                            args_json = (call.get("function", {}) or {}).get(
                                "arguments",
                                "{}",
                            )
                            cid = call.get("id") or "call"
                            if max_calls is not None and scheduled_count >= max_calls:
                                break
                            await tools_data.schedule_base_tool_call(
                                msg0,
                                name=name,
                                args_json=args_json,
                                call_id=cid,
                                call_idx=idx,
                                parent_chat_context=self.parent_chat_context,
                                propagate_chat_context=self.propagate_chat_context,
                                assistant_meta=assistant_meta,
                            )
                            scheduled_count += 1
                        except Exception:
                            continue
                    # Ensure placeholders
                    try:
                        await _ensure_placeholders_for_pending(
                            assistant_msg=msg0,
                            tools_data=tools_data,
                            assistant_meta=assistant_meta,
                            client=self.client,
                            msg_dispatcher=dispatcher,
                        )
                    except Exception:
                        pass
                    # Build watchers; interject waiter only when adapter disabled
                    interject_waiter = (
                        asyncio.create_task(
                            self.interject_queue.get(),
                            name="FirstTurnInterject",
                        )
                        if not enable_interject_adapter
                        else None
                    )
                    cancel_waiter2 = asyncio.create_task(
                        self.cancel_event.wait(),
                        name="FirstTurnCancel",
                    )
                    clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                    notif_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                    for _t in list(tools_data.pending):
                        _inf = tools_data.info.get(_t)
                        if not _inf:
                            continue
                        if (
                            not getattr(_inf, "waiting_for_clarification", False)
                            and _inf.clar_up_queue is not None
                        ):
                            cw = asyncio.create_task(
                                _inf.clar_up_queue.get(),
                                name="FirstTurnClarification",
                            )
                            clar_waiters[cw] = _t
                        if _inf.notification_queue is not None:
                            pw = asyncio.create_task(
                                _inf.notification_queue.get(),
                                name="FirstTurnNotification",
                            )
                            notif_waiters[pw] = _t
                    # Loop to handle multiple completions before handing off
                    canceled = False
                    interjected = False
                    llm_turn_required = False
                    while True:
                        waitset = (
                            tools_data.pending
                            | set(clar_waiters.keys())
                            | set(notif_waiters.keys())
                            | (
                                {interject_waiter}
                                if interject_waiter is not None
                                else set()
                            )
                            | {cancel_waiter2}
                        )
                        if not waitset:
                            break
                        done_first, _ = await asyncio.wait(
                            waitset,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Cleanup unused helpers for this iteration
                        try:
                            for aux in (
                                interject_waiter,
                                cancel_waiter2,
                                *clar_waiters.keys(),
                                *notif_waiters.keys(),
                            ):
                                if aux not in done_first and not aux.done():
                                    aux.cancel()
                            await asyncio.gather(
                                interject_waiter,
                                cancel_waiter2,
                                *clar_waiters.keys(),
                                *notif_waiters.keys(),
                                return_exceptions=True,
                            )
                        except Exception:
                            pass
                        # Handle branches
                        if cancel_waiter2 in done_first:
                            try:
                                await tools_data.cancel_pending_tasks()
                            except Exception:
                                pass
                            canceled = True
                            break
                        if (
                            (interject_waiter in done_first)
                            if interject_waiter is not None
                            else False
                        ) and self.interrupt_llm_with_interjections:
                            try:
                                interject_payload = (
                                    interject_waiter.result()
                                    if interject_waiter
                                    else None
                                )
                                # Append legacy system message for consolidation/visibility
                                sys_msg = {
                                    "role": "system",
                                    "content": self._build_interjection_system_content(
                                        interject_payload,
                                    ),
                                }
                                await dispatcher.append_msgs([sys_msg])
                                await self.interject_queue.put(interject_payload)
                            except Exception:
                                pass
                            interjected = True
                            break
                        # Clarification request(s)
                        if done_first & set(clar_waiters.keys()):
                            for cw in done_first & set(clar_waiters.keys()):
                                try:
                                    q = cw.result()
                                    q_text = (
                                        q.get("question")
                                        if isinstance(q, dict)
                                        else str(q)
                                    )
                                except Exception:
                                    q_text = ""
                                src_task = clar_waiters[cw]
                                try:
                                    call_id = tools_data.info[src_task].call_id
                                    placeholder = tools_data.info[
                                        src_task
                                    ].tool_reply_msg
                                    if placeholder is None:
                                        placeholder = create_tool_call_message(
                                            name=f"clarification_request_{call_id}",
                                            call_id=call_id,
                                            content="",
                                        )
                                        await _insert_tool_message_after_assistant(
                                            assistant_meta,
                                            msg0,
                                            placeholder,
                                            self.client,
                                            dispatcher,
                                        )
                                        tools_data.info[src_task].tool_reply_msg = (
                                            placeholder
                                        )
                                    placeholder["name"] = (
                                        f"clarification_request_{call_id}"
                                    )
                                    placeholder["content"] = (
                                        "Tool incomplete, please answer the following to continue tool execution:\n"
                                        + q_text
                                    )
                                    tools_data.info[
                                        src_task
                                    ].waiting_for_clarification = True
                                except Exception:
                                    pass
                            llm_turn_required = True
                            break
                        # Notification(s)
                        if done_first & set(notif_waiters.keys()):
                            for pw in done_first & set(notif_waiters.keys()):
                                try:
                                    payload = pw.result()
                                    content_payload = (
                                        payload
                                        if isinstance(payload, dict)
                                        else {"message": str(payload)}
                                    )
                                    src_task = notif_waiters[pw]
                                    tool_name = tools_data.info[src_task].name
                                    pretty = _json_pretty(
                                        {"tool": tool_name, **content_payload},
                                        indent=4,
                                    )
                                    placeholder = tools_data.info[
                                        src_task
                                    ].tool_reply_msg
                                    if placeholder is None:
                                        placeholder = create_tool_call_message(
                                            name=tool_name,
                                            call_id=tools_data.info[src_task].call_id,
                                            content=pretty,
                                        )
                                        await _insert_tool_message_after_assistant(
                                            assistant_meta,
                                            msg0,
                                            placeholder,
                                            self.client,
                                            dispatcher,
                                        )
                                        tools_data.info[src_task].tool_reply_msg = (
                                            placeholder
                                        )
                                    else:
                                        placeholder["content"] = pretty
                                except Exception:
                                    pass
                            llm_turn_required = True
                            break
                        # Completed tool tasks (may be multiple)
                        tracker = _FailureTracker(self.max_consecutive_failures)
                        for t in done_first & tools_data.pending:
                            try:
                                await tools_data.process_completed_task(
                                    task=t,
                                    consecutive_failures=tracker,
                                    outer_handle_container=self.outer_handle_container,
                                    assistant_meta=assistant_meta,
                                    msg_dispatcher=dispatcher,
                                )
                            except Exception:
                                pass
                        # Rebuild watchers for remaining tasks
                        clar_waiters.clear()
                        notif_waiters.clear()
                        for _t in list(tools_data.pending):
                            _inf = tools_data.info.get(_t)
                            if not _inf:
                                continue
                            if (
                                not getattr(_inf, "waiting_for_clarification", False)
                                and _inf.clar_up_queue is not None
                            ):
                                cw = asyncio.create_task(
                                    _inf.clar_up_queue.get(),
                                    name="FirstTurnClarification",
                                )
                                clar_waiters[cw] = _t
                            if _inf.notification_queue is not None:
                                pw = asyncio.create_task(
                                    _inf.notification_queue.get(),
                                    name="FirstTurnNotification",
                                )
                                notif_waiters[pw] = _t
                        # Recreate interject/cancel waiters for next loop
                        interject_waiter = (
                            asyncio.create_task(
                                self.interject_queue.get(),
                                name="FirstTurnInterject",
                            )
                            if not enable_interject_adapter
                            else None
                        )
                        cancel_waiter2 = asyncio.create_task(
                            self.cancel_event.wait(),
                            name="FirstTurnCancel",
                        )

                    # Optional: if no pending and no cancellations/interjections and no clarification requested,
                    # we can allow one more LLM turn evented before handoff (broadening subtly)
                    if (
                        not canceled
                        and not interjected
                        and not llm_turn_required
                        and not tools_data.pending
                    ):
                        try:
                            # Second-turn assistant – parity: tool_choice per policy step 1
                            tool_choice_mode2 = "auto"
                            if self.tool_policy is not None:
                                try:
                                    tool_choice_mode2, _ = self.tool_policy(
                                        1,
                                        dict(self.tools or {}),
                                    )
                                except Exception:
                                    tool_choice_mode2 = "auto"
                            await _gwp(
                                self.client,
                                self.preprocess_msgs,
                                return_full_completion=True,
                                tools=[],
                                tool_choice=tool_choice_mode2,
                                stateful=True,
                            )
                        except Exception:
                            pass

                # Subsequent turns: repeat LLM → schedule base tools → wait cycles
                try:
                    step_idx = 1
                    while True:
                        # Build schemas per policy
                        tool_choice_mode2 = "auto"
                        filtered_map2: Dict[str, Callable] = dict(self.tools or {})
                        if self.tool_policy is not None:
                            try:
                                tool_choice_mode2, filtered_map2 = self.tool_policy(
                                    step_idx,
                                    dict(self.tools or {}),
                                )
                            except Exception:
                                tool_choice_mode2 = "auto"
                                filtered_map2 = dict(self.tools or {})

                        schemas2: list[dict] = []
                        for n, f in filtered_map2.items():
                            try:
                                schemas2.append(_method_to_schema(f, tool_name=n))
                            except Exception:
                                continue
                        if self.response_format is not None:
                            try:
                                _answer_schema = _resp_schema(self.response_format)
                                schemas2.append(
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
                                                "properties": {
                                                    "answer": _answer_schema,
                                                },
                                                "required": ["answer"],
                                            },
                                        },
                                    },
                                )
                            except Exception:
                                pass

                        gen_kwargs2 = {
                            "return_full_completion": True,
                            "tools": schemas2,
                            "tool_choice": tool_choice_mode2,
                            "stateful": True,
                        }
                        if self.max_parallel_tool_calls is not None:
                            try:
                                gen_kwargs2["max_tool_calls"] = (
                                    self.max_parallel_tool_calls
                                )
                            except Exception:
                                pass

                        # Enforce time/step limits before generating
                        try:
                            if timer.has_exceeded_time() or timer.has_exceeded_msgs():
                                break
                        except Exception:
                            pass
                        await _gwp(self.client, self.preprocess_msgs, **gen_kwargs2)
                        try:
                            timer.reset()
                        except Exception:
                            pass

                        # Handle structured-output final_answer
                        msg_tail = (
                            self.client.messages[-1]
                            if getattr(self.client, "messages", None)
                            else None
                        )
                        if (
                            isinstance(msg_tail, dict)
                            and self.response_format is not None
                        ):
                            try:
                                for call in list(msg_tail.get("tool_calls") or []):
                                    if (
                                        (call.get("function", {}) or {}).get("name")
                                    ) != "final_answer":
                                        continue
                                    args = (call.get("function", {}) or {}).get(
                                        "arguments",
                                        {},
                                    )
                                    payload = (
                                        args.get("answer")
                                        if isinstance(args, dict)
                                        else None
                                    )
                                    if payload is None:
                                        continue
                                    try:
                                        self.response_format.model_validate(payload)
                                        content_txt = json.dumps(payload)
                                        tool_msg_ok = create_tool_call_message(
                                            name="final_answer",
                                            call_id=call.get("id") or "call",
                                            content=(
                                                _json_pretty(payload, indent=4)
                                                if _json_pretty
                                                else content_txt
                                            ),
                                        )
                                        self.client.append_messages([tool_msg_ok])
                                        try:
                                            idx = self.client.messages.index(msg_tail)
                                            self.client.messages.insert(
                                                idx + 1,
                                                self.client.messages.pop(),
                                            )
                                        except Exception:
                                            pass
                                        return content_txt
                                    except Exception:
                                        # Insert validation failure tool message; continue
                                        try:
                                            tool_msg = create_tool_call_message(
                                                name="final_answer",
                                                call_id=call.get("id") or "call",
                                                content=(
                                                    "⚠️ Validation failed – proceeding with standard formatting step."
                                                ),
                                            )
                                            self.client.append_messages([tool_msg])
                                            try:
                                                idx = self.client.messages.index(
                                                    msg_tail,
                                                )
                                                self.client.messages.insert(
                                                    idx + 1,
                                                    self.client.messages.pop(),
                                                )
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                        # If no tool_calls → final assistant message content
                        if isinstance(msg_tail, dict) and not (
                            msg_tail.get("tool_calls") or []
                        ):
                            return msg_tail.get("content", "")

                        # Insert placeholders for new assistant turn tool calls
                        try:
                            tcs2 = (
                                list(msg_tail.get("tool_calls") or [])
                                if isinstance(msg_tail, dict)
                                else []
                            )
                            if tcs2:
                                self._insert_placeholders_for_calls(msg_tail, tcs2)
                        except Exception:
                            pass

                        # Helper hygiene again
                        try:
                            tool_calls2 = (
                                list(msg_tail.get("tool_calls") or [])
                                if isinstance(msg_tail, dict)
                                else []
                            )
                        except Exception:
                            tool_calls2 = []
                        if tool_calls2:
                            try:
                                remaining2 = []
                                for call in tool_calls2:
                                    name2 = (call.get("function", {}) or {}).get("name")
                                    args_json2 = (call.get("function", {}) or {}).get(
                                        "arguments",
                                        "{}",
                                    )
                                    call_id2 = call.get("id") or "call"
                                    if isinstance(name2, str) and _is_helper_tool(
                                        name2,
                                    ):
                                        if name2 == "wait":
                                            continue
                                        try:
                                            self._insert_helper_ack(
                                                msg_tail,
                                                name2,
                                                args_json2,
                                                str(call_id2),
                                            )
                                        except Exception:
                                            pass
                                    else:
                                        remaining2.append(call)
                                if len(remaining2) != len(tool_calls2):
                                    if remaining2:
                                        msg_tail["tool_calls"] = remaining2
                                    else:
                                        try:
                                            content_present2 = bool(
                                                (msg_tail.get("content") or "").strip(),
                                            )
                                            if (
                                                not content_present2
                                                and self.client.messages
                                                and self.client.messages[-1] is msg_tail
                                            ):
                                                self.client.messages.pop()
                                            else:
                                                msg_tail.pop("tool_calls", None)
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                        # Schedule base tools for this turn and wait similar to first
                        cfg2 = _LoopConfig(
                            self.loop_id,
                            self.lineage,
                            self.lineage or [],
                        )
                        timer2 = _Timer(
                            timeout=self.timeout,
                            max_steps=self.max_steps,
                            raise_on_limit=self.raise_on_limit,
                            client=self.client,
                        )
                        dispatcher2 = _Dispatcher(self.client, cfg2, timer2)
                        logger2 = _LoopLogger(cfg2, self.log_steps)
                        tools_data2 = _ToolsData(
                            self.tools,
                            client=self.client,
                            logger=logger2,
                        )
                        assistant_meta2: Dict[int, Dict[str, Any]] = {}
                        scheduled2 = 0
                        max_calls2 = (
                            int(self.max_parallel_tool_calls)
                            if self.max_parallel_tool_calls is not None
                            else None
                        )
                        for idx2, call in enumerate(
                            list(msg_tail.get("tool_calls") or []),
                        ):
                            try:
                                nm = (call.get("function", {}) or {}).get("name")
                                if not isinstance(nm, str) or _is_helper_tool(nm):
                                    continue
                                aj = (call.get("function", {}) or {}).get(
                                    "arguments",
                                    "{}",
                                )
                                cid2 = call.get("id") or "call"
                                if max_calls2 is not None and scheduled2 >= max_calls2:
                                    break
                                await tools_data2.schedule_base_tool_call(
                                    msg_tail,
                                    name=nm,
                                    args_json=aj,
                                    call_id=cid2,
                                    call_idx=idx2,
                                    parent_chat_context=self.parent_chat_context,
                                    propagate_chat_context=self.propagate_chat_context,
                                    assistant_meta=assistant_meta2,
                                )
                                scheduled2 += 1
                            except Exception:
                                continue
                        try:
                            await _ensure_placeholders_for_pending(
                                assistant_msg=msg_tail,
                                tools_data=tools_data2,
                                assistant_meta=assistant_meta2,
                                client=self.client,
                                msg_dispatcher=dispatcher2,
                            )
                        except Exception:
                            pass

                        # Await first event or completion(s) for this turn
                        interject2 = (
                            asyncio.create_task(
                                self.interject_queue.get(),
                                name="TurnInterject",
                            )
                            if not enable_interject_adapter
                            else None
                        )
                        cancel2 = asyncio.create_task(
                            self.cancel_event.wait(),
                            name="TurnCancel",
                        )
                        clar_w2: Dict[asyncio.Task, asyncio.Task] = {}
                        notif_w2: Dict[asyncio.Task, asyncio.Task] = {}
                        for _t in list(tools_data2.pending):
                            _inf2 = tools_data2.info.get(_t)
                            if not _inf2:
                                continue
                            if (
                                not getattr(_inf2, "waiting_for_clarification", False)
                                and _inf2.clar_up_queue is not None
                            ):
                                cw2 = asyncio.create_task(
                                    _inf2.clar_up_queue.get(),
                                    name="TurnClarification",
                                )
                                clar_w2[cw2] = _t
                            if _inf2.notification_queue is not None:
                                pw2 = asyncio.create_task(
                                    _inf2.notification_queue.get(),
                                    name="TurnNotification",
                                )
                                notif_w2[pw2] = _t
                        wset2 = (
                            tools_data2.pending
                            | set(clar_w2.keys())
                            | set(notif_w2.keys())
                            | ({interject2} if interject2 is not None else set())
                            | {cancel2}
                        )
                        if wset2:
                            # Honor rolling time budget while waiting
                            try:
                                if (
                                    timer.has_exceeded_time()
                                    or timer.has_exceeded_msgs()
                                ):
                                    await tools_data2.cancel_pending_tasks()
                                    break
                            except Exception:
                                pass
                            done2, _ = await asyncio.wait(
                                wset2,
                                return_when=asyncio.FIRST_COMPLETED,
                                timeout=timer.remaining_time(),
                            )
                            # Handle branches akin to first-turn
                            if cancel2 in done2:
                                try:
                                    await tools_data2.cancel_pending_tasks()
                                except Exception:
                                    pass
                                break
                            if (
                                (interject2 in done2)
                                if interject2 is not None
                                else False
                            ) and self.interrupt_llm_with_interjections:
                                try:
                                    interject_payload2 = (
                                        interject2.result() if interject2 else None
                                    )
                                    sys_msg2 = {
                                        "role": "system",
                                        "content": self._build_interjection_system_content(
                                            interject_payload2,
                                        ),
                                    }
                                    await dispatcher2.append_msgs([sys_msg2])
                                    await self.interject_queue.put(interject_payload2)
                                except Exception:
                                    pass
                                break
                            if done2 & set(clar_w2.keys()):
                                break  # clarification triggers next LLM turn
                            if done2 & set(notif_w2.keys()):
                                break  # notification triggers next LLM turn
                            # Process completions then continue loop to potentially await more or proceed to next LLM turn
                            tracker2 = _FailureTracker(self.max_consecutive_failures)
                            for t in done2 & tools_data2.pending:
                                try:
                                    await tools_data2.process_completed_task(
                                        task=t,
                                        consecutive_failures=tracker2,
                                        outer_handle_container=self.outer_handle_container,
                                        assistant_meta=assistant_meta2,
                                        msg_dispatcher=dispatcher2,
                                    )
                                except Exception:
                                    pass
                        # Increment step index
                        step_idx += 1
                except Exception:
                    pass

                # Ensure no dangling tasks remain
                try:
                    for w in (llm_task, interject_w, cancel_w):
                        if w not in done and not w.done():
                            w.cancel()
                    await asyncio.gather(
                        llm_task,
                        interject_w,
                        cancel_w,
                        return_exceptions=True,
                    )
                except Exception:
                    pass

        elif do_slice:
            async with asyncio.TaskGroup() as tg:
                self._tg = tg
                # Optionally expose real tool schemas during experimental slice
                try:
                    expose_tools = json.loads(
                        os.environ.get("UNITY_EVENTED_LLM_TOOLS", "false"),
                    )
                except Exception:
                    expose_tools = False
                # Apply policy on the slice as well for parity
                tool_choice_mode = "auto"
                filtered_map: Dict[str, Callable] = dict(self.tools or {})
                if self.tool_policy is not None:
                    try:
                        tool_choice_mode, filtered_map = self.tool_policy(
                            0,
                            dict(self.tools or {}),
                        )
                    except Exception:
                        tool_choice_mode = "auto"
                        filtered_map = dict(self.tools or {})
                tools_param: list[dict] = []
                if expose_tools:
                    tools_param = [
                        _method_to_schema(fn, tool_name=name)
                        for name, fn in filtered_map.items()
                    ]
                    if self.response_format is not None:
                        try:
                            _answer_schema = _resp_schema(self.response_format)
                            tools_param.append(
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
                        except Exception:
                            pass
                # Parity: add structured-output system_message hint in slice mode as well
                if self.response_format is not None:
                    try:
                        _schema = _resp_schema(self.response_format)
                        _hint = (
                            "\n\nNOTE: After completing all tool calls, your **final** assistant reply must be valid JSON that conforms to the following schema. Do NOT include any extra keys or commentary.\n"
                            + json.dumps(_schema, indent=2)
                        )
                        base_sys = getattr(self.client, "system_message", "") or ""
                        if hasattr(self.client, "set_system_message"):
                            try:
                                self.client.set_system_message(base_sys + _hint)
                            except Exception:
                                pass
                    except Exception:
                        pass
                # Wire adapters: control always; interject gated by env (future use)
                try:
                    ControlAdapter(self).schedule()
                except Exception:
                    pass
                # Interject adapter enabled by default for slice as well
                try:
                    enable_interject_adapter = json.loads(
                        os.environ.get("UNITY_EVENTED_INTERJECT_ADAPTER", "true"),
                    )
                except Exception:
                    enable_interject_adapter = True
                if enable_interject_adapter:
                    try:
                        InterjectAdapter(self).schedule()
                    except Exception:
                        pass

                LLMRunner(self).schedule_generate(
                    {
                        "tools": tools_param,
                        "tool_choice": tool_choice_mode,
                        "response_format": self.response_format,
                    },
                )
                # Wait for a single LLM event
                while True:
                    evt = await self.events.get()
                    t = evt.get("type")
                    if t == "pause_requested":
                        try:
                            self.pause_event.clear()
                            self._set_state(State.PAUSED, on=t)
                        except Exception:
                            pass
                        continue
                    if t == "cancel_requested":
                        try:
                            self._set_state(State.CANCELLING, on=t)
                            await self._cancel_children()
                            if self.stop_event is not None:
                                self.stop_event.set()
                        except Exception:
                            pass
                        break
                    if t == "resume_requested":
                        try:
                            self.pause_event.set()
                            self._set_state(State.WAITING_LLM, on=t)
                        except Exception:
                            pass
                        continue
                    if t == "llm_completed":
                        # Attempt to schedule any tool calls if present
                        msg = evt.get("message") or {}
                        tool_calls = (
                            (msg.get("tool_calls") or [])
                            if isinstance(msg, dict)
                            else []
                        )
                        try:
                            tool_calls = self._prune_over_quota_on_msg(msg)
                        except Exception:
                            pass
                        try:
                            self._insert_placeholders_for_calls(msg, tool_calls)
                        except Exception:
                            pass
                        for call in tool_calls:
                            try:
                                fn_meta = call.get("function", {})
                                name = fn_meta.get("name")
                                args_json = call.get("function", {}).get(
                                    "arguments",
                                    "{}",
                                )
                                args = (
                                    json.loads(args_json)
                                    if isinstance(args_json, str)
                                    else (args_json or {})
                                )
                                call_id = call.get("id") or "call"
                                if isinstance(name, str) and _is_helper_tool(name):
                                    if name == "wait":
                                        continue
                                    try:
                                        self._insert_helper_ack(
                                            msg,
                                            name,
                                            args_json,
                                            str(call_id),
                                        )
                                    except Exception:
                                        pass
                                    continue
                                fn = self.tools.get(name)
                                if callable(
                                    fn,
                                ) and ToolRunner.is_safe_to_schedule_without_clar(fn):
                                    try:
                                        self._call_counts[name] = (
                                            self._call_counts.get(name, 0) + 1
                                        )
                                    except Exception:
                                        pass
                                    ToolRunner(self).schedule_tool(
                                        name=name,
                                        call_id=str(call_id),
                                        fn=fn,
                                        parent_assistant_msg=msg,
                                        **args,
                                    )
                            except Exception:
                                # Best-effort in experimental slice
                                continue
                        break
                    if t in {"llm_failed", "llm_preempted"}:
                        break

        # Delegate to legacy loop for full behaviour and completion –
        # do not mutate transcript by inserting a system message here.
        try:
            LOGGER.info(
                f"orchestrator: handover to legacy; preconsumed_quotas={getattr(self, '_call_counts', {})}",
            )
        except Exception:
            pass
        return await _legacy_tool_loop_inner(
            client=self.client,
            message=self.message,
            tools=self.tools,
            loop_id=self.loop_id,
            lineage=self.lineage,
            interject_queue=self.interject_queue,
            cancel_event=self.cancel_event,
            stop_event=self.stop_event,
            pause_event=self.pause_event,
            max_consecutive_failures=self.max_consecutive_failures,
            prune_tool_duplicates=self.prune_tool_duplicates,
            interrupt_llm_with_interjections=self.interrupt_llm_with_interjections,
            propagate_chat_context=self.propagate_chat_context,
            parent_chat_context=self.parent_chat_context,
            log_steps=self.log_steps,
            max_steps=self.max_steps,
            timeout=self.timeout,
            raise_on_limit=self.raise_on_limit,
            include_class_in_dynamic_tool_names=self.include_class_in_dynamic_tool_names,
            tool_policy=self.tool_policy,
            preprocess_msgs=self.preprocess_msgs,
            outer_handle_container=self.outer_handle_container,
            response_format=self.response_format,
            max_parallel_tool_calls=self.max_parallel_tool_calls,
            semantic_cache=self.semantic_cache,
            images=self.images,
        )

    def _build_tool_schemas(self) -> list[dict]:
        schemas: list[dict] = []
        for name, fn in (self.tools or {}).items():
            try:
                schemas.append(_method_to_schema(fn, tool_name=name))
            except Exception:
                continue
        return schemas

    # ── transition helpers (not yet wired) ─────────────────────────────────
    def _set_state(self, new_state: State, *, on: str) -> None:
        old = self.state.value
        self.state = new_state
        # structured transition log
        self._log_transition(old_state=old, new_state=new_state.value, on=on)

    async def _consume_event_once(self) -> None:  # pragma: no cover - skeleton
        evt = await self.events.get()
        t = evt.get("type", "") if isinstance(evt, dict) else ""
        if t == "llm_completed" and self.state in {State.IDLE, State.WAITING_LLM}:
            self._set_state(State.RUNNING_TOOLS, on=t)
        elif t == "interjected":
            # would preempt LLM and stay in current logical state
            self._set_state(self.state, on=t)
        elif t == "pause_requested":
            self._set_state(State.PAUSED, on=t)
        elif t == "resume_requested":
            # in full impl, restore prior state
            self._set_state(State.WAITING_LLM, on=t)
        elif t in {"cancel_requested", "timeout"}:
            self._set_state(State.CANCELLING, on=t)

    # Structured logging helpers
    def _log_event(self, name: str, **fields) -> None:
        try:
            rec = {"event": name, "loop_id": self.loop_id, **fields}
            LOGGER.info(f"orchestrator {json.dumps(rec, default=str)}")
        except Exception:
            pass

    def _log_transition(self, *, old_state: str, new_state: str, on: str, **kw) -> None:
        self._log_event("transition", old=old_state, new=new_state, on=on, **kw)

    def _register_child(self, task: asyncio.Task) -> None:
        try:
            self._children.add(task)

            def _done(_):
                try:
                    self._children.discard(task)
                except Exception:
                    pass

            task.add_done_callback(_done)
        except Exception:
            pass

    async def _cancel_children(self) -> None:
        try:
            for t in list(self._children):
                try:
                    t.cancel()
                except Exception:
                    pass
            if self._children:
                await asyncio.gather(*list(self._children), return_exceptions=True)
        except Exception:
            pass

    # Insert simple placeholders after the assistant message for each tool call
    def _insert_placeholders_for_calls(
        self,
        assistant_msg: dict,
        tool_calls: list[dict],
    ) -> None:
        if not isinstance(assistant_msg, dict):
            return
        try:
            idx = self.client.messages.index(assistant_msg)
        except Exception:
            # Fallback: append to tail
            idx = len(self.client.messages) - 1
        offset = 0
        for call in tool_calls:
            try:
                fn_meta = call.get("function", {})
                name = fn_meta.get("name")
                cid = str(call.get("id") or "call")
                # Skip placeholders for helper tools to avoid clutter; acknowledgements inserted separately
                if isinstance(name, str) and _is_helper_tool(name):
                    continue
                placeholder = create_tool_call_message(
                    name=name,
                    call_id=cid,
                    content="Pending… tool call accepted. Working on it.",
                )
                # Append and reposition right after the assistant msg (+offset)
                self.client.append_messages([placeholder])
                try:
                    self.client.messages.insert(
                        idx + 1 + offset,
                        self.client.messages.pop(),
                    )
                    offset += 1
                except Exception:
                    pass
            except Exception:
                continue

    def _insert_helper_ack(
        self,
        assistant_msg: dict,
        helper_name: str,
        args_json: Any,
        call_id: str,
    ) -> None:
        ack = create_tool_call_message(
            name=helper_name,
            call_id=call_id,
            content=_helper_ack(helper_name, args_json),
        )
        try:
            idx = self.client.messages.index(assistant_msg)
        except Exception:
            idx = len(self.client.messages) - 1
        self.client.append_messages([ack])
        try:
            self.client.messages.insert(idx + 1, self.client.messages.pop())
        except Exception:
            pass

    def _route_clarify_answer(self, helper_name: str, args_json: Any) -> bool:
        try:
            payload = (
                json.loads(args_json or "{}")
                if isinstance(args_json, str)
                else (args_json or {})
            )
        except Exception:
            payload = {}
        ans = payload.get("answer")
        if ans is None:
            return False
        # Find matching call_id by suffix match
        call_id = None
        try:
            for cid in list(self._clar_down.keys()):
                if isinstance(helper_name, str) and helper_name.endswith(cid):
                    call_id = cid
                    break
        except Exception:
            call_id = None
        if call_id is None:
            return False
        # Route to base tool clarification down-queue if present
        try:
            dq = self._clar_down.get(call_id)
            if dq is not None:
                dq.put_nowait(str(ans))
                return True
        except Exception:
            pass
        # Route to nested handle if present
        try:
            h = self._clar_nested.get(call_id)
            if h is not None and hasattr(h, "answer_clarification"):
                # Call with the call_id suffix if required; accept either full or suffix
                try:
                    return (
                        asyncio.create_task(h.answer_clarification(call_id, str(ans)))
                        is not None
                    )
                except Exception:
                    return False
        except Exception:
            return False
        return False

    def _ensure_system_message(self) -> None:
        # If no system-role message exists, append one using the client's system_message
        try:
            has_sys = any(
                m.get("role") == "system" for m in (self.client.messages or [])
            )
        except Exception:
            has_sys = False
        if not has_sys:
            content = getattr(self.client, "system_message", "") or ""
            if isinstance(content, str) and content.strip():
                sys_msg = {"role": "system", "content": content}
                self.client.append_messages([sys_msg])

    def _prune_over_quota_on_msg(self, assistant_msg: dict) -> list[dict]:
        """Prune tool_calls according to hidden per-tool quotas using local counters."""
        try:
            tool_calls = assistant_msg.get("tool_calls") or []
            if not isinstance(tool_calls, list) or not tool_calls:
                return tool_calls

            remaining: dict[str, int] = {}
            for name, spec in (self._normalized_tools or {}).items():
                lim = getattr(spec, "max_total_calls", None)
                if lim is None:
                    continue
                used = int(self._call_counts.get(name, 0))
                remaining[name] = max(0, int(lim) - used)

            kept: list[dict] = []
            for call in tool_calls:
                try:
                    fn_name = call.get("function", {}).get("name")
                except Exception:
                    fn_name = None
                if isinstance(fn_name, str) and fn_name in remaining:
                    if remaining[fn_name] > 0:
                        kept.append(call)
                        remaining[fn_name] -= 1
                    else:
                        continue
                else:
                    kept.append(call)
            if len(kept) != len(tool_calls):
                assistant_msg["tool_calls"] = kept
            return assistant_msg.get("tool_calls") or []
        except Exception:
            return assistant_msg.get("tool_calls") or []


class LLMRunner:
    """Child runner for LLM turns that posts events instead of bubbling cancels.

    Skeleton only – wired but not invoked yet. Intended usage:
        runner = LLMRunner(orch)
        runner.schedule_generate({"tools": [...], ...})
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orch = orchestrator

    def schedule_generate(self, gen_kwargs: dict) -> None:
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        async def _task():
            try:
                msg = await _gwp(
                    self._orch.client,
                    self._orch.preprocess_msgs,
                    **gen_kwargs,
                )
                evt: LLMCompletedEvent = {"type": "llm_completed", "message": msg}
                await self._orch.events.put(evt)
            except asyncio.CancelledError:
                # Treat as preemption; do NOT re-raise (prevent orchestrator cancel)
                await self._orch.events.put({"type": "llm_preempted"})
            except Exception as e:
                await self._orch.events.put(
                    {"type": "llm_failed", "error": str(e)},
                )

        child = self._orch._tg.create_task(_task())
        self._orch._register_child(child)


class ToolRunner:
    """Child runner for base tool executions that posts completion/failure events.

    Skeleton only – not yet integrated with ToolsData scheduling or placeholders.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orch = orchestrator

    def schedule_tool(self, name: str, call_id: str, fn: Callable, *a, **kw) -> None:
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        # Inject supported, safe kwargs (do not provide clarification queues yet)
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
        except Exception:
            params = {}

        # Optional pause_event support
        if "pause_event" in params and "pause_event" not in kw:
            kw["pause_event"] = self._orch.pause_event

        # Optional notification queue: attach and surface events upward
        notification_up_q = None
        if "notification_up_q" in params and "notification_up_q" not in kw:
            notification_up_q = asyncio.Queue()
            kw["notification_up_q"] = notification_up_q

        # Clarification queues: when supported, create and bridge to orchestrator events
        clar_up_q = None
        clar_down_q = None
        if ("clarification_up_q" in params and "clarification_down_q" in params) and (
            "clarification_up_q" not in kw and "clarification_down_q" not in kw
        ):
            clar_up_q = asyncio.Queue()
            clar_down_q = asyncio.Queue()
            kw["clarification_up_q"] = clar_up_q
            kw["clarification_down_q"] = clar_down_q
            # Register down queue so clarify_* answers can be routed
            self._orch._clar_down[call_id] = clar_down_q

        async def _task():
            try:
                res = await maybe_await(fn(*a, **kw))
                evt: ToolCompletedEvent = {
                    "type": "tool_completed",
                    "call_id": call_id,
                    "name": name,
                    "result": res,
                }
                await self._orch.events.put(evt)
            except asyncio.CancelledError:
                # Treat tool cancellation as a normal stop; do not bubble
                await self._orch.events.put(
                    {
                        "type": "tool_failed",
                        "call_id": call_id,
                        "name": name,
                        "error": "cancelled",
                    },
                )
            except Exception as e:
                await self._orch.events.put(
                    {
                        "type": "tool_failed",
                        "call_id": call_id,
                        "name": name,
                        "error": str(e),
                    },
                )

        self._orch._register_child(self._orch._tg.create_task(_task()))

        # If we attached a notification queue, schedule a watcher
        if notification_up_q is not None:

            async def _watch_notifications():
                try:
                    while True:
                        payload = await notification_up_q.get()
                        msg = None
                        try:
                            if isinstance(payload, dict):
                                msg = payload.get("message")
                            else:
                                msg = str(payload)
                        except Exception:
                            msg = None
                        await self._orch.events.put(
                            {
                                "type": "notification_received",
                                "call_id": call_id,
                                "tool_name": name,
                                "message": msg or "",
                            },
                        )
                except asyncio.CancelledError:
                    return

            self._orch._register_child(
                self._orch._tg.create_task(_watch_notifications()),
            )

        # Clarification watcher: bubble requests
        if clar_up_q is not None:

            async def _watch_clarifications():
                try:
                    while True:
                        q = await clar_up_q.get()
                        # Requests may be raw strings or dicts with images/question
                        question = q.get("question") if isinstance(q, dict) else str(q)
                        await self._orch.events.put(
                            {
                                "type": "clarification_requested",
                                "call_id": call_id,
                                "tool_name": name,
                                "question": question or "",
                            },
                        )
                except asyncio.CancelledError:
                    return

            self._orch._register_child(
                self._orch._tg.create_task(_watch_clarifications()),
            )

    @staticmethod
    def is_safe_to_schedule_without_clar(fn: Callable) -> bool:
        """Return True if the tool does not require clarification/interject queues."""
        try:
            sig = inspect.signature(fn)
        except Exception:
            return True
        req = set()
        for name, p in sig.parameters.items():
            if p.default is inspect._empty and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                req.add(name)
        blocked = {
            "clarification_up_q",
            "clarification_down_q",
            "interject_queue",
        }
        return not (req & blocked)


class InterjectAdapter:
    """Adapter that would forward interjections as events.

    Not scheduled yet to avoid interfering with the legacy loop during
    delegation. When the evented path is fully enabled, this will consume
    `orchestrator.interject_queue` and post `interjected` events.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orch = orchestrator

    def schedule(self) -> None:  # pragma: no cover - skeleton
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        async def _task():
            while True:
                payload = await self._orch.interject_queue.get()
                await self._orch.events.put({"type": "interjected", "content": payload})

        self._orch._tg.create_task(_task())


class ControlAdapter:
    """Adapter that would convert control signals into events.

    Not scheduled yet; when enabled it should observe cancel/pause/resume and
    post `cancel_requested`, `pause_requested`, `resume_requested`.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orch = orchestrator

    def schedule(self) -> None:  # pragma: no cover - skeleton
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        async def _watch_cancel():
            await self._orch.cancel_event.wait()
            await self._orch.events.put({"type": "cancel_requested", "reason": None})

        self._orch._tg.create_task(_watch_cancel())


async def evented_tool_loop_inner(
    client: unify.AsyncUnify,
    message: str | dict | list[str | dict],
    tools: Dict[str, Callable],
    *,
    loop_id: Optional[str] = None,
    lineage: Optional[list[str]] = None,
    interject_queue: asyncio.Queue[dict | str],
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
    semantic_cache: Optional[bool] = False,
    images: Optional[dict[str, Any]] = None,
) -> str:
    """
    Event-driven orchestrator (skeleton): instantiate the orchestrator and run.

    Behaviour is currently identical to legacy because Orchestrator.run delegates
    to the legacy inner loop. This will evolve incrementally.
    """

    orch = Orchestrator(
        client=client,
        message=message,
        tools=tools,
        loop_id=loop_id,
        lineage=lineage,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
        stop_event=stop_event,
        pause_event=pause_event,
        max_consecutive_failures=max_consecutive_failures,
        prune_tool_duplicates=prune_tool_duplicates,
        interrupt_llm_with_interjections=interrupt_llm_with_interjections,
        propagate_chat_context=propagate_chat_context,
        parent_chat_context=parent_chat_context,
        log_steps=log_steps,
        max_steps=max_steps,
        timeout=timeout,
        raise_on_limit=raise_on_limit,
        include_class_in_dynamic_tool_names=include_class_in_dynamic_tool_names,
        tool_policy=tool_policy,
        preprocess_msgs=preprocess_msgs,
        outer_handle_container=outer_handle_container,
        response_format=response_format,
        max_parallel_tool_calls=max_parallel_tool_calls,
        semantic_cache=semantic_cache,
        images=images,
    )
    return await orch.run()
