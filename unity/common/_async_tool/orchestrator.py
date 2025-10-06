from __future__ import annotations

import asyncio
import os
import json
from typing import Optional, Dict, Callable, Tuple, Any, Union
from datetime import timedelta

import unify

# Delegate to the existing inner loop for now (skeleton orchestrator).
from .loop import _check_valid_response_format as _resp_schema
from .messages import generate_with_preprocess as _gwp
from .messages import _is_helper_tool as _is_helper_tool
from .messages import build_helper_ack_content as _helper_ack
from .messages import (
    ensure_placeholders_for_pending as _ensure_placeholders_for_pending,
)
from .messages import (
    find_unreplied_assistant_entries as _find_unreplied,
)
from .messages import (
    schedule_missing_for_message as _schedule_missing_for_message,
)
from .messages import insert_tool_message_after_assistant as _insert_tool_after
from .tools_utils import create_tool_call_message as _create_tool_call_msg
from .messages import (
    insert_tool_message_after_assistant as _insert_tool_message_after_assistant,
)
from .messages import forward_handle_call as _forward_handle_call
from .messages import propagate_stop_once as _propagate_stop_once
from .utils import maybe_await
from ..llm_helpers import method_to_schema as _method_to_schema
from ..llm_helpers import _dumps as _json_pretty
from ..tool_spec import normalise_tools
from ..tool_spec import ToolSpec
from . import semantic_cache as sc
from .tools_utils import create_tool_call_message
from .dynamic_tools_factory import DynamicToolFactory
from .tools_utils import append_source_scoped_images as _append_images
from .tools_utils import default_source_label as _default_img_src
from .tools_data import ToolsData as _ToolsData
from contextlib import suppress
from .message_dispatcher import LoopMessageDispatcher as _Dispatcher
from .loop_config import LoopConfig as _LoopConfig
from .loop_config import TOOL_LOOP_LINEAGE as _TOOL_LOOP_LINEAGE
from .loop_config import LIVE_IMAGES_LOG as _LIVE_IMAGES_LOG
from .timeout_timer import TimeoutTimer as _Timer
from .loop import LoopLogger as _LoopLogger
from .loop import _LoopToolFailureTracker as _FailureTracker
from .orchestrator_events import State, Event
from ...constants import LOGGER


## Events and State are imported from orchestrator_events


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

    async def _append_initial_message(self) -> None:
        """Append the initial message parameter to the client transcript.

        Normalises strings to a user turn and preserves dict/list structures as-is.
        """
        try:

            def _as_msgs(msg) -> list[dict]:
                if isinstance(msg, str):
                    return [{"role": "user", "content": msg}]
                if isinstance(msg, dict):
                    # Assume already in chat message shape
                    return [msg]
                if isinstance(msg, list):
                    out: list[dict] = []
                    for it in msg:
                        if isinstance(it, str):
                            out.append({"role": "user", "content": it})
                        elif isinstance(it, dict):
                            out.append(it)
                    return out
                return []

            msgs = _as_msgs(self.message)
            if not msgs:
                return
            cfg = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
            timer = _Timer(
                timeout=self.timeout,
                max_steps=self.max_steps,
                raise_on_limit=self.raise_on_limit,
                client=self.client,
            )
            dispatcher = _Dispatcher(self.client, cfg, timer)
            # Ensure the transcript begins with a system header carrying parent context
            # when provided – this mirrors legacy ordering used by tests.
            try:
                if self.parent_chat_context:
                    sys_msg = {
                        "role": "system",
                        "_ctx_header": True,
                        "content": (
                            "Broader context (read-only):\n"
                            + json.dumps(self.parent_chat_context, indent=2)
                            + "\n\nResolve the *next* user request in light of this."
                        ),
                    }
                    await dispatcher.append_msgs([sys_msg])
            except Exception:
                pass
            await dispatcher.append_msgs(msgs)

        except Exception:
            # Defensive: never fail appending initial messages
            pass

    async def run(self) -> str:
        # Ensure the initial user/system message(s) are present in the transcript
        try:
            await self._append_initial_message()
        except Exception:
            pass

        # Immediate limits guard (pre-LLM): enforce max_steps/timeout right after
        # the initial message is appended. This mirrors legacy behaviour where
        # USER + ASSISTANT would exceed max_steps=1 and must raise when
        # raise_on_limit=True.
        try:
            cfg0 = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
            timer0 = _Timer(
                timeout=self.timeout,
                max_steps=self.max_steps,
                raise_on_limit=self.raise_on_limit,
                client=self.client,
            )
            try:
                LOGGER.info(
                    "pre_llm_limit_guard: label=%s messages=%s max_steps=%s timeout=%s",
                    getattr(cfg0, "label", "<unknown>"),
                    len(getattr(self.client, "messages", []) or []),
                    self.max_steps,
                    self.timeout,
                )
            except Exception:
                pass
            # These checks will raise when raise_on_limit is True
            timer0.has_exceeded_time()
            timer0.has_exceeded_msgs()
        except Exception:
            # Propagate exception (TimeoutError / RuntimeError) to caller
            raise

        # Refactored: run the evented core path (single code path for all turns)
        return await self._run_evented_core()

    async def _run_evented_core(self) -> str:
        """Simplified, unified evented flow: first LLM → schedule tools → wait → final LLM.

        This consolidates the previously scattered first-turn and subsequent-turn logic
        into a single deterministic path.
        """
        # Build first-turn tool schemas using tool_policy (legacy parity)
        # This allows policies like ("required", {...}) to force a tool call on step 0.
        first_tool_choice: str = "auto"
        # When no policy is provided, default to HIDE-NONE for auto mode
        policy_returned_map: Dict[str, Callable] = {}
        all_tools_map: Dict[str, Callable] = dict(self.tools or {})
        if self.tool_policy is not None:
            try:
                first_tool_choice, policy_returned_map = self.tool_policy(
                    0,
                    dict(all_tools_map),
                )
            except Exception:
                # On policy failure, fall back to auto with no hidden tools
                first_tool_choice, policy_returned_map = "auto", {}

        # Legacy parity for first turn:
        # - If mode == "auto": mapping is the HIDE set → visible = all - returned
        # - Else: mapping is the SHOW set → visible = returned (fallback to all when empty)
        if first_tool_choice == "auto":
            first_visible_map = {
                n: f
                for n, f in all_tools_map.items()
                if n not in (policy_returned_map or {})
            }
        else:
            first_visible_map = dict(policy_returned_map) or dict(all_tools_map)

        # Build schemas from visible tools only
        schemas: list[dict] = []
        for name, fn in first_visible_map.items():
            try:
                schemas.append(_method_to_schema(fn, tool_name=name))
            except Exception:
                continue

        # Debug: record the chosen policy and visibility for the first turn
        try:
            LOGGER.info(
                "first_turn_policy: choice=%s tools=%s visible=%s",
                first_tool_choice,
                list((policy_returned_map or {}).keys()),
                list(first_visible_map.keys()),
            )
        except Exception:
            pass

        # If any interjections arrived before the first LLM turn, append a system message
        # that consolidates user-visible history and the latest interjection. This mirrors
        # legacy behaviour and ensures EVENT_BUS publishes a system entry containing
        # "user: **<interjection>**", and the LLM will incorporate the latest guidance.
        try:
            cfg0 = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
            timer0 = _Timer(
                timeout=self.timeout,
                max_steps=self.max_steps,
                raise_on_limit=self.raise_on_limit,
                client=self.client,
            )
            dispatcher0 = _Dispatcher(self.client, cfg0, timer0)

            last_payload = None
            while True:
                try:
                    payload = self.interject_queue.get_nowait()
                    last_payload = payload
                except Exception:
                    # QueueEmpty or any failure → stop draining
                    break

            if last_payload is not None:
                sys_msg = {
                    "role": "system",
                    "content": self._build_interjection_system_content(last_payload),
                }
                await dispatcher0.append_msgs([sys_msg])
        except Exception:
            pass

        # Track whether we consumed policy step=1 via the follow-up path
        consumed_followup_step1: bool = False

        # Ask for tool calls
        # Enforce a hard wall-clock timeout around the very first LLM turn to
        # preserve legacy semantics: when raise_on_limit=True, exceed → TimeoutError;
        # when False, terminate gracefully with an assistant notice.
        try:
            _first_timer = _Timer(
                timeout=self.timeout,
                max_steps=self.max_steps,
                raise_on_limit=self.raise_on_limit,
                client=self.client,
            )
            _remaining = _first_timer.remaining_time()
            if _remaining is None:
                await _gwp(
                    self.client,
                    self.preprocess_msgs,
                    return_full_completion=True,
                    tools=schemas,
                    tool_choice=first_tool_choice,
                    stateful=True,
                )
            else:
                # Guard against negative drift
                _timeout_secs = max(0.0, float(_remaining))
                await asyncio.wait_for(
                    _gwp(
                        self.client,
                        self.preprocess_msgs,
                        return_full_completion=True,
                        tools=schemas,
                        tool_choice=first_tool_choice,
                        stateful=True,
                    ),
                    timeout=_timeout_secs,
                )
        except asyncio.TimeoutError:
            # Hard timeout on the first turn
            if self.raise_on_limit:
                raise
            try:
                cfgX = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
                timerX = _Timer(
                    timeout=self.timeout,
                    max_steps=self.max_steps,
                    raise_on_limit=self.raise_on_limit,
                    client=self.client,
                )
                dispatcherX = _Dispatcher(self.client, cfgX, timerX)
                notice = {
                    "role": "assistant",
                    "content": f"🔚 Terminating early: timeout ({self.timeout}s) exceeded",
                }
                await dispatcherX.append_msgs([notice])
                return notice["content"]
            except Exception:
                return ""

        last = (
            self.client.messages[-1] if getattr(self.client, "messages", None) else None
        )
        # If the assistant answered directly, consider a policy-driven second turn
        if isinstance(last, dict) and not (last.get("tool_calls") or []):
            # If a policy is present and would reveal any tools on step 1, run a follow-up
            run_follow_up = False
            next_choice = "auto"
            next_policy_map: Dict[str, Callable] = {}
            try:
                if self.tool_policy is not None:
                    try:
                        next_choice, next_policy_map = self.tool_policy(
                            1,
                            dict(self.tools or {}),
                        )
                    except Exception:
                        next_choice, next_policy_map = "auto", {}

                    all_map = dict(self.tools or {})
                    if next_choice == "auto":
                        visible_next = {
                            n: f
                            for n, f in all_map.items()
                            if n not in (next_policy_map or {})
                        }
                    else:
                        visible_next = dict(next_policy_map) or dict(all_map)

                    try:
                        LOGGER.info(
                            "policy_eval_followup: step=1 mode=%s returned=%s visible=%s",
                            next_choice,
                            list((next_policy_map or {}).keys()),
                            list(visible_next.keys()),
                        )
                    except Exception:
                        pass

                    run_follow_up = bool(visible_next)
                else:
                    run_follow_up = False
            except Exception:
                run_follow_up = False

            if not run_follow_up:
                return last.get("content", "") or ""

            # Execute a second LLM turn with the visible tool set for step 1
            try:
                schemas_next: list[dict] = []
                for n, f in visible_next.items():
                    try:
                        schemas_next.append(_method_to_schema(f, tool_name=n))
                    except Exception:
                        continue
                await _gwp(
                    self.client,
                    self.preprocess_msgs,
                    return_full_completion=True,
                    tools=schemas_next,
                    tool_choice=next_choice,
                    stateful=True,
                )
                consumed_followup_step1 = True
            except Exception:
                # If follow-up failed, fall back to returning the first assistant answer
                return last.get("content", "") or ""

            # Refresh pointer to the latest assistant message after the follow-up
            last = (
                self.client.messages[-1]
                if getattr(self.client, "messages", None)
                else None
            )
            if isinstance(last, dict) and not (last.get("tool_calls") or []):
                return last.get("content", "") or ""

        # Locate the assistant turn that requested tools
        assistant_msg = None
        try:
            for _m in reversed(self.client.messages or []):
                if isinstance(_m, dict) and _m.get("role") == "assistant":
                    assistant_msg = _m
                    break
        except Exception:
            assistant_msg = None

        if not isinstance(assistant_msg, dict):
            return ""

        # Schedule tool calls
        cfg = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
        timer = _Timer(
            timeout=self.timeout,
            max_steps=self.max_steps,
            raise_on_limit=self.raise_on_limit,
            client=self.client,
        )
        dispatcher = _Dispatcher(self.client, cfg, timer)
        logger = _LoopLogger(cfg, self.log_steps)
        tools_data = _ToolsData(self.tools, client=self.client, logger=logger)
        assistant_meta: Dict[int, Dict[str, Any]] = {}

        # Enforce hidden per-loop quotas on the assistant turn before scheduling
        try:
            tools_data.prune_over_quota_tool_calls(assistant_msg)
        except Exception:
            pass

        # Seed per-loop quota counters from prior turns (evented-only accumulation)
        try:
            tools_data.call_counts.update(self._call_counts)
        except Exception:
            pass

        # Optionally prune duplicate tool calls within the same assistant turn
        call_list = list(assistant_msg.get("tool_calls") or [])
        if self.prune_tool_duplicates and call_list:
            try:
                seen: set[tuple[str, str]] = set()
                filtered: list[dict] = []
                for c in call_list:
                    fn = c.get("function", {}) or {}
                    nm = fn.get("name")
                    aj = fn.get("arguments", "{}")
                    key = (str(nm), str(aj))
                    if key in seen:
                        continue
                    seen.add(key)
                    filtered.append(c)
                if len(filtered) != len(call_list):
                    call_list = filtered
                    # Update transcript so the LLM no longer expects replies for pruned call_ids
                    try:
                        assistant_msg["tool_calls"] = call_list
                    except Exception:
                        pass
            except Exception:
                pass

        for idx, call in enumerate(call_list):
            try:
                nm = (call.get("function", {}) or {}).get("name")
                if not isinstance(nm, str) or _is_helper_tool(nm):
                    continue
                aj = (call.get("function", {}) or {}).get("arguments", "{}")
                cid = call.get("id") or "call"
                await tools_data.schedule_base_tool_call(
                    assistant_msg,
                    name=nm,
                    args_json=aj,
                    call_id=cid,
                    call_idx=idx,
                    parent_chat_context=self.parent_chat_context,
                    propagate_chat_context=self.propagate_chat_context,
                    assistant_meta=assistant_meta,
                )
                # Track per-loop quota usage across evented turns
                try:
                    self._call_counts[nm] = self._call_counts.get(nm, 0) + 1
                except Exception:
                    pass
            except Exception:
                continue

        # Ensure placeholders exist for all scheduled tool calls before any further LLM turns
        try:
            await _ensure_placeholders_for_pending(
                assistant_msg=assistant_msg,
                tools_data=tools_data,
                assistant_meta=assistant_meta,
                client=self.client,
                msg_dispatcher=dispatcher,
            )
        except Exception:
            pass

        # Strong guard: if any assistant tool_call ids still have no responding tool message,
        # schedule or backfill placeholders so the next assistant turn never triggers API 400s.
        try:
            findings = _find_unreplied(self.client)
            for f in findings:
                if f.get("assistant_msg") is not assistant_msg:
                    continue
                missing_ids = set(f.get("missing") or [])
                if not missing_ids:
                    continue

                # Attempt to schedule any missing ones (if not already scheduled)
                try:
                    await _schedule_missing_for_message(
                        assistant_msg,
                        missing_ids,
                        tools_data=tools_data,
                        parent_chat_context=self.parent_chat_context,
                        propagate_chat_context=self.propagate_chat_context,
                        assistant_meta=assistant_meta,
                        client=self.client,
                        msg_dispatcher=dispatcher,
                    )
                except Exception:
                    pass
                # Ensure placeholders now exist for these
                with suppress(Exception):
                    await _ensure_placeholders_for_pending(
                        assistant_msg=assistant_msg,
                        tools_data=tools_data,
                        assistant_meta=assistant_meta,
                        client=self.client,
                        msg_dispatcher=dispatcher,
                    )
                # As a final safety net, inject minimal tool placeholder messages directly
                # for any still-missing ids to satisfy the API contract.
                try:
                    # Recompute missing after attempts above
                    f2 = next(
                        (
                            x
                            for x in _find_unreplied(self.client)
                            if x.get("assistant_msg") is assistant_msg
                        ),
                        None,
                    )
                    still_missing = set(f2.get("missing") or []) if f2 else set()
                except Exception:
                    still_missing = set()
                if still_missing:
                    for cid in list(still_missing):
                        try:
                            # Find the function name for this call id (if present) for readability
                            fn_name = None
                            try:
                                for c in assistant_msg.get("tool_calls") or []:
                                    if c.get("id") == cid:
                                        fn_name = (c.get("function", {}) or {}).get(
                                            "name",
                                        )
                                        break
                            except Exception:
                                fn_name = None
                            tool_msg = _create_tool_call_msg(
                                name=str(fn_name or "tool"),
                                call_id=cid,
                                content="Pending… tool call accepted. Working on it.",
                            )
                            await _insert_tool_after(
                                assistant_meta,
                                assistant_msg,
                                tool_msg,
                                self.client,
                                dispatcher,
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        # Local early-exit helper for this minimal core
        async def _early_exit_local(reason: str) -> str:
            # Cancel any pending work and append a graceful termination notice
            for t in list(getattr(tools_data, "pending", [])):
                try:
                    inf = tools_data.info.get(t)
                    if (
                        inf is not None
                        and inf.handle is not None
                        and hasattr(inf.handle, "stop")
                    ):
                        await maybe_await(inf.handle.stop())
                except Exception:
                    pass
                try:
                    if not t.done():
                        t.cancel()
                except Exception:
                    pass
            try:
                await asyncio.gather(
                    *list(getattr(tools_data, "pending", [])),
                    return_exceptions=True,
                )
                tools_data.pending.clear()
            except Exception:
                pass
            notice = {
                "role": "assistant",
                "content": f"🔚 Terminating early: {reason}",
            }
            await dispatcher.append_msgs([notice])
            return notice["content"]

        # Wait for tool completions (no clar/notify handling in this minimal core)
        while tools_data.pending:
            done_tasks, _ = await asyncio.wait(
                set(tools_data.pending),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timer.remaining_time(),
            )
            if not done_tasks:
                # No tool finished within the allotted time budget. Mirror legacy
                # behaviour by cancelling pending work and appending a graceful
                # early-termination notice instead of proceeding to another LLM turn.
                try:
                    if timer.has_exceeded_time():
                        try:
                            LOGGER.info(
                                "evented_minimal_core_timeout: label=%s reason=%s",
                                getattr(cfg, "label", "<unknown>"),
                                f"timeout ({self.timeout}s) exceeded",
                            )
                        except Exception:
                            pass
                        return await _early_exit_local(
                            f"timeout ({self.timeout}s) exceeded",
                        )
                    if timer.has_exceeded_msgs():
                        try:
                            LOGGER.info(
                                "evented_minimal_core_timeout: label=%s reason=%s",
                                getattr(cfg, "label", "<unknown>"),
                                f"max_steps ({self.max_steps}) exceeded",
                            )
                        except Exception:
                            pass
                        return await _early_exit_local(
                            f"max_steps ({self.max_steps}) exceeded",
                        )
                except Exception:
                    # When raise_on_limit=True the checks above raise; let that bubble up.
                    raise
                # Fallback: treat as a timeout even if the guard did not trip.
                try:
                    LOGGER.info(
                        "evented_minimal_core_timeout: label=%s reason=%s",
                        getattr(cfg, "label", "<unknown>"),
                        "timeout exceeded",
                    )
                except Exception:
                    pass
                return await _early_exit_local("timeout exceeded")
            tracker = _FailureTracker(self.max_consecutive_failures)
            for t in list(done_tasks):
                try:
                    await tools_data.process_completed_task(
                        task=t,
                        consecutive_failures=tracker,
                        outer_handle_container=self.outer_handle_container,
                        assistant_meta=assistant_meta,
                        msg_dispatcher=dispatcher,
                    )
                except Exception:
                    raise

        # If we already executed a follow-up LLM turn to honour policy step=1 above,
        # finish with a final assistant turn WITHOUT tools to mirror legacy behaviour
        # (prevents an unnecessary second tool call when policy only revealed tools once).
        if consumed_followup_step1:
            try:
                await _gwp(
                    self.client,
                    self.preprocess_msgs,
                    return_full_completion=True,
                    tools=[],
                    tool_choice="auto",
                    stateful=True,
                )
                tail = (
                    self.client.messages[-1]
                    if getattr(self.client, "messages", None)
                    else None
                )
                if isinstance(tail, dict) and not (tail.get("tool_calls") or []):
                    return tail.get("content", "") or ""
            except Exception:
                pass

        # Subsequent LLM turns: keep offering the same schemas until no tool_calls are requested
        # If we already executed a follow-up LLM turn to honour policy step=1 above,
        # start counting subsequent policy turns from step=2 to avoid duplicating step=1.
        step = 2 if consumed_followup_step1 else 1
        while True:
            # Pre-LLM guard (subsequent turns): if we still have pending tools and
            # the very next assistant turn would consume the last remaining step,
            # terminate gracefully instead of letting the LLM emit a noop.
            try:
                cur_msgs = len(getattr(self.client, "messages", []) or [])
                if self.max_steps is not None and tools_data.pending:
                    if cur_msgs + 1 >= int(self.max_steps):
                        try:
                            LOGGER.info(
                                "pre_llm_limit_guard_subseq: pending_tools=%s messages=%s max_steps=%s -> early_exit",
                                len(tools_data.pending),
                                cur_msgs,
                                self.max_steps,
                            )
                        except Exception:
                            pass
                        return await _early_exit_local(
                            f"max_steps ({self.max_steps}) exceeded",
                        )
                # Also guard for wall-clock expiry before scheduling another LLM turn
                timer_guard = _Timer(
                    timeout=self.timeout,
                    max_steps=self.max_steps,
                    raise_on_limit=self.raise_on_limit,
                    client=self.client,
                )
                if timer_guard.has_exceeded_time():
                    return await _early_exit_local(
                        f"timeout ({self.timeout}s) exceeded",
                    )
            except Exception:
                # When raise_on_limit=True, let the exception bubble up
                raise

            # Apply step-specific tool policy (legacy parity):
            # On each subsequent turn, re-evaluate the policy with the current step index.
            # Semantics:
            # - mode == "auto" → returned mapping are tools to HIDE
            # - other modes (e.g. "required") → returned mapping are tools to SHOW
            step_tool_choice = "auto"
            step_policy_map: Dict[str, Callable] = {}
            visible_map_for_step: Dict[str, Callable] = dict(self.tools or {})
            try:
                if self.tool_policy is not None:
                    try:
                        step_tool_choice, step_policy_map = self.tool_policy(
                            step,
                            dict(self.tools or {}),
                        )
                    except Exception:
                        step_tool_choice, step_policy_map = "auto", {}

                    all_map = dict(self.tools or {})
                    if step_tool_choice == "auto":
                        visible_map_for_step = {
                            n: f
                            for n, f in all_map.items()
                            if n not in (step_policy_map or {})
                        }
                    else:
                        visible_map_for_step = dict(step_policy_map) or dict(all_map)

                    try:
                        LOGGER.info(
                            "subseq_policy: step=%s mode=%s returned=%s visible=%s",
                            step,
                            step_tool_choice,
                            list((step_policy_map or {}).keys()),
                            list(visible_map_for_step.keys()),
                        )
                    except Exception:
                        pass
            except Exception:
                # Fall back to exposing all tools with auto choice
                step_tool_choice = "auto"
                visible_map_for_step = dict(self.tools or {})

            # Build schemas for this step from the visible map
            schemas_for_step: list[dict] = []
            for name, fn in visible_map_for_step.items():
                try:
                    schemas_for_step.append(_method_to_schema(fn, tool_name=name))
                except Exception:
                    continue
            # Drain any pending interjections non-blockingly and append a consolidated
            # system message before the next LLM turn so the guidance is considered.
            try:
                _cfg_inter = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
                _timer_inter = _Timer(
                    timeout=self.timeout,
                    max_steps=self.max_steps,
                    raise_on_limit=self.raise_on_limit,
                    client=self.client,
                )
                _dispatcher_inter = _Dispatcher(self.client, _cfg_inter, _timer_inter)
                drained_any = False
                while True:
                    try:
                        _payload = self.interject_queue.get_nowait()
                    except Exception:
                        break
                    try:
                        _sys_msg = {
                            "role": "system",
                            "content": self._build_interjection_system_content(
                                _payload,
                            ),
                        }
                        await _dispatcher_inter.append_msgs([_sys_msg])
                        drained_any = True
                    except Exception:
                        # keep draining
                        pass
                if drained_any:
                    try:
                        LOGGER.info(
                            "interject_drain: appended guidance before LLM turn step=%s",
                            step,
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            await _gwp(
                self.client,
                self.preprocess_msgs,
                return_full_completion=True,
                tools=schemas_for_step,
                tool_choice=step_tool_choice,
                stateful=True,
            )
            tail = (
                self.client.messages[-1]
                if getattr(self.client, "messages", None)
                else None
            )
            if isinstance(tail, dict) and not (tail.get("tool_calls") or []):
                return tail.get("content", "") or ""

            # Schedule any new tool calls from this assistant turn
            if isinstance(tail, dict):
                cfg2 = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
                timer2 = _Timer(
                    timeout=self.timeout,
                    max_steps=self.max_steps,
                    raise_on_limit=self.raise_on_limit,
                    client=self.client,
                )
                dispatcher2 = _Dispatcher(self.client, cfg2, timer2)
                logger2 = _LoopLogger(cfg2, self.log_steps)
                tools_data2 = _ToolsData(self.tools, client=self.client, logger=logger2)
                assistant_meta2: Dict[int, Dict[str, Any]] = {}

                # Seed and enforce hidden per-loop quotas on subsequent assistant turns as well
                try:
                    tools_data2.call_counts.update(self._call_counts)
                except Exception:
                    pass
                try:
                    tools_data2.prune_over_quota_tool_calls(tail)
                except Exception:
                    pass

                # Optionally prune duplicates in subsequent assistant turn
                tail_calls = list(tail.get("tool_calls") or [])
                if self.prune_tool_duplicates and tail_calls:
                    try:
                        seen2: set[tuple[str, str]] = set()
                        filtered2: list[dict] = []
                        for c in tail_calls:
                            fn = c.get("function", {}) or {}
                            nm = fn.get("name")
                            aj = fn.get("arguments", "{}")
                            key = (str(nm), str(aj))
                            if key in seen2:
                                continue
                            seen2.add(key)
                            filtered2.append(c)
                        if len(filtered2) != len(tail_calls):
                            tail_calls = filtered2
                            # Update transcript so the LLM no longer expects replies for pruned call_ids
                            try:
                                tail["tool_calls"] = tail_calls
                            except Exception:
                                pass
                    except Exception:
                        pass

                for idx, call in enumerate(tail_calls):
                    try:
                        nm = (call.get("function", {}) or {}).get("name")
                        if not isinstance(nm, str) or _is_helper_tool(nm):
                            continue
                        aj = (call.get("function", {}) or {}).get("arguments", "{}")
                        cid = call.get("id") or "call"
                        await tools_data2.schedule_base_tool_call(
                            tail,
                            name=nm,
                            args_json=aj,
                            call_id=cid,
                            call_idx=idx,
                            parent_chat_context=self.parent_chat_context,
                            propagate_chat_context=self.propagate_chat_context,
                            assistant_meta=assistant_meta2,
                        )
                        # Track per-loop quota usage across evented turns
                        try:
                            self._call_counts[nm] = self._call_counts.get(nm, 0) + 1
                        except Exception:
                            pass
                    except Exception:
                        continue

                # Ensure placeholders exist for all scheduled tool calls before next LLM turn
                try:
                    await _ensure_placeholders_for_pending(
                        assistant_msg=tail,
                        tools_data=tools_data2,
                        assistant_meta=assistant_meta2,
                        client=self.client,
                        msg_dispatcher=dispatcher2,
                    )
                except Exception:
                    pass

                # Strong guard for subsequent turns as well
                try:
                    findings2 = _find_unreplied(self.client)
                    for f2 in findings2:
                        if f2.get("assistant_msg") is not tail:
                            continue
                        missing_ids2 = set(f2.get("missing") or [])
                        if not missing_ids2:
                            continue

                        try:
                            await _schedule_missing_for_message(
                                tail,
                                missing_ids2,
                                tools_data=tools_data2,
                                parent_chat_context=self.parent_chat_context,
                                propagate_chat_context=self.propagate_chat_context,
                                assistant_meta=assistant_meta2,
                                client=self.client,
                                msg_dispatcher=dispatcher2,
                            )
                        except Exception:
                            pass
                        with suppress(Exception):
                            await _ensure_placeholders_for_pending(
                                assistant_msg=tail,
                                tools_data=tools_data2,
                                assistant_meta=assistant_meta2,
                                client=self.client,
                                msg_dispatcher=dispatcher2,
                            )
                        # Final safety net for subsequent turn
                        try:
                            f3 = next(
                                (
                                    x
                                    for x in _find_unreplied(self.client)
                                    if x.get("assistant_msg") is tail
                                ),
                                None,
                            )
                            still_missing2 = (
                                set(f3.get("missing") or []) if f3 else set()
                            )
                        except Exception:
                            still_missing2 = set()
                        if still_missing2:
                            for cid2 in list(still_missing2):
                                try:
                                    fn2 = None
                                    try:
                                        for c in tail.get("tool_calls") or []:
                                            if c.get("id") == cid2:
                                                fn2 = (c.get("function", {}) or {}).get(
                                                    "name",
                                                )
                                                break
                                    except Exception:
                                        fn2 = None
                                    tool_msg2 = _create_tool_call_msg(
                                        name=str(fn2 or "tool"),
                                        call_id=cid2,
                                        content="Pending… tool call accepted. Working on it.",
                                    )
                                    await _insert_tool_after(
                                        assistant_meta2,
                                        tail,
                                        tool_msg2,
                                        self.client,
                                        dispatcher2,
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass

                while tools_data2.pending:
                    done2, _ = await asyncio.wait(
                        set(tools_data2.pending),
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=timer2.remaining_time(),
                    )
                    if not done2:
                        break
                    tracker2 = _FailureTracker(self.max_consecutive_failures)
                    for t in list(done2):
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

            step += 1
        # First-turn path is always enabled; no env gating

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
        # If a finite timeout is set, we still run the evented path – parity is maintained elsewhere.
        has_finite_timeout = self.timeout is not None and float(self.timeout) > 0

        # Additional parity gating: disable evented path when semantic cache or images are used
        # Semantic cache is supported in evented path: inject messages and a dummy tool like legacy
        sem_cache_hit = None
        if self.semantic_cache:
            try:
                sem_cache_hit = sc.search_semantic_cache(self.message)
            except Exception:
                sem_cache_hit = None

        async def _early_exit(
            reason: str,
            tools_data: _ToolsData,
            dispatcher: _Dispatcher,
        ) -> str:
            # Gracefully stop any nested handles, cancel tasks, and append notice (legacy parity)
            for t in list(getattr(tools_data, "pending", [])):
                try:
                    inf = tools_data.info.get(t)
                    if (
                        inf is not None
                        and inf.handle is not None
                        and hasattr(inf.handle, "stop")
                    ):
                        await maybe_await(inf.handle.stop())
                except Exception:
                    pass
                try:
                    if not t.done():
                        t.cancel()
                except Exception:
                    pass
            try:
                await asyncio.gather(
                    *list(getattr(tools_data, "pending", [])),
                    return_exceptions=True,
                )
                tools_data.pending.clear()
            except Exception:
                pass
            notice = {"role": "assistant", "content": f"🔚 Terminating early: {reason}"}
            await dispatcher.append_msgs([notice])
            return notice["content"]

        # Minimal evented execution path (temporary) – runs even if the detailed
        # evented logic below is accidentally skipped. This ensures simple flows
        # (like a single sync tool) complete without legacy fallback.
        try:
            # 1) First assistant turn asking for tools
            base_tools = {n: f for n, f in (self.tools or {}).items()}
            tool_schemas = []
            for n, f in base_tools.items():
                try:
                    tool_schemas.append(_method_to_schema(f, tool_name=n))
                except Exception:
                    continue
            await _gwp(
                self.client,
                self.preprocess_msgs,
                return_full_completion=True,
                tools=tool_schemas,
                tool_choice="auto",
                stateful=True,
            )

            # 2) If assistant already answered directly, return
            last = (
                self.client.messages[-1]
                if getattr(self.client, "messages", None)
                else None
            )
            if isinstance(last, dict) and not (last.get("tool_calls") or []):
                return last.get("content", "") or ""

            # 3) Schedule any base tool calls from the assistant turn
            assistant_msg = None
            for _m in reversed(self.client.messages or []):
                if isinstance(_m, dict) and _m.get("role") == "assistant":
                    assistant_msg = _m
                    break
            if isinstance(assistant_msg, dict):
                cfg = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
                timer = _Timer(
                    timeout=self.timeout,
                    max_steps=self.max_steps,
                    raise_on_limit=self.raise_on_limit,
                    client=self.client,
                )
                dispatcher = _Dispatcher(self.client, cfg, timer)
                logger = _LoopLogger(cfg, self.log_steps)
                tools_data = _ToolsData(self.tools, client=self.client, logger=logger)
                assistant_meta: Dict[int, Dict[str, Any]] = {}

                for idx, call in enumerate(list(assistant_msg.get("tool_calls") or [])):
                    try:
                        nm = (call.get("function", {}) or {}).get("name")
                        if not isinstance(nm, str) or _is_helper_tool(nm):
                            continue
                        aj = (call.get("function", {}) or {}).get("arguments", "{}")
                        cid = call.get("id") or "call"
                        await tools_data.schedule_base_tool_call(
                            assistant_msg,
                            name=nm,
                            args_json=aj,
                            call_id=cid,
                            call_idx=idx,
                            parent_chat_context=self.parent_chat_context,
                            propagate_chat_context=self.propagate_chat_context,
                            assistant_meta=assistant_meta,
                        )
                    except Exception:
                        continue

                # 4) Wait for all scheduled tools to complete
                while tools_data.pending:
                    done_tasks, _ = await asyncio.wait(
                        set(tools_data.pending),
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=timer.remaining_time(),
                    )
                    if not done_tasks:
                        break
                    tracker = _FailureTracker(self.max_consecutive_failures)
                    for t in list(done_tasks):
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

                # 5) Ask the model for the final answer (no tools needed now)
                await _gwp(
                    self.client,
                    self.preprocess_msgs,
                    return_full_completion=True,
                    tools=[],
                    tool_choice="auto",
                    stateful=True,
                )
                final_msg = (
                    self.client.messages[-1]
                    if getattr(self.client, "messages", None)
                    else None
                )
                if isinstance(final_msg, dict):
                    content = final_msg.get("content", "") or ""
                    if content:
                        return content
        except Exception:
            # Fall through to the comprehensive evented path or legacy fallback
            pass

            # First-turn: run a single LLM step with optional tool schemas and
            # honour interjection preemption. Do not mutate the transcript beyond
            # what the LLM produces; let the legacy loop handle scheduling,
            # placeholders, helper-acks, and all ordering semantics for parity.
            # First-turn: run a single LLM step with optional tool schemas and
            # honour interjection preemption. Do not mutate the transcript beyond
            # what the LLM produces; let the legacy loop handle scheduling,
            # placeholders, helper-acks, and all ordering semantics for parity.
            # Inject parent chat context header (legacy parity)
            try:
                if self.parent_chat_context:
                    cfg0 = _LoopConfig(self.loop_id, self.lineage, self.lineage or [])
                    timer0 = _Timer(
                        timeout=self.timeout,
                        max_steps=self.max_steps,
                        raise_on_limit=self.raise_on_limit,
                        client=self.client,
                    )
                    dispatcher0 = _Dispatcher(self.client, cfg0, timer0)
                    sys_msg = {
                        "role": "system",
                        "_ctx_header": True,
                        "content": (
                            "Broader context (read-only):\n"
                            + json.dumps(self.parent_chat_context, indent=2)
                            + "\n\nResolve the *next* user request in light of this."
                        ),
                    }
                    await dispatcher0.append_msgs([sys_msg])
                    # Pre-LLM early-exit: enforce limits before starting LLM
                    if timer0.has_exceeded_time():
                        notice = {
                            "role": "assistant",
                            "content": f"🔚 Terminating early: timeout ({self.timeout}s) exceeded",
                        }
                        await dispatcher0.append_msgs([notice])
                        return notice["content"]
                    if timer0.has_exceeded_msgs():
                        notice = {
                            "role": "assistant",
                            "content": f"🔚 Terminating early: max_steps ({self.max_steps}) exceeded",
                        }
                        await dispatcher0.append_msgs([notice])
                        return notice["content"]
            except Exception:
                pass
            # Append the initial message(s) if not already present in transcript
            try:
                has_user_already = any(
                    m.get("role") == "user" for m in (self.client.messages or [])
                )
            except Exception:
                has_user_already = False
            if not has_user_already:
                await self._append_initial_message()
            # Determine tool exposure and policy for the first LLM turn
            expose_tools = True

            tool_choice_mode = "auto"
            policy_returned_map: Dict[str, Callable] = dict(self.tools or {})
            full_tools_map: Dict[str, Callable] = dict(self.tools or {})
            if self.tool_policy is not None:
                try:
                    tool_choice_mode, policy_returned_map = self.tool_policy(
                        0,
                        dict(full_tools_map),
                    )
                except Exception:
                    tool_choice_mode = "auto"
                    policy_returned_map = dict(full_tools_map)

            # Legacy semantics (loop.py parity):
            # - For mode == "auto", the policy's returned mapping represents tools to HIDE.
            # - For other modes (e.g., "required"), the returned mapping represents tools to SHOW.
            if tool_choice_mode == "auto":
                effective_visible_map = {
                    name: fn
                    for name, fn in full_tools_map.items()
                    if name not in (policy_returned_map or {})
                }
            else:
                effective_visible_map = (
                    dict(policy_returned_map)
                    if policy_returned_map
                    else dict(full_tools_map)
                )

            try:
                LOGGER.info(
                    "policy_eval_first_turn: mode=%s returned=%s visible=%s",
                    tool_choice_mode,
                    list((policy_returned_map or {}).keys()),
                    list(effective_visible_map.keys()),
                )
            except Exception:
                pass

            tools_param: list[dict] = []
            if expose_tools:
                # Build schemas from the filtered mapping with pre-exposure filtering
                try:
                    normalized = normalise_tools(
                        {n: fn for n, fn in effective_visible_map.items()},
                    )
                except Exception:
                    normalized = {
                        n: ToolSpec(fn=fn) for n, fn in effective_visible_map.items()
                    }
                td0 = _ToolsData(
                    self.tools,
                    client=self.client,
                    logger=_LoopLogger(
                        _LoopConfig(self.loop_id, self.lineage, self.lineage or []),
                        self.log_steps,
                    ),
                )
                try:
                    _self_task = asyncio.current_task()
                    if _self_task is not None and hasattr(_self_task, "task_info"):
                        td0.info = getattr(_self_task, "task_info", {})
                except Exception:
                    pass
                for name, spec in normalized.items():
                    try:
                        limit = getattr(spec, "max_total_calls", None)
                        used = 0
                        try:
                            used = sum(
                                1 for _t, _inf in td0.info.items() if _inf.name == name
                            )
                        except Exception:
                            used = 0
                        if limit is not None and used >= int(limit):
                            continue
                        max_cc = getattr(spec, "max_concurrent", None)
                        active = 0
                        try:
                            active = td0.active_count(name)
                        except Exception:
                            active = 0
                        if max_cc is not None and active >= int(max_cc):
                            continue
                        tools_param.append(_method_to_schema(spec.fn, tool_name=name))
                    except Exception:
                        continue
                # Live image helpers (reuse legacy docstrings; cheap exposure)
                if self.images:
                    try:
                        # Build minimal overview doc with any prior appended images
                        prior_lines = []
                        try:
                            for rec in _LIVE_IMAGES_LOG.get() or []:
                                try:
                                    src, iid_s, span_key = rec.split(":", 2)
                                    prior_lines.append(
                                        f"- source={src}, id={int(iid_s)}, span={span_key}",
                                    )
                                except Exception:
                                    continue
                        except Exception:
                            prior_lines = []
                        overview_doc = (
                            "Live images aligned to the current user_message (visible in this description; calling is optional).\n"
                            + ("\n".join(prior_lines) if prior_lines else "(none)")
                        )

                        async def live_images_overview() -> Dict[str, str]:  # type: ignore[name-defined]
                            return {"status": "ok"}

                        live_images_overview.__doc__ = overview_doc  # type: ignore[attr-defined]

                        # Image helpers
                        id_to_handle: dict[int, Any] = {}
                        try:
                            for span_key, ih in list(self.images.items()):
                                try:
                                    img_id = int(getattr(ih, "image_id", -1))
                                except Exception:
                                    img_id = -1
                                id_to_handle[img_id] = ih
                        except Exception:
                            id_to_handle = {}

                        async def ask_image(*, image_id: int, question: str, images: dict | None = None) -> Any:  # type: ignore[valid-type]
                            ih = id_to_handle.get(int(image_id))
                            if ih is None:
                                return {"error": f"image_id {image_id} not found"}
                            try:
                                _append_images(images, _default_img_src("ask"))
                            except Exception:
                                pass
                            try:
                                return await ih.ask(question)
                            except Exception as _exc:  # noqa: BLE001
                                return {"error": str(_exc)}

                        async def attach_image_raw(*, image_id: int, note: str | None = None) -> Dict[str, Any]:  # type: ignore[valid-type,name-defined]
                            iid = int(image_id)
                            ih = id_to_handle.get(iid)
                            if ih is None:
                                return {"error": f"image_id {iid} not found"}
                            # Build image content block (GCS/URL/data URI/bytes) like legacy
                            try:
                                data_str = getattr(
                                    getattr(ih, "_image", None),
                                    "data",
                                    None,
                                )
                                is_gcs_url = isinstance(data_str, str) and (
                                    data_str.startswith("gs://")
                                    or data_str.startswith(
                                        "https://storage.googleapis.com/",
                                    )
                                )
                                if is_gcs_url:
                                    try:
                                        from urllib.parse import urlparse as _urlparse

                                        parsed_url = _urlparse(data_str)
                                        bucket_name = ""
                                        object_path = ""
                                        if parsed_url.scheme == "gs":
                                            bucket_name = parsed_url.netloc
                                            object_path = parsed_url.path.lstrip("/")
                                        elif (
                                            parsed_url.hostname
                                            == "storage.googleapis.com"
                                        ):
                                            parts = parsed_url.path.lstrip("/").split(
                                                "/",
                                                1,
                                            )
                                            if len(parts) == 2:
                                                bucket_name, object_path = parts
                                        storage_client = getattr(
                                            getattr(ih, "_manager", None),
                                            "storage_client",
                                            None,
                                        )
                                        bucket = storage_client.bucket(bucket_name)
                                        blob = bucket.blob(object_path)
                                        signed_url = blob.generate_signed_url(
                                            version="v4",
                                            expiration=timedelta(hours=1),
                                            method="GET",
                                        )
                                        content_block = {
                                            "type": "image_url",
                                            "image_url": {"url": signed_url},
                                        }
                                    except Exception:
                                        raw = ih.raw()
                                        import base64 as _b64

                                        head = (
                                            bytes(raw[:10])
                                            if isinstance(raw, (bytes, bytearray))
                                            else b""
                                        )
                                        if head.startswith(b"\xff\xd8"):
                                            mime = "image/jpeg"
                                        elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                                            mime = "image/png"
                                        else:
                                            mime = "image/png"
                                        b64 = _b64.b64encode(raw).decode("ascii")
                                        content_block = {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{mime};base64,{b64}",
                                            },
                                        }
                                elif isinstance(data_str, str) and (
                                    data_str.startswith("http://")
                                    or data_str.startswith("https://")
                                    or data_str.startswith("data:image/")
                                ):
                                    content_block = {
                                        "type": "image_url",
                                        "image_url": {"url": data_str},
                                    }
                                else:
                                    raw = ih.raw()
                                    import base64 as _b64

                                    head = (
                                        bytes(raw[:10])
                                        if isinstance(raw, (bytes, bytearray))
                                        else b""
                                    )
                                    if head.startswith(b"\xff\xd8"):
                                        mime = "image/jpeg"
                                    elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                                        mime = "image/png"
                                    else:
                                        mime = "image/png"
                                    b64 = _b64.b64encode(raw).decode("ascii")
                                    content_block = {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{b64}",
                                        },
                                    }

                                # Append as a user block via dispatcher
                                cfgA = _LoopConfig(
                                    self.loop_id,
                                    self.lineage,
                                    self.lineage or [],
                                )
                                timerA = _Timer(
                                    timeout=self.timeout,
                                    max_steps=self.max_steps,
                                    raise_on_limit=self.raise_on_limit,
                                    client=self.client,
                                )
                                dispatcherA = _Dispatcher(self.client, cfgA, timerA)
                                await dispatcherA.append_msgs(
                                    [
                                        {
                                            "role": "user",
                                            "content": (
                                                [content_block]
                                                if note is None
                                                else [
                                                    {"type": "text", "text": note},
                                                    content_block,
                                                ]
                                            ),
                                        },
                                    ],
                                )
                                try:
                                    _append_images(None, _default_img_src("attach"))
                                except Exception:
                                    pass
                                return {"status": "attached", "image_id": iid}
                            except Exception as _exc:  # noqa: BLE001
                                return {"error": str(_exc)}

                        async def align_images_for(*, args: dict, hints: list[dict]) -> dict:  # type: ignore[valid-type]
                            out: dict[str, int] = {}
                            try:
                                arg_texts = {
                                    str(k): str(v) for k, v in dict(args or {}).items()
                                }
                            except Exception:
                                arg_texts = {}

                            def _extract_id(obj: dict) -> int | None:
                                for k in ("image_id", "imageId", "id"):
                                    if k in obj:
                                        try:
                                            return int(obj[k])
                                        except Exception:
                                            return None
                                return None

                            def _extract_arg(obj: dict) -> str | None:
                                for k in ("arg", "argument", "arg_name", "name"):
                                    if k in obj:
                                        return str(obj[k])
                                return None

                            def _extract_substring(obj: dict) -> str | None:
                                for k in ("substring", "text", "span_text"):
                                    if k in obj:
                                        return str(obj[k])
                                return None

                            for item in list(hints or []):
                                if not isinstance(item, dict):
                                    continue
                                iid = _extract_id(item)
                                arg_name = _extract_arg(item)
                                sub = _extract_substring(item)
                                if iid is None or not arg_name or sub is None:
                                    continue
                                base = arg_texts.get(arg_name)
                                if not isinstance(base, str):
                                    continue
                                try:
                                    start = base.find(sub)
                                    if start < 0:
                                        continue
                                    end = start + len(sub)
                                    key = f"{arg_name}[{start}:{end}]"
                                    out[key] = iid
                                except Exception:
                                    continue
                            return {"images": out}

                        # Append schemas
                        try:
                            tools_param.append(
                                _method_to_schema(
                                    live_images_overview,
                                    tool_name="live_images_overview",
                                ),
                            )
                        except Exception:
                            pass
                        try:
                            tools_param.append(
                                _method_to_schema(
                                    align_images_for,
                                    tool_name="align_images_for",
                                ),
                            )
                        except Exception:
                            pass
                        try:
                            tools_param.append(
                                _method_to_schema(ask_image, tool_name="ask_image"),
                            )
                        except Exception:
                            pass
                        try:
                            tools_param.append(
                                _method_to_schema(
                                    attach_image_raw,
                                    tool_name="attach_image_raw",
                                ),
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                # Dynamic helper tools for in-flight calls
                try:
                    tools_data_tmp = _ToolsData(
                        self.tools,
                        client=self.client,
                        logger=_LoopLogger(
                            _LoopConfig(self.loop_id, self.lineage, self.lineage or []),
                            self.log_steps,
                        ),
                    )
                    # Populate with any pending tasks already visible on current task (if any)
                    try:
                        _self_task = asyncio.current_task()
                        if _self_task is not None and hasattr(_self_task, "task_info"):
                            tools_data_tmp.info = getattr(_self_task, "task_info", {})
                            tools_data_tmp.pending = set(
                                list(getattr(_self_task, "task_info", {}).keys()),
                            )
                    except Exception:
                        pass
                    dyn_factory = DynamicToolFactory(tools_data_tmp)
                    dyn_factory.generate()
                    # Hide `wait` if any task awaits clarification
                    try:
                        if any(
                            getattr(_inf, "waiting_for_clarification", False)
                            for _inf in tools_data_tmp.info.values()
                        ):
                            dyn_factory.dynamic_tools.pop("wait", None)
                    except Exception:
                        pass
                    # Merge dynamic helpers into tools_param schema
                    for _nm, _fn in dyn_factory.dynamic_tools.items():
                        try:
                            tools_param.append(_method_to_schema(_fn, tool_name=_nm))
                        except Exception:
                            continue
                except Exception:
                    pass
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

            # Inject semantic cache messages & dummy tool (if any hit)
            if sem_cache_hit:
                try:
                    msgs = await sc.get_dummy_tool(
                        sem_cache_hit,
                        _ToolsData(
                            self.tools,
                            client=self.client,
                            logger=_LoopLogger(
                                _LoopConfig(
                                    self.loop_id,
                                    self.lineage,
                                    self.lineage or [],
                                ),
                                self.log_steps,
                            ),
                        ),
                    )
                    self.client.append_messages(msgs)
                    self.client.set_system_message(
                        (self.client.system_message or "") + sc.get_system_msg_hint(),
                    )
                    try:
                        tools_param.append(
                            _method_to_schema(
                                sc.semantic_search_placeholder,
                                tool_name="semantic_search",
                            ),
                        )
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

            # Global pause gating – do not allow LLM turns while paused
            try:
                if not self.pause_event.is_set():
                    # Wait until resume or cancel, but still allow cancel to win
                    cancel_gate = asyncio.create_task(
                        self.cancel_event.wait(),
                        name="EventedPauseCancelGate",
                    )
                    resume_gate = asyncio.create_task(
                        self.pause_event.wait(),
                        name="EventedPauseResumeGate",
                    )
                    done_gate, _ = await asyncio.wait(
                        {cancel_gate, resume_gate},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Cancel whichever helper didn't complete
                    for g in (cancel_gate, resume_gate):
                        if g not in done_gate and not g.done():
                            g.cancel()
                    await asyncio.gather(
                        cancel_gate,
                        resume_gate,
                        return_exceptions=True,
                    )
                    # If cancelled while paused, respect cancellation before LLM step
                    if self.cancel_event.is_set():
                        if self.stop_event is not None:
                            try:
                                self.stop_event.set()
                            except Exception:
                                pass
                        # Mirror legacy: no extra assistant message here; handoff to legacy cancel path
                        raise asyncio.CancelledError
            except Exception:
                pass

            # Time/step guard for first-turn LLM
            timer0 = _Timer(
                timeout=self.timeout,
                max_steps=self.max_steps,
                raise_on_limit=self.raise_on_limit,
                client=self.client,
            )
            # Limits already enforced above when header was injected; keep here as a guard without raising

            llm_task = asyncio.create_task(
                _gwp(self.client, self.preprocess_msgs, **gen_kwargs),
                name="EventedTurnLLM",
            )
            interject_w = (
                asyncio.create_task(
                    self.interject_queue.get(),
                    name="EventedTurnInterject",
                )
                if not enable_interject_adapter
                else None
            )
            cancel_w = asyncio.create_task(
                self.cancel_event.wait(),
                name="EventedTurnCancel",
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

                # Defer placeholder insertion to ensure_placeholders_for_pending
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
                    assistant_msg = None
                    for _m in reversed(self.client.messages or []):
                        if isinstance(_m, dict) and _m.get("role") == "assistant":
                            assistant_msg = _m
                            break
                except Exception:
                    assistant_msg = None
                if isinstance(assistant_msg, dict):

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

                    async def _handle_helper_call(name: str, call: dict) -> str:
                        args_json = (call.get("function", {}) or {}).get(
                            "arguments",
                            "{}",
                        )
                        call_id = call.get("id") or "call"
                        # wait → drop
                        if name == "wait":
                            return "drop"
                        # stop_*
                        if name.startswith("stop_") and not name.startswith(
                            "_stop_tasks",
                        ):
                            try:
                                suffix = name.split("_")[-1]
                                tgt_task = next(
                                    (
                                        t
                                        for t, info in tools_data.info.items()
                                        if str(info.call_id).endswith(suffix)
                                    ),
                                    None,
                                )
                                payload = (
                                    json.loads(args_json)
                                    if isinstance(args_json, str)
                                    else (args_json or {})
                                )
                                if tgt_task:
                                    nested_handle = tools_data.info[tgt_task].handle
                                    if nested_handle is not None and hasattr(
                                        nested_handle,
                                        "stop",
                                    ):
                                        await _forward_handle_call(
                                            nested_handle,
                                            "stop",
                                            payload,
                                            fallback_positional_keys=["reason"],
                                        )
                                    if not tgt_task.done():
                                        tgt_task.cancel()
                                    tools_data.pop_task(tgt_task)
                                try:
                                    _append_images(
                                        payload.get("images"),
                                        _default_img_src("stop"),
                                    )
                                except Exception:
                                    pass
                                pretty_name = (
                                    f"stop   {tools_data.info[tgt_task].name}({tools_data.info[tgt_task].call_dict['function']['arguments']})"
                                    if tgt_task
                                    else name
                                )
                                tool_msg = create_tool_call_message(
                                    name=pretty_name,
                                    call_id=call_id,
                                    content=f"The tool call [{suffix}] has been stopped successfully.",
                                )
                                await _insert_tool_message_after_assistant(
                                    assistant_meta,
                                    assistant_msg,
                                    tool_msg,
                                    self.client,
                                    dispatcher,
                                )
                            except Exception:
                                pass
                            return "handled"
                        # pause_*
                        if name.startswith("pause_") and not name.startswith(
                            "_pause_tasks",
                        ):
                            try:
                                suffix = name.split("_")[-1]
                                payload = (
                                    json.loads(args_json)
                                    if isinstance(args_json, str)
                                    else (args_json or {})
                                )
                                tgt_task = next(
                                    (
                                        t
                                        for t, info in tools_data.info.items()
                                        if suffix in info.call_id
                                    ),
                                    None,
                                )
                                if tgt_task:
                                    h = tools_data.info[tgt_task].handle
                                    ev = tools_data.info[tgt_task].pause_event
                                    if h is not None and hasattr(h, "pause"):
                                        await _forward_handle_call(h, "pause", payload)
                                    elif ev is not None:
                                        ev.clear()
                                pretty_name = (
                                    f"pause {tools_data.info[tgt_task].name}({tools_data.info[tgt_task].call_dict['function']['arguments']})"
                                    if tgt_task
                                    else name
                                )
                                tool_msg = create_tool_call_message(
                                    name=pretty_name,
                                    call_id=call_id,
                                    content=f"The tool call [{suffix}] has been paused successfully.",
                                )
                                await _insert_tool_message_after_assistant(
                                    assistant_meta,
                                    assistant_msg,
                                    tool_msg,
                                    self.client,
                                    dispatcher,
                                )
                            except Exception:
                                pass
                            return "handled"
                        # resume_*
                        if name.startswith("resume_") and not name.startswith(
                            "_resume_tasks",
                        ):
                            try:
                                suffix = name.split("_")[-1]
                                payload = (
                                    json.loads(args_json)
                                    if isinstance(args_json, str)
                                    else (args_json or {})
                                )
                                tgt_task = next(
                                    (
                                        t
                                        for t, info in tools_data.info.items()
                                        if suffix in info.call_id
                                    ),
                                    None,
                                )
                                if tgt_task:
                                    h = tools_data.info[tgt_task].handle
                                    ev = tools_data.info[tgt_task].pause_event
                                    if h is not None and hasattr(h, "resume"):
                                        await _forward_handle_call(h, "resume", payload)
                                    elif ev is not None:
                                        ev.set()
                                pretty_name = (
                                    f"resume {tools_data.info[tgt_task].name}({tools_data.info[tgt_task].call_dict['function']['arguments']})"
                                    if tgt_task
                                    else name
                                )
                                tool_msg = create_tool_call_message(
                                    name=pretty_name,
                                    call_id=call_id,
                                    content=f"The tool call [{suffix}] has been resumed successfully.",
                                )
                                await _insert_tool_message_after_assistant(
                                    assistant_meta,
                                    assistant_msg,
                                    tool_msg,
                                    self.client,
                                    dispatcher,
                                )
                            except Exception:
                                pass
                            return "handled"
                        # clarify_*
                        if name.startswith("clarify_"):
                            try:
                                payload = (
                                    json.loads(args_json)
                                    if isinstance(args_json, str)
                                    else (args_json or {})
                                )
                                ans = payload.get("answer")
                                suffix = name.split("_")[-1]
                                clar_key = next(
                                    (
                                        k
                                        for k in tools_data.clarification_channels.keys()
                                        if k.endswith(suffix)
                                    ),
                                    None,
                                )
                                if clar_key is not None and ans is not None:
                                    await tools_data.clarification_channels[clar_key][
                                        1
                                    ].put(ans)
                                    for _t, _inf in tools_data.info.items():
                                        if str(_inf.call_id).endswith(suffix):
                                            _inf.waiting_for_clarification = False
                                            break
                                try:
                                    _append_images(
                                        (
                                            payload.get("images")
                                            if isinstance(payload, dict)
                                            else None
                                        ),
                                        _default_img_src("clar_answer"),
                                    )
                                except Exception:
                                    pass
                                tool_reply_msg = create_tool_call_message(
                                    name=name,
                                    call_id=call_id,
                                    content=(
                                        f"Clarification answer sent upstream: {ans!r}\n"
                                        "⏳ Waiting for the original tool to finish…"
                                    ),
                                )
                                await _insert_tool_message_after_assistant(
                                    assistant_meta,
                                    assistant_msg,
                                    tool_reply_msg,
                                    self.client,
                                    dispatcher,
                                )
                            except Exception:
                                pass
                            return "handled"
                        # interject_*
                        if name.startswith("interject_"):
                            try:
                                payload = (
                                    json.loads(args_json)
                                    if isinstance(args_json, str)
                                    else (args_json or {})
                                )
                                new_text = (
                                    payload.get("content")
                                    or payload.get("message")
                                    or ""
                                )
                                suffix = name.split("_")[-1]
                                tgt_task = next(
                                    (
                                        t
                                        for t, inf in tools_data.info.items()
                                        if str(inf.call_id).endswith(suffix)
                                    ),
                                    None,
                                )
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
                                            fallback_positional_keys=[
                                                "content",
                                                "message",
                                            ],
                                        )
                                try:
                                    _append_images(
                                        payload.get("images"),
                                        _default_img_src("interjection"),
                                    )
                                except Exception:
                                    pass
                                pretty_name = (
                                    f"interject {tools_data.info[tgt_task].name}({new_text})"
                                    if tgt_task
                                    else name
                                )
                                tool_msg = create_tool_call_message(
                                    name=pretty_name,
                                    call_id=call_id,
                                    content=f'Guidance "{new_text}" forwarded to the running tool.',
                                )
                                await _insert_tool_message_after_assistant(
                                    assistant_meta,
                                    assistant_msg,
                                    tool_msg,
                                    self.client,
                                    dispatcher,
                                )
                            except Exception:
                                pass
                            return "handled"
                        return "skip"

                    # Process helper calls (drop/handle) and prune from assistant msg
                    try:
                        calls0 = list(assistant_msg.get("tool_calls") or [])
                        remaining_calls0 = []
                        for c in calls0:
                            _nm = (c.get("function", {}) or {}).get("name")
                            if isinstance(_nm, str) and _is_helper_tool(_nm):
                                res = await _handle_helper_call(_nm, c)
                                if res in ("drop", "handled"):
                                    continue
                            remaining_calls0.append(c)
                        if len(remaining_calls0) != len(calls0):
                            assistant_msg["tool_calls"] = remaining_calls0

                        # If there are no tool_calls at all, return final assistant content immediately (legacy parity)
                        if not (assistant_msg.get("tool_calls") or []):
                            _content = assistant_msg.get("content", "")
                            return _content
                    except Exception:
                        pass
                    # Schedule base tool calls (enforce max_parallel_tool_calls)
                    scheduled_count = 0
                    max_calls = (
                        int(self.max_parallel_tool_calls)
                        if self.max_parallel_tool_calls is not None
                        else None
                    )
                    for idx, call in enumerate(
                        list(assistant_msg.get("tool_calls") or []),
                    ):
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
                                assistant_msg,
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
                            assistant_msg=assistant_msg,
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
                            name="TurnInterject",
                        )
                        if not enable_interject_adapter
                        else None
                    )
                    cancel_waiter2 = asyncio.create_task(
                        self.cancel_event.wait(),
                        name="TurnCancel",
                    )
                    clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                    notif_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                    _stop_forwarded_once = False
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
                                name="TurnClarification",
                            )
                            clar_waiters[cw] = _t
                        if _inf.notification_queue is not None:
                            pw = asyncio.create_task(
                                _inf.notification_queue.get(),
                                name="TurnNotification",
                            )
                            notif_waiters[pw] = _t

                    # Loop to handle multiple completions before handing off
                    canceled = False
                    interjected = False
                    llm_turn_required = False
                    while True:
                        # Determine whether we have any real waiters (not just cancel)
                        _has_real_waiters = bool(
                            tools_data.pending
                            or clar_waiters
                            or notif_waiters
                            or (interject_waiter is not None),
                        )
                        if not _has_real_waiters:
                            break

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
                        # Redundant guard (kept for safety)
                        if not waitset:
                            break
                        # Early-limit checks before waiting
                        try:
                            if timer.has_exceeded_time():
                                return await _early_exit(
                                    f"timeout ({self.timeout}s) exceeded",
                                    tools_data,
                                    dispatcher,
                                )
                            if timer.has_exceeded_msgs():
                                return await _early_exit(
                                    f"max_steps ({self.max_steps}) exceeded",
                                    tools_data,
                                    dispatcher,
                                )
                        except Exception:
                            pass

                        done_first, _ = await asyncio.wait(
                            waitset,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=timer.remaining_time(),
                        )
                        if not done_first:
                            return await _early_exit(
                                f"timeout ({self.timeout}s) exceeded",
                                tools_data,
                                dispatcher,
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
                                try:
                                    _stop_forwarded_once = await _propagate_stop_once(
                                        tools_data.info,
                                        _stop_forwarded_once,
                                        "outer-loop cancelled",
                                    )
                                except Exception:
                                    pass
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
                                            assistant_msg,
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
                                    # Forward programmatic event upstream
                                    try:
                                        outer = (
                                            self.outer_handle_container[0]
                                            if self.outer_handle_container
                                            else None
                                        )
                                        if outer is not None and hasattr(
                                            outer,
                                            "_clar_q",
                                        ):
                                            await outer._clar_q.put(
                                                {
                                                    "type": "clarification",
                                                    "call_id": call_id,
                                                    "tool_name": tools_data.info[
                                                        src_task
                                                    ].name,
                                                    "question": q_text,
                                                },
                                            )
                                    except Exception:
                                        pass
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
                                            assistant_msg,
                                            placeholder,
                                            self.client,
                                            dispatcher,
                                        )
                                        tools_data.info[src_task].tool_reply_msg = (
                                            placeholder
                                        )
                                    else:
                                        placeholder["content"] = pretty
                                    # Append images from child notifications (if any)
                                    try:
                                        imgs = (
                                            payload.get("images")
                                            if isinstance(payload, dict)
                                            else None
                                        )
                                        _append_images(
                                            imgs,
                                            _default_img_src("notification"),
                                        )
                                    except Exception:
                                        pass
                                    # Forward programmatic event upstream
                                    try:
                                        outer = (
                                            self.outer_handle_container[0]
                                            if self.outer_handle_container
                                            else None
                                        )
                                        if outer is not None and hasattr(
                                            outer,
                                            "_notification_q",
                                        ):
                                            event_payload = (
                                                payload
                                                if isinstance(payload, dict)
                                                else {"message": str(payload)}
                                            )
                                            await outer._notification_q.put(
                                                {
                                                    "type": "notification",
                                                    "call_id": tools_data.info[
                                                        src_task
                                                    ].call_id,
                                                    "tool_name": tool_name,
                                                    **event_payload,
                                                },
                                            )
                                    except Exception:
                                        pass
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
                                    name="TurnClarification",
                                )
                                clar_waiters[cw] = _t
                            if _inf.notification_queue is not None:
                                pw = asyncio.create_task(
                                    _inf.notification_queue.get(),
                                    name="TurnNotification",
                                )
                                notif_waiters[pw] = _t
                        # Recreate interject/cancel waiters for next loop
                        interject_waiter = (
                            asyncio.create_task(
                                self.interject_queue.get(),
                                name="TurnInterject",
                            )
                            if not enable_interject_adapter
                            else None
                        )
                        cancel_waiter2 = asyncio.create_task(
                            self.cancel_event.wait(),
                            name="TurnCancel",
                        )

                    # Optional: if no pending and no cancellations/interjections and no clarification requested,
                    # we can allow one more LLM turn evented before handoff (broadening subtly)
                    if not canceled and not interjected and not llm_turn_required:
                        # Finalization check (legacy parity): if no tasks pending and the latest assistant
                        # turn contains no tool_calls, return its content immediately.
                        try:
                            if not tools_data.pending:
                                latest_asst = None
                                for _m in reversed(self.client.messages or []):
                                    if (
                                        isinstance(_m, dict)
                                        and _m.get("role") == "assistant"
                                    ):
                                        latest_asst = _m
                                        break
                                if isinstance(latest_asst, dict):
                                    if not (latest_asst.get("tool_calls") or []):
                                        return latest_asst.get("content", "")
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

                        # Legacy semantics (loop.py parity) for subsequent turns as well
                        full_tools_map2: Dict[str, Callable] = dict(self.tools or {})
                        if tool_choice_mode2 == "auto":
                            effective_visible_map2 = {
                                name: fn
                                for name, fn in full_tools_map2.items()
                                if name not in (filtered_map2 or {})
                            }
                        else:
                            effective_visible_map2 = (
                                dict(filtered_map2)
                                if filtered_map2
                                else dict(full_tools_map2)
                            )

                        try:
                            LOGGER.info(
                                "policy_eval_turn: step=%s mode=%s returned=%s visible=%s",
                                step_idx,
                                tool_choice_mode2,
                                list((filtered_map2 or {}).keys()),
                                list(effective_visible_map2.keys()),
                            )
                        except Exception:
                            pass

                        # Pre-exposure filtering by hidden quotas and concurrency
                        try:
                            normalized2 = normalise_tools(
                                {n: f for n, f in effective_visible_map2.items()},
                            )
                        except Exception:
                            normalized2 = {
                                n: ToolSpec(fn=f)
                                for n, f in effective_visible_map2.items()
                            }
                        filtered_schemas2: list[dict] = []
                        for name, spec in normalized2.items():
                            try:
                                limit = getattr(spec, "max_total_calls", None)
                                used = 0
                                try:
                                    used = sum(
                                        1
                                        for _t, _inf in tools_data2.info.items()
                                        if _inf.name == name
                                    )
                                except Exception:
                                    used = 0
                                if limit is not None and used >= int(limit):
                                    continue
                                max_cc = getattr(spec, "max_concurrent", None)
                                active = 0
                                try:
                                    active = tools_data2.active_count(name)
                                except Exception:
                                    active = 0
                                if max_cc is not None and active >= int(max_cc):
                                    continue
                                filtered_schemas2.append(
                                    _method_to_schema(spec.fn, tool_name=name),
                                )
                            except Exception:
                                continue
                        schemas2 = filtered_schemas2
                        # Live image helpers – expose overview and aligner when images present
                        if self.images:
                            try:
                                prior_lines2 = []
                                try:
                                    for rec in _LIVE_IMAGES_LOG.get() or []:
                                        try:
                                            src, iid_s, span_key = rec.split(":", 2)
                                            prior_lines2.append(
                                                f"- source={src}, id={int(iid_s)}, span={span_key}",
                                            )
                                        except Exception:
                                            continue
                                except Exception:
                                    prior_lines2 = []
                                overview_doc2 = (
                                    "Live images aligned to the current user_message (visible in this description; calling is optional).\n"
                                    + (
                                        "\n".join(prior_lines2)
                                        if prior_lines2
                                        else "(none)"
                                    )
                                )

                                async def live_images_overview() -> Dict[str, str]:  # type: ignore[name-defined]
                                    return {"status": "ok"}

                                live_images_overview.__doc__ = overview_doc2  # type: ignore[attr-defined]

                                # Image helpers
                                id_to_handle2: dict[int, Any] = {}
                                try:
                                    for span_key, ih in list(self.images.items()):
                                        try:
                                            img_id = int(getattr(ih, "image_id", -1))
                                        except Exception:
                                            img_id = -1
                                        id_to_handle2[img_id] = ih
                                except Exception:
                                    id_to_handle2 = {}

                                async def ask_image(*, image_id: int, question: str, images: dict | None = None) -> Any:  # type: ignore[valid-type]
                                    ih = id_to_handle2.get(int(image_id))
                                    if ih is None:
                                        return {
                                            "error": f"image_id {image_id} not found",
                                        }
                                    try:
                                        _append_images(images, _default_img_src("ask"))
                                    except Exception:
                                        pass
                                    try:
                                        return await ih.ask(question)
                                    except Exception as _exc:  # noqa: BLE001
                                        return {"error": str(_exc)}

                                async def attach_image_raw(*, image_id: int, note: str | None = None) -> Dict[str, Any]:  # type: ignore[valid-type,name-defined]
                                    iid = int(image_id)
                                    ih = id_to_handle2.get(iid)
                                    if ih is None:
                                        return {"error": f"image_id {iid} not found"}
                                    try:
                                        data_str = getattr(
                                            getattr(ih, "_image", None),
                                            "data",
                                            None,
                                        )
                                        is_gcs_url = isinstance(data_str, str) and (
                                            data_str.startswith("gs://")
                                            or data_str.startswith(
                                                "https://storage.googleapis.com/",
                                            )
                                        )
                                        if is_gcs_url:
                                            try:
                                                from urllib.parse import (
                                                    urlparse as _urlparse,
                                                )

                                                parsed_url = _urlparse(data_str)
                                                bucket_name = ""
                                                object_path = ""
                                                if parsed_url.scheme == "gs":
                                                    bucket_name = parsed_url.netloc
                                                    object_path = (
                                                        parsed_url.path.lstrip("/")
                                                    )
                                                elif (
                                                    parsed_url.hostname
                                                    == "storage.googleapis.com"
                                                ):
                                                    parts = parsed_url.path.lstrip(
                                                        "/",
                                                    ).split("/", 1)
                                                    if len(parts) == 2:
                                                        bucket_name, object_path = parts
                                                storage_client = getattr(
                                                    getattr(ih, "_manager", None),
                                                    "storage_client",
                                                    None,
                                                )
                                                bucket = storage_client.bucket(
                                                    bucket_name,
                                                )
                                                blob = bucket.blob(object_path)
                                                signed_url = blob.generate_signed_url(
                                                    version="v4",
                                                    expiration=timedelta(hours=1),
                                                    method="GET",
                                                )
                                                content_block = {
                                                    "type": "image_url",
                                                    "image_url": {"url": signed_url},
                                                }
                                            except Exception:
                                                raw = ih.raw()
                                                import base64 as _b64

                                                head = (
                                                    bytes(raw[:10])
                                                    if isinstance(
                                                        raw,
                                                        (bytes, bytearray),
                                                    )
                                                    else b""
                                                )
                                                if head.startswith(b"\xff\xd8"):
                                                    mime = "image/jpeg"
                                                elif head.startswith(
                                                    b"\x89PNG\r\n\x1a\n",
                                                ):
                                                    mime = "image/png"
                                                else:
                                                    mime = "image/png"
                                                b64 = _b64.b64encode(raw).decode(
                                                    "ascii",
                                                )
                                                content_block = {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:{mime};base64,{b64}",
                                                    },
                                                }
                                        elif isinstance(data_str, str) and (
                                            data_str.startswith("http://")
                                            or data_str.startswith("https://")
                                            or data_str.startswith("data:image/")
                                        ):
                                            content_block = {
                                                "type": "image_url",
                                                "image_url": {"url": data_str},
                                            }
                                        else:
                                            raw = ih.raw()
                                            import base64 as _b64

                                            head = (
                                                bytes(raw[:10])
                                                if isinstance(raw, (bytes, bytearray))
                                                else b""
                                            )
                                            if head.startswith(b"\xff\xd8"):
                                                mime = "image/jpeg"
                                            elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                                                mime = "image/png"
                                            else:
                                                mime = "image/png"
                                            b64 = _b64.b64encode(raw).decode("ascii")
                                            content_block = {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{mime};base64,{b64}",
                                                },
                                            }

                                        cfgB = _LoopConfig(
                                            self.loop_id,
                                            self.lineage,
                                            self.lineage or [],
                                        )
                                        timerB = _Timer(
                                            timeout=self.timeout,
                                            max_steps=self.max_steps,
                                            raise_on_limit=self.raise_on_limit,
                                            client=self.client,
                                        )
                                        dispatcherB = _Dispatcher(
                                            self.client,
                                            cfgB,
                                            timerB,
                                        )
                                        await dispatcherB.append_msgs(
                                            [
                                                {
                                                    "role": "user",
                                                    "content": (
                                                        [content_block]
                                                        if note is None
                                                        else [
                                                            {
                                                                "type": "text",
                                                                "text": note,
                                                            },
                                                            content_block,
                                                        ]
                                                    ),
                                                },
                                            ],
                                        )
                                        try:
                                            _append_images(
                                                None,
                                                _default_img_src("attach"),
                                            )
                                        except Exception:
                                            pass
                                        return {"status": "attached", "image_id": iid}
                                    except Exception as _exc:  # noqa: BLE001
                                        return {"error": str(_exc)}

                                async def align_images_for(*, args: dict, hints: list[dict]) -> dict:  # type: ignore[valid-type]
                                    out: dict[str, int] = {}
                                    try:
                                        arg_texts = {
                                            str(k): str(v)
                                            for k, v in dict(args or {}).items()
                                        }
                                    except Exception:
                                        arg_texts = {}

                                    def _extract_id(obj: dict) -> int | None:
                                        for k in ("image_id", "imageId", "id"):
                                            if k in obj:
                                                try:
                                                    return int(obj[k])
                                                except Exception:
                                                    return None
                                        return None

                                    def _extract_arg(obj: dict) -> str | None:
                                        for k in (
                                            "arg",
                                            "argument",
                                            "arg_name",
                                            "name",
                                        ):
                                            if k in obj:
                                                return str(obj[k])
                                        return None

                                    def _extract_substring(obj: dict) -> str | None:
                                        for k in ("substring", "text", "span_text"):
                                            if k in obj:
                                                return str(obj[k])
                                        return None

                                    for item in list(hints or []):
                                        if not isinstance(item, dict):
                                            continue
                                        iid = _extract_id(item)
                                        arg_name = _extract_arg(item)
                                        sub = _extract_substring(item)
                                        if iid is None or not arg_name or sub is None:
                                            continue
                                        base = arg_texts.get(arg_name)
                                        if not isinstance(base, str):
                                            continue
                                        try:
                                            start = base.find(sub)
                                            if start < 0:
                                                continue
                                            end = start + len(sub)
                                            key = f"{arg_name}[{start}:{end}]"
                                            out[key] = iid
                                        except Exception:
                                            continue
                                    return {"images": out}

                                try:
                                    schemas2.append(
                                        _method_to_schema(
                                            live_images_overview,
                                            tool_name="live_images_overview",
                                        ),
                                    )
                                except Exception:
                                    pass
                                try:
                                    schemas2.append(
                                        _method_to_schema(
                                            align_images_for,
                                            tool_name="align_images_for",
                                        ),
                                    )
                                except Exception:
                                    pass
                                try:
                                    schemas2.append(
                                        _method_to_schema(
                                            ask_image,
                                            tool_name="ask_image",
                                        ),
                                    )
                                except Exception:
                                    pass
                                try:
                                    schemas2.append(
                                        _method_to_schema(
                                            attach_image_raw,
                                            tool_name="attach_image_raw",
                                        ),
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        # Dynamic helper tools for current pending set
                        try:
                            dyn_factory2 = DynamicToolFactory(tools_data2)
                            dyn_factory2.generate()
                            # Hide `wait` if any task awaits clarification
                            try:
                                if any(
                                    getattr(_inf, "waiting_for_clarification", False)
                                    for _inf in tools_data2.info.values()
                                ):
                                    dyn_factory2.dynamic_tools.pop("wait", None)
                            except Exception:
                                pass
                            for _nm, _fn in dyn_factory2.dynamic_tools.items():
                                try:
                                    schemas2.append(
                                        _method_to_schema(_fn, tool_name=_nm),
                                    )
                                except Exception:
                                    continue
                        except Exception:
                            pass
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
                        # Global pause gating – do not allow LLM turns while paused
                        try:
                            if not self.pause_event.is_set():
                                cancel_gate2 = asyncio.create_task(
                                    self.cancel_event.wait(),
                                    name="EventedPauseCancelGate2",
                                )
                                resume_gate2 = asyncio.create_task(
                                    self.pause_event.wait(),
                                    name="EventedPauseResumeGate2",
                                )
                                done_gate2, _ = await asyncio.wait(
                                    {cancel_gate2, resume_gate2},
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                                for g in (cancel_gate2, resume_gate2):
                                    if g not in done_gate2 and not g.done():
                                        g.cancel()
                                await asyncio.gather(
                                    cancel_gate2,
                                    resume_gate2,
                                    return_exceptions=True,
                                )
                                if self.cancel_event.is_set():
                                    if self.stop_event is not None:
                                        try:
                                            self.stop_event.set()
                                        except Exception:
                                            pass
                                    raise asyncio.CancelledError
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
                                        # Handle helper tools via same semantics as first turn
                                        try:

                                            async def _handle_helper_2(
                                                nm: str,
                                                c: dict,
                                            ) -> str:
                                                return await _handle_helper_call(nm, c)  # type: ignore[name-defined]

                                            res2 = await _handle_helper_2(name2, call)
                                            if res2 in ("drop", "handled"):
                                                continue
                                        except Exception:
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
                            if not done2:
                                return await _early_exit(
                                    f"timeout ({self.timeout}s) exceeded",
                                    tools_data2,
                                    dispatcher2,
                                )
                            # Handle branches akin to first-turn
                            if cancel2 in done2:
                                try:
                                    try:
                                        _ = await _propagate_stop_once(
                                            tools_data2.info,
                                            False,
                                            "outer-loop cancelled",
                                        )
                                    except Exception:
                                        pass
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

        # Evented-only: return final assistant content if present; else empty string.
        try:
            latest_asst = None
            for _m in reversed(self.client.messages or []):
                if isinstance(_m, dict) and _m.get("role") == "assistant":
                    latest_asst = _m
                    break
            if latest_asst is not None:
                return latest_asst.get("content", "") or ""
        except Exception:
            pass
        # No assistant content, return empty string (explicit evented-only behaviour)
        return ""

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
        except Exception:
            rec = {"event": name}

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


## Runners and adapters are imported from orchestrator_runners and orchestrator_adapters


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
    # Ensure nested loops inherit lineage via contextvar (legacy parity)
    cfg = _LoopConfig(loop_id, lineage, _TOOL_LOOP_LINEAGE.get([]))
    _token = _TOOL_LOOP_LINEAGE.set(cfg.lineage)
    try:
        return await orch.run()
    finally:
        with suppress(Exception):
            _TOOL_LOOP_LINEAGE.reset(_token)
