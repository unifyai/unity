"""
Contains classes and helpers for manipulating and managing messages in an async tool loop.
"""

import copy
import json
import unify
from typing import Callable, Optional, Any
from .utils import maybe_await
from ...constants import LOGGER
from contextlib import suppress
from .tools_utils import create_tool_call_message


# TODO: Some of these helpers should not be placed here, but in utils.py or their own files


# Helper: scan transcript for assistant messages that have tool_calls with
# missing tool replies (before the next assistant message).
def find_unreplied_assistant_entries(client: unify.AsyncUnify) -> list[dict]:
    findings: list[dict] = []
    try:
        for i, m in enumerate(client.messages):
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            if not tcs:
                continue
            ids = [tc.get("id") for tc in tcs if isinstance(tc, dict)]
            if not ids:
                continue
            responded: set[str] = set()
            j = i + 1
            while (
                j < len(client.messages)
                and client.messages[j].get("role") != "assistant"
            ):
                mm = client.messages[j]
                if mm.get("role") == "tool":
                    tcid = mm.get("tool_call_id")
                    if tcid in ids:
                        responded.add(tcid)
                j += 1
            missing = [c for c in ids if c not in responded]
            if missing:
                findings.append(
                    {
                        "assistant_index": i,
                        "assistant_msg": m,
                        "missing": missing,
                    },
                )
    except Exception:
        pass
    return findings


# Helper: call `client.generate` with optional preprocessing
async def generate_with_preprocess(
    client: unify.AsyncUnify,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]],
    **gen_kwargs,
):
    if preprocess_msgs is None:
        return await maybe_await(client.generate(**gen_kwargs))

    original_msgs = client.messages  # reference to canonical log
    msgs_copy = copy.deepcopy(original_msgs)

    try:
        patched = preprocess_msgs(msgs_copy) or msgs_copy
    except Exception as exc:  # resilience – don't fail the loop
        LOGGER.error(
            f"preprocess_msgs raised {exc!r}; using original messages.",
        )
        patched = msgs_copy

    start_len = len(patched)

    # ------------------------------------------------------------------
    # Some ``AsyncUnify`` implementations (the real one) keep their chat
    # transcript in a **private** attribute ``_messages`` which is what
    # ``.generate`` reads from, while lightweight test doubles (e.g.
    # ``SpyAsyncUnify`` in the test-suite) expose only a public
    # ``messages`` list.  To remain compatible with *both* variants we
    # detect the attribute that is actually consumed by the downstream
    # ``generate`` call and patch **that** for the duration of the call.
    # ------------------------------------------------------------------
    target_attr = "_messages" if hasattr(client, "_messages") else "messages"

    original_container = getattr(client, target_attr)
    setattr(client, target_attr, patched)
    try:
        result = await maybe_await(client.generate(**gen_kwargs))

        # Append any new messages the LLM produced back to canonical log
        current_msgs = getattr(client, target_attr)
        if len(current_msgs) > start_len:
            original_msgs.extend(copy.deepcopy(current_msgs[start_len:]))

        return result
    finally:
        # Always restore the canonical chat log so the outer loop remains
        # consistent irrespective of whether we patched `_messages` or
        # `messages`.
        setattr(client, target_attr, original_container)


def chat_context_repr(
    parent_ctx: Optional[list[dict]],
    current_msgs: list[dict],
) -> list[dict]:
    """
    Combine **existing** ``parent_ctx`` with the *current* chat history
    (``current_msgs``) into a depth-aware nested structure:

        root_msg0
        root_msg1
        root_msg2
          └── children:
              ├── child_msg0
              └── child_msg1

    Strategy – keep the original list untouched and attach the new
    messages as ``children`` of the *last* element.
    """
    ctx_block = [
        {"role": m.get("role"), "content": m.get("content")} for m in current_msgs
    ]
    if not parent_ctx:
        return ctx_block

    combined = copy.deepcopy(parent_ctx)
    combined[-1].setdefault("children", []).extend(ctx_block)
    return combined


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


async def forward_handle_call(
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


# Helper: detect helper-tool names (wait/stop_/pause_/resume_/clarify_/interject_)
def _is_helper_tool(name: str) -> bool:
    return (
        name == "wait"
        or name.startswith("stop_")
        or name.startswith("pause_")
        or name.startswith("resume_")
        or name.startswith("clarify_")
        or name.startswith("interject_")
    )


# Helper: build human-readable acknowledgement content for helper tools
def build_helper_ack_content(name: str, args_json: Any) -> str:
    ack_content = "Acknowledged."
    try:
        payload = (
            json.loads(args_json or "{}")
            if isinstance(args_json, str)
            else (args_json or {})
        )
    except Exception:
        payload = {}

    if name == "wait":
        ack_content = "Waiting acknowledged. Keeping current tool calls in flight."
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
async def insert_tool_message_after_assistant(
    assistant_meta: dict,
    parent_msg: dict,
    tool_msg,
    client,
    msg_dispatcher,
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
            h = _inf.handle
            if h is not None and hasattr(h, "stop"):
                try:
                    await forward_handle_call(
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


async def propagate_stop_once(
    task_info,
    stop_forward_once,
    reason: Optional[str],
) -> bool:
    if stop_forward_once:
        return stop_forward_once
    await _propagate_stop_to_nested_handles(task_info, reason)
    return True


# Helper: insert a tool-acknowledgement message for helper tools
async def acknowledge_helper_call(
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
        content=build_helper_ack_content(name, args_json),
    )
    await insert_tool_message_after_assistant(
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
async def ensure_placeholders_for_pending(
    assistant_msg: Optional[dict] = None,
    *,
    content: Optional[str] = None,
    tools_data,
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
        if _inf.tool_reply_msg or _inf.clarify_placeholder:
            continue

        placeholder = create_tool_call_message(
            name=_inf.name,
            call_id=_inf.call_id,
            content=placeholder_content,
        )
        await insert_tool_message_after_assistant(
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
async def schedule_missing_for_message(
    asst_msg: dict,
    only_ids: set[str],
    *,
    tools_data,
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
                # Special-case: `wait` should not clutter the transcript.
                if name == "wait":
                    # Prune the wait tool call from the assistant message; if it was the
                    # only tool call and content is empty, drop the assistant message.
                    try:
                        tool_calls = asst_msg.get("tool_calls") or []
                        remaining = [c for c in tool_calls if c.get("id") != cid]
                        content_present = bool((asst_msg.get("content") or "").strip())
                        if not remaining:
                            if not content_present:
                                try:
                                    idx_in_log = client.messages.index(asst_msg)
                                    client.messages.pop(idx_in_log)
                                except Exception:
                                    pass
                            else:
                                asst_msg.pop("tool_calls", None)
                        else:
                            asst_msg["tool_calls"] = remaining
                    except Exception:
                        pass
                    # Mark as handled without emitting any tool reply
                    scheduled.append(cid)
                    continue

                # Other helpers: acknowledge but do not execute during backfill
                try:
                    await acknowledge_helper_call(
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
        await ensure_placeholders_for_pending(
            assistant_msg=asst_msg,
            tools_data=tools_data,
            assistant_meta=assistant_meta,
            client=client,
            msg_dispatcher=msg_dispatcher,
        )
    return scheduled
