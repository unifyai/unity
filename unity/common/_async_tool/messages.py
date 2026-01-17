"""
Contains classes and helpers for manipulating and managing messages in an async tool loop.
"""

import copy
import json
import unillm
from typing import Callable, Optional, Any
from .utils import maybe_await
from ...constants import LOGGER
from contextlib import suppress, contextmanager
from .tools_utils import create_tool_call_message
from .images import append_images_with_source


@contextmanager
def _preserve_canonical_messages(client, canonical_msgs):
    """Context manager to ensure client.messages returns canonical_msgs during the block.

    Properties defined at class level cannot be shadowed by instance attributes,
    so we temporarily patch the class-level property to check for a special
    `_canonical_messages` attribute first.
    """
    prop_class = None
    orig_prop = None
    try:
        client._canonical_messages = canonical_msgs
        for klass in type(client).__mro__:
            if "messages" in klass.__dict__:
                prop_class = klass
                orig_prop = klass.__dict__["messages"]
                break
        if prop_class is not None and orig_prop is not None:

            def _patched_getter(self, _orig=orig_prop):
                cm = getattr(self, "_canonical_messages", None)
                return cm if cm is not None else _orig.fget(self)

            prop_class.messages = property(_patched_getter)
    except Exception:
        pass
    try:
        yield
    finally:
        if prop_class is not None and orig_prop is not None:
            with suppress(Exception):
                prop_class.messages = orig_prop
        with suppress(Exception):
            del client._canonical_messages


# TODO: Some of these helpers should not be placed here, but in utils.py or their own files


# Helper: scan transcript for assistant messages that have tool_calls with
# missing tool replies (before the next assistant message).


def is_non_final_tool_reply(msg: dict) -> bool:
    """Return True when a tool message looks like a placeholder/progress, not a final result.

    Rules:
    - Clarification wrappers (name startswith "clarification_request_") are non-final.
    - Any tool message whose content parses to a dict containing the top-level key
      "_placeholder" is non-final (used for pending/progress/nested-start placeholders).
    """
    try:
        if msg.get("role") != "tool":
            return False
        name = str(msg.get("name") or "")
        if name.startswith("clarification_request_"):
            return True
        content = msg.get("content")
        if isinstance(content, str):
            try:
                import json as _json

                parsed = _json.loads(content)
                if isinstance(parsed, dict) and "_placeholder" in parsed:
                    return True
            except Exception:
                pass
    except Exception:
        return False
    return False


def transform_tool_calls_to_context(
    msgs: list[dict],
    *,
    marker_key: str = "_transformed_context",
    context_header: str = "[Prior tool execution context]",
    context_footer: str = "[Continue with the original request]",
    predicate: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Transform assistant tool_calls into a system context message.

    This unified function handles two scenarios:
    1. Seeded transcripts for Claude reasoning models that require
       provider-specific metadata (thinking blocks) which we lack when
       replaying manually constructed tool calls.
    2. Claude extended thinking re-enablement after forced-tool turns where
       thinking was disabled (incompatible with tool_choice="required").

    Parameters
    ----------
    msgs : list[dict]
        The list of messages to transform.
    marker_key : str
        Key to set on the context system message for identification.
    context_header : str
        Header text for the context message.
    context_footer : str
        Footer text for the context message.
    predicate : callable | None
        Optional function(msg) -> bool to determine which assistant messages
        need transformation. If None, transforms ALL assistant messages with
        tool_calls.

    Returns
    -------
    list[dict]
        Transformed message list with matching tool_calls converted to context.
    """
    if not msgs:
        return msgs

    # Default predicate: transform all assistant messages with tool_calls
    if predicate is None:

        def predicate(m: dict) -> bool:
            return (
                isinstance(m, dict)
                and m.get("role") == "assistant"
                and bool(m.get("tool_calls"))
            )

    # Check if any messages need transformation
    if not any(predicate(m) for m in msgs):
        return msgs

    # Build a mapping of tool_call_id -> tool result content
    tool_results: dict[str, dict] = {}
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if isinstance(tcid, str) and tcid:
                tool_results[tcid] = {
                    "name": m.get("name", "unknown"),
                    "content": m.get("content", ""),
                }

    # Collect IDs of tool_calls from messages that need transformation
    transformed_call_ids: set[str] = set()
    tool_call_descriptions: list[str] = []

    for m in msgs:
        if not predicate(m):
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id", "")
            transformed_call_ids.add(tc_id)
            func = tc.get("function") or {}
            name = func.get("name", "unknown")
            args = func.get("arguments", "{}")
            result_info = tool_results.get(tc_id)
            if result_info:
                result_content = result_info.get("content", "(no result)")
                tool_call_descriptions.append(
                    f"• Called `{name}({args})` → {result_content}",
                )
            else:
                tool_call_descriptions.append(
                    f"• Called `{name}({args})` → (pending/no result)",
                )

    # Build transformed message list
    transformed: list[dict] = []
    context_inserted = False

    for m in msgs:
        if not isinstance(m, dict):
            transformed.append(m)
            continue

        role = m.get("role")

        if role == "user":
            transformed.append(m)

        elif role == "assistant":
            if predicate(m):
                # Insert context AT THIS POSITION (where the transformed turn was)
                # This maintains chronological order so Claude sees preserved turns
                # before the synthetic summary of non-thinking turns.
                if not context_inserted and tool_call_descriptions:
                    context_msg = {
                        "role": "system",
                        "content": (
                            context_header
                            + "\n"
                            + "\n".join(tool_call_descriptions)
                            + "\n"
                            + context_footer
                        ),
                        marker_key: True,
                    }
                    transformed.append(context_msg)
                    context_inserted = True
                # Skip the assistant message itself - replaced by context
            else:
                transformed.append(m)

        elif role == "tool":
            # Skip tool messages for transformed calls
            tcid = m.get("tool_call_id")
            if tcid in transformed_call_ids:
                continue
            else:
                transformed.append(m)

        else:
            transformed.append(m)

    return transformed


def find_unreplied_assistant_entries(client: unillm.AsyncUnify) -> list[dict]:
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
                    # Count as responded only when the tool reply looks **final**.
                    if tcid in ids and not is_non_final_tool_reply(mm):
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
    client: unillm.AsyncUnify,
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

    # Capture the system message for potential patching
    sys_txt = getattr(client, "system_message", "") or ""
    sys_patched = sys_txt

    # ──────────────────────────────────────────────────────────────────────
    # Fix: Ensure the original system message is always at the front of
    # patched messages. The Unify client's generate() checks if ANY system
    # message exists in messages[], and if so, doesn't prepend system_message.
    # This means if preprocessing adds a system message (like Claude's
    # thinking-block context), the original system prompt gets dropped.
    #
    # We explicitly prepend the original system_message to patched messages
    # if it's not already there, ensuring it's always sent to the LLM.
    # ──────────────────────────────────────────────────────────────────────
    if sys_txt:
        # Check if the first message is already the original system message
        first_is_original_system = (
            patched
            and patched[0].get("role") == "system"
            and patched[0].get("content") == sys_txt
        )
        if not first_is_original_system:
            patched = [{"role": "system", "content": sys_txt}] + patched

    start_len = len(patched)

    # ------------------------------------------------------------------
    # Some ``AsyncUnify`` implementations (the real one) keep their chat
    # transcript in a **private** attribute ``_messages`` which is what
    # ``.generate`` reads from, while lightweight test doubles (e.g.
    # ``SpyAsyncUnify`` in the test-suite) expose only a public
    # ``messages`` list.  To remain compatible with *both* variants we
    # detect the attribute that is actually consumed by the downstream
    # ``generate`` call and patch **that** for the duration of the call.
    #
    # When we swap ``_messages``, the public ``messages`` property would
    # also return the patched list, causing a race condition for external
    # code polling ``client.messages``. We use _preserve_canonical_messages
    # to ensure external observers see the canonical log during the swap.
    # ------------------------------------------------------------------
    target_attr = "_messages" if hasattr(client, "_messages") else "messages"
    original_system_message = getattr(client, "system_message", None)
    with suppress(Exception):
        if original_system_message is not None:
            setattr(client, "system_message", sys_patched)

    original_container = getattr(client, target_attr)

    # Use context manager to preserve canonical messages visibility when swapping
    preserve_ctx = (
        _preserve_canonical_messages(client, original_container)
        if target_attr == "_messages"
        else suppress()
    )

    with preserve_ctx:
        setattr(client, target_attr, patched)
        try:
            result = await maybe_await(client.generate(**gen_kwargs))

            # Append any new messages the LLM produced back to canonical log
            current_msgs = getattr(client, target_attr)
            if len(current_msgs) > start_len:
                original_msgs.extend(copy.deepcopy(current_msgs[start_len:]))

            return result
        finally:
            setattr(client, target_attr, original_container)
            with suppress(Exception):
                if original_system_message is not None:
                    setattr(client, "system_message", original_system_message)


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
    call_args: list | tuple | None = None,
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
        args = list(call_args or [])
        normalised = _normalise_kwargs_for_bound_method(bound, kwargs or {})
        return await maybe_await(bound(*args, **normalised))
    except TypeError:
        # Fallbacks: try positional-only, then kwargs-only, then legacy single-key
        # positional extraction via fallback_positional_keys for maximum tolerance.
        try:
            args2 = list(call_args or [])
            return await maybe_await(bound(*args2))  # type: ignore[misc]
        except Exception:
            pass
        try:
            return await maybe_await(bound(**(normalised if isinstance(normalised, dict) else {})))  # type: ignore[misc]
        except Exception:
            pass
        for k in fallback_positional_keys:
            if kwargs and k in kwargs:
                try:
                    # Preserve additional kwargs (e.g., images) alongside the positional message
                    rest_kwargs = (
                        dict(normalised) if isinstance(normalised, dict) else {}
                    )
                except Exception:
                    rest_kwargs = {}
                try:
                    # Avoid passing the alias key twice if it accidentally matched a parameter
                    rest_kwargs.pop(k, None)
                except Exception:
                    pass
                try:
                    return await maybe_await(bound(kwargs.get(k), **rest_kwargs))  # type: ignore[misc]
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
        # NOTE: ask_* proxies are treated as base tools so they can be scheduled/executed
        # during backfill when symbolically injected by the outer ask loop.
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


# Helper: prune a `wait` tool call from an assistant message. If it was the
# only tool call and there is no content, drop the assistant message from the
# client's transcript where possible.
def prune_wait_tool_call(
    asst_msg: dict,
    call_id: str,
    *,
    client: unillm.AsyncUnify | None = None,
) -> None:
    try:
        tool_calls = asst_msg.get("tool_calls") or []
        remaining = [c for c in tool_calls if c.get("id") != call_id]
        content_present = bool((asst_msg.get("content") or "").strip())
        if not remaining:
            if not content_present:
                if client is not None:
                    try:
                        if client.messages and client.messages[-1] is asst_msg:
                            client.messages.pop()
                        else:
                            idx_in_log = client.messages.index(asst_msg)
                            client.messages.pop(idx_in_log)
                    except Exception:
                        pass
                else:
                    asst_msg.pop("tool_calls", None)
            else:
                asst_msg.pop("tool_calls", None)
        else:
            asst_msg["tool_calls"] = remaining
    except Exception:
        pass


# ── small helper: keep assistant→tool chronology DRY ────────────────────
async def insert_tool_message_after_assistant(
    assistant_meta: dict,
    parent_msg: dict,
    tool_msg,
    client,
    msg_dispatcher,
    *,
    skip_event_bus: bool = False,
) -> None:
    """
    Append *tool_msg* and move it directly after *parent_msg*, while
    updating the per-assistant `results_count` bookkeeping.

    If *skip_event_bus* is True, the message is appended to the client
    transcript but NOT published to the EventBus. This is used for
    placeholder messages that will be updated in-place later.
    """
    meta = assistant_meta.setdefault(
        id(parent_msg),
        {"results_count": 0},
    )
    await msg_dispatcher.append_msgs([tool_msg], skip_event_bus=skip_event_bus)
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
    tools_data,
    assistant_meta,
    client,
    msg_dispatcher,
) -> list[str]:
    created: list[str] = []
    # Sort by call_idx to ensure deterministic placeholder ordering matching
    # the original tool_calls array order. This makes the "at tail" check in
    # process_completed_task behave consistently regardless of set iteration.
    for task in sorted(
        list(tools_data.pending),
        key=lambda t: getattr(tools_data.info.get(t), "call_idx", 0),
    ):
        _inf = tools_data.info.get(task)
        if not _inf:
            continue
        if assistant_msg is not None and _inf.assistant_msg is not assistant_msg:
            continue
        # Reuse any existing tool reply message in the transcript for this call_id
        try:
            if _inf.tool_reply_msg is None:
                existing = None
                msgs = client.messages or []
                for m in msgs:
                    try:
                        if m.get("role") == "tool" and str(
                            m.get("tool_call_id"),
                        ) == str(_inf.call_id):
                            existing = m
                            break
                    except Exception:
                        continue
                if existing is not None:
                    _inf.tool_reply_msg = existing
        except Exception:
            pass
        if _inf.tool_reply_msg or _inf.clarify_placeholder:
            continue

        placeholder = create_tool_call_message(
            name=_inf.name,
            call_id=_inf.call_id,
            content=json.dumps({"_placeholder": "pending"}, indent=4),
        )
        await insert_tool_message_after_assistant(
            assistant_meta,
            _inf.assistant_msg,
            placeholder,
            client,
            msg_dispatcher,
            skip_event_bus=True,  # Don't publish placeholders; publish when final
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
    initial_paused: bool = False,
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
                    try:
                        prune_wait_tool_call(asst_msg, cid, client=client)
                    except Exception:
                        pass
                    scheduled.append(cid)
                    continue

                # If helper arguments include images, append them to the live images registry immediately
                with suppress(Exception):
                    payload = (
                        json.loads(args_json or "{}")
                        if isinstance(args_json, str)
                        else (args_json or {})
                    )
                    imgs = payload.get("images") if isinstance(payload, dict) else None
                    if imgs is None and isinstance(payload, dict):
                        imgs = payload.get("images")
                    append_images_with_source(imgs)

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
                initial_paused=initial_paused,
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
