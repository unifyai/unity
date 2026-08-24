"""
Contains classes and helpers for manipulating and managing messages in an async tool loop.
"""

import copy
import hashlib
import json
import unillm
from typing import Callable, Optional, Any
from .utils import maybe_await
from ...logger import LOGGER
from ...common.hierarchical_logger import DEFAULT_ICON
from contextlib import suppress, contextmanager
from .tools_utils import create_tool_call_message
from ..context_dump import make_messages_safe_for_context_dump

# ── sent-watermark invariant ────────────────────────────────────────────
#
# A message is immutable once it has been included in any dispatched LLM
# request; everything from the watermark index onward is still free to
# mutate. Provider prefix caching matches serialized requests byte-for-byte
# from position 0, so any edit below the watermark invalidates the cached
# prefix for every subsequent request.


def _message_index(client, msg: dict) -> Optional[int]:
    """Identity-index of *msg* within ``client.messages``, or ``None`` if absent.

    Identity (``is``), not equality — two structurally-identical dicts must
    not be confused, and message dicts are mutated in place over their
    lifetime so equality is not stable anyway.
    """
    msgs = getattr(client, "messages", None) or []
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i] is msg:
            return i
    return None


def is_mutable(client, msg: dict) -> bool:
    """True when *msg* has not yet been included in any dispatched request.

    Fails closed: a message absent from the transcript (e.g. a swapped-out
    canonical log during a concurrent dispatch) is treated as immutable, so
    callers route its content through the tail-append paths that reach the
    model instead of writing into a dict the transcript will never contain.
    """
    idx = _message_index(client, msg)
    if idx is None:
        return False
    watermark = getattr(client, "_sent_watermark", 0)
    return idx >= watermark


def loop_user_notice(content: Any, **extra: Any) -> dict:
    """Build a ``role="user"`` message the loop itself authors — status,
    threshold, quota, or context-continuation notices — never a genuine
    user turn.

    This is the only place that stamps ``_loop_authored``, the marker
    ``is_loop_authored_message`` checks. Every loop-authored user-role
    message must be built here rather than as an inline dict literal, so
    the marker can never be forgotten at a new call site the way it was
    at three of them before this existed. ``extra`` still accepts the
    older, purpose-specific markers (``_progress_msg``, ``_clarify_msg``,
    ``_lifecycle_msg``, ``_ctx_header``) for callers that also need those
    for their own coalescing/filtering logic — ``_loop_authored`` is
    stamped regardless, so the boundary check never depends on which of
    those a given caller remembered to pass.

    A genuine user interjection (``_interjection``) is built directly at
    its call site, never through here — that asymmetry is what makes it
    a real turn boundary.
    """
    return {"role": "user", "content": content, "_loop_authored": True, **extra}


def is_loop_authored_message(msg: dict) -> bool:
    """True for a ``role="user"`` message the loop itself appended, never
    a genuine new user turn.

    Every message built by ``loop_user_notice`` carries ``_loop_authored``,
    so this checks a single flag rather than an inline list of marker
    keys. Every consumer that needs to tell "the user said something"
    apart from "the loop said something" (e.g. a boundary check that must
    not treat loop-authored status as the start of a new request) should
    use this predicate.
    """
    return bool(msg.get("_loop_authored"))


def extract_substantive_text(content: Any) -> Optional[str]:
    """Normalize assistant content to the text a user would read, for
    deciding whether a turn carries a substantive answer.

    Handles both a plain string and a multimodal content-block list (only
    ``"text"`` blocks contribute); returns ``None`` when the result is
    empty or whitespace-only in either shape, so callers can use a plain
    ``is None`` check rather than relying on truthiness — which passes a
    whitespace-only string and misreports a non-empty block list as
    substantive even when every block's text is blank. When a block list
    does carry substantive text, the extracted text is returned rather
    than the raw list, since every consumer downstream treats the answer
    as plain text. Shared by the final-answer walk-back and the parent
    context snapshot filter so both apply one definition of "substantive".
    """
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        joined = "".join(texts)
        return joined if joined.strip() else None
    return None


def _hash_msgs_slice(msgs: list) -> str:
    try:
        blob = json.dumps(msgs, sort_keys=True, default=str)
    except Exception:
        blob = repr(msgs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Diagnostic switch, not a rollout flag: prod behavior of the loop itself is
# identical either way. Off by default so the below-watermark hashing (real
# per-dispatch CPU on long transcripts) never ships to prod; the test suite's
# conftest turns it on so CI still catches an unsanctioned mutation.
_INVARIANT_CHECKS_ENV = "UNIFY_TRANSCRIPT_INVARIANT_CHECKS"


def _invariant_checks_enabled() -> bool:
    import os

    return os.environ.get(_INVARIANT_CHECKS_ENV) == "1"


def _call_id_has_reply(client, call_id: Optional[str]) -> bool:
    """True if some tool-role message in the transcript already answers *call_id*."""
    if not call_id:
        return False
    for m in getattr(client, "messages", None) or []:
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id:
            return True
    return False


def _rebaseline_watermark_hash(client) -> None:
    """Recompute the stored watermark hash after a sanctioned escape-hatch splice.

    A ``bypass_watermark`` splice deliberately shifts content at indices the
    previous hash already covered — that's the escape hatch's whole point
    (legality beats cache). Without re-baselining here, the next dispatch's
    integrity check would read that sanctioned shift as an unsanctioned
    mutation and raise.
    """
    if not _invariant_checks_enabled():
        return
    watermark = getattr(client, "_sent_watermark", 0)
    with suppress(Exception):
        client._sent_watermark_hash = _hash_msgs_slice(client.messages[:watermark])


_REVIEW_COMPACTION_MARKER = "[compacted after skill review:"
_REVIEW_COMPACTION_MIN_CHARS = 800
_REVIEW_COMPACTION_HEAD_CHARS = 300


_REASONING_PAYLOAD_KEYS = ("provider_specific_fields", "reasoning_details", "reasoning")


def strip_reasoning_payloads(msg: dict) -> int:
    """Drop provider reasoning machinery from one message, in place.

    Encrypted reasoning blobs and reasoning summaries exist so a provider
    can continue an in-flight chain of thought; once the turn that
    produced them is over they are pure re-billed bulk — often the
    largest single component of a long-lived transcript. The visible
    ``content`` is never touched. Returns the serialized characters
    removed (approximate, for accounting).
    """
    saved = 0
    for key in _REASONING_PAYLOAD_KEYS:
        if key in msg and msg[key] is not None:
            try:
                saved += len(json.dumps(msg[key], default=str))
            except (TypeError, ValueError):
                saved += 0
            msg.pop(key, None)
    return saved


def compact_reviewed_messages(client, reviewed_message_count: int) -> int:
    """Shed the bulk of an already-reviewed transcript span, in place.

    Once a storage review has consolidated a stretch of the transcript into
    stored functions, guidance and claims, that stretch's raw machinery is
    dead weight: every later dispatch of a long-lived session re-pays it,
    and so does every later review. Within the first
    ``reviewed_message_count`` messages this pass:

    * replaces bulky *tool* result contents with a head slice plus an
      omission marker, and
    * strips provider reasoning payloads (encrypted blobs, reasoning
      summaries) from assistant messages — a completed turn's chain of
      thought is not needed to continue the session.

    Message identity, ordering and tool_call pairing are untouched, so
    nothing holding a reference to a message dict ever sees it disappear.
    User-facing words — requests, requirements, the assistant's visible
    replies — stay verbatim. Placeholders/progress replies, small
    contents, image-bearing parts, and already-compacted messages are
    left alone.

    Mutating below the sent watermark is sanctioned here the same way an
    escape-hatch splice is: the watermark hash is re-baselined afterwards,
    trading provider prefix cache for a permanently smaller transcript.

    Returns the number of characters removed.
    """
    saved = 0
    messages = list(getattr(client, "messages", None) or [])
    span = messages[: max(0, min(reviewed_message_count, len(messages)))]
    for msg in span:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            saved += strip_reasoning_payloads(msg)
            continue
        if msg.get("role") != "tool":
            continue
        if is_non_final_tool_reply(msg):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            if any(
                not (isinstance(part, dict) and part.get("type") == "text")
                for part in content
            ):
                continue
            text = "\n".join(str(part.get("text") or "") for part in content)
        else:
            continue
        if len(text) < _REVIEW_COMPACTION_MIN_CHARS:
            continue
        if _REVIEW_COMPACTION_MARKER in text or "[img:" in text:
            continue
        stub = (
            f"{text[:_REVIEW_COMPACTION_HEAD_CHARS]}\n… "
            f"{_REVIEW_COMPACTION_MARKER} "
            f"{len(text) - _REVIEW_COMPACTION_HEAD_CHARS} chars omitted]"
        )
        msg["content"] = stub
        saved += len(text) - len(stub)
    if saved:
        _rebaseline_watermark_hash(client)
    return saved


async def emit_completion_pair(
    result: str,
    call_id: str,
    msg_dispatcher: Any,
) -> dict:
    """
    Append a synthetic assistant→tool pair carrying *result* for *call_id*
    at the tail of the transcript, instead of splicing it into an
    already-dispatched (below-watermark) position. This is the sole
    below-watermark delivery path for both late tool results and, via
    ``insert_tool_message_after_assistant``'s gate, any other reply that
    would otherwise land below the mark.
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
    - Clarification wrappers (name startswith "clarification_request_") are
      non-final. Nothing creates this shape anymore — ToolsData.record_clarification
      delivers the question as a "[clarification <call_id>]" user-role tail
      message instead — but a transcript persisted before that change can
      still contain one, so this stays for backward compatibility.
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

    This function handles scenarios where assistant messages with tool_calls
    need to be transformed into context messages for provider compatibility
    (e.g., when replaying manually constructed tool calls that lack required
    provider-specific metadata).

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
                # Insert context AT THIS POSITION (where the transformed turn was).
                # This maintains chronological order so the model sees preserved
                # turns before the synthetic summary of transformed turns.
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
    # Sent watermark: everything below this index has been (or is about to
    # be) included in a dispatched request and must never be mutated again.
    # Set here — the one place both llm_task dispatch sites funnel through —
    # on the pre-copy length, since the deep copy taken below is what
    # actually gets serialized. Monotonic, and set unconditionally *before*
    # the request goes out (not after it returns) so a cancelled/interrupted
    # dispatch still advances it: the provider may have cached the prefix of
    # a stream that never finished.
    prev_watermark = getattr(client, "_sent_watermark", 0)
    _checks_on = _invariant_checks_enabled()
    if _checks_on:
        # Integrity check (diagnostic switch — see _invariant_checks_enabled):
        # the below-watermark slice must be byte-identical to what was hashed
        # at the last dispatch, UNLESS a sanctioned escape-hatch splice
        # re-baselined it in between (see _rebaseline_watermark_hash). An
        # unsanctioned mutation is the only thing left that can still trip
        # this.
        prev_hash = getattr(client, "_sent_watermark_hash", None)
        if prev_hash is not None:
            assert _hash_msgs_slice(client.messages[:prev_watermark]) == prev_hash, (
                "Append-only transcript invariant violated: a message below "
                "the sent watermark was mutated between dispatches."
            )
    pre_copy_len = len(client.messages)
    client._sent_watermark = max(prev_watermark, pre_copy_len)
    if _checks_on:
        client._sent_watermark_hash = _hash_msgs_slice(
            client.messages[: client._sent_watermark],
        )

    # Stamp the in-flight window on the client. ``handle.ask()`` snapshots
    # ``client.messages``, which dead-ends silently while a request is out; the
    # stamp lets the inspection transcript say "the loop is waiting on an LLM
    # response since T" instead. finally-cleared so cancellation (interjection
    # pre-emption) can never leave a stale stamp behind.
    import time as _time

    client._llm_inflight_since = _time.time()
    try:
        return await _generate_with_preprocess_inner(
            client,
            preprocess_msgs,
            **gen_kwargs,
        )
    finally:
        client._llm_inflight_since = None


async def _generate_with_preprocess_inner(
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
            f"{DEFAULT_ICON} preprocess_msgs raised {exc!r}; using original messages.",
        )
        patched = msgs_copy

    # Capture the system message for potential patching
    sys_txt = getattr(client, "system_message", "") or ""
    sys_patched = sys_txt

    # ──────────────────────────────────────────────────────────────────────
    # Fix: Ensure the original system message is always at the front of
    # patched messages. The Unify client's generate() checks if ANY system
    # message exists in messages[], and if so, doesn't prepend system_message.
    # This means if preprocessing adds a system message (e.g., for provider
    # compatibility), the original system prompt gets dropped.
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
    safe_parent_ctx = make_messages_safe_for_context_dump(parent_ctx)
    safe_current_msgs = make_messages_safe_for_context_dump(current_msgs)
    ctx_block = [
        {"role": m.get("role"), "content": m.get("content")} for m in safe_current_msgs
    ]
    if not safe_parent_ctx:
        return ctx_block

    combined = copy.deepcopy(safe_parent_ctx)
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

        # 5) Coerce values to match type annotations (best-effort).
        #    LLMs sometimes pass all args as strings even when the signature
        #    expects int, float, bool, or dict.  Annotations may be actual
        #    types OR strings (when `from __future__ import annotations` is
        #    in effect), so we check both forms.
        import json as _json

        for param_name, param in params.items():
            if param_name not in kw or param_name == "self":
                continue
            annotation = param.annotation
            if annotation is _inspect.Parameter.empty:
                continue
            val = kw[param_name]
            try:
                ann_str = annotation if isinstance(annotation, str) else ""
                origin = getattr(annotation, "__origin__", None)

                is_int = annotation is int or ann_str == "int"
                is_float = annotation is float or ann_str == "float"
                is_bool = annotation is bool or ann_str == "bool"
                is_dict = (
                    annotation is dict
                    or ann_str == "dict"
                    or ann_str.startswith("Dict[")
                    or (origin is not None and origin is dict)
                )

                if is_int and isinstance(val, str):
                    kw[param_name] = int(val)
                elif is_float and isinstance(val, str):
                    kw[param_name] = float(val)
                elif is_bool and isinstance(val, str):
                    kw[param_name] = val.lower() in ("true", "1", "yes")
                elif is_dict and isinstance(val, str):
                    kw[param_name] = _json.loads(val)
            except (ValueError, _json.JSONDecodeError):
                pass

        return kw
    except Exception:
        # Best-effort; return original
        return dict(incoming_kw or {})


def apply_llm_soft_required_defaults(bound_method, kwargs: dict) -> dict:
    """Backfill arguments advertised as required but optional at runtime.

    See :func:`unify.common.tool_spec.llm_soft_required`. For each parameter a
    tool declared as soft-required, if the model omitted it from *kwargs* the
    configured default is filled in. Parameters not declared soft-required are
    left untouched, so a genuine omission of a functional argument still raises
    the usual error the model can self-correct against.

    Mutates and returns *kwargs* for convenience.
    """
    from unify.common.tool_spec import LLM_SOFT_REQUIRED_DEFAULTS_ATTR

    defaults = getattr(bound_method, LLM_SOFT_REQUIRED_DEFAULTS_ATTR, None)
    if not defaults:
        return kwargs

    import inspect as _inspect

    params = _inspect.signature(bound_method).parameters
    for name, default in defaults.items():
        if name in params and name not in kwargs:
            kwargs[name] = default
    return kwargs


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
                    # Preserve additional kwargs alongside the positional message
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


# Helper: detect helper-tool names — the static steering/inspection surface
# (steer/wait/ask_about_completed_tool) that ack-during-backfill rather than
# actually re-dispatching/re-executing (the underlying async work is gone on
# restart regardless of which action was requested).
def _is_helper_tool(name: str) -> bool:
    return name in ("wait", "steer", "ask_about_completed_tool")


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
    elif name == "steer":
        action = str(payload.get("action") or "").strip().lower()
        steer_payload = payload.get("payload")
        if action == "stop":
            ack_content = "Stop request acknowledged. If the underlying call is still running, it will be stopped."
        elif action == "pause":
            ack_content = "Pause request acknowledged. If the underlying call is still running, it will be paused."
        elif action == "resume":
            ack_content = "Resume request acknowledged. If the underlying call was paused, it will be resumed."
        elif action == "clarify":
            ack_content = (
                f"Clarification answer received: {steer_payload!r}. Waiting for the original tool to proceed."
                if steer_payload is not None
                else "Clarification helper acknowledged. Waiting for the original tool to proceed."
            )
        elif action == "interject":
            ack_content = (
                f"Guidance forwarded to the running tool: {steer_payload!r}."
                if steer_payload
                else "Interjection acknowledged and forwarded to the running tool."
            )
        elif action == "ask":
            ack_content = "Ask request acknowledged and forwarded to the running tool."
        elif action == "call":
            method = payload.get("method")
            ack_content = (
                f"Call to {method!r} acknowledged and forwarded to the running tool."
                if method
                else "Custom method call acknowledged and forwarded to the running tool."
            )
        else:
            ack_content = "Steering request acknowledged."
    elif name == "ask_about_completed_tool":
        ack_content = "Follow-up question acknowledged and forwarded for retrospective inspection."
    else:
        # Default acknowledgement for custom write-only helpers
        ack_content = (
            f"Operation {name!r} acknowledged and forwarded to the running tool."
        )
    return ack_content


# Helper: prune a `wait` tool call from an assistant message. If it was the
# only tool call and there is no content, drop the assistant message from the
# client's transcript where possible.
#
# Below the sent watermark, *asst_msg* was already included in a dispatched
# request — popping the wait's tool_calls entry, or editing the array in
# place, would mutate already-cached bytes. Instead the stale wait is left
# untouched and acknowledged via an appended tool reply (spliced directly
# after asst_msg, bypassing the watermark gate): a one-time prefix break,
# accepted because the alternative — a wait tool_calls entry with no reply
# anywhere — is a permanently illegal transcript, not just an uncached one.
async def prune_wait_tool_call(
    asst_msg: dict,
    call_id: str,
    *,
    client: unillm.AsyncUnify | None = None,
    assistant_meta: Optional[dict] = None,
    msg_dispatcher: Any = None,
) -> None:
    if client is not None and not is_mutable(client, asst_msg):
        if assistant_meta is None or msg_dispatcher is None:
            # A missing param here must never silently fall through to the
            # pop/in-place edit below — that's precisely the mutation this
            # branch exists to prevent. Fail loudly instead of corrupting
            # an already-dispatched message. Logged explicitly because every
            # known caller wraps this in a broad suppress/except-pass, which
            # would otherwise swallow the raise along with the failure.
            _msg = (
                "prune_wait_tool_call: asst_msg is already below the sent "
                "watermark, so popping or editing its tool_calls in place "
                "would mutate already-dispatched bytes — but assistant_meta "
                "and msg_dispatcher (needed to route the ack through the "
                "escape hatch) were not both provided."
            )
            LOGGER.error(f"{DEFAULT_ICON} {_msg}")
            raise ValueError(_msg)
        await acknowledge_helper_call(
            asst_msg,
            call_id,
            "wait",
            "{}",
            assistant_meta=assistant_meta,
            client=client,
            msg_dispatcher=msg_dispatcher,
            bypass_watermark=True,
        )
        return

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
    bypass_watermark: bool = False,
) -> None:
    """
    Append *tool_msg* and move it directly after *parent_msg*, while
    updating the per-assistant `results_count` bookkeeping.

    If *skip_event_bus* is True, the message is appended to the client
    transcript but NOT published to the EventBus. This is used for
    placeholder messages that will be updated in-place later.

    If the computed insertion position falls below the client's sent
    watermark, splicing there would shift every already-dispatched message
    that follows — breaking the provider's cached prefix from that point
    on. The message is instead delivered as a check_status pair appended
    at the tail, leaving everything below the watermark untouched —
    *unless* the transcript would otherwise become illegal: when
    ``tool_msg``'s call_id has no reply anywhere yet, this insertion IS the
    first-ever reply, and redirecting it would permanently orphan the
    original ``tool_calls`` entry (a check_status pair answers a different,
    synthesized call_id). That case always splices, whether or not the
    caller passed *bypass_watermark* — legality beats cache, enforced here
    rather than trusted to every call site. A caller with its own reason to
    force the splice (e.g. the backfill/restore escape hatch) may still
    pass *bypass_watermark* explicitly.

    A sanctioned below-watermark splice re-baselines the stored watermark
    hash immediately, so the next dispatch's integrity check (when enabled)
    reads it as the new legitimate state rather than a violation.
    """
    call_id = tool_msg.get("tool_call_id") if isinstance(tool_msg, dict) else None

    if (
        not bypass_watermark
        and client is not None
        and call_id is not None
        and not _call_id_has_reply(client, call_id)
    ):
        bypass_watermark = True

    watermark = getattr(client, "_sent_watermark", 0) if client is not None else 0
    parent_idx = _message_index(client, parent_msg) if client is not None else None
    existing_meta = assistant_meta.get(id(parent_msg))
    results_count = existing_meta["results_count"] if existing_meta else 0
    insert_pos = (parent_idx + 1 + results_count) if parent_idx is not None else None
    below_watermark = insert_pos is not None and insert_pos < watermark

    if below_watermark and not bypass_watermark:
        content = (
            tool_msg.get("content") if isinstance(tool_msg, dict) else str(tool_msg)
        )
        await emit_completion_pair(
            content,
            call_id or "unknown",
            msg_dispatcher,
        )
        return

    # Only now mark the parent handled — a reply diverted to check_status
    # above must not suppress the preflight repair that looks for
    # unanswered tool_calls entries.
    meta = assistant_meta.setdefault(id(parent_msg), {"results_count": 0})
    await msg_dispatcher.append_msgs([tool_msg], skip_event_bus=skip_event_bus)
    final_insert_pos = _message_index(client, parent_msg) + 1 + meta["results_count"]
    client.messages.insert(final_insert_pos, client.messages.pop())
    meta["results_count"] += 1

    if below_watermark:
        _rebaseline_watermark_hash(client)


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
    bypass_watermark: bool = False,
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
        bypass_watermark=bypass_watermark,
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
    time_ctx=None,
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

        # Self-describing so a permanently-frozen stub (below-watermark by
        # the time the result arrives) still reads truthfully: the result
        # never rewrites this message, it always arrives as a check_status
        # pair appended below. "meta:"-prefixed keys are the established
        # convention for annotations that don't change what this placeholder
        # fundamentally *is* (still "pending").
        ph_content: dict = {
            "_placeholder": "pending",
            "meta:status": "async — result arrives as a check_status message below",
        }
        if time_ctx is not None:
            with suppress(Exception):
                ph_content["meta:started"] = time_ctx.offset_at(_inf.scheduled_time)

        placeholder = create_tool_call_message(
            name=_inf.name,
            call_id=_inf.call_id,
            content=json.dumps(ph_content, indent=4),
        )
        # The first-ever reply to a call_id must always sit immediately
        # after its assistant message — that's the API's own legality
        # requirement, not a caching nicety, so this always bypasses the
        # watermark gate (see the escape hatch on insert_tool_message_after_assistant).
        await insert_tool_message_after_assistant(
            assistant_meta,
            _inf.assistant_msg,
            placeholder,
            client,
            msg_dispatcher,
            skip_event_bus=True,  # Don't publish placeholders; publish when final
            bypass_watermark=True,
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
    context_state,
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
                        await prune_wait_tool_call(
                            asst_msg,
                            cid,
                            client=client,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=msg_dispatcher,
                        )
                    except Exception:
                        pass
                    scheduled.append(cid)
                    continue

                # Other helpers: acknowledge but do not execute during backfill.
                # This is the backfill/restore escape hatch — the call_id has
                # no reply anywhere yet, so the ack must splice adjacently
                # regardless of watermark; legality beats cache.
                try:
                    await acknowledge_helper_call(
                        asst_msg,
                        cid,
                        name,
                        args_json,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=msg_dispatcher,
                        bypass_watermark=True,
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
                context_state=context_state,
                propagate_chat_context=propagate_chat_context,
                assistant_meta=assistant_meta,
                msg_dispatcher=msg_dispatcher,
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
