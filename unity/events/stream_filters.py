"""Pub/Sub stream noise filters.

Rules that identify ToolLoop events which are internal bookkeeping artifacts
and should be suppressed from the real-time Pub/Sub stream (and its SSE
frontend feed) to save bandwidth and reduce UI clutter.

Unify context logs remain **unfiltered** -- these rules only gate the
Pub/Sub publishing path in ``EventBus.publish()``.

SAFETY: ManagerMethod events are never touched.  They define the hierarchy
tree (incoming/outgoing/action) and must always reach the frontend.  Only
ToolLoop events -- leaf content *within* a tree node -- are candidates.
Dropping them cannot orphan, disconnect, or corrupt the hierarchy tree.
"""

from __future__ import annotations

import json
from typing import Callable

# ---------------------------------------------------------------------------
#  Individual noise-detection rules
# ---------------------------------------------------------------------------
# Each rule takes a single ``msg: dict`` (the raw LLM message from
# ``ToolLoopPayload.message``) and returns ``True`` when the message is
# noise that should be suppressed.


def is_synthetic_status_check(msg: dict) -> bool:
    """Synthetic ``check_status_*`` completion pairs.

    Injected by ``ToolsData._emit_completion_pair`` when an async tool
    finishes out of order.  Always come in adjacent assistant+tool pairs.
    """
    for tc in msg.get("tool_calls") or []:
        name = (tc.get("function") or {}).get("name", "")
        if isinstance(name, str) and name.startswith("check_status_"):
            return True
    if msg.get("role") == "tool":
        name = msg.get("name") or ""
        if isinstance(name, str) and name.startswith("check_status_"):
            return True
    return False


def is_placeholder_message(msg: dict) -> bool:
    """Pending/progress/nested-start placeholder tool replies.

    These are superseded by real content once the tool completes.
    Known ``_placeholder`` values: ``"pending"``, ``"progress"``,
    ``"completed"``, ``"nested_start"``.
    """
    if msg.get("role") != "tool":
        return False
    try:
        parsed = json.loads(msg.get("content", ""))
        return isinstance(parsed, dict) and "_placeholder" in parsed
    except Exception:
        return False


def is_runtime_context_header(msg: dict) -> bool:
    """Parent chat context blobs forwarded to inner loops.

    Large system messages (often multi-KB) carrying ``_runtimeContext: true``.
    Purely LLM-internal; never user-facing.
    """
    return msg.get("_runtimeContext") is True


def is_visibility_guidance(msg: dict) -> bool:
    """Interjection visibility guidance system prompt.

    System message with ``_visibility_guidance: true`` explaining user
    visibility to the LLM during interjections.  Purely LLM-internal.
    """
    return msg.get("_visibility_guidance") is True


# ---------------------------------------------------------------------------
#  Rule registry -- append new rules here
# ---------------------------------------------------------------------------

_STREAM_NOISE_RULES: list[Callable[[dict], bool]] = [
    is_synthetic_status_check,
    is_placeholder_message,
    is_runtime_context_header,
    is_visibility_guidance,
]

# ---------------------------------------------------------------------------
#  Public predicate
# ---------------------------------------------------------------------------


def is_streaming_noise(event_type: str, payload_dict: dict) -> bool:
    """Return ``True`` if this event should be suppressed from Pub/Sub streaming.

    Only ``ToolLoop`` events are candidates for filtering.  ``ManagerMethod``
    events always pass through -- they define the hierarchy tree and must
    never be dropped.

    Parameters
    ----------
    event_type:
        The ``Event.type`` string (e.g. ``"ToolLoop"``, ``"ManagerMethod"``).
    payload_dict:
        The already-serialised payload dict (same object passed to
        ``_stream_action_to_pubsub``).
    """
    if event_type != "ToolLoop":
        return False
    msg = payload_dict.get("message")
    if not isinstance(msg, dict):
        return False
    return any(rule(msg) for rule in _STREAM_NOISE_RULES)
