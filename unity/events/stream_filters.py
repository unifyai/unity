"""Pub/Sub stream noise filters.

Rules that identify events which are internal bookkeeping artifacts and
should be suppressed from the real-time Pub/Sub stream (and its SSE
frontend feed) to save bandwidth and reduce UI clutter.

Unify context logs remain **unfiltered** -- these rules only gate the
Pub/Sub publishing path in ``EventBus.publish()``.

Two categories of filtering exist:

1. **ToolLoop noise** -- individual LLM messages (status checks, placeholders,
   runtime context headers, visibility guidance) that are leaf content within
   an action tree node.  Dropping them cannot orphan or corrupt the tree.

2. **Disjoint tree suppression** -- entire trees spawned by background
   managers (currently ``MemoryManager``) that run asynchronously alongside
   the user-initiated action.  Both their ``ManagerMethod`` *and* ``ToolLoop``
   events are suppressed because they form completely independent parallel
   trees with no hierarchy linkage to the primary action.
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
#  ToolLoop noise rule registry -- append new per-message rules here
# ---------------------------------------------------------------------------

_STREAM_NOISE_RULES: list[Callable[[dict], bool]] = [
    # is_synthetic_status_check intentionally excluded — the frontend needs
    # these events to resolve pending parallel tool calls via
    # resolvedToolCallIds.  Display filtering is handled client-side by
    # event-filters.ts isToolLoopNoise.
    is_placeholder_message,
    is_runtime_context_header,
    is_visibility_guidance,
]

# ---------------------------------------------------------------------------
#  Disjoint tree suppression -- managers whose entire event trees are hidden
# ---------------------------------------------------------------------------

_SUPPRESSED_MANAGERS: frozenset[str] = frozenset({"MemoryManager"})


def _hierarchy_root_is_suppressed(payload_dict: dict) -> bool:
    """Check whether the first segment of ``hierarchy`` belongs to a suppressed manager."""
    hierarchy = payload_dict.get("hierarchy")
    if isinstance(hierarchy, list) and hierarchy:
        root = hierarchy[0]
        if isinstance(root, str):
            return any(root.startswith(f"{m}.") for m in _SUPPRESSED_MANAGERS)
    return False


def is_suppressed_manager_tree(event_type: str, payload_dict: dict) -> bool:
    """Return ``True`` if this event belongs to a disjoint background manager tree.

    ``MemoryManager`` methods fire asynchronously via EventBus callbacks and
    produce independent parallel trees with no hierarchy linkage to the
    user-initiated action.  **All** events in the tree are suppressed —
    ``ManagerMethod`` nodes, inner nested ``ManagerMethod`` nodes from
    sub-managers, and ``ToolLoop`` leaf messages alike.

    Detection for both ``ManagerMethod`` and ``ToolLoop``:
    - Primary: ``payload_dict["manager"]`` is in ``_SUPPRESSED_MANAGERS``
      (direct match for the manager's own events).
    - Fallback: the root segment of ``payload_dict["hierarchy"]`` starts
      with a suppressed manager name (catches nested sub-manager calls).
    """
    if event_type not in ("ManagerMethod", "ToolLoop"):
        return False

    if payload_dict.get("manager") in _SUPPRESSED_MANAGERS:
        return True

    return _hierarchy_root_is_suppressed(payload_dict)


# ---------------------------------------------------------------------------
#  Public predicate
# ---------------------------------------------------------------------------


def is_streaming_noise(event_type: str, payload_dict: dict) -> bool:
    """Return ``True`` if this event should be suppressed from Pub/Sub streaming.

    Checks two layers:

    1. **Disjoint tree suppression** -- entire trees from background managers
       (e.g. ``MemoryManager``) are dropped for both ``ManagerMethod`` *and*
       ``ToolLoop`` event types.
    2. **ToolLoop noise** -- individual LLM messages matching the per-message
       noise rules (status checks, placeholders, context headers, etc.).

    Parameters
    ----------
    event_type:
        The ``Event.type`` string (e.g. ``"ToolLoop"``, ``"ManagerMethod"``).
    payload_dict:
        The already-serialised payload dict (same object passed to
        ``_stream_action_to_pubsub``).
    """
    if is_suppressed_manager_tree(event_type, payload_dict):
        return True

    if event_type != "ToolLoop":
        return False
    msg = payload_dict.get("message")
    if not isinstance(msg, dict):
        return False
    return any(rule(msg) for rule in _STREAM_NOISE_RULES)
