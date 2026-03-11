"""Pub/Sub stream noise filters.

Rules that identify events which are internal bookkeeping artifacts and
should be suppressed from the real-time Pub/Sub stream (and its SSE
frontend feed) to save bandwidth and reduce UI clutter.

Unify context logs remain **unfiltered** -- these rules only gate the
Pub/Sub publishing path in ``EventBus.publish()``.

Two categories of filtering exist:

1. **ToolLoop noise** -- identified by the ``kind`` field on
   ``ToolLoopPayload``.  Events whose kind is in ``_STREAM_NOISE_KINDS``
   are leaf content within an action tree node; dropping them cannot
   orphan or corrupt the tree.

2. **Disjoint tree suppression** -- entire trees spawned by background
   managers (currently ``MemoryManager``) that run asynchronously alongside
   the user-initiated action.  Both their ``ManagerMethod`` *and* ``ToolLoop``
   events are suppressed because they form completely independent parallel
   trees with no hierarchy linkage to the primary action.
"""

from __future__ import annotations

from .types.tool_loop import ToolLoopKind

# ---------------------------------------------------------------------------
#  ToolLoop noise -- kinds that are internal bookkeeping
# ---------------------------------------------------------------------------
# ``STATUS_CHECK`` is intentionally excluded: the frontend needs those
# events to resolve pending parallel tool calls via resolvedToolCallIds.
# Display filtering is handled client-side by event-filters.ts.

_STREAM_NOISE_KINDS: frozenset[str] = frozenset(
    {
        ToolLoopKind.PLACEHOLDER,
        ToolLoopKind.RUNTIME_CONTEXT,
        ToolLoopKind.TIME_EXPLANATION,
        ToolLoopKind.VISIBILITY_GUIDANCE,
    },
)

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
    user-initiated action.  **All** events in the tree are suppressed --
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
    2. **ToolLoop noise** -- events whose ``kind`` is in the noise set.

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

    return payload_dict.get("kind", "") in _STREAM_NOISE_KINDS
