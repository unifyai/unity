import time
import unify

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event


@pytest.mark.asyncio
@_handle_project
async def test_join_initialization_sets_flag() -> None:
    """Calling ``join_initialization`` must complete the background hydration
    and set :pyattr:`EventBus.initialized` to *True*.
    """

    bus = EventBus()

    # The flag must be a boolean (might already be set depending on timing)
    assert isinstance(bus.initialized, bool)

    # Await readiness – must not raise and afterwards flag is True
    await bus.join_initialization()
    assert bus.initialized is True


@pytest.mark.asyncio
@_handle_project
async def test_join_initialization_idempotent() -> None:
    """Subsequent calls to ``join_initialization`` after the first one should
    return quickly and leave the state unchanged."""

    bus = EventBus()
    await bus.join_initialization()
    assert bus.initialized is True

    # Capture time for a second immediate call – should be near-instant
    t0 = time.perf_counter()
    await bus.join_initialization()
    t1 = time.perf_counter()

    assert (t1 - t0) < 0.05, "Second join_initialization call took unexpectedly long"


@pytest.mark.asyncio
@_handle_project
async def test_reset_deletes_type_contexts_and_clears_events() -> None:
    """reset(delete_contexts=True) must drop per-type contexts and clear history.

    After publishing an event to create the per-type context, calling reset should
    remove that context from Unify and re-initialise the in-memory state so that
    subsequent searches return no results for the old event-type.
    """

    bus = EventBus()

    # Publish to create per-type context and persist an event
    await bus.publish(Event(type="ResetProbe", payload={"ok": True}))
    bus.join_published()

    # Sanity: event is retrievable before reset
    out_before = await bus.search(filter='type == "ResetProbe"', limit=10)
    assert len(out_before) == 1

    # Derive global Events context from the per-type context path
    per_type_ctx = bus.ctxs["ResetProbe"]
    global_ctx = per_type_ctx.rsplit("/", 1)[0]

    # Verify contexts exist prior to reset
    ctxs_before = set(unify.get_contexts(prefix=global_ctx))
    assert per_type_ctx in ctxs_before
    assert any(c.endswith("/_callbacks") for c in ctxs_before)

    # Reset: delete contexts and re-initialise this instance in-place
    bus.clear(delete_contexts=True)

    # After reset, the specific per-type context must be gone
    ctxs_after = set(unify.get_contexts(prefix=global_ctx))
    assert per_type_ctx not in ctxs_after

    # Old events should no longer be found
    out_after = await bus.search(filter='type == "ResetProbe"', limit=10)
    assert out_after == []
