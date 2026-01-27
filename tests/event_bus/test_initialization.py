import time

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event
from unity.events.types.comms import CommsPayload


@pytest.mark.asyncio
@_handle_project
async def test_join_sets_flag() -> None:
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
async def test_join_idempotent() -> None:
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
async def test_reset_clears_history() -> None:
    """reset(delete_contexts=True) must clear event history.

    After publishing an event, calling reset should re-initialise the in-memory
    state so that subsequent searches return no results for the old events.

    Note: With eager context creation for known types, contexts may be recreated
    after clear(), but the event data within them should be deleted.
    """

    bus = EventBus()

    # Publish to create per-type context and persist an event
    await bus.publish(Event(type="Comms", payload=CommsPayload(ok=True)))
    bus.join_published()

    # Sanity: event is retrievable before reset
    out_before = await bus.search(filter='type == "Comms"', limit=10)
    assert len(out_before) >= 1

    # Reset: delete contexts and re-initialise this instance in-place
    bus.clear(delete_contexts=True)

    # Old events should no longer be found (contexts may be recreated but empty)
    out_after = await bus.search(filter='type == "Comms"', limit=10)
    assert out_after == []
