import time

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus


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
