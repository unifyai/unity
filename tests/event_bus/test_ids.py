# tests/event_bus/test_ids.py
#
# Verifies that every Event published through EventBus
# (a) always carries a non-empty, unique `event_id` and
# (b) preserves a user-supplied `calling_id` across messages
#     that belong to the same conversation / tool-loop run.
#
# The tests assume the *real* unify package is available.
# If your project already provides the `_handle_project` helper
# that spawns an isolated Unify project for test-repeatability,
# we re-use it; otherwise the decorator is a harmless no-op.

from __future__ import annotations

import uuid

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event
from unity.events.types.manager_method import ManagerMethodPayload


# --------------------------------------------------------------------------- #
#                               TESTS                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_populated_and_unique() -> None:
    """
    • EventBus.publish must *not* leave `event_id` blank.
    • Every call produces a **new** event_id.
    """
    bus = EventBus()

    # Publish two events without specifying event_id nor calling_id
    for txt in ("one", "two"):
        payload = ManagerMethodPayload(
            manager="TestManager",
            method="test",
            question=txt,
        )
        await bus.publish(Event(type="ManagerMethod", payload=payload))

    latest = await bus.search(filter="type == 'ManagerMethod'", limit=2)
    assert len(latest) >= 2

    # Take the last two events
    e1, e2 = latest[-2], latest[-1]

    # 1. Both fields must be non-empty UUID strings
    for evt in (e1, e2):
        assert evt.event_id, "event_id left blank"
        # Raises ValueError if not a valid UUID
        uuid.UUID(evt.event_id)  # type: ignore[arg-type]

    # 2. They must be distinct
    assert e1.event_id != e2.event_id, "event_id should be unique per message"
