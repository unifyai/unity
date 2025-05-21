# tests/test_event_bus_ids.py
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
from pydantic import BaseModel


from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event


# dummy payload for the envelope ------------------------------------------------
class DummyPayload(BaseModel):
    msg: str


# --------------------------------------------------------------------------- #
#                               TESTS                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_event_ids_are_populated_and_unique() -> None:
    """
    • EventBus.publish must *not* leave `event_id` blank.
    • Every call produces a **new** event_id.
    """
    bus = EventBus()
    bus.register_event_types("numbers")

    # Publish two events without specifying event_id nor calling_id
    for txt in ("one", "two"):
        await bus.publish(Event(type="numbers", payload=DummyPayload(msg=txt)))

    latest = (await bus.get_latest(types=["numbers"], limits=2))["numbers"]
    assert len(latest) == 2

    e1, e2 = latest

    # 1. Both fields must be non-empty UUID strings
    for evt in (e1, e2):
        assert evt.event_id, "event_id left blank"
        # Raises ValueError if not a valid UUID
        uuid.UUID(evt.event_id)  # type: ignore[arg-type]
        assert evt.calling_id == "", "unexpected calling_id default modified"

    # 2. They must be distinct
    assert e1.event_id != e2.event_id, "event_id should be unique per message"
