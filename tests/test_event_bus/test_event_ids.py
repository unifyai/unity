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
from typing import Any

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
    • Every call produces a **new** event_id, even when the caller supplies none.
    """
    bus = EventBus()
    bus.register_event_types("IDS")

    # Publish two events without specifying event_id nor calling_id
    for txt in ("one", "two"):
        await bus.publish(Event(type="IDS", payload=DummyPayload(msg=txt)))

    latest = await bus.get_latest(types=["IDS"], limit=2)
    assert len(latest) == 2

    # newest-first → reverse for creation order
    e1, e2 = reversed(latest)

    # 1. Both fields must be non-empty UUID strings
    for evt in (e1, e2):
        assert evt.event_id, "event_id left blank"
        # Raises ValueError if not a valid UUID
        uuid.UUID(evt.event_id)  # type: ignore[arg-type]
        assert evt.calling_id == "", "unexpected calling_id default modified"

    # 2. They must be distinct
    assert e1.event_id != e2.event_id, "event_id should be unique per message"


@pytest.mark.asyncio
@_handle_project
async def test_calling_id_is_preserved_across_messages() -> None:
    """
    When the publisher sets `calling_id`, every event for the same logical
    conversation should keep that value (the EventBus must *not* overwrite it).
    """
    bus = EventBus()
    bus.register_event_types("FLOW")

    cid = "conversation-42"

    for word in ("alpha", "beta", "gamma"):
        await bus.publish(
            Event(type="FLOW", payload=DummyPayload(msg=word), calling_id=cid)
        )

    latest = await bus.get_latest(types=["FLOW"], limit=10)
    assert latest, "no events returned"

    # All retrieved events belong to the requested flow
    assert {evt.calling_id for evt in latest} == {cid}, "calling_id not preserved"
