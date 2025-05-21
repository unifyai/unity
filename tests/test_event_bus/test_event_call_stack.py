# tests/test_event_bus_call_stack.py
#
# Validates EventBus.get_event_call_stack for the *happy-path* in which all
# required events already live inside the in-memory deques (no Unify fetch
# necessary).
#
# Requires the real EventBus implementation and the Unify SDK.

from __future__ import annotations

import asyncio
import pytest
from pydantic import BaseModel

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event


class Payload(BaseModel):
    txt: str


@pytest.mark.asyncio
@_handle_project
async def test_get_event_call_stack_local() -> None:
    """
    Build a simple three-level call chain:

        root (no calling_id)
          └── mid  (calling_id = root.event_id)
                └── leaf (calling_id = mid.event_id)

    Publish them in order and confirm that `get_event_call_stack(leaf)`
    yields [root, mid, leaf].
    """
    bus = EventBus()
    bus.register_event_types("STACK")

    # ── construct chain ──────────────────────────────────────────────────
    root_evt = Event(type="STACK", payload=Payload(txt="root"))
    mid_evt  = Event(type="STACK", payload=Payload(txt="mid"),  calling_id=root_evt.event_id)
    leaf_evt = Event(type="STACK", payload=Payload(txt="leaf"), calling_id=mid_evt.event_id)

    # publish -> now all three live inside the relevant deque
    for e in (root_evt, mid_evt, leaf_evt):
        await bus.publish(e)

    # ── exercise new helper ──────────────────────────────────────────────
    stack = await bus.get_event_call_stack(leaf_evt.event_id)

    # Expect exact order: top-down
    assert [ev.event_id for ev in stack] == [
        root_evt.event_id,
        mid_evt.event_id,
        leaf_evt.event_id,
    ]

    # Sanity: root is indeed the chain origin
    assert stack[0].calling_id == ""
    assert stack[-1].event_id == leaf_evt.event_id
