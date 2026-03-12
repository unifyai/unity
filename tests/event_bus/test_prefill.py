import pytest
import datetime as dt

from unity.events.event_bus import EventBus, Event
from unity.events.types.comms import CommsPayload
from unity.transcript_manager.types.message import Message
from unity.conversation_manager.types import Medium
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_prefill_populates_comms_deque():
    """Comms events should be prefilled into the in-memory deque on startup.

    Comms is in _PREFILL_TYPES, so a new EventBus instance should hydrate
    recent Comms events from Orchestra into its deque without a backend
    fallback call.
    """
    window = 10
    bus1 = EventBus()
    bus1.set_default_window(window)

    base_ts = dt.datetime.now(dt.UTC)
    published: list[Event] = []

    for i in range(5):
        evt = Event(
            type="Comms",
            timestamp=base_ts + dt.timedelta(seconds=i),
            payload=CommsPayload(ok=True, seq=i),
        )
        published.append(evt)
        await bus1.publish(evt)

    bus1.join_published()

    bus2 = EventBus()
    bus2.set_window("Comms", window)

    latest = await bus2.search(filter="type == 'Comms'", limit=window)

    for sent in published:
        assert any(
            rec.timestamp == sent.timestamp for rec in latest
        ), f"Event with ts {sent.timestamp} not found in prefilled window"


@pytest.mark.asyncio
@_handle_project
async def test_non_prefilled_types_still_searchable():
    """Event types outside _PREFILL_TYPES should still be retrievable via
    search(), which falls back to the Orchestra backend when the in-memory
    deque is empty.
    """
    bus1 = EventBus()

    evt = Event(
        type="Message",
        payload=Message(
            medium=Medium.UNIFY_MESSAGE,
            sender_id=1,
            receiver_ids=[2],
            timestamp=dt.datetime.now(dt.UTC),
            content="hello",
            exchange_id=0,
        ),
    )
    await bus1.publish(evt)
    bus1.join_published()

    bus2 = EventBus()
    results = await bus2.search(filter="type == 'Message'", limit=10)

    assert any(
        r.timestamp == evt.timestamp for r in results
    ), "Message event not found via backend fallback"


@pytest.mark.asyncio
@_handle_project
async def test_row_id_seeded_for_non_prefilled_types():
    """Non-prefilled types should have their row_id counter seeded from
    the backend so new events get monotonically increasing IDs.
    """
    bus1 = EventBus()

    evt = Event(
        type="Message",
        payload=Message(
            medium=Medium.UNIFY_MESSAGE,
            sender_id=1,
            receiver_ids=[2],
            timestamp=dt.datetime.now(dt.UTC),
            content="seed-test",
            exchange_id=0,
        ),
    )
    await bus1.publish(evt)
    bus1.join_published()
    first_row_id = evt.row_id

    bus2 = EventBus()
    await bus2.join_initialization()

    next_id = bus2._next_row_ids.get("Message", 0)
    assert next_id > first_row_id, (
        f"row_id counter ({next_id}) should be seeded above the "
        f"persisted row_id ({first_row_id})"
    )
