import pytest
import datetime as dt
from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event
from unity.events.types.comms import CommsPayload
from unity.events.types.manager_method import ManagerMethodPayload

# -------------------------------------------------------------------
#  helpers
# -------------------------------------------------------------------


def mk_evt(**kw):
    return Event(type="Comms", payload=CommsPayload(**kw))


def ts(i: int) -> str:
    """Deterministic, strictly-monotonic timestamps."""
    return (
        dt.datetime(2025, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=i)
    ).isoformat()


# -------------------------------------------------------------------
#  test suite
# -------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_deque_only():
    bus = EventBus()
    await bus.publish(mk_evt(level="INFO", msg="one"))
    await bus.publish(mk_evt(level="WARN", msg="two"))
    res = await bus.search(
        filter='type == "Comms" and payload["level"] == "WARN"',
        limit=5,
    )
    assert len(res) == 1
    # Payload is always a dict
    assert res[0].payload.get("msg") == "two"


@pytest.mark.asyncio
@_handle_project
async def test_grouped_and_limit_dict():
    bus = EventBus()
    await bus.publish(mk_evt(level="INFO"))
    await bus.publish(mk_evt(level="WARN"))
    await bus.publish(
        Event(
            type="ManagerMethod",
            payload=ManagerMethodPayload(manager="Test", method="heartbeat"),
        ),
    )
    out = await bus.search(
        limit={"Comms": 1, "ManagerMethod": 5},
        grouped_by_type=True,
    )
    # Correct shape
    assert "Comms" in out
    # Per-type slicing respected
    assert len(out["Comms"]) == 1


@pytest.mark.asyncio
@_handle_project
async def test_hybrid_reads():
    bus = EventBus()
    bus.set_default_window(1)  # deque keeps 1

    for seq in range(4):  # publish 4 events
        await bus.publish(Event(type="Comms", payload=CommsPayload(seq=seq)))
        bus.join_published()

    out = await bus.search(
        limit=3,
        filter='type == "Comms"',
    )
    assert [e.payload.get("seq") for e in out] == [3, 2, 1]


@pytest.mark.asyncio
@_handle_project
async def test_with_offset_across_backend():
    """
    Skip 2 newest rows (offset=2) and return the next 2
    even though the deque window is only 1 deep.
    """
    bus = EventBus()
    bus.set_default_window(1)  # deque holds just the newest row

    for seq in range(5):  # seq 0..4  (4 is newest)
        await bus.publish(
            Event(
                type="Comms",
                timestamp=ts(seq),
                payload=CommsPayload(seq=seq),
            ),
        )
        bus.join_published()

    out = await bus.search(
        limit=2,
        offset=2,
        filter='type == "Comms"',
    )

    assert [e.payload.get("seq") for e in out] == [2, 1]


@pytest.mark.asyncio
@_handle_project
async def test_flat_ordering():
    """
    With no filter and a scalar limit, verify that the flat list is
    newest-first across *all* event-types.
    """
    bus = EventBus()

    # older ManagerMethod event
    await bus.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(0),
            payload=ManagerMethodPayload(manager="Test", method="heartbeat"),
        ),
    )
    # newer Comms event
    await bus.publish(
        Event(
            type="Comms",
            timestamp=ts(1),
            payload=CommsPayload(seq=0),
        ),
    )
    bus.join_published()

    out = await bus.search(limit=2)  # flat list
    assert [(e.type, e.payload.get("seq")) for e in out] == [
        ("Comms", 0),
        ("ManagerMethod", None),
    ]


@pytest.mark.asyncio
@_handle_project
async def test_type_alias_in_filter():
    """
    Local evaluator must accept `event_type` as an alias for `type`.
    """
    bus = EventBus()
    await bus.publish(
        Event(
            type="ManagerMethod",
            payload=ManagerMethodPayload(manager="Test", method="heartbeat"),
        ),
    )
    bus.join_published()

    res = await bus.search(filter='event_type == "ManagerMethod"', limit=10)
    assert len(res) >= 1 and res[0].type == "ManagerMethod"


@pytest.mark.asyncio
@_handle_project
async def test_limit_dict_unknown_type():
    """
    If the caller specifies a per-type limit for a type that doesn't
    have events in the deque, the search should handle it gracefully.
    """
    bus = EventBus()
    await bus.publish(Event(type="Comms", payload=CommsPayload(seq=0)))
    bus.join_published()

    out = await bus.search(
        limit={"Comms": 5, "ToolLoop": 3},  # ToolLoop may have no events
        grouped_by_type=True,
    )
    assert "Comms" in out and len(out["Comms"]) >= 1
