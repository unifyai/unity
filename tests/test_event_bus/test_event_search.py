import pytest
import datetime as dt
from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event


# -------------------------------------------------------------------
#  helpers
# -------------------------------------------------------------------


def mk_evt(**kw):
    return Event(type="Alerts", payload=kw)


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
async def test_search_deque_only():
    bus = EventBus()
    await bus.publish(mk_evt(level="INFO", msg="one"))
    await bus.publish(mk_evt(level="WARN", msg="two"))
    res = await bus.search(
        filter='type == "Alerts" and payload["level"] == "WARN"',
        limit=5,
    )
    assert len(res) == 1
    assert res[0].payload["msg"] == "two"


@pytest.mark.asyncio
@_handle_project
async def test_search_grouped_and_limit_dict():
    bus = EventBus()
    await bus.publish(mk_evt(level="INFO"))
    await bus.publish(mk_evt(level="WARN"))
    await bus.publish(Event(type="Heartbeat", payload={"ok": True}))
    out = await bus.search(limit={"Alerts": 1, "Heartbeat": 5}, grouped_by_type=True)
    # Correct shape
    assert set(out) == {"Alerts", "Heartbeat"}
    # Per-type slicing respected
    assert len(out["Alerts"]) == 1


@pytest.mark.asyncio
@_handle_project
async def test_search_hybrid_reads():
    bus = EventBus()
    bus.set_default_window(1)  # deque keeps 1

    for seq in range(4):  # publish 4 events
        await bus.publish(Event(type="Alerts", payload={"seq": seq}))
    bus.join_published()

    out = await bus.search(
        limit=3,
        filter='type == "Alerts"',
    )
    assert [e.payload["seq"] for e in out] == [3, 2, 1]


@pytest.mark.asyncio
@_handle_project
async def test_search_with_offset_across_backend():
    """
    Skip 2 newest rows (offset=2) and return the next 2
    even though the deque window is only 1 deep.
    """
    bus = EventBus()
    bus.set_default_window(1)  # deque holds just the newest row

    for seq in range(5):  # seq 0..4  (4 is newest)
        await bus.publish(
            Event(
                type="Alerts",
                timestamp=ts(seq),
                payload={"seq": seq},
            ),
        )
    bus.join_published()

    out = await bus.search(
        limit=2,
        offset=2,
        filter='type == "Alerts"',
    )

    assert [e.payload["seq"] for e in out] == [2, 1]


@pytest.mark.asyncio
@_handle_project
async def test_flat_ordering_across_types():
    """
    With no filter and a scalar limit, verify that the flat list is
    newest-first across *all* event-types.
    """
    bus = EventBus()

    # older heartbeat
    await bus.publish(
        Event(
            type="Heartbeat",
            timestamp=ts(0),
            payload={"ok": True},
        ),
    )
    # newer alert
    await bus.publish(
        Event(
            type="Alerts",
            timestamp=ts(1),
            payload={"seq": 0},
        ),
    )
    bus.join_published()

    out = await bus.search(limit=2)  # flat list
    assert [(e.type, e.payload.get("seq", None)) for e in out] == [
        ("Alerts", 0),
        ("Heartbeat", None),
    ]


@pytest.mark.asyncio
@_handle_project
async def test_event_type_alias_in_filter():
    """
    Local evaluator must accept `event_type` as an alias for `type`.
    """
    bus = EventBus()
    await bus.publish(Event(type="Heartbeat", payload={"ok": True}))
    bus.join_published()

    res = await bus.search(filter='event_type == "Heartbeat"', limit=10)
    assert len(res) == 1 and res[0].type == "Heartbeat"


@pytest.mark.asyncio
@_handle_project
async def test_limit_dict_with_unknown_type():
    """
    If the caller specifies a per-type limit for a type that doesn't
    exist, the search should silently ignore it.
    """
    bus = EventBus()
    await bus.publish(Event(type="Alerts", payload={"seq": 0}))
    bus.join_published()

    out = await bus.search(
        limit={"Alerts": 5, "NonExistent": 3},
        grouped_by_type=True,
    )
    assert set(out) == {"Alerts"} and len(out["Alerts"]) == 1
