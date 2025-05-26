import pytest
from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event


# -------------------------------------------------------------------
#  helpers
# -------------------------------------------------------------------
class DummyLog:
    """Mimic unify.Log minimal surface."""

    def __init__(self, entries):
        self.entries = entries


def mk_evt(**kw):
    return Event(type="Alerts", payload=kw)


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
