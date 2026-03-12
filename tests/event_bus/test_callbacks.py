# Verifies the two trigger-mechanisms implemented by EventBus.register_callback:
#
#   • frequency-based ("every N events")
#   • time-based ("every X seconds")
#
# Each mechanism is tested
#   1. inside a single EventBus lifetime (no persistence required)
#   2. across two independent EventBus lifetimes (state re-hydrated from Unify)
#
# and then repeated with a payload-level *filter* so that only a subset of
# published events contribute towards the trigger counter / timer.
#
# The tests rely on the Subscription state being persisted to the dedicated
# "…/Events/_callbacks" context (created by the patched EventBus) and on
# auto-incremented `row_id` / timestamp metadata coming back from Unify.
#
# Uses different event types for test isolation:
#   - Frequency tests: Comms
#   - Time tests: ToolLoop
#   - Filter tests: ManagerMethod
#
# --------------------------------------------------------------------------- #

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event
from unity.events.types.comms import CommsPayload
from unity.events.types.tool_loop import ToolLoopPayload, ToolLoopKind
from unity.events.types.manager_method import ManagerMethodPayload


def _get_seq(payload) -> int:
    """Extract seq from payload (always a dict)."""
    return payload.get("seq")


# --------------------------------------------------------------------------- #
#  deterministic timestamp helper                                             #
# --------------------------------------------------------------------------- #


def ts(offset: int) -> str:
    """T0 is 2025-03-01 00:00:00Z; *offset* is seconds."""
    return dt.datetime(2025, 3, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=offset)


# --------------------------------------------------------------------------- #
#  1. frequency-based trigger, single session                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_frequency_single_session() -> None:
    bus = EventBus()
    triggered: list[int] = []

    async def cb(events):  # noqa: ANN001
        # store the seq of the event that caused the trigger
        triggered.append(_get_seq(events[0].payload))

    await bus.register_callback(
        event_type="Comms",
        callback=cb,
        every_n=3,
    )

    # Publish five events → expect one trigger at seq==2
    for seq in range(5):
        await bus.publish(Event(type="Comms", payload=CommsPayload(seq=seq)))
    bus.join_published()
    await asyncio.sleep(0.05)

    assert triggered == [2]


# --------------------------------------------------------------------------- #
#  2. frequency-based trigger, across sessions (row_id sync)                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_frequency_across_sessions() -> None:
    # ── first run ────────────────────────────────────────────────────────────
    bus1 = EventBus()
    trig1: list[int] = []

    async def cb1(events):  # noqa: ANN001
        trig1.append(_get_seq(events[0].payload))

    await bus1.register_callback(
        event_type="Comms",
        callback=cb1,
        every_n=3,
    )

    # Publish first three → one trigger
    for seq in range(3):
        await bus1.publish(Event(type="Comms", payload=CommsPayload(seq=seq)))
    bus1.join_published()
    await asyncio.sleep(0.05)
    assert trig1 == [2]

    # ── second (fresh) EventBus instance ─────────────────────────────────────
    bus2 = EventBus()
    trig2: list[int] = []

    async def cb2(events):  # noqa: ANN001
        trig2.append(_get_seq(events[0].payload))

    # Re-register identical subscription → attaches to persisted state
    await bus2.register_callback(
        event_type="Comms",
        callback=cb2,
        every_n=3,
    )

    # Two more events: not enough yet
    for seq in range(3, 5):
        await bus2.publish(Event(type="Comms", payload=CommsPayload(seq=seq)))
    bus2.join_published()
    await asyncio.sleep(0.05)
    assert trig2 == []  # threshold not reached

    # Third additional event reaches "next 3" ⇒ second trigger
    await bus2.publish(Event(type="Comms", payload=CommsPayload(seq=5)))
    bus2.join_published()
    await asyncio.sleep(0.05)
    assert trig2 == [5]


# --------------------------------------------------------------------------- #
#  3. time-based trigger, single session (uses ToolLoop for isolation)        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_time_single_session() -> None:
    bus = EventBus()
    triggered: list[int] = []

    async def cb(events):  # noqa: ANN001
        triggered.append(_get_seq(events[0].payload))

    await bus.register_callback(
        event_type="ToolLoop",
        callback=cb,
        every_seconds=2,
    )

    # Three events with controlled timestamps:
    #   t0 → triggers immediately (empty baseline)
    #   t0+1s → no trigger
    #   t0+3s → Δ=3 ≥ 2s → second trigger
    await bus.publish(
        Event(
            type="ToolLoop",
            timestamp=ts(0),
            payload=ToolLoopPayload(
                kind=ToolLoopKind.REQUEST,
                message={"role": "user"},
                method="test",
                hierarchy=[],
                hierarchy_label="",
                seq=0,
            ),
        ),
    )
    await bus.publish(
        Event(
            type="ToolLoop",
            timestamp=ts(1),
            payload=ToolLoopPayload(
                kind=ToolLoopKind.REQUEST,
                message={"role": "user"},
                method="test",
                hierarchy=[],
                hierarchy_label="",
                seq=1,
            ),
        ),
    )
    await bus.publish(
        Event(
            type="ToolLoop",
            timestamp=ts(3),
            payload=ToolLoopPayload(
                kind=ToolLoopKind.REQUEST,
                message={"role": "user"},
                method="test",
                hierarchy=[],
                hierarchy_label="",
                seq=2,
            ),
        ),
    )
    bus.join_published()
    await asyncio.sleep(0.05)

    assert triggered == [0, 2]


# --------------------------------------------------------------------------- #
#  4. time-based trigger, across sessions (uses ToolLoop for isolation)       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_time_across_sessions() -> None:
    # ── initial run ──────────────────────────────────────────────────────────
    bus1 = EventBus()
    trig1: list[int] = []

    async def cb1(events):  # noqa: ANN001
        trig1.append(_get_seq(events[0].payload))

    await bus1.register_callback(
        event_type="ToolLoop",
        callback=cb1,
        every_seconds=2,
    )

    await bus1.publish(
        Event(
            type="ToolLoop",
            timestamp=ts(0),
            payload=ToolLoopPayload(
                kind=ToolLoopKind.REQUEST,
                message={"role": "user"},
                method="test",
                hierarchy=[],
                hierarchy_label="",
                seq=0,
            ),
        ),
    )  # triggers
    bus1.join_published()
    await asyncio.sleep(0.05)
    assert trig1 == [0]

    # ── new interpreter session ──────────────────────────────────────────────
    bus2 = EventBus()
    trig2: list[int] = []

    async def cb2(events):  # noqa: ANN001
        trig2.append(_get_seq(events[0].payload))

    await bus2.register_callback(
        event_type="ToolLoop",
        callback=cb2,
        every_seconds=2,
    )

    # Event within 1 s of last trigger → shouldn't fire
    await bus2.publish(
        Event(
            type="ToolLoop",
            timestamp=ts(1),
            payload=ToolLoopPayload(
                kind=ToolLoopKind.REQUEST,
                message={"role": "user"},
                method="test",
                hierarchy=[],
                hierarchy_label="",
                seq=1,
            ),
        ),
    )
    # Event 3 s later → should fire
    await bus2.publish(
        Event(
            type="ToolLoop",
            timestamp=ts(3),
            payload=ToolLoopPayload(
                kind=ToolLoopKind.REQUEST,
                message={"role": "user"},
                method="test",
                hierarchy=[],
                hierarchy_label="",
                seq=2,
            ),
        ),
    )
    bus2.join_published()
    await asyncio.sleep(0.05)
    assert trig2 == [2]


# --------------------------------------------------------------------------- #
#  5–8.  same four scenarios but with a payload-filter (uses ManagerMethod)   #
# --------------------------------------------------------------------------- #


def _mk_msg(sender: int, receiver: int, seq: int) -> Event:
    return Event(
        type="ManagerMethod",
        payload=ManagerMethodPayload(
            manager="Test",
            method="filter_test",
            sender_id=sender,
            receiver_ids=[receiver],
            seq=seq,
        ),
    )


FILTER = "evt.payload.get('sender_id') == 1 and evt.payload.get('receiver_ids') == [2]"


def _get_sender_id(payload) -> int:
    """Extract sender_id from payload (always a dict)."""
    return payload.get("sender_id")


@pytest.mark.asyncio
@_handle_project
async def test_frequency_single_session_with_filter() -> None:
    bus = EventBus()
    trig: list[int] = []

    async def cb(evts):  # noqa: ANN001
        trig.append(_get_seq(evts[0].payload))

    await bus.register_callback(
        event_type="ManagerMethod",
        callback=cb,
        every_n=2,
        filter=FILTER,
    )

    # two matching + one non-matching + one matching  → triggers once
    await bus.publish(_mk_msg(1, 2, 0))  # match   (1/2)
    await bus.publish(_mk_msg(1, 2, 1))  # match   → trigger
    await bus.publish(_mk_msg(9, 9, 2))  # non-match
    await bus.publish(_mk_msg(1, 2, 3))  # match   (1/2 of next batch)

    bus.join_published()
    await asyncio.sleep(0.05)
    assert trig == [1]


@pytest.mark.asyncio
@_handle_project
async def test_frequency_across_sessions_with_filter() -> None:
    # Session 1
    bus1 = EventBus()
    await bus1.register_callback(
        event_type="ManagerMethod",
        callback=lambda _: None,  # dummy
        every_n=2,
        filter=FILTER,
    )
    await bus1.publish(_mk_msg(1, 2, 0))
    await bus1.publish(_mk_msg(1, 2, 1))  # triggers once
    bus1.join_published()
    await asyncio.sleep(0.05)

    # Session 2
    bus2 = EventBus()
    trig: list[int] = []

    async def cb(evts):  # noqa: ANN001
        trig.append(_get_seq(evts[0].payload))

    await bus2.register_callback(
        event_type="ManagerMethod",
        callback=cb,
        every_n=2,
        filter=FILTER,
    )

    await bus2.publish(_mk_msg(1, 2, 2))
    await bus2.publish(_mk_msg(1, 2, 3))  # second trigger
    bus2.join_published()
    await asyncio.sleep(0.05)
    assert trig == [3]


@pytest.mark.asyncio
@_handle_project
async def test_time_single_session_with_filter() -> None:
    bus = EventBus()
    trig: list[int] = []

    async def cb(evts):  # noqa: ANN001
        trig.append(_get_seq(evts[0].payload))

    await bus.register_callback(
        event_type="ManagerMethod",
        callback=cb,
        every_seconds=2,
        filter=FILTER,
    )

    await bus.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(0),
            payload=ManagerMethodPayload(
                manager="Test",
                method="filter_test",
                sender_id=1,
                receiver_ids=[2],
                seq=0,
            ),
        ),
    )  # trigger
    await bus.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(1),
            payload=ManagerMethodPayload(
                manager="Test",
                method="filter_test",
                sender_id=1,
                receiver_ids=[2],
                seq=1,
            ),
        ),
    )  # ignore (Δ=1)
    await bus.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(3),
            payload=ManagerMethodPayload(
                manager="Test",
                method="filter_test",
                sender_id=1,
                receiver_ids=[2],
                seq=2,
            ),
        ),
    )  # trigger (Δ=3)
    bus.join_published()
    await asyncio.sleep(0.05)
    assert trig == [0, 2]


@pytest.mark.asyncio
@_handle_project
async def test_time_across_sessions_with_filter() -> None:
    # First run
    bus1 = EventBus()
    await bus1.register_callback(
        event_type="ManagerMethod",
        callback=lambda _: None,
        every_seconds=2,
        filter=FILTER,
    )
    await bus1.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(0),
            payload=ManagerMethodPayload(
                manager="Test",
                method="filter_test",
                sender_id=1,
                receiver_ids=[2],
                seq=0,
            ),
        ),
    )  # trigger
    bus1.join_published()
    await asyncio.sleep(0.05)

    # Second run
    bus2 = EventBus()
    trig: list[int] = []

    async def cb(evts):  # noqa: ANN001
        trig.append(_get_seq(evts[0].payload))

    await bus2.register_callback(
        event_type="ManagerMethod",
        callback=cb,
        every_seconds=2,
        filter=FILTER,
    )

    await bus2.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(1),
            payload=ManagerMethodPayload(
                manager="Test",
                method="filter_test",
                sender_id=1,
                receiver_ids=[2],
                seq=1,
            ),
        ),
    )  # too soon
    await bus2.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(3),
            payload=ManagerMethodPayload(
                manager="Test",
                method="filter_test",
                sender_id=1,
                receiver_ids=[2],
                seq=2,
            ),
        ),
    )  # Δ=3 ⇒ trigger
    bus2.join_published()
    await asyncio.sleep(0.05)
    assert trig == [2]
