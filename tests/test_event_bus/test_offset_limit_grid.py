# Exhaustive coverage of the 2 × 2 grid of ``offset``/``limit`` type-combinations
# accepted by :pymeth:`unity.events.event_bus.EventBus.search`.  The table below
# comes straight from the updated docstring; every row now has a dedicated test.
#
# ┌──────────────┬──────────────┬────────────────────────────────────────────┐
# │ ``offset``   │ ``limit``    │ Semantics verified by this file            │
# ├──────────────┼──────────────┼────────────────────────────────────────────┤
# │ int          │ int          │ global window (combined event stream)      │
# │ dict         │ dict         │ per-type window (both values independent)  │
# │ dict         │ int          │ per-type *offset* + shared *limit*         │
# │ int          │ dict         │ shared *offset* + per-type *limit*         │
# └──────────────┴──────────────┴────────────────────────────────────────────┘

from __future__ import annotations

import datetime as dt

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event


# --------------------------------------------------------------------------- #
#  deterministic timestamps                                                   #
# --------------------------------------------------------------------------- #


def ts(i: int) -> str:
    """Monotonic, deterministic timestamps in UTC."""
    return (
        dt.datetime(2025, 2, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=i)
    ).isoformat()


# --------------------------------------------------------------------------- #
#  1. offset =int, limit =int  → global window                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_offset_int_limit_int_global_window() -> None:
    """
    With both parameters as scalars, ``search`` must apply the window **after**
    merging all event-types into one timeline.

    Timeline (newest last):

        ts0 ─ A          ← oldest
        ts1 ─ B
        ts2 ─ A
        ts3 ─ B          ← newest

    With ``offset=1`` we skip *ts3/B* and then return the next two newest
    events overall – *ts2/A* followed by *ts1/B*.
    """

    bus = EventBus()

    await bus.publish(
        Event(type="A", timestamp=ts(0), payload={"idx": "a0"}),
        blocking=True,
    )
    await bus.publish(
        Event(type="B", timestamp=ts(1), payload={"idx": "b0"}),
        blocking=True,
    )
    await bus.publish(
        Event(type="A", timestamp=ts(2), payload={"idx": "a1"}),
        blocking=True,
    )
    await bus.publish(
        Event(type="B", timestamp=ts(3), payload={"idx": "b1"}),
        blocking=True,
    )

    out = await bus.search(offset=1, limit=2)
    ids = [e.payload["idx"] for e in out]

    # Newest-first across the combined stream after applying the global offset
    assert ids == ["a1", "b0"]


# --------------------------------------------------------------------------- #
#  2. offset =dict, limit =dict  → per-type window                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_offset_dict_limit_dict_per_type_window() -> None:
    """
    Independent offset/limit per event-type.

    * Type **A**:         3 events (seq 0,1,2) – skip 1 newest → keep 1 & 0
    * Type **B**:         3 events (seq 0,1,2) – keep newest two → 2 & 1
    """

    bus = EventBus()

    # three A-events
    for i in range(3):
        await bus.publish(
            Event(type="A", timestamp=ts(i), payload={"seq": i}),
            blocking=True,
        )
    # three B-events, later timestamps so they don't interfere with A ordering
    for i in range(3):
        await bus.publish(
            Event(type="B", timestamp=ts(10 + i), payload={"seq": i}),
            blocking=True,
        )

    out = await bus.search(
        offset={"A": 1, "B": 0},
        limit={"A": 2, "B": 2},
        grouped_by_type=True,
    )

    assert set(out) == {"A", "B"}
    assert [e.payload["seq"] for e in out["A"]] == [1, 0]
    assert [e.payload["seq"] for e in out["B"]] == [2, 1]


# --------------------------------------------------------------------------- #
#  3. offset =dict, limit =int  → per-type offset + shared limit               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_offset_dict_limit_int_mixed_window() -> None:
    """
    Only *offset* varies per type; *limit* caps the **total** number returned.

    We publish five events:

        ts0  A seq0
        ts1  B seq0
        ts2  A seq1
        ts3  B seq1
        ts4  A seq2   ← newest A (to be skipped for A)

    With ``offset={'A': 1}`` and ``limit=4`` we expect:

        ts3 B-1, ts2 A-1, ts1 B-0, ts0 A-0
    """

    bus = EventBus()

    events = [
        ("A", 0),
        ("B", 0),
        ("A", 1),
        ("B", 1),
        ("A", 2),
    ]
    for idx, (typ, seq) in enumerate(events):
        await bus.publish(
            Event(type=typ, timestamp=ts(idx), payload={"seq": seq}),
            blocking=True,
        )

    out = await bus.search(offset={"A": 1}, limit=4)
    typ_seq = [(e.type, e.payload["seq"]) for e in out]

    assert typ_seq == [("B", 1), ("A", 1), ("B", 0), ("A", 0)]


# --------------------------------------------------------------------------- #
#  4. offset =int, limit =dict  → shared offset + per-type limit               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_offset_int_limit_dict_mixed_window() -> None:
    """
    Scalar *offset* applies **inside each type**, because *limit* is a dict.

    Publish three events per type (seq 0,1,2).  With ``offset=1`` we discard the
    newest of each type.  ``limit={'A': 2, 'B': 2}`` then returns the next two
    per type, grouped by type.
    """

    bus = EventBus()

    for i in range(3):
        await bus.publish(
            Event(type="A", timestamp=ts(i), payload={"seq": i}),
            blocking=True,
        )
    for i in range(3):
        await bus.publish(
            Event(type="B", timestamp=ts(10 + i), payload={"seq": i}),
            blocking=True,
        )

    out = await bus.search(
        offset=1,
        limit={"A": 2, "B": 2},
        grouped_by_type=True,
    )

    assert set(out) == {"A", "B"}
    # newest-first within each list, after skipping one
    assert [e.payload["seq"] for e in out["A"]] == [1, 0]
    assert [e.payload["seq"] for e in out["B"]] == [1, 0]
