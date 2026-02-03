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
from unity.events.types.comms import CommsPayload
from unity.events.types.manager_method import ManagerMethodPayload

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
async def test_int_offset_int_limit() -> None:
    """
    With both parameters as scalars, ``search`` must apply the window **after**
    merging all event-types into one timeline.

    Timeline (newest last):

        ts0 ─ Comms          ← oldest
        ts1 ─ ManagerMethod
        ts2 ─ Comms
        ts3 ─ ManagerMethod  ← newest

    With ``offset=1`` we skip *ts3/ManagerMethod* and then return the next two
    newest events overall – *ts2/Comms* followed by *ts1/ManagerMethod*.
    """

    bus = EventBus()

    await bus.publish(
        Event(type="Comms", timestamp=ts(0), payload=CommsPayload(idx="c0")),
        blocking=True,
    )
    await bus.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(1),
            payload=ManagerMethodPayload(manager="Test", method="test", idx="m0"),
        ),
        blocking=True,
    )
    await bus.publish(
        Event(type="Comms", timestamp=ts(2), payload=CommsPayload(idx="c1")),
        blocking=True,
    )
    await bus.publish(
        Event(
            type="ManagerMethod",
            timestamp=ts(3),
            payload=ManagerMethodPayload(manager="Test", method="test", idx="m1"),
        ),
        blocking=True,
    )

    out = await bus.search(offset=1, limit=2)
    ids = [e.payload.get("idx") for e in out]

    # Newest-first across the combined stream after applying the global offset
    assert ids == ["c1", "m0"]


# --------------------------------------------------------------------------- #
#  2. offset =dict, limit =dict  → per-type window                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_dict_offset_dict_limit() -> None:
    """
    Independent offset/limit per event-type.

    * Type **Comms**:         3 events (seq 0,1,2) – skip 1 newest → keep 1 & 0
    * Type **ManagerMethod**: 3 events (seq 0,1,2) – keep newest two → 2 & 1
    """

    bus = EventBus()

    # three Comms events
    for i in range(3):
        await bus.publish(
            Event(type="Comms", timestamp=ts(i), payload=CommsPayload(seq=i)),
            blocking=True,
        )
    # three ManagerMethod events, later timestamps
    for i in range(3):
        await bus.publish(
            Event(
                type="ManagerMethod",
                timestamp=ts(10 + i),
                payload=ManagerMethodPayload(manager="Test", method="test", seq=i),
            ),
            blocking=True,
        )

    out = await bus.search(
        offset={"Comms": 1, "ManagerMethod": 0},
        limit={"Comms": 2, "ManagerMethod": 2},
        grouped_by_type=True,
    )

    assert set(out) == {"Comms", "ManagerMethod"}
    assert [e.payload.get("seq") for e in out["Comms"]] == [1, 0]
    assert [e.payload.get("seq") for e in out["ManagerMethod"]] == [2, 1]


# --------------------------------------------------------------------------- #
#  3. offset =dict, limit =int  → per-type offset + shared limit               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_dict_offset_int_limit() -> None:
    """
    Only *offset* varies per type; *limit* caps the **total** number returned.

    We publish five events:

        ts0  Comms seq0
        ts1  ManagerMethod seq0
        ts2  Comms seq1
        ts3  ManagerMethod seq1
        ts4  Comms seq2   ← newest Comms (to be skipped for Comms)

    With ``offset={'Comms': 1}`` and ``limit=4`` we expect:

        ts3 ManagerMethod-1, ts2 Comms-1, ts1 ManagerMethod-0, ts0 Comms-0
    """

    bus = EventBus()

    events = [
        ("Comms", 0),
        ("ManagerMethod", 0),
        ("Comms", 1),
        ("ManagerMethod", 1),
        ("Comms", 2),
    ]
    for idx, (typ, seq) in enumerate(events):
        if typ == "Comms":
            payload = CommsPayload(seq=seq)
        else:
            payload = ManagerMethodPayload(manager="Test", method="test", seq=seq)
        await bus.publish(
            Event(type=typ, timestamp=ts(idx), payload=payload),
            blocking=True,
        )

    out = await bus.search(offset={"Comms": 1}, limit=4)
    typ_seq = [(e.type, e.payload.get("seq")) for e in out]

    assert typ_seq == [
        ("ManagerMethod", 1),
        ("Comms", 1),
        ("ManagerMethod", 0),
        ("Comms", 0),
    ]


# --------------------------------------------------------------------------- #
#  4. offset =int, limit =dict  → shared offset + per-type limit               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_int_offset_dict_limit() -> None:
    """
    Scalar *offset* applies **inside each type**, because *limit* is a dict.

    Publish three events per type (seq 0,1,2).  With ``offset=1`` we discard the
    newest of each type.  ``limit={'Comms': 2, 'ManagerMethod': 2}`` then returns
    the next two per type, grouped by type.
    """

    bus = EventBus()

    for i in range(3):
        await bus.publish(
            Event(type="Comms", timestamp=ts(i), payload=CommsPayload(seq=i)),
            blocking=True,
        )
    for i in range(3):
        await bus.publish(
            Event(
                type="ManagerMethod",
                timestamp=ts(10 + i),
                payload=ManagerMethodPayload(manager="Test", method="test", seq=i),
            ),
            blocking=True,
        )

    out = await bus.search(
        offset=1,
        limit={"Comms": 2, "ManagerMethod": 2},
        grouped_by_type=True,
    )

    assert set(out) == {"Comms", "ManagerMethod"}
    # newest-first within each list, after skipping one
    assert [e.payload.get("seq") for e in out["Comms"]] == [1, 0]
    assert [e.payload.get("seq") for e in out["ManagerMethod"]] == [1, 0]
