from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event
import unity.memory_manager.memory_manager as mm_mod
import unity.events.manager_event_logging as mel


# --------------------------------------------------------------------------- #
# deterministic timestamp helper                                              #
# --------------------------------------------------------------------------- #


def ts(offset: int) -> str:
    """T0 is 2025-04-01 00:00:00Z; *offset* is seconds."""
    return (
        dt.datetime(2025, 4, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=offset)
    ).isoformat()


# --------------------------------------------------------------------------- #
# reusable publish helper                                                     #
# --------------------------------------------------------------------------- #


async def _emit_events(bus: EventBus, count: int, start_seq: int = 0) -> None:
    """
    Publish *count* synthetic ManagerMethod events (ContactManager → ask).

    Each event is timestamped 1 s apart so we exercise the count-windows
    without ever touching the time-windows.
    """
    for seq in range(start_seq, start_seq + count):
        await bus.publish(
            Event(
                type="ManagerMethod",
                timestamp=ts(seq),
                payload={
                    "manager": "ContactManager",
                    "method": "ask",
                    "phase": "outgoing",
                    "seq": seq,
                },
            ),
        )
    bus.join_published()
    # give callbacks a chance to run
    await asyncio.sleep(0.05)


# --------------------------------------------------------------------------- #
#  main test                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_rolling_activity_across_sessions(monkeypatch) -> None:
    # ───────────────────────────  Session 1  ────────────────────────────
    bus1 = EventBus()
    # Make *this* bus the global one used by MemoryManager + event logging
    monkeypatch.setattr(mm_mod, "EVENT_BUS", bus1, raising=True)
    monkeypatch.setattr(mel, "EVENT_BUS", bus1, raising=False)

    mm1 = mm_mod.MemoryManager()  # registers callbacks
    await asyncio.sleep(0.05)  # wait for registration

    # 1️⃣  nine events → ONLY "Past Interaction" should exist
    await _emit_events(bus1, 9)
    txt = mm1.get_rolling_activity(mode="interaction")
    assert "Past Interaction" in txt
    assert "Past 10 Interactions" not in txt

    # 2️⃣  tenth event crosses the threshold → "Past 10 Interactions" appears
    await _emit_events(bus1, 1, start_seq=9)
    txt = mm1.get_rolling_activity(mode="interaction")
    assert "Past 10 Interactions" in txt
    assert "Past 40 Interactions" not in txt

    # ───────────────────────────  Session 2  ────────────────────────────
    #
    # Fresh EventBus + MemoryManager – simulates a full process restart.
    #
    bus2 = EventBus()
    monkeypatch.setattr(mm_mod, "EVENT_BUS", bus2, raising=True)
    monkeypatch.setattr(mel, "EVENT_BUS", bus2, raising=False)

    mm2 = mm_mod.MemoryManager()
    await asyncio.sleep(0.05)

    # Existing summaries must still be visible after the "restart".
    txt = mm2.get_rolling_activity(mode="interaction")
    assert "Past 10 Interactions" in txt
    assert "Past 40 Interactions" not in txt

    # 3️⃣  produce 40 more events → triggers "Past 40 Interactions"
    await _emit_events(bus2, 40, start_seq=10)
    txt = mm2.get_rolling_activity(mode="interaction")
    assert "Past 40 Interactions" in txt
