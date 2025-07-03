"""
Verifies that MemoryManager automatically keeps the *RollingActivity*
context up-to-date via the EventBus callback mechanism.

We trigger two thresholds for **ContactManager**:

• `past_interaction`        – fires on *every* event (every_n = 1)
• `past_10_interactions`    – fires after 10 events (every_n = 10)

For each trigger we assert that

1.  a new row is inserted into *RollingActivity*, and
2.  the relevant column contains the (stubbed) LLM summary.
"""

from __future__ import annotations

import asyncio
import pytest
import unify

from tests.helpers import _handle_project

from unity.memory_manager.memory_manager import MemoryManager
from unity.events.event_bus import Event, EVENT_BUS


# --------------------------------------------------------------------------- #
#  Dummy LLM – avoids network traffic in unit-tests                           #
# --------------------------------------------------------------------------- #
class _DummyLLM:  # noqa: D101 – tiny stub
    def __init__(self, *_, **__):
        pass

    def set_system_message(self, *_):
        pass

    async def chat(self, _prompt: str) -> str:  # noqa: D401
        return "DUMMY-SUMMARY"


# --------------------------------------------------------------------------- #
#  test                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_rolling_activity_updates(monkeypatch) -> None:
    # Patch *every* AsyncUnify constructor with the stub
    monkeypatch.setattr(unify, "AsyncUnify", _DummyLLM, raising=True)

    # ── 1.  Instantiate MemoryManager → registers all callbacks ------------
    mm = MemoryManager()

    # Helper – publish *n* synthetic ContactManager events
    async def _publish(n: int) -> None:
        for seq in range(n):
            await EVENT_BUS.publish(
                Event(
                    type="ManagerMethod",
                    payload={
                        "manager": "ContactManager",
                        "method": "ask",
                        "phase": "outgoing",
                        "seq": seq,
                    },
                ),
            )
        EVENT_BUS.join_published()
        await asyncio.sleep(0.05)  # let callbacks run

    # ── 2.  First event → past_interaction should fire ---------------------
    await _publish(1)

    logs = unify.get_logs(context=mm._rolling_ctx, sorting={"row_id": "ascending"})
    assert len(logs) >= 1, "RollingActivity should contain at least one row"
    row1 = logs[-1].entries
    col_pi = "contact_manager/past_interaction"
    assert col_pi in row1 and row1[col_pi] == "DUMMY-SUMMARY"

    # ── 3.  Nine more events (total 10) → past_10_interactions ------------
    await _publish(9)

    logs = unify.get_logs(context=mm._rolling_ctx, sorting={"row_id": "ascending"})
    assert len(logs) >= 2, "A second row should be created after 10 events"
    row2 = logs[-1].entries
    col_p10 = "contact_manager/past_10_interactions"
    assert col_p10 in row2 and row2[col_p10] == "DUMMY-SUMMARY"
