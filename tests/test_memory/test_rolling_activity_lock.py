import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_rolling_activity_lock(monkeypatch):
    """Ensure that only one coroutine at a time updates RollingActivity."""
    from unity.memory_manager.memory_manager import MemoryManager

    # ── 1.  Stub heavy methods to avoid side-effects / external I/O
    async def _noop(self, *_, **__):
        return None

    monkeypatch.setattr(MemoryManager, "_setup_rolling_callbacks", _noop, raising=True)
    monkeypatch.setattr(
        MemoryManager,
        "_ensure_rolling_context",
        classmethod(lambda cls: "ctx"),
        raising=True,
    )

    # ── 2.  Instantiate the manager _after_ we patched the helpers
    mm = MemoryManager()

    # ── 3.  Instrument the *body* method to detect overlap
    started: list[float] = []
    finished: list[float] = []
    row_ids: list[int] = []

    async def fake_body(self, column, events):  # noqa: ARG001 – unused params
        started.append(time.monotonic())
        # Assign a pseudo row_id in *call* order (serialised by the lock)
        row_ids.append(len(row_ids))
        # Give the scheduler a chance to interleave if the lock were absent
        await asyncio.sleep(0.05)
        finished.append(time.monotonic())

    monkeypatch.setattr(
        MemoryManager,
        "_record_rolling_activity_body",
        fake_body,
        raising=True,
    )

    # ── 4.  Kick off *concurrent* updates --------------------------------
    tasks = [
        asyncio.create_task(
            mm._record_rolling_activity("contact_manager/past_day", []),
        ),
        asyncio.create_task(
            mm._record_rolling_activity("contact_manager/past_week", []),
        ),
    ]
    await asyncio.gather(*tasks)

    # ── 5.  Validate serial execution ------------------------------------
    assert len(started) == 2 == len(finished)
    # The second coroutine must start *after* the first one has finished
    assert (
        started[1] >= finished[0]
    ), "RollingActivity updates overlapped – lock not enforced"

    # ── 6.  Verify row_id sequencing -------------------------------------
    assert row_ids == [0, 1], "row_id sequence incorrect – concurrent writes detected"
