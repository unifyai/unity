"""
tests/event_bus/test_periodic_flush_lifecycle.py
================================================

The periodic flusher must survive event-loop replacement, and an embedded
teardown must persist what is still buffered.

The failure being locked out: the flusher is a task on whatever loop first
touched the bus, and the bus is a process-lifetime singleton. An embedder
that replaces its event loop mid-process (an in-process CM reboot between
benchmark weeks) stranded the flusher on the dead loop; every later publish
buffered into ``_pending_writes`` forever, no Comms row reached Orchestra,
and the next boot's hydration truthfully found nothing to restore.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from tests.helpers import _handle_project
from unify.events.event_bus import Event, EventBus
from unify.transcript_manager.types.message import Message


async def _arm(bus: EventBus) -> None:
    bus._ensure_periodic_flush_task()


def _make_event() -> Event:
    return Event(
        type="Message",
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        payload=Message.model_construct(),
    )


@_handle_project
def test_flusher_rebinds_after_its_loop_dies():
    bus = EventBus()

    loop_a = asyncio.new_event_loop()
    loop_a.run_until_complete(_arm(bus))
    task_a = bus._periodic_flush_task
    assert task_a is not None
    assert task_a.get_loop() is loop_a
    loop_a.close()

    # The old loop is dead: the next touch on a new loop must replace the
    # stranded flusher rather than trusting the not-None task object.
    loop_b = asyncio.new_event_loop()
    try:
        loop_b.run_until_complete(_arm(bus))
        task_b = bus._periodic_flush_task
        assert task_b is not task_a
        assert task_b.get_loop() is loop_b

        # A live flusher is left alone — no churn on every publish.
        loop_b.run_until_complete(_arm(bus))
        assert bus._periodic_flush_task is task_b
    finally:
        loop_b.close()


@_handle_project
def test_publish_revives_flusher_and_keeps_buffering_on_new_loop():
    """Publishing after a loop replacement re-arms the flusher in place.

    The buffer itself is loop-independent, so rows published before and
    after the replacement must both still be pending — nothing is lost,
    and the revived flusher is what will drain them.
    """
    bus = EventBus()
    # Prefill already resolved: publish must not kick backend hydration here.
    bus._prefill_done.set()

    loop_a = asyncio.new_event_loop()
    loop_a.run_until_complete(bus.publish(_make_event()))
    task_a = bus._periodic_flush_task
    assert task_a is not None
    pending_after_first = len(bus._pending_writes)
    assert pending_after_first >= 1
    loop_a.close()

    loop_b = asyncio.new_event_loop()
    try:
        loop_b.run_until_complete(bus.publish(_make_event()))
        assert bus._periodic_flush_task is not task_a
        assert bus._periodic_flush_task.get_loop() is loop_b
        assert len(bus._pending_writes) == pending_after_first + 1
    finally:
        loop_b.close()


@pytest.mark.asyncio
@_handle_project
async def test_comms_row_lands_in_orchestra_after_flush():
    """A published Comms event reaches the backend once flushed.

    This is the write path hydration depends on end-to-end: publish
    buffers the row, flush writes it through ``unisdk.create_logs``, and a
    plain read of the bus's own Comms context sees it. If this holds and a
    deployment still hydrates nothing, the gap is in that deployment's
    lifecycle (a stranded flusher, a teardown that never flushed), not in
    the write path itself.
    """
    import unisdk

    from unify.conversation_manager.events import SMSReceived

    bus = EventBus()
    ev = SMSReceived(
        contact={"contact_id": 2, "first_name": "Alice", "surname": "Smith"},
        content="persist probe",
    ).to_bus_event()

    await bus.publish(ev)
    assert any(
        entries.get("event_id") == ev.event_id for entries, _ctx in bus._pending_writes
    ), "publish must buffer the Comms row for Orchestra persistence"

    bus.flush()
    assert bus._pending_writes == []

    rows = unisdk.get_logs(context=bus._specific_ctxs["Comms"], limit=100)
    assert any(
        (log.entries or {}).get("event_id") == ev.event_id for log in rows
    ), "the flushed Comms row must be readable from the backend"


@pytest.mark.asyncio
@_handle_project
async def test_stop_async_flushes_pending_writes(monkeypatch):
    """An embedded teardown persists the buffer.

    The pod-exit path (main.py) flushes before its hard exit; lifecycles
    that stop through ``stop_async`` — sandboxes, tests, benchmark
    harnesses — must flush too, or the rows a later boot's hydration needs
    die with the process.
    """
    from unittest.mock import AsyncMock, MagicMock

    import unify.conversation_manager as cm_pkg

    bus = MagicMock()
    monkeypatch.setattr("unify.events.event_bus.EVENT_BUS", bus)

    stub_cm = MagicMock()
    stub_cm.cleanup = AsyncMock()
    monkeypatch.setattr(cm_pkg, "_conversation_manager", stub_cm)

    await cm_pkg.stop_async(reason="test teardown")

    bus.flush.assert_called_once()
