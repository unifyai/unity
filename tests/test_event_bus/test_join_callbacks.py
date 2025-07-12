import asyncio
import time

import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event


@pytest.mark.asyncio
@_handle_project
async def test_join_callbacks_waits_for_pending() -> None:
    """join_callbacks must block until callbacks already running are done."""

    bus = EventBus()
    done_evt = asyncio.Event()

    async def cb(_):  # noqa: ANN001
        # Artificial delay to keep the task in "running" state
        await asyncio.sleep(0.05)
        done_evt.set()

    # Trigger on every event of type Pending
    await bus.register_callback(
        event_type="Pending",
        callback=cb,
        every_n=1,
    )

    # Publish one event → schedules the callback above
    await bus.publish(Event(type="Pending", payload={}))
    bus.join_published()

    # `join_callbacks` is now a synchronous, blocking method – run it in a
    # background thread so we can await it without blocking the event-loop.
    join_task = asyncio.create_task(asyncio.to_thread(bus.join_callbacks))

    # Shortly after starting, the callback should still be running
    await asyncio.sleep(0.01)
    assert not done_evt.is_set(), "Callback already finished unexpectedly fast"
    assert not join_task.done(), "join_callbacks returned before callback finished"

    # Now wait for join to return – it should only do so *after* cb finished
    await join_task
    assert done_evt.is_set(), "Callback did not finish before join_callbacks returned"


@pytest.mark.asyncio
@_handle_project
async def test_join_callbacks_ignores_future_callbacks() -> None:
    """join_callbacks must only wait for callbacks that were *already* pending at invocation."""

    bus = EventBus()
    done_first = asyncio.Event()
    done_second = asyncio.Event()

    async def cb(evts):  # noqa: ANN001
        seq = evts[0].payload.get("seq")
        if seq == 1:
            await asyncio.sleep(0.05)  # short – should be awaited by join
            done_first.set()
        else:
            await asyncio.sleep(0.2)  # long – should *not* block join
            done_second.set()

    await bus.register_callback(
        event_type="Future",
        callback=cb,
        every_n=1,
    )

    # -- first event (seq=1) -------------------------------------------------
    await bus.publish(Event(type="Future", payload={"seq": 1}))
    bus.join_published()

    # Invoke join_callbacks while first callback is still running
    t0 = time.perf_counter()
    # `join_callbacks` blocks synchronously; execute in a background thread
    join_task = asyncio.create_task(asyncio.to_thread(bus.join_callbacks))

    # Give join_callbacks a chance to capture current tasks
    await asyncio.sleep(0.01)

    # -- second event (seq=2) ------------------------------------------------
    await bus.publish(Event(type="Future", payload={"seq": 2}))
    bus.join_published()

    # Wait for join – should not wait for the second long callback
    await join_task

    # Verify callbacks state
    assert (
        done_first.is_set()
    ), "First callback did not finish before join_callbacks returned"
    # Second callback should still be pending at this moment
    assert (
        not done_second.is_set()
    ), "join_callbacks incorrectly waited for a callback started after its invocation"

    # Allow the second callback to finish to keep the loop clean
    await done_second.wait()
