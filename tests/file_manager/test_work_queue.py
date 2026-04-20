from __future__ import annotations

import asyncio
import time

import pytest

from unity.common.pipeline import (
    InMemoryWorkQueue,
    LocalQueueWorker,
    RetryWorkItem,
)


@pytest.mark.asyncio
async def test_in_memory_work_queue_round_trips_and_acks_messages():
    queue = InMemoryWorkQueue()
    message_id = await queue.publish(
        topic="parse",
        payload={"file_path": "repairs.csv"},
    )

    received = await queue.receive(topics=["parse"])

    assert len(received) == 1
    assert received[0].message_id == message_id
    assert received[0].payload["file_path"] == "repairs.csv"
    await queue.ack(received[0].receipt_id)
    assert await queue.receive(topics=["parse"]) == []


@pytest.mark.asyncio
async def test_in_memory_work_queue_retry_requeues_with_incremented_attempt():
    queue = InMemoryWorkQueue()
    await queue.publish(topic="parse", payload={"file_path": "repairs.csv"})

    first = (await queue.receive(topics=["parse"]))[0]
    await queue.retry(first.receipt_id, error="temporary failure")

    second = (await queue.receive(topics=["parse"]))[0]
    assert second.attempt == 1
    assert second.last_error == "temporary failure"


@pytest.mark.asyncio
async def test_local_queue_worker_acks_retries_and_dead_letters():
    queue = InMemoryWorkQueue()
    await queue.publish(topic="parse", payload={"id": "ok"})
    await queue.publish(topic="parse", payload={"id": "retry"})
    await queue.publish(topic="parse", payload={"id": "dead"})
    attempts: dict[str, int] = {}

    async def _handler(item):
        item_id = item.payload["id"]
        attempts[item_id] = attempts.get(item_id, 0) + 1
        if item_id == "retry" and attempts[item_id] == 1:
            raise RetryWorkItem("transient", delay_seconds=0.0)
        if item_id == "dead":
            raise RuntimeError("fatal")

    worker = LocalQueueWorker(queue=queue, handler=_handler)

    assert await worker.run_once(max_messages=3, topics=["parse"]) == 3
    assert await worker.run_once(max_messages=3, topics=["parse"]) == 1
    assert attempts == {"ok": 1, "retry": 2, "dead": 1}
    assert len(queue.dead_letters) == 1
    assert queue.dead_letters[0].payload["id"] == "dead"
    assert queue.dead_letters[0].last_error == "fatal"


@pytest.mark.asyncio
async def test_in_memory_work_queue_close_cancels_pending_requeues():
    """close() must cancel delayed-requeue tasks so a worker can exit cleanly.

    When a message is retried with a non-zero ``delay_seconds``, the
    queue schedules a background task to re-enqueue it after the delay.
    A graceful worker shutdown must be able to cancel those pending
    tasks without waiting for the full delay to elapse; otherwise the
    worker process would block on stray background tasks during exit.
    """
    queue = InMemoryWorkQueue()
    await queue.publish(topic="parse", payload={"id": "slow-retry"})
    item = (await queue.receive(topics=["parse"]))[0]

    await queue.retry(item.receipt_id, error="transient", delay_seconds=60.0)
    await asyncio.sleep(0)  # let the requeue task register
    pending = [t for t in queue._background_tasks if not t.done()]
    assert len(pending) == 1, "expected one scheduled requeue task"

    started = time.monotonic()
    await queue.close()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"close() should be near-instant, took {elapsed}s"
    assert len(queue._background_tasks) == 0, "background tasks should be cleared"
    assert all(t.done() for t in pending), "pending tasks should be done/cancelled"
