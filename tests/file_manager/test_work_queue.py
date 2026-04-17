from __future__ import annotations

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
