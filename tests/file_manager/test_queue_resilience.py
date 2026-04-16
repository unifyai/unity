"""Queue resilience tests for InMemoryWorkQueue and LocalQueueWorker.

Validates consecutive retry escalation, dead-lettering after max attempts,
topic isolation, and concurrent receive behavior.
"""

from __future__ import annotations

import asyncio

import pytest

from unity.file_manager.pipeline import (
    InMemoryWorkQueue,
    LocalQueueWorker,
    RetryWorkItem,
)


@pytest.mark.asyncio
async def test_consecutive_retries_escalate_attempt_count():
    """A message retried N times should have attempt == N on the next receive."""
    queue = InMemoryWorkQueue()
    await queue.publish(topic="parse", payload={"file": "big.csv"})

    for expected_attempt in range(5):
        items = await queue.receive(topics=["parse"])
        assert len(items) == 1
        assert items[0].attempt == expected_attempt
        await queue.retry(items[0].receipt_id, error=f"fail #{expected_attempt}")

    final = (await queue.receive(topics=["parse"]))[0]
    assert final.attempt == 5
    assert final.last_error == "fail #4"
    await queue.ack(final.receipt_id)


@pytest.mark.asyncio
async def test_worker_dead_letters_after_persistent_failures():
    """A handler that always raises should dead-letter the message on first attempt."""
    queue = InMemoryWorkQueue()
    await queue.publish(topic="ingest", payload={"table": "repairs"})

    async def _always_fail(item):
        raise RuntimeError("persistent DB error")

    worker = LocalQueueWorker(queue=queue, handler=_always_fail)
    processed = await worker.run_once(max_messages=1, topics=["ingest"])

    assert processed == 1
    assert len(queue.dead_letters) == 1
    assert queue.dead_letters[0].payload["table"] == "repairs"
    assert "persistent DB error" in queue.dead_letters[0].last_error


@pytest.mark.asyncio
async def test_worker_retry_then_succeed_pattern():
    """A handler that fails once then succeeds should retry and eventually ack."""
    queue = InMemoryWorkQueue()
    await queue.publish(topic="parse", payload={"file": "data.xlsx"})
    call_count = 0

    async def _fail_once(item):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RetryWorkItem("transient network error", delay_seconds=0.0)

    worker = LocalQueueWorker(queue=queue, handler=_fail_once)

    await worker.run_once(max_messages=1, topics=["parse"])
    assert call_count == 1

    await worker.run_once(max_messages=1, topics=["parse"])
    assert call_count == 2

    remaining = await queue.receive(topics=["parse"])
    assert remaining == []


@pytest.mark.asyncio
async def test_topic_isolation():
    """Messages published to different topics are not visible to each other."""
    queue = InMemoryWorkQueue()
    await queue.publish(topic="parse", payload={"type": "parse"})
    await queue.publish(topic="ingest", payload={"type": "ingest"})

    parse_items = await queue.receive(topics=["parse"], max_messages=10)
    assert len(parse_items) == 1
    assert parse_items[0].payload["type"] == "parse"

    ingest_items = await queue.receive(topics=["ingest"], max_messages=10)
    assert len(ingest_items) == 1
    assert ingest_items[0].payload["type"] == "ingest"

    empty = await queue.receive(topics=["parse"])
    assert empty == []


@pytest.mark.asyncio
async def test_receive_respects_max_messages():
    """receive(max_messages=N) returns at most N items even if more are available."""
    queue = InMemoryWorkQueue()
    for i in range(10):
        await queue.publish(topic="work", payload={"i": i})

    batch = await queue.receive(topics=["work"], max_messages=3)
    assert len(batch) == 3


@pytest.mark.asyncio
async def test_receive_returns_empty_on_no_topics():
    """Receiving from a topic with no published messages returns empty."""
    queue = InMemoryWorkQueue()
    result = await queue.receive(topics=["nonexistent"])
    assert result == []


@pytest.mark.asyncio
async def test_dead_letter_preserves_original_payload():
    """Dead-lettered items retain the original message payload and metadata."""
    queue = InMemoryWorkQueue()
    msg_id = await queue.publish(
        topic="parse",
        payload={"file": "huge.csv", "size_mb": 500},
    )

    item = (await queue.receive(topics=["parse"]))[0]
    assert item.message_id == msg_id

    await queue.dead_letter(item.receipt_id, error="OOM killed")

    assert len(queue.dead_letters) == 1
    dl = queue.dead_letters[0]
    assert dl.message_id == msg_id
    assert dl.payload == {"file": "huge.csv", "size_mb": 500}
    assert dl.last_error == "OOM killed"
    assert dl.state == "dead_lettered"
    assert dl.dead_lettered_at is not None


@pytest.mark.asyncio
async def test_concurrent_publish_and_receive():
    """Multiple coroutines can publish and receive without data loss."""
    queue = InMemoryWorkQueue()
    n = 50

    async def _publish(i: int):
        await queue.publish(topic="work", payload={"i": i})

    await asyncio.gather(*[_publish(i) for i in range(n)])

    received_ids = set()
    while True:
        items = await queue.receive(topics=["work"], max_messages=10)
        if not items:
            break
        for item in items:
            received_ids.add(item.payload["i"])
            await queue.ack(item.receipt_id)

    assert received_ids == set(range(n))
