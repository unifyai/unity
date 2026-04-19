from __future__ import annotations

import asyncio
import inspect
from datetime import timedelta
from typing import Any, Awaitable, Callable, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from ._utils import utc_now, utc_now_iso

__all__ = [
    "DeadLetterWorkItem",
    "InMemoryWorkQueue",
    "LocalQueueWorker",
    "ReceivedWorkItem",
    "RetryWorkItem",
    "WorkQueue",
    "WorkQueueMessage",
]

QueueMessageState = Literal["queued", "leased", "dead_lettered"]


class WorkQueueMessage(BaseModel):
    """Typed queue message used for parse/ingest orchestration."""

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    state: QueueMessageState = "queued"
    published_at: str = Field(default_factory=utc_now_iso)
    available_at: str = Field(default_factory=utc_now_iso)
    last_error: str | None = None


class ReceivedWorkItem(WorkQueueMessage):
    """Leased queue message returned to a worker."""

    receipt_id: str
    received_at: str = Field(default_factory=utc_now_iso)
    state: QueueMessageState = "leased"


class DeadLetterWorkItem(WorkQueueMessage):
    """Terminal queue record retained for operator inspection."""

    dead_lettered_at: str = Field(default_factory=utc_now_iso)
    state: QueueMessageState = "dead_lettered"


class WorkQueue(Protocol):
    """Port for queue-backed parse/ingest orchestration."""

    async def publish(self, *, topic: str, payload: dict[str, Any]) -> str: ...

    async def receive(
        self,
        *,
        max_messages: int = 1,
        topics: list[str] | None = None,
    ) -> list[ReceivedWorkItem]: ...

    async def ack(self, receipt_id: str) -> None: ...

    async def retry(
        self,
        receipt_id: str,
        *,
        error: str,
        delay_seconds: float = 0.0,
    ) -> None: ...

    async def dead_letter(self, receipt_id: str, *, error: str) -> None: ...

    async def is_cancelled(self, run_id: str) -> bool:
        """Check whether the job identified by *run_id* has been cancelled.

        Workers call this between stages to detect operator-initiated
        cancellation.  The default implementation returns ``False`` so
        that callers not using cancellation are unaffected.
        """
        ...


class RetryWorkItem(Exception):
    """Signal that a queue item should be retried instead of dead-lettered."""

    def __init__(self, message: str, *, delay_seconds: float = 0.0):
        super().__init__(message)
        self.delay_seconds = delay_seconds


class InMemoryWorkQueue:
    """Local in-memory work queue for development and tests."""

    def __init__(self):
        self._topic_queues: dict[str, asyncio.Queue[WorkQueueMessage]] = {}
        self._leased: dict[str, WorkQueueMessage] = {}
        self._dead_letters: list[DeadLetterWorkItem] = []
        self._cancelled: set[str] = set()
        self._lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def dead_letters(self) -> list[DeadLetterWorkItem]:
        return list(self._dead_letters)

    async def publish(self, *, topic: str, payload: dict[str, Any]) -> str:
        message = WorkQueueMessage(topic=topic, payload=dict(payload or {}))
        await self._queue_for_topic(topic).put(message)
        return message.message_id

    async def receive(
        self,
        *,
        max_messages: int = 1,
        topics: list[str] | None = None,
    ) -> list[ReceivedWorkItem]:
        leased: list[ReceivedWorkItem] = []

        async with self._lock:
            topic_names = (
                list(self._topic_queues.keys()) if topics is None else list(topics)
            )
            while len(leased) < max_messages:
                claimed = False
                for topic in topic_names:
                    queue = self._topic_queues.get(topic)
                    if queue is None:
                        continue
                    try:
                        message = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    receipt_id = uuid4().hex
                    leased_item = ReceivedWorkItem(
                        **message.model_dump(),
                        receipt_id=receipt_id,
                    )
                    self._leased[receipt_id] = message
                    leased.append(leased_item)
                    claimed = True
                    if len(leased) >= max_messages:
                        break
                if not claimed:
                    break
        return leased

    async def ack(self, receipt_id: str) -> None:
        async with self._lock:
            message = self._leased.pop(receipt_id, None)
            if message is None:
                return
            self._queue_for_topic(message.topic).task_done()

    async def retry(
        self,
        receipt_id: str,
        *,
        error: str,
        delay_seconds: float = 0.0,
    ) -> None:
        async with self._lock:
            message = self._leased.pop(receipt_id)
            message.attempt += 1
            message.last_error = error
            message.available_at = (
                utc_now() + timedelta(seconds=delay_seconds)
            ).isoformat()
            message.state = "queued"
            self._queue_for_topic(message.topic).task_done()

            if delay_seconds > 0:
                self._schedule_requeue(message, delay_seconds)
                return
            await self._queue_for_topic(message.topic).put(message)

    async def dead_letter(self, receipt_id: str, *, error: str) -> None:
        async with self._lock:
            message = self._leased.pop(receipt_id)
            dead_letter = DeadLetterWorkItem(
                **message.model_dump(exclude={"last_error", "state"}),
                last_error=error,
            )
            self._dead_letters.append(dead_letter)
            self._queue_for_topic(message.topic).task_done()

    async def cancel(self, run_id: str) -> None:
        """Mark *run_id* as cancelled so workers can detect it."""
        self._cancelled.add(run_id)

    async def is_cancelled(self, run_id: str) -> bool:
        return run_id in self._cancelled

    def _queue_for_topic(self, topic: str) -> asyncio.Queue[WorkQueueMessage]:
        queue = self._topic_queues.get(topic)
        if queue is None:
            queue = asyncio.Queue()
            self._topic_queues[topic] = queue
        return queue

    def _schedule_requeue(
        self,
        message: WorkQueueMessage,
        delay_seconds: float,
    ) -> None:
        task = asyncio.create_task(self._requeue_after_delay(message, delay_seconds))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _requeue_after_delay(
        self,
        message: WorkQueueMessage,
        delay_seconds: float,
    ) -> None:
        await asyncio.sleep(delay_seconds)
        await self._queue_for_topic(message.topic).put(message)


WorkItemHandler = Callable[[ReceivedWorkItem], Awaitable[Any] | Any]


class LocalQueueWorker:
    """Simple local worker that drains work from a WorkQueue implementation."""

    def __init__(self, *, queue: WorkQueue, handler: WorkItemHandler):
        self._queue = queue
        self._handler = handler

    async def run_once(
        self,
        *,
        max_messages: int = 1,
        topics: list[str] | None = None,
    ) -> int:
        items = await self._queue.receive(max_messages=max_messages, topics=topics)
        for item in items:
            try:
                maybe_awaitable = self._handler(item)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except RetryWorkItem as exc:
                await self._queue.retry(
                    item.receipt_id,
                    error=str(exc),
                    delay_seconds=exc.delay_seconds,
                )
            except Exception as exc:
                await self._queue.dead_letter(item.receipt_id, error=str(exc))
            else:
                await self._queue.ack(item.receipt_id)
        return len(items)
