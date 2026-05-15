from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class ActiveWorkSnapshot:
    active_count: int
    oldest_started_at: float | None
    newest_heartbeat_at: float | None
    oldest_elapsed_s: float
    works: tuple[dict[str, Any], ...]


@dataclass
class _ActiveWorkRecord:
    work_id: str
    label: str
    started_at: float
    last_heartbeat_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    last_user_notification_at: float | None = None
    last_fallback_notification_at: float | None = None
    fallback_notification_count: int = 0

    def to_public_dict(self, now: float) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "label": self.label,
            "started_at": self.started_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "elapsed_s": now - self.started_at,
            "last_user_notification_at": self.last_user_notification_at,
            "last_fallback_notification_at": self.last_fallback_notification_at,
            "fallback_notification_count": self.fallback_notification_count,
            "metadata": dict(self.metadata),
        }


class ActiveWorkHandle:
    def __init__(self, registry: "ActiveWorkRegistry", work_id: str) -> None:
        self._registry = registry
        self.work_id = work_id

    def heartbeat(self) -> None:
        self._registry.heartbeat(self.work_id)

    def record_user_notification(self) -> None:
        self._registry.record_user_notification(self.work_id)

    def fallback_notification_due(
        self,
        *,
        initial_delay_s: float,
        repeat_interval_s: float,
    ) -> bool:
        return self._registry.fallback_notification_due(
            self.work_id,
            initial_delay_s=initial_delay_s,
            repeat_interval_s=repeat_interval_s,
        )

    def record_fallback_notification(self) -> None:
        self._registry.record_fallback_notification(self.work_id)

    def end(self) -> None:
        self._registry.end(self.work_id)


class ActiveWorkRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[str, _ActiveWorkRecord] = {}

    def begin(
        self,
        *,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> ActiveWorkHandle:
        now = time.monotonic()
        work_id = uuid.uuid4().hex
        with self._lock:
            self._records[work_id] = _ActiveWorkRecord(
                work_id=work_id,
                label=label,
                started_at=now,
                last_heartbeat_at=now,
                metadata=dict(metadata or {}),
            )
        return ActiveWorkHandle(self, work_id)

    def heartbeat(self, work_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            record = self._records.get(work_id)
            if record is not None:
                record.last_heartbeat_at = now

    def record_user_notification(self, work_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            record = self._records.get(work_id)
            if record is not None:
                record.last_user_notification_at = now

    def fallback_notification_due(
        self,
        work_id: str,
        *,
        initial_delay_s: float,
        repeat_interval_s: float,
    ) -> bool:
        now = time.monotonic()
        with self._lock:
            record = self._records.get(work_id)
            if record is None:
                return False

            if record.last_fallback_notification_at is None:
                anchor = record.last_user_notification_at or record.started_at
                return now - anchor >= initial_delay_s

            anchor = max(
                record.last_fallback_notification_at,
                record.last_user_notification_at or record.started_at,
            )
            return now - anchor >= repeat_interval_s

    def record_fallback_notification(self, work_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            record = self._records.get(work_id)
            if record is not None:
                record.last_fallback_notification_at = now
                record.fallback_notification_count += 1

    def end(self, work_id: str) -> None:
        with self._lock:
            self._records.pop(work_id, None)

    def snapshot(self) -> ActiveWorkSnapshot:
        now = time.monotonic()
        with self._lock:
            records = list(self._records.values())
            if not records:
                return ActiveWorkSnapshot(
                    active_count=0,
                    oldest_started_at=None,
                    newest_heartbeat_at=None,
                    oldest_elapsed_s=0.0,
                    works=(),
                )

            oldest_started_at = min(record.started_at for record in records)
            newest_heartbeat_at = max(record.last_heartbeat_at for record in records)
            works = tuple(record.to_public_dict(now) for record in records)
            return ActiveWorkSnapshot(
                active_count=len(records),
                oldest_started_at=oldest_started_at,
                newest_heartbeat_at=newest_heartbeat_at,
                oldest_elapsed_s=now - oldest_started_at,
                works=works,
            )

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


ACTIVE_WORK = ActiveWorkRegistry()
