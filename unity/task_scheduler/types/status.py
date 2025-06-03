from enum import StrEnum


class Status(StrEnum):
    scheduled = "scheduled"
    queued = "queued"
    paused = "paused"
    active = "active"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"
