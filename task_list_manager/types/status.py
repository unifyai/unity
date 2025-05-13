from enum import StrEnum


class Status(StrEnum):
    scheduled = "scheduled"
    queued = "queued"
    paused = "paused"
    in_progress = "in_progress"
    done = "done"
    cancelled = "cancelled"
    failed = "failed"
