from enum import StrEnum


class Status(StrEnum):
    scheduled = "scheduled"
    queued = "queued"
    in_progress = "in_progress"
    done = "done"
    cancelled = "cancelled"
    failed = "failed"
