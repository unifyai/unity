"""Enumerated states for a task lifecycle."""

from enum import StrEnum


class Status(StrEnum):
    scheduled = "scheduled"
    queued = "queued"
    paused = "paused"
    primed = "primed"
    active = "active"
    triggerable = "triggerable"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"
