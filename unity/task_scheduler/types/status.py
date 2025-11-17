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


def to_status(value: Status | str | None) -> Status:
    """Convert a status-like value to a Status enum.

    Treat None as 'queued'.
    """
    if isinstance(value, Status):
        return value
    try:
        return Status(value)
    except Exception:
        return Status.queued
