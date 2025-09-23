"""Priority levels for tasks."""

from enum import StrEnum


class Priority(StrEnum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"
