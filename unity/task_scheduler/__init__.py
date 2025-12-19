"""
TaskScheduler package providing task scheduling abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseTaskScheduler",
    "TaskScheduler",
    "SimulatedTaskScheduler",
]

_lazy_map = {
    "BaseTaskScheduler": "unity.task_scheduler.base",
    "TaskScheduler": "unity.task_scheduler.task_scheduler",
    "SimulatedTaskScheduler": "unity.task_scheduler.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseTaskScheduler
    from .task_scheduler import TaskScheduler
    from .simulated import SimulatedTaskScheduler
