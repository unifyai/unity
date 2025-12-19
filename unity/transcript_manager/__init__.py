"""
TranscriptManager package providing transcript management abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseTranscriptManager",
    "TranscriptManager",
    "SimulatedTranscriptManager",
]

_lazy_map = {
    "BaseTranscriptManager": "unity.transcript_manager.base",
    "TranscriptManager": "unity.transcript_manager.transcript_manager",
    "SimulatedTranscriptManager": "unity.transcript_manager.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseTranscriptManager
    from .transcript_manager import TranscriptManager
    from .simulated import SimulatedTranscriptManager
