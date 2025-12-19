"""
GuidanceManager package providing guidance management abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseGuidanceManager",
    "GuidanceManager",
    "SimulatedGuidanceManager",
]

_lazy_map = {
    "BaseGuidanceManager": "unity.guidance_manager.base",
    "GuidanceManager": "unity.guidance_manager.guidance_manager",
    "SimulatedGuidanceManager": "unity.guidance_manager.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseGuidanceManager
    from .guidance_manager import GuidanceManager
    from .simulated import SimulatedGuidanceManager
