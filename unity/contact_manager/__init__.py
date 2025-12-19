"""
ContactManager package providing contact management abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseContactManager",
    "ContactManager",
    "SimulatedContactManager",
]

_lazy_map = {
    "BaseContactManager": "unity.contact_manager.base",
    "ContactManager": "unity.contact_manager.contact_manager",
    "SimulatedContactManager": "unity.contact_manager.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseContactManager
    from .contact_manager import ContactManager
    from .simulated import SimulatedContactManager
