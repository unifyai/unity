"""
EnvironmentManager package providing custom environment storage and loading.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseEnvironmentManager",
    "EnvironmentManager",
    "SimulatedEnvironmentManager",
]

_lazy_map = {
    "BaseEnvironmentManager": "unity.customization.environments.base",
    "EnvironmentManager": "unity.customization.environments.environment_manager",
    "SimulatedEnvironmentManager": "unity.customization.environments.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseEnvironmentManager
    from .environment_manager import EnvironmentManager
    from .simulated import SimulatedEnvironmentManager
