"""
ConfigManager package providing DB-stored actor configuration.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseConfigManager",
    "ConfigManager",
    "SimulatedConfigManager",
]

_lazy_map = {
    "BaseConfigManager": "unity.customization.configs.base",
    "ConfigManager": "unity.customization.configs.config_manager",
    "SimulatedConfigManager": "unity.customization.configs.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseConfigManager
    from .config_manager import ConfigManager
    from .simulated import SimulatedConfigManager
