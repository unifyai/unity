"""
SecretManager package providing secret management abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseSecretManager",
    "SecretManager",
    "SimulatedSecretManager",
]

_lazy_map = {
    "BaseSecretManager": "unity.secret_manager.base",
    "SecretManager": "unity.secret_manager.secret_manager",
    "SimulatedSecretManager": "unity.secret_manager.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseSecretManager
    from .secret_manager import SecretManager
    from .simulated import SimulatedSecretManager
