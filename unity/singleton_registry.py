from __future__ import annotations

from abc import ABCMeta
from threading import Lock
from typing import Any, Dict, Type

__all__ = [
    "SingletonRegistry",
    "SingletonABCMeta",
]


class SingletonRegistry:
    """Central registry that stores *exactly one* instance per class.

    Instances are stored until :pyfunc:`clear` is invoked – typically via the
    pytest fixture that runs after every test – so that fresh, independent
    singletons are created for the next test run.
    """

    _instances: Dict[Type[Any], Any] = {}
    _lock: Lock = Lock()

    @classmethod
    def get(cls, klass: Type[Any]) -> Any | None:
        """Return the cached instance for *klass* or *None* if none exists."""
        with cls._lock:
            return cls._instances.get(klass)

    @classmethod
    def register(
        cls,
        klass: Type[Any],
        instance: Any,
    ) -> None:
        """Register *instance* as the singleton for *klass*."""
        with cls._lock:
            cls._instances[klass] = instance

    @classmethod
    def clear(cls) -> None:
        """Remove **all** cached singletons – primarily for test isolation."""
        with cls._lock:
            cls._instances.clear()


class SingletonABCMeta(ABCMeta):
    """Metaclass that enforces the Singleton pattern *and* tracks instances.

    Any concrete subclass that uses :class:`SingletonABCMeta` as its metaclass
    will only ever be instantiated **once** (until the registry is cleared).
    Subsequent constructor calls return the *existing* instance without calling
    ``__init__`` again.
    """

    def __call__(cls, *args, **kwargs):
        from .singleton_registry import SingletonRegistry

        existing = SingletonRegistry.get(cls)
        if existing is not None:
            return existing

        # First instantiation – create the object and register it
        instance = super().__call__(*args, **kwargs)
        SingletonRegistry.register(cls, instance)
        return instance
