"""
KnowledgeManager package providing knowledge management abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseKnowledgeManager",
    "KnowledgeManager",
    "SimulatedKnowledgeManager",
]

_lazy_map = {
    "BaseKnowledgeManager": "unity.knowledge_manager.base",
    "KnowledgeManager": "unity.knowledge_manager.knowledge_manager",
    "SimulatedKnowledgeManager": "unity.knowledge_manager.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseKnowledgeManager
    from .knowledge_manager import KnowledgeManager
    from .simulated import SimulatedKnowledgeManager
