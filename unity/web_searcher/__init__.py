"""
WebSearcher package providing web search abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseWebSearcher",
    "WebSearcher",
    "SimulatedWebSearcher",
]

_lazy_map = {
    "BaseWebSearcher": "unity.web_searcher.base",
    "WebSearcher": "unity.web_searcher.web_searcher",
    "SimulatedWebSearcher": "unity.web_searcher.simulated",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .base import BaseWebSearcher
    from .web_searcher import WebSearcher
    from .simulated import SimulatedWebSearcher
