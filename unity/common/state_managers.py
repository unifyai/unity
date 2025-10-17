from __future__ import annotations

from abc import ABC
from typing import Dict, Callable, Optional


class BaseStateManager(ABC):
    """
    Central marker base class for all state managers.

    This abstract base exists solely to provide a single common ancestor for
    manager interfaces such as ContactManager, TranscriptManager, KnowledgeManager,
    TaskScheduler, FileManager, FunctionManager, GuidanceManager, ImageManager,
    SecretManager, WebSearcher, and Conductor.

    Purpose
    -------
    - Enable straightforward `isinstance(obj, BaseStateManager)` checks.
    - Allow expressive and maintainable type hints (e.g., unions or generics
      bounded to `BaseStateManager`).

    The class intentionally defines no abstract methods to avoid constraining
    individual manager contracts.
    """

    def __init__(self):
        self._tools = {}
        self._tools_frozen = False

    def add_tools(self, method: str, tools: Dict[str, Callable]):
        if self._tools_frozen:
            raise RuntimeError("Tools are frozen, cannot add more tools")
        self._tools[method] = tools

    def get_tools(self, method: Optional[str] = None) -> Dict[str, Callable]:
        if method is None:
            return self._tools

        return self._tools.get(method, {})

    def build(self):
        self._tools_frozen = True
