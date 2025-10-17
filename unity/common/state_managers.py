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

    def add_tools(self, method: str, tools: Dict[str, Callable]):
        """
        Store tools for a given manager method. must be called in manager's __init__ method.
        Any tools added after the manager has been initialised may not be available for semantic cache re-execution.

        Parameters
        ----------
        method : str
            The name of the manager method to store tools for.
        tools : Dict[str, Callable]
            A dictionary of tools to store for the given manager method.
        """
        self._tools[method] = tools

    def get_tools(self, method: Optional[str] = None) -> Dict[str, Callable]:
        """
        Get tools for a given manager method.

        Parameters
        ----------
        method : Optional[str], default ``None``
            The name of the manager method to get tools for. If ``None``, return all tools.
        """
        if method is None:
            return self._tools

        return self._tools.get(method, {})
