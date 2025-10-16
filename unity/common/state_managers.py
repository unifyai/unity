from __future__ import annotations

from abc import ABC


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
