from __future__ import annotations

from abc import ABC
from typing import Dict, Callable, Optional, Type, Iterable
import importlib
import pkgutil
import sys


class BaseStateManager(ABC):
    """
    Central marker base class for all state managers.

    This abstract base exists solely to provide a single common ancestor for
    manager interfaces such as ContactManager, TranscriptManager, KnowledgeManager,
    TaskScheduler, FileManager, FunctionManager, GuidanceManager, ImageManager,
    SecretManager, and WebSearcher.

    Purpose
    -------
    - Enable straightforward `isinstance(obj, BaseStateManager)` checks.
    - Allow expressive and maintainable type hints (e.g., unions or generics
      bounded to `BaseStateManager`).

    The class intentionally defines no abstract methods to avoid constraining
    individual manager contracts.

    Caller Context
    --------------
    When a manager invokes another manager's tool loop, the ``_as_caller_description``
    class attribute provides a one-liner describing this manager from the perspective
    of the callee. This is injected into the system message so the LLM understands
    who the "user" messages are coming from.
    """

    # Global registry of discovered manager classes keyed by class name
    _registry: Dict[str, Type["BaseStateManager"]] = {}

    # Override in subclasses to describe this manager when it's the caller of another.
    # Used to explain who the "user" is in nested tool loops.
    _as_caller_description: str = "another component of the assistant system"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register concrete subclasses by their class name for stable lookup
        # We intentionally key by the bare class name to match snapshot entrypoints.
        try:
            BaseStateManager._registry[cls.__name__] = cls
        except Exception:
            # Registration is best-effort; avoid raising at import time
            pass

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

    def get_tools(
        self,
        method: Optional[str] = None,
        include_sub_tools: bool = False,
    ) -> Dict[str, Callable]:
        """
        Get tools for a given manager method.

        Parameters
        ----------
        method : Optional[str], default ``None``
            The name of the manager method to get tools for. If ``None``, return all tools.
        """
        if method is None:
            ret = {}
            for sub_tools in self._tools.values():
                ret.update(sub_tools)
            return ret

        if include_sub_tools:
            # Return all sub tools that starts with `method.`
            ret = self._tools.get(method, {})
            for sub_tool in self._tools.keys():
                if sub_tool.startswith(f"{method}."):
                    ret.update(self._tools[sub_tool])
            return ret

        return self._tools.get(method, {})


def get_manager_registry() -> Dict[str, Type[BaseStateManager]]:
    """
    Return a snapshot of the current manager class registry.

    The registry is populated via BaseStateManager.__init_subclass__ and can be
    extended by calling `discover_manager_modules()` to import manager packages.
    """
    return dict(BaseStateManager._registry)


def get_caller_description(manager_class_name: str) -> Optional[str]:
    """
    Look up the caller description for a manager by class name.

    Parameters
    ----------
    manager_class_name : str
        The class name of the manager (e.g., "TaskScheduler", "ContactManager").

    Returns
    -------
    str | None
        The ``_as_caller_description`` for the manager, or None if not found.
    """
    cls = BaseStateManager._registry.get(manager_class_name)
    if cls is None:
        return None
    return getattr(cls, "_as_caller_description", None)


def _iter_unity_subpackages() -> Iterable[str]:
    """
    Yield qualified module names for all subpackages under `unity`.

    This avoids hard-coding specific manager names. We only import packages that
    look like manager packages to keep discovery minimal and fast.
    """
    try:
        import unity  # local import to resolve package path dynamically
    except Exception:
        return []

    for mod in pkgutil.walk_packages(unity.__path__, unity.__name__ + "."):
        name = mod.name
        # We consider any package directly under unity whose name ends with "_manager"
        # as a candidate (e.g., unity.contact_manager, unity.task_scheduler, ...).
        try:
            base = name.rsplit(".", 1)[-1]
        except Exception:
            base = name
        if base.endswith("_manager"):
            yield name


def discover_manager_modules() -> None:
    """
    Import manager packages under `unity/*_manager/` to populate the registry.

    This replaces brittle, hard-coded import lists with a pattern-based discovery.
    Safe to call multiple times; repeated imports are ignored by Python's module cache.
    """
    for pkg_name in _iter_unity_subpackages():
        try:
            # Import the package itself
            importlib.import_module(pkg_name)
            # Prefer importing a module with the same name as the package for
            # conventional layouts like `unity.contact_manager.contact_manager`.
            leaf = pkg_name.rsplit(".", 1)[-1]
            candidate = f"{pkg_name}.{leaf}"
            if candidate not in sys.modules:
                try:
                    importlib.import_module(candidate)
                except Exception:
                    # Fall back: import all immediate submodules to trigger subclass registration
                    for sm in pkgutil.iter_modules(
                        sys.modules[pkg_name].__path__,
                        pkg_name + ".",
                    ):
                        try:
                            importlib.import_module(sm.name)
                        except Exception:
                            continue
        except Exception:
            # Best-effort discovery; individual import failures are ignored
            continue
