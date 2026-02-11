"""
Runtime primitives interface for state managers.

This module provides:
- `ComputerPrimitives` - Computer use (web/desktop) control capabilities
- `Primitives` - Scoped runtime interface for accessing state manager primitives
- `_AsyncPrimitiveWrapper` - Async wrapper for sync managers

All manager configuration (aliases, excluded methods, class paths) is defined in
`unity.function_manager.primitives.registry`. This module only handles runtime instantiation
and async wrapping.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from typing import Any, Callable, Optional, TYPE_CHECKING

from unity.function_manager.primitives.scope import (
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
)
from unity.function_manager.primitives.registry import (
    get_registry,
    _CLASS_PATH_TO_ALIAS,
)
from unity.manager_registry import SingletonABCMeta

if TYPE_CHECKING:
    from unity.function_manager.computer import Computer
    from unity.contact_manager.contact_manager import ContactManager
    from unity.transcript_manager.transcript_manager import TranscriptManager
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.secret_manager.secret_manager import SecretManager
    from unity.guidance_manager.guidance_manager import GuidanceManager
    from unity.web_searcher.web_searcher import WebSearcher

logger = logging.getLogger(__name__)


# =============================================================================
# ComputerPrimitives - Computer Use (Web/Desktop) Control
# =============================================================================

# Default agent-service URL for local development
DEFAULT_AGENT_SERVER_URL = "http://localhost:3000"

# Gate that blocks lazy MagnitudeBackend initialization until the managed VM is
# confirmed ready.  Set immediately for localhost / mock (no VM to wait for),
# or by _startup_sequence after log_job_startup confirms VM readiness.
_vm_ready = threading.Event()


class ComputerPrimitives(metaclass=SingletonABCMeta):
    """
    Provides a library of high-level, agentic actions for the Actor.
    Each public method is a tool that the actor can incorporate into its generated code.

    There is exactly one VM and one screen per assistant, so this class is a
    singleton (via ``SingletonABCMeta`` / ``ManagerRegistry``).  All actors —
    including nested sub-agents — share the same backend connection.  Call
    ``ManagerRegistry.clear()`` to reset the singleton (e.g. between tests).
    """

    # Methods dynamically created from the backend (single source of truth)
    _DYNAMIC_METHODS = (
        "act",
        "observe",
        "query",
        "navigate",
        "get_links",
        "get_content",
    )
    # All primitive methods (used for discovery)
    _PRIMITIVE_METHODS = _DYNAMIC_METHODS

    @staticmethod
    def _resolve_agent_server_url(explicit_url: str | None) -> str:
        """
        Resolve agent_server_url with priority:
        1. Explicit non-default override (user's personal desktop)
        2. SESSION_DETAILS.assistant.desktop_url (managed VM)
        3. Fallback to DEFAULT_AGENT_SERVER_URL (local dev)
        """
        # If user explicitly provided a non-default URL, honor it
        if explicit_url is not None and explicit_url != DEFAULT_AGENT_SERVER_URL:
            return explicit_url

        # Try SESSION_DETAILS.assistant.desktop_url
        try:
            from unity.session_details import SESSION_DETAILS

            if SESSION_DETAILS.assistant.desktop_url:
                return SESSION_DETAILS.assistant.desktop_url.rstrip("/") + "/api"
        except Exception:
            pass

        return DEFAULT_AGENT_SERVER_URL

    def __init__(
        self,
        headless: bool = False,
        computer_mode: str = "magnitude",
        agent_mode: str = "web",
        agent_server_url: str | None = None,
        *,
        connect_now: bool = False,
        # Deprecated parameters (kept for backward compatibility, ignored)
        session_connect_url: str | None = None,
        controller_mode: str = "hybrid",
    ):
        # Resolve URL centrally from SESSION_DETAILS or explicit override
        resolved_url = self._resolve_agent_server_url(agent_server_url)

        # Cache computer configuration for lazy initialization
        computer_kwargs = {
            "magnitude": {
                "headless": headless,
                "agent_mode": agent_mode,
                "agent_server_url": resolved_url,
            },
            "mock": {
                # MockComputerBackend accepts optional url, screenshot, etc.
                # but works fine with no kwargs
            },
        }

        self._secret_manager = None
        self._computer = None
        self._computer_mode = computer_mode
        self._computer_kwargs_map = computer_kwargs

        # No VM to wait for when using mock backend or co-located agent-service
        if computer_mode == "mock" or resolved_url == DEFAULT_AGENT_SERVER_URL:
            _vm_ready.set()

        # Lazily create the Computer (and thus avoid connecting to agent-service) unless requested
        if connect_now:
            from unity.function_manager.computer import Computer

            self._computer = Computer(
                mode=self._computer_mode,
                secret_manager=self.secret_manager,
                **self._computer_kwargs_map[self._computer_mode],
            )
        self._setup_computer_methods()

    @property
    def secret_manager(self):
        """Lazily initialize and return the SecretManager via ManagerRegistry."""
        if self._secret_manager is None:
            from unity.manager_registry import ManagerRegistry

            self._secret_manager = ManagerRegistry.get_secret_manager()
        return self._secret_manager

    def _setup_computer_methods(self):
        """Dynamically create tool methods without forcing an early backend connection."""
        from unity.function_manager.computer_backends import (
            MagnitudeBackend,
            MockComputerBackend,
        )

        if self._computer_mode == "magnitude":
            backend_class = MagnitudeBackend
        elif self._computer_mode == "mock":
            backend_class = MockComputerBackend
        else:
            raise ValueError(
                f"Unknown computer_mode: '{self._computer_mode}'. Must be 'magnitude' or 'mock'.",
            )

        def _make_lazy_wrapper(method_name: str, backend_class):
            async def wrapper(*args, **kwargs):
                # Internal-only kwargs may be injected by environment wrappers (e.g.
                # clarification queue propagation). Backend implementations (notably
                # MagnitudeBackend) do not accept these, so strip them here.
                kwargs.pop("_clarification_up_q", None)
                kwargs.pop("_clarification_down_q", None)
                # Block until the managed VM is confirmed ready (instant for
                # localhost / mock since the event is pre-set).
                if not _vm_ready.is_set():
                    ready = await asyncio.to_thread(_vm_ready.wait, 300)
                    if not ready:
                        raise RuntimeError(
                            "Managed VM did not become ready within 5 minutes",
                        )
                backend_method = getattr(self.computer.backend, method_name)
                return await backend_method(*args, **kwargs)

            wrapper.__name__ = method_name
            wrapper.__qualname__ = method_name
            backend_method = getattr(backend_class, method_name, None)
            if backend_method and hasattr(backend_method, "__doc__"):
                wrapper.__doc__ = backend_method.__doc__
            return wrapper

        for method_name in self._DYNAMIC_METHODS:
            setattr(
                self,
                method_name,
                _make_lazy_wrapper(method_name, backend_class),
            )

    @property
    def computer(self) -> "Computer":
        """Lazily initialize and return the Computer instance."""
        if self._computer is None:
            from unity.function_manager.computer import Computer

            self._computer = Computer(
                mode=self._computer_mode,
                secret_manager=self.secret_manager,
                **self._computer_kwargs_map[self._computer_mode],
            )
        return self._computer


# =============================================================================
# Async Wrapper for Sync Managers
# =============================================================================


class _AsyncPrimitiveWrapper:
    """
    Wrapper that provides async versions of sync manager methods.

    Delegates to the original manager without modifying it,
    ensuring internal code using the manager synchronously continues to work.

    Uses asyncio.to_thread() for sync methods to avoid blocking the event loop.
    """

    def __init__(self, manager: Any, manager_alias: str):
        """
        Initialize the wrapper.

        Args:
            manager: The original sync manager instance.
            manager_alias: The manager alias to look up primitive methods.
        """
        object.__setattr__(self, "_wrapped_manager", manager)
        object.__setattr__(self, "_manager_alias", manager_alias)
        # Get primitive methods from registry
        registry = get_registry()
        object.__setattr__(
            self,
            "_primitive_methods",
            set(registry.primitive_methods(manager_alias=manager_alias)),
        )

    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute - returns async wrapper for primitive methods, else delegates.
        """
        attr = getattr(self._wrapped_manager, name)

        # Only wrap methods that are in our primitive methods set
        if name not in self._primitive_methods:
            return attr

        # Non-callable attributes pass through directly
        if not callable(attr):
            return attr

        # Create async wrapper that uses to_thread for sync methods
        @functools.wraps(attr)
        async def async_method_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(attr):
                return await attr(*args, **kwargs)
            else:
                return await asyncio.to_thread(attr, *args, **kwargs)

        return async_method_wrapper


def _create_async_wrapper(manager: Any, manager_alias: str) -> _AsyncPrimitiveWrapper:
    """
    Create an async wrapper for a sync manager.

    Args:
        manager: The original sync manager instance.
        manager_alias: The manager alias for registry lookup.

    Returns:
        An async wrapper around the manager.
    """
    return _AsyncPrimitiveWrapper(manager, manager_alias)


# =============================================================================
# Manager Registry Key Mapping
# =============================================================================

# Maps manager_alias to ManagerRegistry getter method name
_ALIAS_TO_GETTER: dict[str, str] = {
    "contacts": "get_contact_manager",
    "data": "get_data_manager",
    "transcripts": "get_transcript_manager",
    "knowledge": "get_knowledge_manager",
    "tasks": "get_task_scheduler",
    "secrets": "get_secret_manager",
    "guidance": "get_guidance_manager",
    "web": "get_web_searcher",
    "files": "get_file_manager",
}

# Managers that need async wrapping (sync implementations)
_SYNC_MANAGERS: frozenset[str] = frozenset({"data", "files"})


# =============================================================================
# Primitives Runtime Class (Scoped)
# =============================================================================


class Primitives:
    """
    Scoped runtime interface to state manager primitives.

    Only managers in the provided `primitive_scope` are accessible.
    Attempting to access an out-of-scope manager raises AttributeError.

    All state managers are obtained via ManagerRegistry typed methods
    to respect IMPL settings (real vs simulated).

    Sync managers (DataManager, FileManager) are wrapped with async interfaces
    for consistency - the LLM can safely use `await` on all primitives.

    Usage:
        scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
        primitives = Primitives(primitive_scope=scope)

        # Accessible:
        await primitives.files.describe(file_path="...")
        await primitives.contacts.ask(text="...")

        # Raises AttributeError:
        primitives.tasks  # not in scope
    """

    def __init__(self, *, primitive_scope: Optional[PrimitiveScope] = None) -> None:
        """
        Initialize primitives with the given scope.

        Args:
            primitive_scope: Defines which managers are accessible.
                           If None, all managers are accessible.
        """
        self._primitive_scope = primitive_scope or PrimitiveScope.all_managers()
        # Lazy-initialized manager instances
        self._managers: dict[str, Any] = {}
        # ComputerPrimitives handled separately (not in primitives registry)
        self._computer: Optional[ComputerPrimitives] = None

    @property
    def primitive_scope(self) -> PrimitiveScope:
        """The scope controlling which managers are accessible."""
        return self._primitive_scope

    def _get_manager(self, alias: str) -> Any:
        """
        Get or create a manager instance by alias.

        Raises AttributeError if alias is not in scope.
        """
        if alias not in self._primitive_scope.scoped_managers:
            available = sorted(self._primitive_scope.scoped_managers)
            raise AttributeError(
                f"primitives.{alias} is not available in this scope. "
                f"Available managers: {available}",
            )

        if alias in self._managers:
            return self._managers[alias]

        getter_name = _ALIAS_TO_GETTER.get(alias)
        if not getter_name:
            raise AttributeError(f"Unknown manager alias: {alias}")

        from unity.manager_registry import ManagerRegistry

        getter = getattr(ManagerRegistry, getter_name)
        manager = getter()

        # Wrap sync managers with async interface
        if alias in _SYNC_MANAGERS:
            manager = _create_async_wrapper(manager, alias)

        self._managers[alias] = manager
        return manager

    def __getattr__(self, name: str) -> Any:
        """
        Attribute access for manager retrieval.

        Special handling for 'computer' which is not part of the scoped registry.
        All other names are treated as manager aliases.
        """
        # Computer primitives are not in the primitives registry.
        # ComputerPrimitives uses SingletonABCMeta, so the constructor
        # returns the shared instance automatically.
        if name == "computer":
            if self._computer is None:
                self._computer = ComputerPrimitives()
            return self._computer

        # Check if it's a valid manager alias
        if name in VALID_MANAGER_ALIASES:
            return self._get_manager(name)

        raise AttributeError(f"'Primitives' object has no attribute '{name}'")

    # Convenience properties for type hints (IDE support)
    # These are optional and just provide better autocomplete

    @property
    def contacts(self) -> "ContactManager":
        """Contact management primitives (ask, update)."""
        return self._get_manager("contacts")

    @property
    def data(self) -> "_AsyncPrimitiveWrapper":
        """Data operations primitives (filter, search, reduce, join, etc.)."""
        return self._get_manager("data")

    @property
    def transcripts(self) -> "TranscriptManager":
        """Transcript management primitives (ask)."""
        return self._get_manager("transcripts")

    @property
    def knowledge(self) -> "KnowledgeManager":
        """Knowledge management primitives (ask, update, refactor)."""
        return self._get_manager("knowledge")

    @property
    def tasks(self) -> "TaskScheduler":
        """Task scheduling primitives (ask, update, execute)."""
        return self._get_manager("tasks")

    @property
    def secrets(self) -> "SecretManager":
        """Secret management primitives (ask, update)."""
        return self._get_manager("secrets")

    @property
    def guidance(self) -> "GuidanceManager":
        """Guidance management primitives (ask, update)."""
        return self._get_manager("guidance")

    @property
    def web(self) -> "WebSearcher":
        """Web search primitives (ask)."""
        return self._get_manager("web")

    @property
    def files(self) -> "_AsyncPrimitiveWrapper":
        """File management primitives (describe, reduce, filter_files, etc.)."""
        return self._get_manager("files")


# =============================================================================
# Primitive Callable Resolution (for FunctionManager execution)
# =============================================================================


def get_primitive_callable(
    primitive_data: dict[str, Any],
    computer_primitives: Optional[ComputerPrimitives] = None,
    primitives: Optional[Primitives] = None,
) -> Optional[Callable]:
    """
    Resolve a primitive metadata dict to its actual callable.

    For ComputerPrimitives methods, uses the provided computer_primitives instance.
    For state manager methods, uses the provided primitives instance or ManagerRegistry.

    Args:
        primitive_data: Primitive metadata with primitive_class and primitive_method.
        computer_primitives: ComputerPrimitives instance (for ComputerPrimitives primitives).
        primitives: Optional scoped Primitives instance for state manager resolution.

    Returns:
        The callable method, or None if resolution fails.
    """
    class_path = primitive_data.get("primitive_class")
    method_name = primitive_data.get("primitive_method")

    if not class_path or not method_name:
        return None

    # Special case: ComputerPrimitives methods use the provided instance
    if "ComputerPrimitives" in class_path:
        if computer_primitives is None:
            logger.warning(
                "Cannot resolve ComputerPrimitives primitive without computer_primitives instance",
            )
            return None
        return getattr(computer_primitives, method_name, None)

    # Derive manager_alias from primitive_class using the registry mapping
    manager_alias = _CLASS_PATH_TO_ALIAS.get(class_path)

    # State managers: use provided primitives instance if available
    if primitives is not None and manager_alias:
        try:
            manager = getattr(primitives, manager_alias)
            return getattr(manager, method_name, None)
        except AttributeError:
            # Manager not in scope, fall through to ManagerRegistry
            pass

    # Fallback: use ManagerRegistry directly
    if manager_alias:
        getter_name = _ALIAS_TO_GETTER.get(manager_alias)
        if getter_name:
            try:
                from unity.manager_registry import ManagerRegistry

                getter = getattr(ManagerRegistry, getter_name)
                instance = getter()
                return getattr(instance, method_name, None)
            except Exception as e:
                logger.warning(f"Could not get manager via '{getter_name}': {e}")

    return None
