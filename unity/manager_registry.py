"""
unity/manager_registry.py
=========================

Centralized manager infrastructure for Unity's state managers.

This module is the single source of truth for:
  - Manager implementation registration (real, simulated, etc.)
  - Settings-based IMPL resolution
  - Singleton instance caching
  - Factory method for obtaining manager instances

Usage:
    # Get a manager (auto-resolves IMPL from settings, returns singleton):
    contact_manager = ManagerRegistry.get("contacts")

    # Get a manager with dependencies:
    transcript_manager = ManagerRegistry.get(
        "transcripts",
        contact_manager=contact_manager,
    )

    # For simulated managers, pass description:
    ManagerRegistry.get("contacts", description="test scenario")

    # Clear all singletons (for test isolation):
    ManagerRegistry.clear()
"""

from __future__ import annotations

from abc import ABCMeta
from threading import Lock
from typing import Any, Callable, Dict, Type

__all__ = [
    "ManagerRegistry",
    "SingletonABCMeta",
]


class ManagerRegistry:
    """Central registry for manager implementations and singleton instances.

    Handles:
    - Implementation class registration (real, simulated, etc.)
    - Settings-based IMPL resolution
    - Singleton instance caching
    - Factory method for obtaining correctly-configured manager instances
    """

    # (manager_key, impl_name) -> class
    _classes: Dict[tuple[str, str], Type] = {}

    # class -> singleton instance
    _instances: Dict[Type, Any] = {}

    # manager_key -> callable that returns the settings object for that manager
    _settings_map: Dict[str, Callable[[], Any]] = {}

    _lock: Lock = Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Registration API
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def register_class(cls, manager_key: str, impl_name: str, klass: Type) -> None:
        """Register a manager implementation class.

        Parameters
        ----------
        manager_key : str
            Logical name for the manager (e.g., "contacts", "transcripts").
        impl_name : str
            Implementation variant (e.g., "real", "simulated").
        klass : Type
            The concrete class to register.
        """
        cls._classes[(manager_key, impl_name)] = klass

    @classmethod
    def register_settings(cls, manager_key: str, settings_accessor: Callable[[], Any]) -> None:
        """Register a settings accessor for a manager key.

        Parameters
        ----------
        manager_key : str
            Logical name for the manager.
        settings_accessor : Callable
            A callable that returns the settings object for this manager.
            Expected to have an `IMPL` attribute.
        """
        cls._settings_map[manager_key] = settings_accessor

    # ──────────────────────────────────────────────────────────────────────────
    # Factory API
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def get(
        cls,
        manager_key: str,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Get or create the singleton instance for a manager.

        Automatically resolves the implementation based on settings (IMPL).

        Parameters
        ----------
        manager_key : str
            Logical name for the manager (e.g., "contacts", "transcripts").
        description : str | None
            For simulated managers, the scenario description. Ignored for real
            managers.
        simulation_guidance : str | None
            For simulated managers, additional guidance for simulation behavior.
            Ignored for real managers.
        _force_new : bool
            If True, bypass the singleton cache and create a fresh instance.
            Primarily for testing.
        **kwargs
            Additional keyword arguments passed to the manager constructor.

        Returns
        -------
        Any
            The manager instance (singleton unless _force_new=True).

        Raises
        ------
        ValueError
            If the manager_key is unknown or no implementation is registered
            for the resolved IMPL.
        """
        # 1. Resolve IMPL from settings
        impl_name = cls._resolve_impl(manager_key)

        # 2. Get the class
        klass = cls.get_class(manager_key, impl_name)

        # 3. Check singleton cache (unless forced)
        if not _force_new:
            with cls._lock:
                existing = cls._instances.get(klass)
                if existing is not None:
                    return existing

        # 4. Build constructor kwargs
        ctor_kwargs = dict(kwargs)
        # Simulated-only parameters
        if impl_name == "simulated":
            if description is not None:
                ctor_kwargs["description"] = description
            if simulation_guidance is not None:
                ctor_kwargs["simulation_guidance"] = simulation_guidance

        # 5. Create instance
        instance = klass(**ctor_kwargs)

        # 6. Cache (unless forced)
        if not _force_new:
            with cls._lock:
                # Double-check pattern for thread safety
                existing = cls._instances.get(klass)
                if existing is not None:
                    return existing
                cls._instances[klass] = instance

        return instance

    @classmethod
    def get_class(cls, manager_key: str, impl_name: str | None = None) -> Type:
        """Get the class for a manager without instantiating.

        Parameters
        ----------
        manager_key : str
            Logical name for the manager.
        impl_name : str | None
            Implementation variant. If None, resolves from settings.

        Returns
        -------
        Type
            The manager class.

        Raises
        ------
        ValueError
            If no implementation is registered for the given key/impl.
        """
        if impl_name is None:
            impl_name = cls._resolve_impl(manager_key)

        key = (manager_key, impl_name)
        if key not in cls._classes:
            available = [k[1] for k in cls._classes if k[0] == manager_key]
            raise ValueError(
                f"Unknown implementation '{impl_name}' for manager '{manager_key}'. "
                f"Available: {available}",
            )
        return cls._classes[key]

    @classmethod
    def _resolve_impl(cls, manager_key: str) -> str:
        """Resolve the IMPL setting for a manager key."""
        settings_accessor = cls._settings_map.get(manager_key)
        if settings_accessor is None:
            raise ValueError(
                f"No settings registered for manager '{manager_key}'. "
                f"Available: {list(cls._settings_map.keys())}",
            )
        settings = settings_accessor()
        return getattr(settings, "IMPL", "real")

    # ──────────────────────────────────────────────────────────────────────────
    # Singleton Management (for direct class instantiation & test isolation)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, klass: Type) -> Any | None:
        """Get a cached singleton instance by class.

        Used by SingletonABCMeta for direct class instantiation.
        """
        with cls._lock:
            return cls._instances.get(klass)

    @classmethod
    def register_instance(cls, klass: Type, instance: Any) -> None:
        """Register an instance as the singleton for a class.

        Used by SingletonABCMeta for direct class instantiation.
        """
        with cls._lock:
            cls._instances[klass] = instance

    @classmethod
    def clear(cls) -> None:
        """Remove all cached singleton instances.

        Call this between tests to ensure isolation.
        """
        with cls._lock:
            cls._instances.clear()


class SingletonABCMeta(ABCMeta):
    """Metaclass that enforces the Singleton pattern via ManagerRegistry.

    Any concrete subclass that uses this metaclass will only ever be
    instantiated once (until ManagerRegistry.clear() is called). Subsequent
    constructor calls return the existing instance without calling __init__
    again.

    Note: This supports direct class instantiation (e.g., `ContactManager()`).
    For settings-aware instantiation that respects IMPL, use
    `ManagerRegistry.get("contacts")` instead.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        existing = ManagerRegistry.get_instance(cls)
        if existing is not None:
            return existing

        # First instantiation – create and register
        instance = super().__call__(*args, **kwargs)
        ManagerRegistry.register_instance(cls, instance)
        return instance


# ──────────────────────────────────────────────────────────────────────────────
# Registry Population
# ──────────────────────────────────────────────────────────────────────────────


def _populate_registry() -> None:
    """Populate the registry with all known implementations and settings.

    Imports are deferred to avoid circular dependencies.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Settings mappings (manager_key -> settings accessor)
    # ─────────────────────────────────────────────────────────────────────────
    from .settings import SETTINGS

    ManagerRegistry.register_settings("actor", lambda: SETTINGS.actor)
    ManagerRegistry.register_settings("contacts", lambda: SETTINGS.contact)
    ManagerRegistry.register_settings("transcripts", lambda: SETTINGS.transcript)
    ManagerRegistry.register_settings("tasks", lambda: SETTINGS.task)
    ManagerRegistry.register_settings("conversation", lambda: SETTINGS.conversation)
    ManagerRegistry.register_settings("knowledge", lambda: SETTINGS.knowledge)
    ManagerRegistry.register_settings("guidance", lambda: SETTINGS.guidance)
    ManagerRegistry.register_settings("secrets", lambda: SETTINGS.secret)
    ManagerRegistry.register_settings("skills", lambda: SETTINGS.skill)
    ManagerRegistry.register_settings("web_search", lambda: SETTINGS.web)
    ManagerRegistry.register_settings("files", lambda: SETTINGS.file)
    ManagerRegistry.register_settings("memory", lambda: SETTINGS.memory)

    # ─────────────────────────────────────────────────────────────────────────
    # Actor implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .actor.hierarchical_actor import HierarchicalActor
    from .actor.single_function_actor import SingleFunctionActor
    from .actor.code_act_actor import CodeActActor
    from .actor.simulated import SimulatedActor

    ManagerRegistry.register_class("actor", "hierarchical", HierarchicalActor)
    ManagerRegistry.register_class("actor", "single_function", SingleFunctionActor)
    ManagerRegistry.register_class("actor", "code_act", CodeActActor)
    ManagerRegistry.register_class("actor", "simulated", SimulatedActor)

    # ─────────────────────────────────────────────────────────────────────────
    # ContactManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .contact_manager.contact_manager import ContactManager
    from .contact_manager.simulated import SimulatedContactManager

    ManagerRegistry.register_class("contacts", "real", ContactManager)
    ManagerRegistry.register_class("contacts", "simulated", SimulatedContactManager)

    # ─────────────────────────────────────────────────────────────────────────
    # TranscriptManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .transcript_manager.transcript_manager import TranscriptManager
    from .transcript_manager.simulated import SimulatedTranscriptManager

    ManagerRegistry.register_class("transcripts", "real", TranscriptManager)
    ManagerRegistry.register_class("transcripts", "simulated", SimulatedTranscriptManager)

    # ─────────────────────────────────────────────────────────────────────────
    # TaskScheduler implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .task_scheduler.task_scheduler import TaskScheduler
    from .task_scheduler.simulated import SimulatedTaskScheduler

    ManagerRegistry.register_class("tasks", "real", TaskScheduler)
    ManagerRegistry.register_class("tasks", "simulated", SimulatedTaskScheduler)

    # ─────────────────────────────────────────────────────────────────────────
    # ConversationManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .conversation_manager.handle import ConversationManagerHandle
    from .conversation_manager.simulated import SimulatedConversationManagerHandle

    ManagerRegistry.register_class("conversation", "real", ConversationManagerHandle)
    ManagerRegistry.register_class("conversation", "simulated", SimulatedConversationManagerHandle)

    # ─────────────────────────────────────────────────────────────────────────
    # KnowledgeManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .knowledge_manager.knowledge_manager import KnowledgeManager
    from .knowledge_manager.simulated import SimulatedKnowledgeManager

    ManagerRegistry.register_class("knowledge", "real", KnowledgeManager)
    ManagerRegistry.register_class("knowledge", "simulated", SimulatedKnowledgeManager)

    # ─────────────────────────────────────────────────────────────────────────
    # GuidanceManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .guidance_manager.guidance_manager import GuidanceManager
    from .guidance_manager.simulated import SimulatedGuidanceManager

    ManagerRegistry.register_class("guidance", "real", GuidanceManager)
    ManagerRegistry.register_class("guidance", "simulated", SimulatedGuidanceManager)

    # ─────────────────────────────────────────────────────────────────────────
    # SecretManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .secret_manager.secret_manager import SecretManager
    from .secret_manager.simulated import SimulatedSecretManager

    ManagerRegistry.register_class("secrets", "real", SecretManager)
    ManagerRegistry.register_class("secrets", "simulated", SimulatedSecretManager)

    # ─────────────────────────────────────────────────────────────────────────
    # SkillManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .skill_manager.skill_manager import SkillManager
    from .skill_manager.simulated import SimulatedSkillManager

    ManagerRegistry.register_class("skills", "real", SkillManager)
    ManagerRegistry.register_class("skills", "simulated", SimulatedSkillManager)

    # ─────────────────────────────────────────────────────────────────────────
    # WebSearcher implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .web_searcher.web_searcher import WebSearcher
    from .web_searcher.simulated import SimulatedWebSearcher

    ManagerRegistry.register_class("web_search", "real", WebSearcher)
    ManagerRegistry.register_class("web_search", "simulated", SimulatedWebSearcher)

    # ─────────────────────────────────────────────────────────────────────────
    # GlobalFileManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .file_manager.global_file_manager import GlobalFileManager
    from .file_manager.simulated import SimulatedGlobalFileManager

    ManagerRegistry.register_class("files", "real", GlobalFileManager)
    ManagerRegistry.register_class("files", "simulated", SimulatedGlobalFileManager)

    # ─────────────────────────────────────────────────────────────────────────
    # MemoryManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .memory_manager.memory_manager import MemoryManager
    from .memory_manager.simulated import SimulatedMemoryManager

    ManagerRegistry.register_class("memory", "real", MemoryManager)
    ManagerRegistry.register_class("memory", "simulated", SimulatedMemoryManager)


# Populate on first import
_populate_registry()
