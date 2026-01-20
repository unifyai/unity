"""
unity/manager_registry.py
=========================

Centralized manager infrastructure for Unity's state managers.

This module is the single source of truth for:
  - Manager implementation registration (real, simulated, etc.)
  - Settings-based IMPL resolution
  - Singleton instance caching
  - Typed factory methods for obtaining manager instances

Usage:
    # Get a manager via typed method (auto-resolves IMPL, returns singleton):
    contact_manager = ManagerRegistry.get_contact_manager()
    task_scheduler = ManagerRegistry.get_task_scheduler()

    # For simulated managers, pass description:
    ManagerRegistry.get_contact_manager(description="test scenario")

    # Clear all singletons (for test isolation):
    ManagerRegistry.clear()

Available typed methods:
    - get_actor()
    - get_contact_manager()
    - get_conversation_manager_handle()
    - get_data_manager()
    - get_file_manager()
    - get_function_manager()
    - get_guidance_manager()
    - get_image_manager()
    - get_knowledge_manager()
    - get_memory_manager()
    - get_secret_manager()
    - get_task_scheduler()
    - get_transcript_manager()
    - get_web_searcher()
"""

from __future__ import annotations

from abc import ABCMeta
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Type

if TYPE_CHECKING:
    from .actor.base import BaseActor
    from .contact_manager.base import BaseContactManager
    from .conversation_manager.base import BaseConversationManagerHandle
    from .data_manager.base import BaseDataManager
    from .file_manager.managers.base import BaseFileManager
    from .function_manager.base import BaseFunctionManager
    from .guidance_manager.base import BaseGuidanceManager
    from .image_manager.base import BaseImageManager
    from .knowledge_manager.base import BaseKnowledgeManager
    from .memory_manager.base import BaseMemoryManager
    from .secret_manager.base import BaseSecretManager
    from .task_scheduler.base import BaseTaskScheduler
    from .transcript_manager.base import BaseTranscriptManager
    from .web_searcher.base import BaseWebSearcher

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

    # Flag to track if the registry has been populated
    _populated: bool = False

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
    def register_settings(
        cls,
        manager_key: str,
        settings_accessor: Callable[[], Any],
    ) -> None:
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
    # Lazy Population
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def _ensure_populated(cls) -> None:
        """Ensure the registry is populated with all known implementations.

        This is called lazily on first access to avoid circular imports.
        The managers import SingletonABCMeta from this module, so we can't
        import them at module load time.
        """
        if cls._populated:
            return
        with cls._lock:
            if cls._populated:
                return
            _populate_registry()
            cls._populated = True

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
        cls._ensure_populated()

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
        cls._ensure_populated()

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

    # Mapping from manager_key to the environment variable name for IMPL.
    # Used to support test-time overrides since SETTINGS is frozen at import time.
    _impl_env_vars: Dict[str, str] = {
        "actor": "UNITY_ACTOR_IMPL",
        "contacts": "UNITY_CONTACT_IMPL",
        "transcripts": "UNITY_TRANSCRIPT_IMPL",
        "tasks": "UNITY_TASK_IMPL",
        "conversation": "UNITY_CONVERSATION_IMPL",
        "knowledge": "UNITY_KNOWLEDGE_IMPL",
        "guidance": "UNITY_GUIDANCE_IMPL",
        "secrets": "UNITY_SECRET_IMPL",
        "web_search": "UNITY_WEB_IMPL",
        "data": "UNITY_DATA_IMPL",
        "files": "UNITY_FILE_IMPL",
        "functions": "UNITY_FUNCTION_IMPL",
        "images": "UNITY_IMAGE_IMPL",
        "memory": "UNITY_MEMORY_IMPL",
    }

    @classmethod
    def _resolve_impl(cls, manager_key: str) -> str:
        """Resolve the IMPL setting for a manager key.

        Checks environment variables at runtime first (via SESSION_DETAILS) to
        support test-time overrides. SETTINGS is frozen at import time, so test
        conftests that set os.environ after import won't affect SETTINGS values.
        """
        from unity.session_details import SESSION_DETAILS

        # First, check for runtime environment variable override
        env_var = cls._impl_env_vars.get(manager_key)
        if env_var:
            env_value = SESSION_DETAILS.get_impl_setting(env_var, default="")
            if env_value:
                return env_value

        # Fall back to SETTINGS (frozen at import time)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Typed Factory Methods
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_actor(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseActor":
        """Get the Actor singleton (respects IMPL settings)."""
        return cls.get(
            "actor",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_contact_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseContactManager":
        """Get the ContactManager singleton (respects IMPL settings)."""
        return cls.get(
            "contacts",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_conversation_manager_handle(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseConversationManagerHandle":
        """Get the ConversationManagerHandle singleton (respects IMPL settings)."""
        return cls.get(
            "conversation",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_data_manager(
        cls,
        *,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseDataManager":
        """Get the DataManager singleton (respects IMPL settings).

        DataManager provides canonical data operations (filter, search, reduce,
        join, vectorize, plot) that work on any Unify context. It owns the
        Data/* namespace but can operate on any context including Files/*.
        """
        return cls.get(
            "data",
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_file_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseFileManager":
        """Get the FileManager singleton (respects IMPL settings)."""
        return cls.get(
            "files",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_function_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseFunctionManager":
        """Get the FunctionManager singleton (respects IMPL settings)."""
        return cls.get(
            "functions",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_guidance_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseGuidanceManager":
        """Get the GuidanceManager singleton (respects IMPL settings)."""
        return cls.get(
            "guidance",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_image_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseImageManager":
        """Get the ImageManager singleton (respects IMPL settings)."""
        return cls.get(
            "images",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_knowledge_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseKnowledgeManager":
        """Get the KnowledgeManager singleton (respects IMPL settings)."""
        return cls.get(
            "knowledge",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_memory_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseMemoryManager":
        """Get the MemoryManager singleton (respects IMPL settings)."""
        return cls.get(
            "memory",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_secret_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseSecretManager":
        """Get the SecretManager singleton (respects IMPL settings)."""
        return cls.get(
            "secrets",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_task_scheduler(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseTaskScheduler":
        """Get the TaskScheduler singleton (respects IMPL settings)."""
        return cls.get(
            "tasks",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_transcript_manager(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseTranscriptManager":
        """Get the TranscriptManager singleton (respects IMPL settings)."""
        return cls.get(
            "transcripts",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )

    @classmethod
    def get_web_searcher(
        cls,
        *,
        description: str | None = None,
        simulation_guidance: str | None = None,
        _force_new: bool = False,
        **kwargs: Any,
    ) -> "BaseWebSearcher":
        """Get the WebSearcher singleton (respects IMPL settings)."""
        return cls.get(
            "web_search",
            description=description,
            simulation_guidance=simulation_guidance,
            _force_new=_force_new,
            **kwargs,
        )


class SingletonABCMeta(ABCMeta):
    """Metaclass that enforces the Singleton pattern via ManagerRegistry.

    Any concrete subclass that uses this metaclass will only ever be
    instantiated once (until ManagerRegistry.clear() is called). Subsequent
    constructor calls return the existing instance without calling __init__
    again.

    Note: This supports direct class instantiation (e.g., `ContactManager()`).
    For settings-aware instantiation that respects IMPL, use the typed methods
    like `ManagerRegistry.get_contact_manager()` instead.
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
    ManagerRegistry.register_settings("web_search", lambda: SETTINGS.web)
    ManagerRegistry.register_settings("data", lambda: SETTINGS.data)
    ManagerRegistry.register_settings("files", lambda: SETTINGS.file)
    ManagerRegistry.register_settings("functions", lambda: SETTINGS.function)
    ManagerRegistry.register_settings("images", lambda: SETTINGS.image)
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
    ManagerRegistry.register_class(
        "transcripts",
        "simulated",
        SimulatedTranscriptManager,
    )

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
    ManagerRegistry.register_class(
        "conversation",
        "simulated",
        SimulatedConversationManagerHandle,
    )

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
    # WebSearcher implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .web_searcher.web_searcher import WebSearcher
    from .web_searcher.simulated import SimulatedWebSearcher

    ManagerRegistry.register_class("web_search", "real", WebSearcher)
    ManagerRegistry.register_class("web_search", "simulated", SimulatedWebSearcher)

    # ─────────────────────────────────────────────────────────────────────────
    # DataManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .data_manager.data_manager import DataManager
    from .data_manager.simulated import SimulatedDataManager

    ManagerRegistry.register_class("data", "real", DataManager)
    ManagerRegistry.register_class("data", "simulated", SimulatedDataManager)

    # ─────────────────────────────────────────────────────────────────────────
    # FileManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .file_manager.managers.file_manager import FileManager
    from .file_manager.simulated import SimulatedFileManager

    ManagerRegistry.register_class("files", "real", FileManager)
    ManagerRegistry.register_class("files", "simulated", SimulatedFileManager)

    # ─────────────────────────────────────────────────────────────────────────
    # MemoryManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .memory_manager.memory_manager import MemoryManager
    from .memory_manager.simulated import SimulatedMemoryManager

    ManagerRegistry.register_class("memory", "real", MemoryManager)
    ManagerRegistry.register_class("memory", "simulated", SimulatedMemoryManager)

    # FunctionManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .function_manager.function_manager import FunctionManager
    from .function_manager.simulated import SimulatedFunctionManager

    ManagerRegistry.register_class("functions", "real", FunctionManager)
    ManagerRegistry.register_class("functions", "simulated", SimulatedFunctionManager)

    # ─────────────────────────────────────────────────────────────────────────
    # ImageManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from .image_manager.image_manager import ImageManager

    ManagerRegistry.register_class("images", "real", ImageManager)
    # Note: No simulated implementation exists for ImageManager
