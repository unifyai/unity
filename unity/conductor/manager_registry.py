"""
Manager implementation registry for environment-driven Conductor configuration.

Maps (manager_key, impl_name) tuples to concrete implementation classes.
"""

from __future__ import annotations

from typing import Type, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Registry mapping: (manager_key, impl_name) -> class
_REGISTRY: Dict[tuple[str, str], Type] = {}


def register(manager_key: str, impl_name: str, cls: Type) -> None:
    """Register a manager implementation class."""
    _REGISTRY[(manager_key, impl_name)] = cls


def get_class(manager_key: str, impl_name: str) -> Type:
    """Look up a manager implementation class by key and impl name."""
    key = (manager_key, impl_name)
    if key not in _REGISTRY:
        available = [k[1] for k in _REGISTRY if k[0] == manager_key]
        raise ValueError(
            f"Unknown implementation '{impl_name}' for manager '{manager_key}'. "
            f"Available: {available}",
        )
    return _REGISTRY[key]


def _populate_registry() -> None:
    """Populate the registry with all known implementations.

    Imports are deferred to avoid circular dependencies.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Actor implementations
    # ─────────────────────────────────────────────────────────────────────────
    from ..actor.hierarchical_actor import HierarchicalActor
    from ..actor.single_function_actor import SingleFunctionActor
    from ..actor.code_act_actor import CodeActActor
    from ..actor.simulated import SimulatedActor

    register("actor", "hierarchical", HierarchicalActor)
    register("actor", "single_function", SingleFunctionActor)
    register("actor", "code_act", CodeActActor)
    register("actor", "simulated", SimulatedActor)

    # ─────────────────────────────────────────────────────────────────────────
    # ContactManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from ..contact_manager.contact_manager import ContactManager
    from ..contact_manager.simulated import SimulatedContactManager

    register("contacts", "real", ContactManager)
    register("contacts", "simulated", SimulatedContactManager)

    # ─────────────────────────────────────────────────────────────────────────
    # TranscriptManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from ..transcript_manager.transcript_manager import TranscriptManager
    from ..transcript_manager.simulated import SimulatedTranscriptManager

    register("transcripts", "real", TranscriptManager)
    register("transcripts", "simulated", SimulatedTranscriptManager)

    # ─────────────────────────────────────────────────────────────────────────
    # TaskScheduler implementations
    # ─────────────────────────────────────────────────────────────────────────
    from ..task_scheduler.task_scheduler import TaskScheduler
    from ..task_scheduler.simulated import SimulatedTaskScheduler

    register("tasks", "real", TaskScheduler)
    register("tasks", "simulated", SimulatedTaskScheduler)

    # ─────────────────────────────────────────────────────────────────────────
    # ConversationManager implementations
    # ─────────────────────────────────────────────────────────────────────────
    from ..conversation_manager.handle import ConversationManagerHandle
    from ..conversation_manager.simulated import SimulatedConversationManagerHandle

    register("conversation", "real", ConversationManagerHandle)
    register("conversation", "simulated", SimulatedConversationManagerHandle)

    # ─────────────────────────────────────────────────────────────────────────
    # KnowledgeManager implementations (optional manager)
    # ─────────────────────────────────────────────────────────────────────────
    from ..knowledge_manager.knowledge_manager import KnowledgeManager
    from ..knowledge_manager.simulated import SimulatedKnowledgeManager

    register("knowledge", "real", KnowledgeManager)
    register("knowledge", "simulated", SimulatedKnowledgeManager)

    # ─────────────────────────────────────────────────────────────────────────
    # GuidanceManager implementations (optional manager)
    # ─────────────────────────────────────────────────────────────────────────
    from ..guidance_manager.guidance_manager import GuidanceManager
    from ..guidance_manager.simulated import SimulatedGuidanceManager

    register("guidance", "real", GuidanceManager)
    register("guidance", "simulated", SimulatedGuidanceManager)

    # ─────────────────────────────────────────────────────────────────────────
    # SecretManager implementations (optional manager)
    # ─────────────────────────────────────────────────────────────────────────
    from ..secret_manager.secret_manager import SecretManager
    from ..secret_manager.simulated import SimulatedSecretManager

    register("secrets", "real", SecretManager)
    register("secrets", "simulated", SimulatedSecretManager)

    # ─────────────────────────────────────────────────────────────────────────
    # SkillManager implementations (optional manager)
    # ─────────────────────────────────────────────────────────────────────────
    from ..skill_manager.skill_manager import SkillManager
    from ..skill_manager.simulated import SimulatedSkillManager

    register("skills", "real", SkillManager)
    register("skills", "simulated", SimulatedSkillManager)

    # ─────────────────────────────────────────────────────────────────────────
    # WebSearcher implementations (optional manager)
    # ─────────────────────────────────────────────────────────────────────────
    from ..web_searcher.web_searcher import WebSearcher
    from ..web_searcher.simulated import SimulatedWebSearcher

    register("web_search", "real", WebSearcher)
    register("web_search", "simulated", SimulatedWebSearcher)

    # ─────────────────────────────────────────────────────────────────────────
    # GlobalFileManager implementations (optional manager)
    # ─────────────────────────────────────────────────────────────────────────
    from ..file_manager.global_file_manager import GlobalFileManager
    from ..file_manager.simulated import SimulatedGlobalFileManager

    register("files", "real", GlobalFileManager)
    register("files", "simulated", SimulatedGlobalFileManager)


# Populate on first import
_populate_registry()
