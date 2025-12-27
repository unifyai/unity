from __future__ import annotations

from typing import Optional

from .conductor import Conductor
from ..settings import SETTINGS

# Base contracts (for type hints)
from ..contact_manager.base import BaseContactManager
from ..transcript_manager.base import BaseTranscriptManager
from ..knowledge_manager.base import BaseKnowledgeManager
from ..task_scheduler.base import BaseTaskScheduler
from ..web_searcher.base import BaseWebSearcher
from ..actor.base import BaseActor
from ..guidance_manager.base import BaseGuidanceManager
from ..manager_registry import ManagerRegistry
from ..secret_manager.base import BaseSecretManager
from ..secret_manager.simulated import SimulatedSecretManager

# Simulated implementations (defaults for this subclass)
from ..contact_manager.simulated import SimulatedContactManager
from ..transcript_manager.simulated import SimulatedTranscriptManager
from ..knowledge_manager.simulated import SimulatedKnowledgeManager
from ..task_scheduler.simulated import SimulatedTaskScheduler
from ..web_searcher.simulated import SimulatedWebSearcher
from ..actor.simulated import SimulatedActor
from ..conversation_manager.base import BaseConversationManagerHandle
from ..conversation_manager.simulated import SimulatedConversationManagerHandle
from ..file_manager.base import BaseGlobalFileManager
from ..file_manager.simulated import SimulatedGlobalFileManager, SimulatedFileManager

__all__ = [
    "SimulatedConductor",
    # Re-export so tests can monkeypatch this symbol on this module
    "SimulatedTaskScheduler",
]


class SimulatedConductor(Conductor):
    """
    Conductor variant that defaults to simulated back-ends for all managers and actor.

    Optional manager overrides can still be supplied to use real managers selectively.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        # Optional manager overrides – fall back to simulated defaults in this subclass
        contact_manager: Optional[BaseContactManager] = None,
        transcript_manager: Optional[BaseTranscriptManager] = None,
        knowledge_manager: Optional[BaseKnowledgeManager] = None,
        guidance_manager: Optional[BaseGuidanceManager] = None,
        secret_manager: Optional[BaseSecretManager] = None,
        task_scheduler: Optional[BaseTaskScheduler] = None,
        web_searcher: Optional[BaseWebSearcher] = None,
        actor: Optional[BaseActor] = None,
        conversation_manager: Optional[BaseConversationManagerHandle] = None,
        global_file_manager: Optional[BaseGlobalFileManager] = None,
    ) -> None:
        # Instantiate simulated components unless caller provided overrides
        _actor = (
            actor
            if actor is not None
            else SimulatedActor(
                steps=SETTINGS.actor.SIMULATED_STEPS,
                duration=None,
                simulation_guidance=simulation_guidance,
            )
        )

        _contact_manager = (
            contact_manager
            if contact_manager is not None
            else SimulatedContactManager(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        _transcript_manager = (
            transcript_manager
            if transcript_manager is not None
            else SimulatedTranscriptManager(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        _knowledge_manager = (
            knowledge_manager
            if knowledge_manager is not None
            else SimulatedKnowledgeManager(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        _guidance_manager = (
            guidance_manager
            if guidance_manager is not None
            else ManagerRegistry.get_guidance_manager(
                rolling_summary_in_prompts=rolling_summary_in_prompts,
            )
        )

        _secret_manager = (
            secret_manager
            if secret_manager is not None
            else SimulatedSecretManager(
                description=description,
                log_events=log_events,
                simulation_guidance=simulation_guidance,
            )
        )

        _task_scheduler = (
            task_scheduler
            if task_scheduler is not None
            else SimulatedTaskScheduler(
                description=description,
                log_events=log_events,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                simulation_guidance=simulation_guidance,
            )
        )

        _web_searcher = (
            web_searcher
            if web_searcher is not None
            else SimulatedWebSearcher(
                description=description,
                log_events=log_events,
            )
        )

        _conversation_manager = (
            conversation_manager
            if conversation_manager is not None
            else SimulatedConversationManagerHandle(
                assistant_id="simulated-assistant",
                contact_id="simulated-contact",
                description=description,
                simulation_guidance=simulation_guidance,
            )
        )

        _global_file_manager = (
            global_file_manager
            if global_file_manager is not None
            else SimulatedGlobalFileManager(
                [SimulatedFileManager(), SimulatedFileManager()],
            )
        )

        # Delegate to the real Conductor with our simulated defaults
        super().__init__(
            description=description,
            log_events=log_events,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
            simulation_guidance=simulation_guidance,
            contact_manager=_contact_manager,
            transcript_manager=_transcript_manager,
            knowledge_manager=_knowledge_manager,
            guidance_manager=_guidance_manager,
            secret_manager=_secret_manager,
            task_scheduler=_task_scheduler,
            web_searcher=_web_searcher,
            actor=_actor,
            global_file_manager=_global_file_manager,
            conversation_manager=_conversation_manager,
        )
