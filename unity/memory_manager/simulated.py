# memory_manager/simulated.py
"""
Thin wrapper around the unified `MemoryManager` that wires in simulated
sub-managers by default and disables real guardrails/callbacks.
"""

from __future__ import annotations

from typing import Optional

from .memory_manager import MemoryManager
from ..contact_manager.simulated import SimulatedContactManager
from ..transcript_manager.simulated import SimulatedTranscriptManager
from ..knowledge_manager.simulated import SimulatedKnowledgeManager
from ..task_scheduler.simulated import SimulatedTaskScheduler


class SimulatedMemoryManager(MemoryManager):
    """Unified simulated variant that delegates to `MemoryManager`.

    - Uses simulated sub-managers by default (share the same description)
    - Disables guardrails and callbacks via configuration
    - Purges transcript tools entirely (handled in MemoryManager)
    """

    def __init__(
        self,
        description: str = "imaginary scenario",
        *,
        contact_manager: Optional[SimulatedContactManager] = None,
        transcript_manager: Optional[SimulatedTranscriptManager] = None,
        knowledge_manager: Optional[SimulatedKnowledgeManager] = None,
        task_scheduler: Optional[SimulatedTaskScheduler] = None,
    ) -> None:
        super().__init__(
            contact_manager=contact_manager
            or SimulatedContactManager(description=description),
            transcript_manager=transcript_manager
            or SimulatedTranscriptManager(description=description),
            knowledge_manager=knowledge_manager
            or SimulatedKnowledgeManager(description=description),
            task_scheduler=task_scheduler
            or SimulatedTaskScheduler(description=description),
            config=MemoryManager.MemoryConfig(
                enable_callbacks=False,
            ),
        )
