from __future__ import annotations

from typing import Optional

from .memory_manager import MemoryManager

# Base contracts (for type hints)
from ..contact_manager.base import BaseContactManager
from ..transcript_manager.base import BaseTranscriptManager
from ..knowledge_manager.base import BaseKnowledgeManager
from ..task_scheduler.base import BaseTaskScheduler

# Simulated defaults
from ..contact_manager.simulated import SimulatedContactManager
from ..transcript_manager.simulated import SimulatedTranscriptManager
from ..knowledge_manager.simulated import SimulatedKnowledgeManager
from ..task_scheduler.simulated import SimulatedTaskScheduler

__all__ = [
    "SimulatedMemoryManager",
]


class SimulatedMemoryManager(MemoryManager):
    """
    MemoryManager variant that defaults to simulated sub-managers and disables callbacks.

    Optional manager overrides (real or simulated) can be supplied.
    """

    def __init__(
        self,
        description: str = "imaginary scenario",
        *,
        contact_manager: Optional[BaseContactManager] = None,
        transcript_manager: Optional[BaseTranscriptManager] = None,
        knowledge_manager: Optional[BaseKnowledgeManager] = None,
        task_scheduler: Optional[BaseTaskScheduler] = None,
        config: Optional["MemoryManager.MemoryConfig"] = None,
    ) -> None:
        cm = contact_manager or SimulatedContactManager(description=description)
        tm = transcript_manager or SimulatedTranscriptManager(description=description)
        km = knowledge_manager or SimulatedKnowledgeManager(description=description)
        ts = task_scheduler or SimulatedTaskScheduler(description=description)

        # Preserve simulated behavior: callbacks disabled unless explicitly provided
        cfg = (
            config
            if config is not None
            else MemoryManager.MemoryConfig(
                enable_callbacks=False,
            )
        )

        super().__init__(
            contact_manager=cm,
            transcript_manager=tm,
            knowledge_manager=km,
            task_scheduler=ts,
            config=cfg,
        )
