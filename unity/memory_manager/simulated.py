from __future__ import annotations

from typing import Any, Optional

from .memory_manager import MemoryManager
import functools

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
from ..common.simulated import (
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
    SimulatedLineage,
)

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
        # Accept but ignore parameters that real MemoryManager may use
        loop: Any = None,
        **kwargs: Any,
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

    # ------------------------------------------------------------------ #
    # Public methods -- add simulated logging wrappers                    #
    # ------------------------------------------------------------------ #
    @functools.wraps(MemoryManager.update_contacts, updated=())
    async def update_contacts(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        sched = maybe_tool_log_scheduled(
            "SimulatedMemoryManager.update_contacts",
            "update_contacts",
            {
                "transcript_chars": (
                    len(transcript) if isinstance(transcript, str) else 0
                ),
                "has_guidance": guidance is not None,
            },
        )
        result = await super().update_contacts(transcript, guidance)
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "update_contacts",
                {
                    "result_preview": SimulatedLineage.preview(str(result)),
                },
                t0,
            )
        return result

    @functools.wraps(MemoryManager.update_knowledge, updated=())
    async def update_knowledge(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        sched = maybe_tool_log_scheduled(
            "SimulatedMemoryManager.update_knowledge",
            "update_knowledge",
            {
                "transcript_chars": (
                    len(transcript) if isinstance(transcript, str) else 0
                ),
                "has_guidance": guidance is not None,
            },
        )
        result = await super().update_knowledge(transcript, guidance)
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "update_knowledge",
                {"result_preview": SimulatedLineage.preview(str(result))},
                t0,
            )
        return result

    @functools.wraps(MemoryManager.update_tasks, updated=())
    async def update_tasks(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        sched = maybe_tool_log_scheduled(
            "SimulatedMemoryManager.update_tasks",
            "update_tasks",
            {
                "transcript_chars": (
                    len(transcript) if isinstance(transcript, str) else 0
                ),
                "has_guidance": guidance is not None,
            },
        )
        result = await super().update_tasks(transcript, guidance)
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "update_tasks",
                {"result_preview": SimulatedLineage.preview(str(result))},
                t0,
            )
        return result

    @functools.wraps(MemoryManager.process_chunk, updated=())
    async def process_chunk(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        sched = maybe_tool_log_scheduled(
            "SimulatedMemoryManager.process_chunk",
            "process_chunk",
            {
                "transcript_chars": (
                    len(transcript) if isinstance(transcript, str) else 0
                ),
                "has_guidance": guidance is not None,
            },
        )
        result = await super().process_chunk(transcript, guidance)
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "process_chunk",
                {"result_preview": SimulatedLineage.preview(str(result))},
                t0,
            )
        return result

    @functools.wraps(MemoryManager.reset, updated=())
    async def reset(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedMemoryManager.reset",
            "reset",
            {},
        )
        await super().reset()
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "reset",
                {"outcome": "reset"},
                t0,
            )
