# memory_manager/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from ..manager_registry import SingletonABCMeta


class BaseMemoryManager(ABC, metaclass=SingletonABCMeta):
    """
    *Offline* memory-maintenance helper that is invoked every 50 messages (by default).

    All public methods consume **one** request and return a final value
    (they do **not** expose live, steerable handles).
    """

    _as_caller_description: str = (
        "the MemoryManager, performing offline memory maintenance"
    )

    @abstractmethod
    async def update_contacts(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    async def update_knowledge(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    async def update_tasks(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    async def process_chunk(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str: ...
