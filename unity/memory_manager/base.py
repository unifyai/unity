# memory_manager/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseMemoryManager(ABC):
    """
    *Offline* memory-maintenance helper that is invoked every 50 messages.

    All public methods consume **one** request and return a final value
    (they do **not** expose live, steerable handles).
    """

    @abstractmethod
    async def update_contacts(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str: ...

    async def update_contact_bio(
        self,
        transcript: str,
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str: ...

    async def update_contact_rolling_summary(
        self,
        transcript: str,
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    async def update_knowledge(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str: ...
