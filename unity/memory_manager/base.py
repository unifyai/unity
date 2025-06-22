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
    async def update_contacts(self, transcript: str) -> str:  # new / updated contacts
        ...

    @abstractmethod
    async def update_contact_bio(
        self,
        transcript: str,
        latest_bio: Optional[str] = None,
    ) -> str:  # new bio
        ...

    @abstractmethod
    async def update_contact_rolling_summary(
        self,
        transcript: str,
        latest_rolling_summary: Optional[str] = None,
    ) -> str:  # new rolling summary
        ...
