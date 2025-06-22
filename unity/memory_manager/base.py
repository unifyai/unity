from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.llm_helpers import SteerableToolHandle


class BaseMemoryManager(ABC):
    """Public contract for the *MemoryManager* family.

    The manager is responsible for *maintaining* auxiliary per‑contact memory
    (bios and rolling summaries) as well as performing bulk contact updates
    derived from freshly ingested transcripts.
    """

    # ––– Public interface ––––––––––––––––––––––––––––––––––––––––––––––––
    @abstractmethod
    async def update_contacts(
        self,
        transcript: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle: ...

    @abstractmethod
    async def update_contact_bio(
        self,
        transcript: str,
        latest_bio: Optional[str] = None,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle: ...

    @abstractmethod
    async def update_contact_rolling_summary(
        self,
        transcript: str,
        latest_rolling_summary: Optional[str] = None,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle: ...
