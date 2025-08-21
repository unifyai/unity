from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from unity.common.llm_helpers import SteerableToolHandle

logger = logging.getLogger(__name__)

__all__ = [
    "BaseActor",
    "PhoneCallHandle",
    "BrowserSessionHandle",
    "ComsManager",
]

# --------------------------------------------------------------------------- #
# BaseActor
# --------------------------------------------------------------------------- #


class BaseActor(ABC):
    """
    Abstract contract that every concrete actor must satisfy.

    An actor is a component capable of performing work based on a natural
    language description. It returns a steerable handle that can be paused,
    resumed, interjected, or stopped. This type is intentionally decoupled
    from any task-specific terminology or lifecycle.
    """

    # ─────────────────────────── Work management ────────────────────────── #

    @abstractmethod
    async def act(
        self,
        description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Start performing work from a free‑form natural language description and
        return a steerable handle for controlling the ongoing activity.
        """
