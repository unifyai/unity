from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from unity.common.async_tool_loop import SteerableToolHandle

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

    Purpose and positioning
    -----------------------
    The Actor provides a direct, real-time handle to "act" in the world and
    get things done – e.g. open a browser, click UI elements, or perform a
    short-lived sandbox session during a conversation.

    Intended use
    ------------
    Use the Actor for interactive, ephemeral sessions within a live
    conversation (onboarding, guided walkthroughs, ad‑hoc demonstrations).
    It returns a steerable handle suitable for pause/resume/interject/stop.
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

        Use this for live, conversational, sandbox-style execution within the
        current session. The returned handle supports pause/resume/interject/
        stop and ultimately yields a single result string.
        """
