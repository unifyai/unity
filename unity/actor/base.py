from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Type
from pydantic import BaseModel

from unity.common.async_tool_loop import SteerableToolHandle

logger = logging.getLogger(__name__)

__all__ = [
    "BaseActor",
    "BaseActorHandle",
    "PhoneCallHandle",
    "BrowserSessionHandle",
    "ComsManager",
]

# --------------------------------------------------------------------------- #
# BaseActor
# --------------------------------------------------------------------------- #


class BaseActorHandle(SteerableToolHandle, ABC):
    """
    Marker base class for all actor handles returned by Actor.act().

    This provides a common nominal type across actor implementations while
    preserving the unified steerable surface inherited from SteerableToolHandle.
    Implementations are free to add additional helpers or properties, but the
    core pause/resume/stop/interject/ask/result interface must remain intact.
    """


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

    Usage guidance (LLM‑facing)
    ---------------------------
    Prefer calling ``Actor.act`` when the user's instruction implies a live,
    ad‑hoc, conversational session that should happen "now" inside the current
    chat, especially when the activity involves controlling tools or a UI in
    short iterative steps. Typical phrasings include:

    - "open a browser", "open a window", "navigate/click/show me"
    - "walk me through", "let's set this up together", "guide me live"
    - "troubleshoot together", "pair on this", "step‑by‑step now"

    This interface starts a live session and returns a steerable handle; it does
    not create durable records or schedules.
    """

    _as_caller_description: str = (
        "the Actor, performing a live action on behalf of the end user"
    )

    # ─────────────────────────── Work management ────────────────────────── #

    @abstractmethod
    async def act(
        self,
        description: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Start performing work from a free‑form natural language description and
        return a steerable handle for controlling the ongoing activity.

        Use this for live, conversational, sandbox-style execution within the
        current session. The returned handle supports pause/resume/interject/
        stop and ultimately yields a single result string.
        """
