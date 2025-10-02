from __future__ import annotations

from abc import abstractmethod
from typing import Literal
from unity.common.async_tool_loop import SteerableToolHandle


class BaseConversationManagerHandle(SteerableToolHandle):
    """
    Abstract interface for steering a live ConversationManager instance.

    This handle enables external processes (like the Conductor or Actor)
    to inject information into the conversation flow in real-time. The primary
    methods for bidirectional communication are:

    - `ask()`: For the external process to get information FROM the user. (Bottom-up)
    - `send_notification()`: For an external system to send information TO the user. (Top-down)
    """

    # ─────────────────────────────────────────────────────────────
    # Conversation-Specific Operations
    # ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def send_notification(
        self,
        content: str,
        *,
        level: Literal["info", "warning", "urgent"] = "info",
        source: str = "system",
    ) -> dict:
        """
        Sends a notification to the live conversation.

        The notification will be added to the conversation state and
        included in the next LLM context, allowing the conversation
        to react appropriately. This is the primary method for a top-down
        flow of information.

        Parameters
        ----------
        content : str
            The notification message content.
        level : Literal["info", "warning", "urgent"]
            Urgency level affecting how the conversation prioritizes it.
        source : str
            Identifier of the notification source (e.g., "conductor", "task_scheduler").

        Returns
        -------
        dict
            Confirmation with a timestamp and notification ID.
        """
