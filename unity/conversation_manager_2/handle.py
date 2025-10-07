from __future__ import annotations

import asyncio
import json
import uuid
import time
from typing import Optional, Type, TypeVar
from datetime import datetime, timezone
import os
import redis.asyncio as redis
from pydantic import BaseModel

import unify
from unity.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.common.tool_spec import ToolSpec
from .base import BaseConversationManagerHandle
from .new_events import NotificationInjectedEvent
import logging

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


# Helper function to format timestamps for transcript queries
def _to_iso(ts: float) -> str:
    """Converts a UNIX timestamp to a timezone-aware ISO 8601 string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class ConversationManagerHandle(BaseConversationManagerHandle):
    """
    The concrete implementation for steering a live ConversationManager instance.

    This handle communicates with the ConversationManager over a Redis event broker,
    allowing external processes like the Actor or Conductor to steer the conversation
    by publishing and subscribing to specific event channels.
    """

    def __init__(
        self,
        event_broker: redis.Redis,
        conversation_id: str,
        contact_id: str,
        *,
        transcript_manager: TranscriptManager | None = None,
    ):
        """
        Initializes the handle for a specific conversation.
        """
        self.event_broker = event_broker
        self.conversation_id = conversation_id
        self.contact_id = contact_id
        self._tm = transcript_manager or TranscriptManager()

        self._input_channel = f"app:conversation_manager:input:{self.conversation_id}"
        self._stopped = False
        self._final_result = "Handle is active."


    async def interject(self, message: str) -> None:
        """A simplified interjection that sends a notification."""
        await self.send_notification(message, source="interjection")

    def stop(self, reason: Optional[str] = None) -> str:
        """Stops the handle."""
        if self._stopped:
            return "Handle already stopped."
        self._stopped = True
        self._final_result = (
            f"Handle stopped. Reason: {reason or 'No reason provided.'}"
        )
        return self._final_result

    def done(self) -> bool:
        return self._stopped

    async def result(self) -> str:
        while not self._stopped:
            await asyncio.sleep(0.1)
        return self._final_result

    # --- Other SteerableToolHandle methods (no-op for this handle) ---

    def pause(self) -> str:
        return "ConversationManagerHandle does not support pausing."

    def resume(self) -> str:
        return "ConversationManagerHandle does not support resuming."

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        pass
