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

        self._steering_channel = "app:comms:steering"
        self._stopped = False
        self._final_result = "Handle is active."

    # ────────────────────────────────────────────────────────────────────
    # Non-Blocking Tools for LLM-Orchestrated Polling
    # ────────────────────────────────────────────────────────────────────

    async def _tool_interject_conversation(self, text: str) -> dict:
        """
        Tool to inject a notification into the live conversation. Returns immediately.
        """
        await self.interject(text)
        logger.info(f"TOOL: Interjected '{text}'.")
        return {
            "status": "ok",
            "message": f"Successfully sent '{text}'. Use _tool_get_latest_user_messages to check for a reply.",
        }

    async def _tool_get_latest_user_messages(
        self,
        delay: float = 2.0,
        max_messages: int = 5,
        since_ts: float | None = None,
    ) -> dict:
        """
        Polls the durable transcript store for recent user messages in this conversation.
        """
        if delay > 0:
            await asyncio.sleep(delay)

        clauses = [f"sender_id != 0"]  # 0 is the assistant id
        if since_ts is not None:
            clauses.append(f"timestamp >= '{_to_iso(since_ts)}'")

        filter_expr = " and ".join(clauses)
        logger.info(f'TOOL: Polling transcript with filter: "{filter_expr}"')

        # _filter_messages is synchronous, so we run it in a thread to avoid blocking.
        def _fetch_from_transcript():
            return self._tm._filter_messages(filter=filter_expr, limit=max_messages)

        try:
            # Await the thread-based call
            results = await asyncio.to_thread(_fetch_from_transcript)
        except Exception as e:
            return {"status": "error", "message": f"Transcript read failed: {e}"}

        # Format the results into a clean JSON shape for the LLM
        messages = [
            {
                "message_id": m.message_id,
                "timestamp": getattr(
                    m.timestamp,
                    "isoformat",
                    lambda: str(m.timestamp),
                )(),
                "content": m.content,
                "medium": m.medium.value,
            }
            for m in (results or [])
        ]

        if messages:
            logger.info(f"TOOL: Found {len(messages)} user message(s).")
        else:
            logger.info("TOOL: No new user messages found yet.")

        return {
            "status": "ok",
            "messages": messages,
            "count": len(messages),
        }

    # ─────────────────────────────────────────────────────────────
    # Conversation-Specific Operations
    # ─────────────────────────────────────────────────────────────

    async def send_notification(
        self,
        content: str,
        *,
        source: str = "system",
    ) -> dict:
        """
        Sends a notification to the live conversation by publishing an event.
        """
        if self._stopped:
            return {"status": "error", "message": "Handle is stopped."}

        # Include target conversation ID so CM knows if the event is for it
        event = NotificationInjectedEvent(
            content=content,
            source=source,
            target_conversation_id=self.conversation_id,
        )
        # Publish to unified steering channel (picked up by app:comms:* subscription)
        await self.event_broker.publish(self._steering_channel, event.to_json())

        return {
            "status": "ok",
            "message": "Notification event published.",
            "notification_id": f"notif_{uuid.uuid4().hex[:8]}",
        }

    # ─────────────────────────────────────────────────────────────
    # Standard SteerableToolHandle Methods
    # ─────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        *,
        response_format: Optional[Type[T]] = None,
        overall_timeout: int = 300,
    ) -> SteerableToolHandle:
        """
        Asks a question to the user and returns a handle to the running sub-conversation.
        """
        if self._stopped:
            raise RuntimeError("Cannot ask a stopped handle.")

        ask_start_ts = time.time()
        llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # Build the schema requirement section only if response_format is provided
        schema_requirement = ""
        if response_format:
            schema_requirement = f"""
        Once you have the answer, you MUST respond with a JSON object matching the following Pydantic schema:
        {response_format.model_json_schema()}
        """

        system_prompt = f"""
        You are a sub-agent focused on a single mission: getting the user's answer to a specific question.

        YOUR MISSION: Get the user's answer to: '{question}'
        {schema_requirement}
        YOUR TOOLS:
        1. `_tool_interject_conversation(text: str)` -> Sends a message to the user. Returns instantly.
        2. `_tool_get_latest_user_messages(delay: float, since_ts: float)` -> Waits, then checks for new user messages from the transcript.

        YOUR WORKFLOW (LLM-Orchestrated Polling):
        1. Start by calling `_tool_interject_conversation` to ask the main question.
        2. Call `_tool_get_latest_user_messages`, providing a `delay` (e.g., 3.0s) and `since_ts` (the timestamp when the 'ask' began: {ask_start_ts}) to check for a reply.
        3. Analyze the result:
           - If you get a message that answers your question -> STOP using tools and respond {'with a JSON object conforming to the schema above' if response_format else 'with the answer'}. MISSION COMPLETE.
           - If the list of messages is empty -> call `_tool_get_latest_user_messages` again to continue waiting.
           - If the user is confused -> `_tool_interject_conversation` to clarify, then resume polling.

        CRITICAL:
        - You are in control of the polling loop. Be patient and persistent.
        {'- Once you have the user\'s answer, your final response MUST be a JSON object that strictly conforms to the provided Pydantic model schema. Do not add any extra keys or commentary.' if response_format else '- Once you have the user\'s answer, respond with a clear and concise summary of what they said.'}
        """
        llm.set_system_message(system_prompt)
        tools = {
            "_tool_interject_conversation": ToolSpec(
                fn=self._tool_interject_conversation,
            ),
            "_tool_get_latest_user_messages": ToolSpec(
                fn=self._tool_get_latest_user_messages,
            ),
        }

        handle = start_async_tool_loop(
            client=llm,
            message=f"Start the process to get an answer for: '{question}'. The operation started at timestamp {ask_start_ts}.",
            tools=tools,
            response_format=response_format,
        )

        original_result = handle.result

        async def _wrapped_result() -> T | str:
            try:
                async with asyncio.timeout(overall_timeout):
                    final_result_str = await original_result()
                    logger.info("INFO: Tool loop finished, parsing final result.")

                    if response_format:
                        # If the result is already the correct model, return it directly
                        if isinstance(final_result_str, response_format):
                            return final_result_str

                        # Otherwise, try to parse it as JSON
                        try:
                            validated_model = response_format.model_validate_json(
                                final_result_str,
                            )
                            logger.info(
                                f"INFO: Successfully validated response as {response_format.__name__}",
                            )
                            return validated_model
                        except Exception as e:
                            logger.warning(
                                f"WARN: Could not parse final result into model: {e}",
                            )
                            # Fall back to returning the raw string
                            return final_result_str
                    else:
                        # No validation needed
                        return final_result_str

            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"The 'ask' method timed out after {overall_timeout}s.",
                )
            except Exception as e:
                raise RuntimeError(
                    f"An unexpected error occurred in the 'ask' tool loop: {e}",
                )

        handle.result = _wrapped_result
        return handle

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
