from __future__ import annotations

import asyncio
import uuid
import time
import inspect
from typing import Optional, Type, TypeVar, TYPE_CHECKING
from pydantic import BaseModel
from enum import Enum
from unity.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle
from unity.common.llm_client import new_llm_client
from unity.manager_registry import ManagerRegistry
from .base import BaseConversationManagerHandle
from .events import (
    NotificationInjectedEvent,
    NotificationUnpinnedEvent,
    DirectMessageEvent,
)
from .prompt_builders import build_ask_handle_prompt
import logging

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker
    from unity.transcript_manager.base import BaseTranscriptManager

T = TypeVar("T", bound=[BaseModel, Enum])

logger = logging.getLogger(__name__)


class ConversationManagerHandle(BaseConversationManagerHandle):
    """
    The concrete implementation for steering a live ConversationManager instance.

    This handle communicates with the ConversationManager over the event broker,
    allowing components like the Actor to steer the conversation
    by publishing and subscribing to specific event channels.
    """

    def __init__(
        self,
        event_broker: "InMemoryEventBroker",
        conversation_id: str,
        contact_id: int,
        *,
        transcript_manager: "BaseTranscriptManager | None" = None,
        conversation_manager: "ConversationManager",
    ):
        """
        Initializes the handle for a specific conversation.
        """
        self.event_broker = event_broker
        self.conversation_id = conversation_id
        self.contact_id = contact_id
        self._tm = transcript_manager or ManagerRegistry.get_transcript_manager()
        self.conversation_manager = conversation_manager

        self._steering_channel = "app:comms:steering"
        self._stopped = False
        self._final_result = "Handle is active."

    # ─────────────────────────────────────────────────────────────
    # Conversation-Specific Operations
    # ─────────────────────────────────────────────────────────────
    async def get_full_transcript(
        self,
        max_messages: int = 20,
    ) -> dict:
        """
        Polls the durable transcript store for recent messages in this conversation.
        """

        # _filter_messages is synchronous, so we run it in a thread to avoid blocking.
        def _fetch_from_transcript():
            return self._tm._filter_messages(limit=max_messages)["messages"]

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

    async def send_notification(
        self,
        content: str,
        *,
        source: str = "system",
        interjection_id: Optional[str] = None,
        pinned: bool = False,
    ) -> dict:
        """
        Sends a notification to the live conversation by publishing an event.
        """
        if self._stopped:
            return {"status": "error", "message": "Handle is stopped."}

        # Generate ID if not provided
        if interjection_id is None:
            interjection_id = str(uuid.uuid4().hex[:12])

        event = NotificationInjectedEvent(
            content=content,
            source=source,
            target_conversation_id=self.conversation_id,
            interjection_id=interjection_id,
            pinned=pinned,
        )
        # Publish to unified steering channel (picked up by app:comms:* subscription)
        await self.event_broker.publish(self._steering_channel, event.to_json())

        return {
            "status": "ok",
            "message": "Notification event published.",
            "interjection_id": interjection_id,
        }

    # ─────────────────────────────────────────────────────────────
    # Standard SteerableToolHandle Methods
    # ─────────────────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        *,
        response_format: Optional[Type[T]] = None,
    ) -> SteerableToolHandle:
        """
        Asks a question to the user and returns a handle to the running sub-conversation.

        Args:
            question: The question to ask the user
            response_format: Optional Pydantic model or Enum type for structured responses
        """
        if self._stopped:
            raise RuntimeError("Cannot ask a stopped handle.")

        cm_handle = self

        ask_start_ts = time.time()

        # Build recent transcript from contact_index for LLM context
        recent_transcript_for_prompt: str = ""
        try:
            contact = (
                self.conversation_manager.call_manager.call_contact
                or self.conversation_manager.contact_index.get_contact(
                    contact_id=self.contact_id,
                )
            )

            conversation_turns, _ = self.conversation_manager.get_recent_transcript(
                contact=contact,
                max_messages=20,
            )

            if conversation_turns:
                prompt_lines = [
                    f"- {turn['role']}: {turn['content']}"
                    for turn in conversation_turns
                ]
                recent_transcript_for_prompt = (
                    "Recent Transcript (last 20 messages):\n" + "\n".join(prompt_lines)
                )
            else:
                recent_transcript_for_prompt = "Recent Transcript: (none)"
        except Exception as e:
            logger.error(f"Could not fetch transcript context: {e}")
            recent_transcript_for_prompt = "Recent Transcript: (error)"

        # Build prompts using prompt_builders
        static_prompt, dynamic_prompt = build_ask_handle_prompt(
            question=question,
            recent_transcript=recent_transcript_for_prompt,
        )

        # Build content array with optional handler context
        content_parts = [
            {
                "type": "text",
                "text": static_prompt,
            },
            {
                "type": "text",
                "text": dynamic_prompt,
            },
        ]

        system_header_msg = {
            "role": "system",
            "content": content_parts,
        }
        kickoff_user_msg = {
            "role": "user",
            "content": f"Answer the question: '{question}'",
        }
        seeded_messages: list[dict] = [system_header_msg, kickoff_user_msg]

        user_reply_future = asyncio.Future()

        # This handles PATH 2 (Ask & Wait).
        async def ask_question(text: str):
            """
            Asks the user a question and WAITS for a reply.
            This tool BLOCKS until the user speaks.
            Use this when you need to ask a clarifying question (PATH 2).
            """
            nonlocal user_reply_future
            # Speak to user via direct speech (bypasses Main CM Brain)
            await self.event_broker.publish(
                "app:comms:direct_speech",
                DirectMessageEvent(content=text).to_json(),
            )

            try:
                # If an interjection already provided the answer (patient mode),
                # return it immediately instead of creating a new future.
                if user_reply_future.done():
                    user_msg = user_reply_future.result()
                    # Reset for potential future questions
                    user_reply_future = asyncio.Future()
                    return f"User replied: {user_msg}"

                user_msg = await asyncio.wait_for(user_reply_future, timeout=120)
                return f"User replied: {user_msg}"
            except asyncio.TimeoutError:
                return "Timed out waiting for user reply."

        tools = {
            "ask_question": ask_question,
            "ask_historic_transcript": self._tm.ask,
        }

        # ──────────────────────────────────────────────────────────────────
        # 3. START THE LOOP
        # ──────────────────────────────────────────────────────────────────
        llm = new_llm_client(
            return_full_completion=False,
        )

        # Get the parent lineage from the ConversationManager's session logger
        parent_lineage: list[str] = []
        if hasattr(self.conversation_manager, "_session_logger"):
            parent_lineage = self.conversation_manager._session_logger.child_lineage()

        # Pass response_format directly - the async tool loop handles
        # final_answer tool injection automatically
        inner_handle = start_async_tool_loop(
            client=llm,
            message=seeded_messages,
            tools=tools,
            response_format=response_format,
            interrupt_llm_with_interjections=True,
            loop_id="ConversationManager.ask",
            parent_lineage=parent_lineage,
        )

        # ──────────────────────────────────────────────────────────────────
        # 4. THE WRAPPER (The Bridge)
        # ──────────────────────────────────────────────────────────────────
        class InterceptingHandle(SteerableToolHandle):
            def __init__(self):
                pass

            # Delegate standard lifecycle methods
            async def stop(self, reason: Optional[str] = None, **kwargs):
                return await inner_handle.stop(reason, **kwargs)

            async def pause(self):
                return await inner_handle.pause()

            async def resume(self):
                return await inner_handle.resume()

            def done(self):
                return inner_handle.done()

            # Delegate event APIs
            async def next_clarification(self) -> dict:
                return await inner_handle.next_clarification()

            async def next_notification(self) -> dict:
                return await inner_handle.next_notification()

            async def answer_clarification(self, call_id: str, answer: str) -> None:
                return await inner_handle.answer_clarification(call_id, answer)

            async def ask(self, question: str, **kwargs) -> SteerableToolHandle:
                return await inner_handle.ask(question, **kwargs)

            # INTERJECT HANDLER (Triggered by ConversationManager)
            async def interject(self, message: str, **kwargs):
                await inner_handle.interject(
                    message,
                    trigger_immediate_llm_turn=False,
                    **kwargs,
                )
                if not user_reply_future.done():
                    user_reply_future.set_result(message)

            async def result(self):
                try:
                    raw_result = await inner_handle.result()

                    # Handle the standard stop notice from async tool loop
                    if raw_result == "processed stopped early, no result":
                        return None

                    # Handle null/None results
                    if raw_result is None:
                        return None

                    # Convert result to dict for processing
                    if hasattr(raw_result, "model_dump"):
                        final_payload = raw_result.model_dump()
                    # Parse result if response_format was specified
                    if response_format:
                        if isinstance(raw_result, str):
                            # Parse JSON string into the Pydantic model
                            if inspect.isclass(response_format) and issubclass(
                                response_format,
                                BaseModel,
                            ):
                                return response_format.model_validate_json(raw_result)
                            elif inspect.isclass(response_format) and issubclass(
                                response_format,
                                Enum,
                            ):
                                import json

                                data = json.loads(raw_result)
                                if isinstance(data, dict) and "value" in data:
                                    return response_format(data["value"])
                                return response_format(data)
                        elif isinstance(raw_result, dict):
                            if inspect.isclass(response_format) and issubclass(
                                response_format,
                                BaseModel,
                            ):
                                return response_format.model_validate(raw_result)
                            elif inspect.isclass(response_format) and issubclass(
                                response_format,
                                Enum,
                            ):
                                if "value" in raw_result:
                                    return response_format(raw_result["value"])
                                return response_format(raw_result)
                        elif isinstance(raw_result, response_format):
                            return raw_result
                    return raw_result

                finally:
                    if cm_handle.conversation_manager.active_ask_handle == self:
                        cm_handle.conversation_manager.active_ask_handle = None

        # Register with CM
        wrapped_handle = InterceptingHandle()
        self.conversation_manager.active_ask_handle = wrapped_handle

        return wrapped_handle

    async def interject(
        self,
        message: str,
        *,
        pinned: bool = False,
        interjection_id: Optional[str] = None,
    ) -> None:
        """
        Send an interjection to the conversation.

        Args:
            message: The message content to inject
            pinned: If True, the interjection persists for the entire session
            interjection_id: Optional explicit ID (auto-generated if not provided)
        """
        await self.send_notification(
            message,
            source="interjection",
            interjection_id=interjection_id,
            pinned=pinned,
        )

    async def unpin_interjection(self, interjection_id: str) -> dict:
        """
        Unpin a previously pinned interjection.

        Args:
            interjection_id: The ID of the interjection to unpin

        Returns:
            Dict with status indicating success
        """
        if self._stopped:
            return {"status": "error", "message": "Handle is stopped."}

        event = NotificationUnpinnedEvent(
            interjection_id=interjection_id,
            target_conversation_id=self.conversation_id,
        )
        await self.event_broker.publish(self._steering_channel, event.to_json())

        return {
            "status": "ok",
            "message": f"Unpin request sent for interjection {interjection_id}",
            "interjection_id": interjection_id,
        }

    async def stop(self, reason: Optional[str] = None, **kwargs) -> str:
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

    async def pause(self) -> str:
        return "ConversationManagerHandle does not support pausing."

    async def resume(self) -> str:
        return "ConversationManagerHandle does not support resuming."

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        pass
