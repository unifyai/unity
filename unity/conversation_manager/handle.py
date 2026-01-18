from __future__ import annotations

import asyncio
import json
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
        overall_timeout: int = 300,
        task_instructions: Optional[str] = None,
    ) -> SteerableToolHandle:
        """
        Asks a question to the user and returns a handle to the running sub-conversation.

        Args:
            question: The question to ask the user
            response_format: Optional Pydantic model or Enum type for structured responses
            overall_timeout: Maximum time to wait for a response (seconds)
            task_instructions: Optional specific instructions for this task to be injected into the prompt
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

            conversation_turns, _ = (
                self.conversation_manager.get_recent_voice_transcript(
                    contact=contact,
                    max_messages=20,
                )
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
        response_format_schema = (
            response_format.model_json_schema() if response_format else None
        )
        static_prompt, dynamic_prompt = build_ask_handle_prompt(
            question=question,
            recent_transcript=recent_transcript_for_prompt,
            response_format_schema=response_format_schema,
            task_instructions=task_instructions,
        )

        # Build content array with optional handler context
        content_parts = [
            {
                "type": "text",
                "text": static_prompt,
                # "cache_control": {"type": "ephemeral"},
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
            "content": f"Start by attempting to infer the answer from the RECENT TRANSCRIPT using strong common-sense reasoning. If you can infer with 90%+ confidence, use PATH 1 (return the single `{{\"acknowledgment\": ..., \"final_answer\": ...}}` JSON). Only use PATH 2 (call `ask_question`) if inference is genuinely impossible. CRITICAL: Your acknowledgments MUST use PAST TENSE to confirm actions are complete (e.g., \"I've selected **X** and we're proceeding\"). For corrections, use explicit completion language (e.g., \"I've updated the room to **Y** as per your correction. Thanks for clarifying, and we're continuing\"). Before generating ANY acknowledgment, check the recent transcript (last 3-5 messages) to see if you've ALREADY acknowledged this issue—if so, use a shorter navigation message instead. If you use PATH 2 and the user replies, VERIFY their response actually answers your question before returning it—if it doesn't, ask a focused follow-up question (you can call the tool multiple times in the same path). Question: '{question}'. Started at {ask_start_ts}.",
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
                if user_reply_future.done():
                    user_reply_future = asyncio.Future()

                user_msg = await asyncio.wait_for(user_reply_future, timeout=120)
                return f"User replied: {user_msg}"
            except asyncio.TimeoutError:
                return "Timed out waiting for user reply."

        tools = {
            "ask_question": ask_question,
            "ask_historic_transcript": self._tm.ask,
        }

        # ──────────────────────────────────────────────────────────────────
        # Dynamic Response Format Wrapper for PATH 1 Acknowledgments
        # ──────────────────────────────────────────────────────────────────
        wrapped_response_format = None
        if response_format:
            from typing import Optional
            from pydantic import Field, create_model

            # Create a wrapper model dynamically
            # The model will have: acknowledgment (optional) + final_answer (the original type)
            wrapped_response_format = create_model(
                f"{response_format.__name__}WithAcknowledgment",
                acknowledgment=(
                    Optional[str],
                    Field(
                        default=None,
                        description="Optional 2-3 sentence acknowledgment message in the user's language (only for PATH 1 inference)",
                    ),
                ),
                final_answer=(
                    response_format,
                    Field(
                        description="The actual answer conforming to the required schema",
                    ),
                ),
                __base__=None,
            )

        # ──────────────────────────────────────────────────────────────────
        # 3. START THE LOOP
        # ──────────────────────────────────────────────────────────────────
        llm = new_llm_client(
            "gemini-2.5-flash@vertex-ai",
            return_full_completion=False,
            reasoning_effort=None,
            service_tier=None,
        )

        # Get the parent lineage from the ConversationManager's session logger
        parent_lineage: list[str] = []
        if hasattr(self.conversation_manager, "_session_logger"):
            parent_lineage = self.conversation_manager._session_logger.child_lineage()

        inner_handle = start_async_tool_loop(
            client=llm,
            message=seeded_messages,
            tools=tools,
            response_format=wrapped_response_format,
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
            def stop(self, reason: Optional[str] = None, **kwargs):
                return inner_handle.stop(reason, **kwargs)

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

                return "Interjection processed"

            async def result(self):
                try:
                    raw_result = await inner_handle.result()

                    # Handle the standard stop notice from async tool loop
                    if raw_result == "processed stopped early, no result":
                        return None

                    # Convert result to dict for processing
                    if hasattr(raw_result, "model_dump"):
                        final_payload = raw_result.model_dump()
                    elif isinstance(raw_result, dict):
                        final_payload = raw_result
                    elif isinstance(raw_result, str):
                        # Strip markdown code fences if present (```json ... ```)
                        json_str = raw_result.strip()
                        if json_str.startswith("```"):
                            # Remove opening fence (```json or ```)
                            first_newline = json_str.find("\n")
                            if first_newline != -1:
                                json_str = json_str[first_newline + 1 :]
                            # Remove closing fence
                            if json_str.rstrip().endswith("```"):
                                json_str = json_str.rstrip()[:-3].rstrip()
                        try:
                            final_payload = json.loads(json_str)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON result: {raw_result}")
                    else:
                        raise ValueError(f"Unexpected result type: {type(raw_result)}")

                    # Send PATH 1 acknowledgment if present
                    if (
                        "acknowledgment" in final_payload
                        and final_payload["acknowledgment"]
                    ):
                        await cm_handle.event_broker.publish(
                            "app:comms:direct_speech",
                            DirectMessageEvent(
                                content=final_payload["acknowledgment"],
                            ).to_json(),
                        )

                    # Unwrap final_answer if wrapped
                    answer_payload = final_payload.get("final_answer", final_payload)

                    # Validate and return
                    if response_format:
                        if inspect.isclass(response_format) and issubclass(
                            response_format,
                            BaseModel,
                        ):
                            return response_format.model_validate(answer_payload)
                        elif inspect.isclass(response_format) and issubclass(
                            response_format,
                            Enum,
                        ):
                            if (
                                isinstance(answer_payload, dict)
                                and "value" in answer_payload
                            ):
                                return response_format(answer_payload["value"])
                            return response_format(answer_payload)

                    return answer_payload

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
    ) -> dict:
        """
        Send an interjection to the conversation.

        Args:
            message: The message content to inject
            pinned: If True, the interjection persists for the entire session
            interjection_id: Optional explicit ID (auto-generated if not provided)

        Returns:
            Dict with status and the interjection_id
        """
        return await self.send_notification(
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
