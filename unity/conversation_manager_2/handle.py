from __future__ import annotations

import asyncio
import json
import uuid
import time
from typing import Optional, Type, TypeVar, Literal
from datetime import datetime, timezone
import os
import redis.asyncio as redis
from pydantic import BaseModel
from enum import Enum
import unify
from unity.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.common.tool_spec import ToolSpec
from .base import BaseConversationManagerHandle
from .new_events import NotificationInjectedEvent, NotificationUnpinnedEvent
import logging

T = TypeVar("T", bound=[BaseModel, Enum])

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
        contact_id: int,
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

    async def _tool_get_latest_user_messages(
        self,
        delay: float = 2.0,
        max_messages: int = 20,
        since_ts: float | None = None,
        sender_filter: Literal["user", "assistant", "all"] = "user",
    ) -> dict:
        """
        Polls the durable transcript store for recent messages in this conversation.
        """
        if delay > 0:
            await asyncio.sleep(delay)

        clauses = []
        if sender_filter == "user":
            clauses.append("sender_id != 0")  # 0 is the assistant id
        elif sender_filter == "assistant":
            clauses.append("sender_id == 0")

        filter_expr = " and ".join(clauses)
        logger.info(f'TOOL: Polling transcript with filter: "{filter_expr}"')

        # _filter_messages is synchronous, so we run it in a thread to avoid blocking.
        def _fetch_from_transcript():
            return self._tm._filter_messages(filter=filter_expr, limit=max_messages)[
                "messages"
            ]

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

    async def get_full_transcript(
        self,
        max_messages: int = 50,
    ) -> dict:
        """
        Retrieves the full conversation transcript from the rolling window,
        including both user and assistant messages.
        """
        return await self._tool_get_latest_user_messages(
            delay=0,
            max_messages=max_messages,
            sender_filter="all",
        )

    # ─────────────────────────────────────────────────────────────
    # Conversation-Specific Operations
    # ─────────────────────────────────────────────────────────────

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

        # Include target conversation ID so CM knows if the event is for it
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
    ) -> SteerableToolHandle:
        """
        Asks a question to the user and returns a handle to the running sub-conversation.
        """
        if self._stopped:
            raise RuntimeError("Cannot ask a stopped handle.")

        ask_start_ts = time.time()
        llm = unify.AsyncUnify(
            "claude-4.5-sonnet@anthropic",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
        )

        # Build the schema requirement section only if response_format is provided
        schema_requirement = ""
        if response_format:
            schema_requirement = f"""
        Once you have the answer, you MUST respond with a JSON object matching the following Pydantic schema:
        {response_format.model_json_schema()}
        """

        final_requirement = (
            "- Once you have the user's answer, your final response MUST be a JSON object that strictly conforms to the provided Pydantic model schema. Do not add any extra keys or commentary."
            if response_format
            else "- Once you have the user's answer, respond with a clear and concise summary of what they said."
        )

        system_prompt = f"""
        You are an intelligent sub-agent embedded within a larger conversational system. Your **sole mission** is to determine the user's answer to a *single, specific question* based primarily on the existing conversation history. You must be efficient, accurate, and avoid asking the user if the answer can be reasonably inferred.

        ---
        ### **🎯 YOUR CURRENT MISSION**
        Determine the user's answer to the question: **'{question}'**

        {schema_requirement}
        ---
        ### **🛠️ YOUR TOOLS**
        1.  `_tool_get_latest_user_messages(delay: float, since_ts: float)` -> Waits `delay` seconds, then fetches recent user messages from the transcript occurring *after* the `since_ts` timestamp. **Always use this first.**
        2.  `_tool_interject_conversation(text: str)` -> Sends a message *to* the user. Use this **only as a last resort** if the answer cannot be inferred. Returns the timestamp of when the message was sent.

        ---
        ### **📜 CORE PRINCIPLES & DECISION PROCESS**

        **1. 🏛️ The Golden Rule: Analyze History First, Infer Actively.**
            * **Mandatory First Step:** ALWAYS start by calling `_tool_get_latest_user_messages` to retrieve the relevant conversation history. Analyze the *entire* retrieved transcript carefully.
            * **Prioritize Recency:** Pay close attention to message timestamps. **More recent user statements generally reflect their current state or intention** and should be weighted more heavily than older, potentially outdated information.
            * **Identify Corrections:** Actively look for user corrections (phrases like "Actually, I meant...", "Sorry, ignore that...", "No, it should be...", "My mistake, it's..."). **Explicit corrections *override* previous conflicting statements** from the user. They are your strongest signal for the current truth.
            * **Infer from Implication:** Use common sense and contextual reasoning. If the user's recent statements *clearly imply* the answer to your mission question (even if not stated verbatim), you **MUST** infer the answer. Don't just match keywords; understand the *meaning* in the context of the ongoing conversation.

        **2. ✅ Provide the Answer Immediately if Found/Inferred.**
            * If your analysis of the transcript (considering recency and corrections) provides a confident answer to your mission question, your task is **complete**.
            * Immediately call `final_answer` with the appropriate structured response. **DO NOT use any more tools.**

        **3. ❓ Ask Only When Genuinely Necessary.**
            * Only if, after thorough analysis of the *latest* relevant messages, the answer is **truly missing** OR there's an **unresolvable ambiguity** (e.g., conflicting statements *of similar recency* with no explicit correction making one clearly dominant), should you proceed to ask the user.
            * **Formulate a Clear Question:** If you must ask, use `_tool_interject_conversation` to send a concise question that directly addresses the missing information needed for *your specific mission*.
            * **Wait for Reply:** After asking, use `_tool_get_latest_user_messages` in a loop (with appropriate delays) to wait for the user's response. Analyze their reply using the same principles (recency, corrections).
            * **Avoid Leading Questions:** Don't ask questions that suggest an answer (e.g., "So you want option B, right?"). Just ask for the specific information needed based on the options available or the nature of the question.
            * **Do Not Invent Intentions:** Do not infer complex actions or choices (like assuming the user wants to backtrack or cancel) *unless the user explicitly states it* or the context *strongly and unambiguously* implies it. Your primary job is information gathering for *your* question.

        **4. 🛑 Stop When Answer is Acquired.**
            * As soon as you obtain a confident answer (either through initial inference or after asking and receiving a reply), **immediately stop using tools** and call `final_answer`.

        ---
        ### **✨ GUIDING EXAMPLES**

        #### **Example 1: Proactive Inference**

        **Scenario:** Your mission is to determine "What is the user's desired product category?" (Options: Electronics, Clothing, Groceries).
        Expected response format: `{{"category": "Electronics" | "Clothing" | "Groceries"}}`

        **Recent Transcript:**
        - Agent: "What kind of item are you looking for today?"
        - User: "I need a new pair of running shoes."
        - User: "And maybe some athletic socks if you have them."

        **❌ INCORRECT Behavior (Missing Obvious Context):**
        1.  Call `_tool_interject_conversation("Is that Clothing or something else?")`
        2.  Wait for response
            *Problem:* Running shoes and socks clearly fall under "Clothing". Asking explicitly shows a lack of basic reasoning.

        **✅ CORRECT Behavior (Common-Sense Inference):**
        1.  Call `_tool_get_latest_user_messages` to review transcript.
        2.  Analyze: "Running shoes" and "athletic socks" are both types of apparel.
        3.  Apply reasoning: Apparel belongs to the "Clothing" category.
        4.  Immediately call `final_answer` with `{{"category": "Clothing"}}`.
            *Result:* Natural conversation flow, demonstrates understanding.

        #### **Example 2: Handling Corrections**

        **Scenario:** Your mission is "What is the user's account type?" (Options: Personal, Business).
        Expected response format: `{{"account_type": "Personal" | "Business"}}`

        **Recent Transcript:**
        - User (earlier): "I need help with my business account."
        - Agent: "Okay, looking at your business account..."
        - User (latest): "**Actually, wait, no, sorry**, this is for my **personal** account. My mistake."

        **❌ INCORRECT Behavior:**
        1.  Analyze transcript. See "business account". See "personal account".
        2.  LLM gets confused by conflicting info.
        3.  Calls `_tool_interject_conversation("Sorry, is this for your Personal or Business account?")`
            *Problem:* Fails to recognize the explicit correction ("Actually, wait, no, sorry...") and prioritize the latest statement defining the correct context.

        **✅ CORRECT Behavior:**
        1.  Analyze transcript using `_tool_get_latest_user_messages`.
        2.  Identify "Actually, wait, no, sorry..." as an explicit correction signal.
        3.  Identify "...this is for my **personal** account" as the latest, superseding information.
        4.  Immediately call `final_answer` with `{{"account_type": "Personal"}}`.
            *Result:* Correctly handles the user's change of mind without unnecessary questions.

        ---
        ### **🚨 CRITICAL FINAL STEP**
        {final_requirement}

        ---
        ### **⚙️ Operational Notes:**
        * Use the `since_ts` parameter in `_tool_get_latest_user_messages` effectively, especially after you've sent a message, to only poll for *new* replies. The timestamp is returned by `_tool_interject_conversation`.
        * Be patient when polling for user replies. Use reasonable delays (e.g., 3-7 seconds).
        * Adhere strictly to the required `response_format` if one is specified. No extra text or explanations in the final JSON.
        """
        llm.set_system_message(system_prompt)

        async def _tool_interject_conversation(text: str) -> dict:
            """
            Tool to inject a notification into the live conversation. Returns immediately.
            """
            interject_ts = time.time()
            await self.interject(text)
            logger.info(f"TOOL: Interjected '{text}' at {interject_ts}.")
            return {
                "status": "ok",
                "message": f"Successfully sent '{text}'. Use _tool_get_latest_user_messages to check for a reply.",
                "timestamp": interject_ts,
            }

        tools = {
            "_tool_interject_conversation": ToolSpec(
                fn=_tool_interject_conversation,
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
                    logger.info(
                        f"INFO: Tool loop finished, parsing final result. Final result: {final_result_str}",
                    )

                    if response_format:
                        cleaned_str = final_result_str.strip()
                        if cleaned_str.startswith("```json"):
                            cleaned_str = cleaned_str[7:].strip()
                        if cleaned_str.startswith("```"):
                            cleaned_str = cleaned_str[3:].strip()
                        if cleaned_str.endswith("```"):
                            cleaned_str = cleaned_str[:-3].strip()

                        try:
                            final_payload = json.loads(cleaned_str)

                            # Handle Pydantic Models
                            if issubclass(response_format, BaseModel):
                                validated_model = response_format.model_validate(
                                    final_payload,
                                )
                                logger.info(
                                    f"INFO: Successfully validated response as {response_format.__name__}",
                                )
                                return validated_model

                            # Handle Enums
                            elif issubclass(response_format, Enum):
                                if (
                                    isinstance(final_payload, dict)
                                    and "value" in final_payload
                                ):
                                    enum_member = response_format(
                                        final_payload["value"],
                                    )
                                else:
                                    enum_member = response_format(final_payload)

                                logger.info(
                                    f"INFO: Successfully validated response as {response_format.__name__}",
                                )
                                return enum_member

                        except (
                            json.JSONDecodeError,
                            TypeError,
                            KeyError,
                            ValueError,
                        ) as e:
                            logger.warning(
                                f"WARN: Could not parse final result into model after cleaning: {e}",
                            )
                            raise e
                    else:
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
