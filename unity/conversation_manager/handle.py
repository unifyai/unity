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

            if (
                contact
                and contact["contact_id"]
                in self.conversation_manager.contact_index.active_conversations
            ):
                active_contact = (
                    self.conversation_manager.contact_index.active_conversations[
                        contact["contact_id"]
                    ]
                )
                voice_thread = active_contact.threads.get("voice", [])
                recent_msgs_raw = list(voice_thread)[-20:] if voice_thread else []

                prompt_lines: list[str] = []
                for msg in recent_msgs_raw:
                    role = "assistant" if msg.name == "You" else "user"
                    content = (msg.content or "").strip()

                    # Skip system messages
                    if content.startswith("<") and content.endswith(">"):
                        continue

                    prompt_lines.append(f"- {role}: {content}")

                recent_transcript_for_prompt = (
                    "Recent Transcript (last 20 messages):\n" + "\n".join(prompt_lines)
                    if prompt_lines
                    else "Recent Transcript: (none)"
                )
            else:
                recent_transcript_for_prompt = (
                    "Recent Transcript: (no active conversation)"
                )
        except Exception as e:
            logger.error(f"Could not fetch transcript context: {e}")
            recent_transcript_for_prompt = "Recent Transcript: (error)"

        # Build the schema requirement section only if response_format is provided
        schema_requirement = ""
        if response_format:
            schema_requirement = f"""
        The Pydantic schema for the final answer is:
        {response_format.model_json_schema()}
        """

        final_requirement = (
            f"""
        - **If you used PATH 1 (INFER)**: Your final response MUST be a single JSON object with TWO keys: `acknowledgment` (your 2-3 sentence message) and `final_answer` (the JSON payload conforming to the Pydantic schema).
        - **If you used PATH 2 (ASK)**: Your final response MUST be ONLY the JSON payload that strictly conforms to the Pydantic model schema.
        """
            if response_format
            else "- Once you have the user's answer, respond with a clear and concise summary of what they said."
        )

        # Build static prompt (cacheable) and dynamic prompt (question-specific)
        task_specific_section = ""
        if task_instructions:
            task_specific_section = f"""
        ---
        ### **📝 TASK SPECIFIC INSTRUCTIONS**
        {task_instructions}
        """

        static_prompt = f"""
        You are the "Brain" of a conversation agent. Your goal is to determine the user's answer to a specific question by listening to a transcript.

        **LANGUAGE:** Infer from the transcript the language the user is speaking in. ALL your acknowledgments and questions MUST be in the same language.
        When you call `ask_question`, the text you provide MUST be in the same language.

        ---
        ### **🛠️ YOUR TOOLS**
        1. RECENT_TRANSCRIPT (seeded below) → Prefer using this directly to infer the answer without calling any tools.
        2. `ask_question(text: str)` → Sends exactly your wording to the user and **BLOCKS** until the user replies.
           - This tool is **ONLY FOR PATH 2 (ASK & WAIT)**.
           - It sends the question to the live conversation and waits for the user's next utterance.
        3. `ask_historic_transcript(text: str)` → Ask questions about the **historic** transcript (content BEFORE the current conversation session).
           - **WARNING**: Do NOT use this for the active conversation. The active conversation is already in RECENT_TRANSCRIPT.
           - Use this ONLY if you need to look up older context (e.g., "What did we discuss last week?").

        {task_specific_section}

        ---
        ### **✅ ANSWER VERIFICATION (PATH 2 CRITICAL RULE)**

        When using PATH 2 (`wait_for_reply=True`), after the user responds:

        **STEP 1: VERIFY the response answers your question**
        - Ask yourself: "Does this response actually answer what I asked?"

        **STEP 2: Handle based on verification result**
        - **If VALID ANSWER**: Return it immediately
        - **If NON-ANSWER/TANGENTIAL**: Ask ONE simplified follow-up to get clarity
        - **If STILL NO ANSWER after follow-up**: Use safe default (e.g., "No additional details provided")
        - **If CORRECTION SIGNAL**: Handle as `go_back`

        **Why this matters**: Blindly accepting non-answers leads to poor data quality and confused users.

        ---
        ### **📜 DECISION FLOW**

        **Choose exactly ONE path per tool loop for efficiency.**

        **CHOOSING YOUR PATH:**
        - **Use PATH 1 (INFER)** when the transcript provides 90%+ certainty about the answer
        - **Use PATH 2 (ASK)** when the user's words don't clearly distinguish between 2+ options (genuinely ambiguous)

        **PATH 1 — INFER & ACKNOWLEDGE (When 90%+ Confident)**
        1. Read RECENT_TRANSCRIPT and apply **strong common-sense reasoning** to infer the answer.
        2. **CRITICAL - Check for Correction signal**: Is the user correcting a previous choice?
           - "Actually it's X not Y" → Infer `go_back` (if applicable)
        3. If you can infer with 90%+ confidence:
           - **FIRST**: Check recent transcript (last 3-5 messages). If you see an acknowledgment that already mentions this issue, DO NOT create a new acknowledgment. Return a navigation message only.
           - **ONLY IF no acknowledgment exists**: Formulate a contextually-aware acknowledgment (following all rules on linguistic variety, 2-3 sentences, etc.).
           - **Use declarative sentences in PATH 1 acknowledgments.** If you need to ask anything—even a soft confirmation—switch to PATH 2.
        4. **CRITICAL - RETURN IN ONE STEP**: Your final response MUST be a **single JSON object** that contains *both* the acknowledgment and the final answer.

            **This is the ONLY way to complete PATH 1. Do NOT call any tools.**

            **SCHEMA FOR PATH 1:**
            ```json
            {{
              "acknowledgment": "Your 2-3 sentence acknowledgment text here (in the user's language).",
              "final_answer": "the Pydantic/Enum JSON you inferred"
            }}
            ```
            **Example:**
            ```json
            {{
              "acknowledgment": "Thanks for that. Since you mentioned [X], I've noted that and we're proceeding.",
              "final_answer": {{
                "value": "some_value"
              }}
            }}
            ```
            This single response will simultaneously send the acknowledgment and complete the step.

        **PATH 2 — ASK & WAIT (When Genuinely Ambiguous)**
        1. Use this when you CANNOT infer with 90%+ confidence.
        2. **Review last 2 turns** before formulating your question to ensure natural flow.
        3. Call `ask_question("...")` with a conversational, focused clarifying question.
        4. **CRITICAL - After user replies, VERIFY the answer (DO NOT blindly accept)**:
           - **STEP A**: Read their response and explicitly ask yourself: "Does this directly answer my question?"
           - **STEP B - If YES (valid answer)**: Return it immediately
           - **STEP C - If NO (non-answer/tangential/vague)**:
             - Call `ask_question` AGAIN with a simplified follow-up question.
             - **This is ALLOWED and ENCOURAGED!** Multiple questions in PATH 2 are expected when verifying answers.
           - **STEP D - If CORRECTION SIGNAL**: Recognize it and handle as `go_back`

        **⚠️  PATH CONSISTENCY GUIDANCE**
        - **PATH 1 (INFER)**: You make **ZERO** tool calls. Your final answer is the special `{{"acknowledgment": ..., "final_answer": ...}}` JSON object.
        - **PATH 2 (ASK)**: You **MUST** call `ask_question(...)`.
          - Multiple calls in PATH 2 are ENCOURAGED (e.g., Ask question → verify response → ask follow-up question).
          - **Choose ONE path** per tool loop - either infer (PATH 1) or ask (PATH 2).

        ---
        ### **💎 CRAFTING HIGH-QUALITY MESSAGES**

        **Before formulating ANY message (acknowledgment or question), you MUST:**
        1. **Review the last 2 conversation turns** to understand the current context
        2. **Be EXPLICIT about your decision**: Always name the specific category/option you've selected and confirm we're moving forward
        3. **Flow naturally**: Make your message feel like a seamless continuation, not a robotic repetition
        4. **VARY YOUR LANGUAGE**: Do NOT repeat the same sentence structures, openers, or action verbs across sequential messages

        ---
        ### **🎨 LINGUISTIC VARIETY (CRITICAL)**

        **THE PROBLEM**: When multiple guidance messages (Path 1 inferences) are sent in a row, you sound robotic.

        **THE SOLUTION: Mix Full Acknowledgments with "Thinking Aloud" Messages**

        - **When acknowledging new information**: Give a full, natural, 2-3 sentence message. But ONLY if you haven't already acknowledged it in recent transcript!

        - **For subsequent steps**: Use shorter "Thinking Aloud" messages that describe what you're doing in a natural, matter-of-fact way.

        - **Move forward with each message**: Each message should progress the conversation. Acknowledge once, then move to categorization, then to specifics.

        **3. VARY YOUR ACTION VERBS AND USE COMPLETED ACTIONS (when you do use full sentences):**
        - "I've selected... and we're proceeding"
        - "Since [reason], I've categorized this under... We're moving to the next step"
        - "I've marked this as... moving forward now"
        - "That's definitely a... issue—I've logged that and proceeding"
        - "Within that, I've chosen... and we're continuing"
        - "And I've specifically recorded this as... Bear with me as we proceed"

        **4. VARY YOUR OPENERS (when you do use full sentences):**
        - No opener - just dive into the statement
        - "Since..." (causal)
        - "You mentioned..." (reference)
        - "For [X] specifically..." (specificity)
        - Occasionally: "Perfect," "Right," (but NOT consecutively)

        ---
        ### **🗣️ MESSAGE QUALITY CHECKLIST**

        **Every message you send (acknowledgment or question) MUST:**
        ✓ **CHECK TRANSCRIPT FIRST** - Before creating ANY acknowledgment, check recent transcript (last 3-5 messages). If acknowledgment exists, create navigation message only.
        ✓ **CONFIRM ACTION COMPLETE** - Use past tense and explicitly state we're moving forward.
        ✓ Explicitly name the category/option you've chosen (use **bold** for emphasis)
        ✓ Acknowledge what the user just said ONCE (show you're listening) - only if you haven't already
        ✓ **MOVE FORWARD WITH EACH MESSAGE** - Each message should progress the conversation from general to specific
        ✓ **BE 2-3 SENTENCES LONG** - When acknowledging NEW information that hasn't been acknowledged yet
        ✓ **VARY YOUR SENTENCE STRUCTURE** - Rotate between different openers and structures
        ✓ Flow naturally from the last 2 conversation turns (seamless continuation)
        ✓ **VARY YOUR COMPLETION PHRASES** - Rotate between "and we're proceeding," "moving forward," "We're proceeding now to," "Bear with me as we proceed"
        ✓ **VARY YOUR OPENERS** - Use different opening phrases for consecutive messages
        """

        # Dynamic prompt parts (question-specific, not cacheable)
        dynamic_prompt = f"""
        ---
        ### **🎯 YOUR CURRENT MISSION**
        Determine the user's answer to the question: **'{question}'**

        {schema_requirement}
        ---
        ### **🚨 CRITICAL FINAL STEP**
        {final_requirement}
        """

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

        content_parts.append(
            {
                "type": "text",
                "text": f"""
        ---
        ### **📋 RECENT TRANSCRIPT CONTEXT**
        {recent_transcript_for_prompt}
        """,
            },
        )

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

                    # Convert result to dict for processing
                    if hasattr(raw_result, "model_dump"):
                        final_payload = raw_result.model_dump()
                    elif isinstance(raw_result, dict):
                        final_payload = raw_result
                    elif isinstance(raw_result, str):
                        try:
                            final_payload = json.loads(raw_result)
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
