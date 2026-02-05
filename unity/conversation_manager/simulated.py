from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional, Type, Any
from pydantic import BaseModel
import unillm

from ..common.llm_client import new_llm_client
from ..common.simulated import (
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
)
from .base import BaseConversationManagerHandle
from ..common import SteerableToolHandle


class SimulatedConversationManagerHandle(
    BaseConversationManagerHandle,
    SimulatedHandleMixin,
):
    """
    Simulated conversation manager handle for testing and demos.

    Uses a stateful LLM to simulate conversation steering without
    actual Redis pub/sub or real conversation state.
    """

    def __init__(
        self,
        assistant_id: str = "",
        contact_id: int = 0,
        *,
        description: str = "A simulated conversation between an AI assistant and a user.",
        simulation_guidance: Optional[str] = None,
        # Accept but ignore parameters that real ConversationManagerHandle uses
        event_broker: Any = None,
        conversation_id: Any = None,
        transcript_manager: Any = None,
        conversation_manager: Any = None,
        **kwargs: Any,
    ):
        self.assistant_id = assistant_id
        self.contact_id = contact_id
        self._description = description
        self._simulation_guidance = simulation_guidance

        # A shared, stateful LLM for maintaining conversation context
        self._llm = new_llm_client(stateful=True)

        # Initialize the system message for the stateful LLM
        system_msg = self._build_system_message()
        self._llm.set_system_message(system_msg)

        # Internal state management
        self._stopped = False
        self._paused = False
        self._final_result = "Conversation is active."

        # Human-friendly log label for consistent hierarchical logging
        self._log_label = SimulatedLineage.make_label("SimulatedConversationManager")

    async def get_full_transcript(self, **kwargs) -> dict:
        """Simulates retrieving the full conversation transcript."""
        if self._stopped:
            return {"status": "error", "message": "Handle is stopped."}

        # The stateful LLM will generate a plausible transcript based on the scenario description.
        prompt = (
            "Based on the conversation scenario, provide a plausible, recent transcript "
            "that includes both user and assistant messages. Return it as a JSON object "
            'with a "messages" key, like {"messages": [{"role": "user", "content": "..."}, ...]}.'
        )
        response = await self._llm.generate(prompt)

        try:
            cleaned_str = response.strip()
            if cleaned_str.startswith("```json"):
                cleaned_str = cleaned_str[7:].strip()
            if cleaned_str.startswith("```"):
                cleaned_str = cleaned_str[3:].strip()
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3].strip()

            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse LLM response into JSON: {response}")

    def _build_system_message(self) -> str:
        """Builds a detailed system message for the stateful LLM, instructing it on how to simulate the conversation."""
        return f"""You are a simulated ConversationManager. Your role is to maintain the state of a conversation and respond to steering commands.

### Conversation Scenario
**Description:** {self._description}
**Assistant ID:** {self.assistant_id}
**Contact ID:** {self.contact_id}

### Core Responsibilities
1.  **Maintain Internal State:** You must remember the conversation history, the user's mood, and any notifications you receive.
2.  **Respond to `ask`:** When you receive a question via `ask`, provide a plausible, in-character response from the simulated user. The response should be concise and directly answer the question.
3.  **Acknowledge `send_notification`:** When a notification is sent, incorporate its content into your internal state and provide a simple JSON confirmation. For example, if you receive "Task 'X' is complete," your subsequent `ask` responses should reflect this knowledge.
4.  **Adhere to Simulation Guidance:** {self._simulation_guidance or "No specific guidance provided."}

### Response Formats
- For `ask` calls, provide a direct, first-person answer as the simulated user.
- For `send_notification` calls, respond with a JSON object like: `{{"status": "ok", "notification_id": "...", "timestamp": "..."}}`
"""

    # ─────────────────────────────────────────────────────────────
    # Conversation-Specific Operations (Minimal Set)
    # ─────────────────────────────────────────────────────────────

    async def send_notification(
        self,
        content: str,
        *,
        source: str = "system",
        interjection_id: Optional[str] = None,
        pinned: bool = False,
    ) -> dict:
        """Simulates sending a notification to the conversation."""
        if self._stopped:
            return {"status": "error", "message": "Handle is stopped."}

        # Generate ID if not provided
        if interjection_id is None:
            interjection_id = str(uuid.uuid4().hex[:12])

        prompt = f"""A notification has been sent to the conversation. Acknowledge it by updating your internal state and returning a JSON confirmation.
- **Content:** {content}
- **Source:** {source}
- **Pinned:** {pinned}
- **Interjection ID:** {interjection_id}
"""
        response = await self._llm.generate(prompt)

        try:
            result = json.loads(response)
            result["interjection_id"] = interjection_id
            return result
        except json.JSONDecodeError:
            # Fallback for non-JSON LLM responses
            return {
                "status": "ok",
                "interjection_id": interjection_id,
                "timestamp": "2024-01-01T00:00:00Z",
                "acknowledged": True,
            }

    # ─────────────────────────────────────────────────────────────
    # Standard SteerableToolHandle Methods
    # ─────────────────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> "SteerableToolHandle":
        """
        Asks a question to the simulated user in the conversation.
        Supports both simple string responses and structured Pydantic models.
        """
        if self._stopped:
            raise RuntimeError("Cannot ask a stopped conversation.")

        # Log the ask request
        ask_label = SimulatedLineage.make_label("SimulatedConversationManager.ask")
        SimulatedLog.log_request("ask", ask_label, question)

        prompt = f"""The external process is asking the user a question. Based on your persona and the conversation history, provide a direct and plausible answer.
**Question:** "{question}"
"""
        if response_format:
            prompt += "\n**FORMAT INSTRUCTIONS:** Your response MUST be a JSON object that strictly conforms to the provided Pydantic model schema."

        class _AnswerHandle(SteerableToolHandle, SimulatedHandleMixin):
            def __init__(
                inner_self,
                stateful_llm: unillm.AsyncUnify,
                prompt_str: str,
                pydantic_model: Optional[Type[BaseModel]],
                log_label: str,
            ):
                inner_self._llm = stateful_llm
                inner_self._prompt = prompt_str
                inner_self._model = pydantic_model
                inner_self._result_cache: Optional[Any] = None
                inner_self._done = False
                inner_self._log_label = log_label

            async def result(inner_self) -> Any:
                if inner_self._result_cache is None:
                    response = await simulated_llm_roundtrip(
                        inner_self._llm,
                        label=inner_self._log_label,
                        prompt=inner_self._prompt,
                        response_format=inner_self._model,
                    )
                    if inner_self._model:
                        # simulated_llm_roundtrip may return a validated model instance
                        # directly, so check before attempting to parse again
                        if isinstance(response, inner_self._model):
                            inner_self._result_cache = response
                        else:
                            try:
                                inner_self._result_cache = (
                                    inner_self._model.model_validate_json(
                                        response,
                                    )
                                )
                            except Exception as e:
                                raise ValueError(
                                    f"Failed to parse LLM response into {inner_self._model.__name__}: {e}\nResponse: {response}",
                                )
                    else:
                        inner_self._result_cache = response
                    inner_self._done = True
                return inner_self._result_cache

            def done(inner_self) -> bool:
                return inner_self._done

            async def stop(inner_self, reason: str | None = None, *args, **kwargs):
                inner_self._log_stop(reason)

            async def pause(inner_self):
                inner_self._log_pause()

            async def resume(inner_self):
                inner_self._log_resume()

            async def interject(inner_self, message: str, *args, **kwargs):
                inner_self._log_interject(message)

            async def ask(inner_self, *args, **kwargs):
                return inner_self

            async def next_clarification(inner_self) -> dict:
                return {}

            async def next_notification(inner_self) -> dict:
                return {}

            async def answer_clarification(
                inner_self,
                call_id: str,
                answer: str,
            ) -> None:
                pass

        return _AnswerHandle(self._llm, prompt, response_format, ask_label)

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
        self._log_interject(message)
        return await self.send_notification(
            message,
            source="external_interjection",
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
        # Simulated implementation - just acknowledge the unpin request
        return {
            "status": "ok",
            "message": f"Unpin simulated for interjection {interjection_id}",
            "interjection_id": interjection_id,
        }

    async def pause(self) -> str:
        """Pauses the simulated conversation."""
        self._log_pause()
        self._paused = True
        return "Simulated conversation is paused."

    async def resume(self) -> str:
        """Resumes a paused conversation."""
        self._log_resume()
        self._paused = False
        return "Simulated conversation is resumed."

    async def stop(self, reason: Optional[str] = None, **kwargs) -> str:
        """Stops the simulated conversation."""
        self._log_stop(reason)
        self._stopped = True
        self._final_result = (
            f"Conversation stopped. Reason: {reason or 'No reason provided.'}"
        )
        return self._final_result

    def done(self) -> bool:
        """Checks if the conversation is stopped."""
        return self._stopped

    async def result(self) -> str:
        """Returns the final state of the conversation."""
        while not self._stopped:
            await asyncio.sleep(0.1)
        return self._final_result

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        pass
