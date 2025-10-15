from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Literal, Optional, Type, Any
from pydantic import BaseModel
import unify

from .base import BaseConversationManagerHandle
from ..common import SteerableToolHandle


class SimulatedConversationManagerHandle(BaseConversationManagerHandle):
    """
    Simulated conversation manager handle for testing and demos.

    Uses a stateful LLM to simulate conversation steering without
    actual Redis pub/sub or real conversation state.ß
    """

    def __init__(
        self,
        assistant_id: str,
        contact_id: int,
        *,
        description: str = "A simulated conversation between an AI assistant and a user.",
        simulation_guidance: Optional[str] = None,
    ):
        self.assistant_id = assistant_id
        self.contact_id = contact_id
        self._description = description
        self._simulation_guidance = simulation_guidance

        # A shared, stateful LLM for maintaining conversation context
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )

        # Initialize the system message for the stateful LLM
        system_msg = self._build_system_message()
        self._llm.set_system_message(system_msg)

        # Internal state management
        self._stopped = False
        self._paused = False
        self._final_result = "Conversation is active."

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
        level: Literal["info", "warning", "urgent"] = "info",
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
- **Level:** {level}
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
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":
        """
        Asks a question to the simulated user in the conversation.
        Supports both simple string responses and structured Pydantic models.
        """
        if self._stopped:
            raise RuntimeError("Cannot ask a stopped conversation.")

        ask_client = unify.AsyncUnify("gpt-4o@openai")
        ask_client.set_system_message(self._llm.system_message)

        prompt = f"""The external process is asking the user a question. Based on your persona and the conversation history, provide a direct and plausible answer.
**Question:** "{question}"
"""
        if response_format:
            ask_client.set_response_format(response_format)
            prompt += "\n**FORMAT INSTRUCTIONS:** Your response MUST be a JSON object that strictly conforms to the provided Pydantic model schema."

        class _AnswerHandle(SteerableToolHandle):
            def __init__(
                self,
                client: unify.AsyncUnify,
                parent_llm: unify.AsyncUnify,
                prompt_str: str,
                pydantic_model: Optional[Type[BaseModel]],
            ):
                self._client = client
                self._parent_llm = parent_llm
                self._prompt = prompt_str
                self._model = pydantic_model
                self._result_cache: Optional[Any] = None
                self._done = False

            async def result(self) -> Any:
                if self._result_cache is None:
                    response_str = await self._client.generate(
                        self._prompt,
                        messages=self._parent_llm.messages,
                    )
                    if self._model:
                        try:
                            self._result_cache = self._model.model_validate_json(
                                response_str,
                            )
                        except Exception as e:
                            # In a real scenario, we would want a retry loop here.
                            raise ValueError(
                                f"Failed to parse LLM response into {self._model.__name__}: {e}\nResponse: {response_str}",
                            )
                    else:
                        self._result_cache = response_str
                    self._done = True
                return self._result_cache

            def done(self) -> bool:
                return self._done

            def stop(self, *args, **kwargs):
                pass

            def pause(self):
                pass

            def resume(self):
                pass

            async def interject(self, *args, **kwargs):
                pass

            async def ask(self, *args, **kwargs):
                return self

            async def next_clarification(self) -> dict:
                return {}

            async def next_notification(self) -> dict:
                return {}

            async def answer_clarification(self, call_id: str, answer: str) -> None:
                pass

        return _AnswerHandle(ask_client, self._llm, prompt, response_format)

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

    def pause(self) -> str:
        """Pauses the simulated conversation."""
        self._paused = True
        return "Simulated conversation is paused."

    def resume(self) -> str:
        """Resumes a paused conversation."""
        self._paused = False
        return "Simulated conversation is resumed."

    def stop(self, reason: Optional[str] = None) -> str:
        """Stops the simulated conversation."""
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
