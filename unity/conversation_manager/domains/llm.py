from collections.abc import Callable
from typing import Type

from pydantic import BaseModel

from unity.common.llm_client import new_llm_client
from unity.common.single_shot import single_shot_tool_decision, SingleShotResult


class LLM:
    def __init__(self, model: str, event_broker=None):
        self.model = model
        self.event_broker = event_broker

    async def run(
        self,
        system_prompt: str,
        messages: str | dict | list,
        response_model: Type[BaseModel],
        *,
        _tools: dict[str, Callable] | None = None,
    ) -> SingleShotResult:
        """Run the Main CM Brain and return the result.

        This is a single-shot LLM call that:
        1. Receives the current state as a message
        2. Returns structured output (thoughts, call_guidance)
        3. Optionally executes ONE action tool (send_sms, make_call, etc.)

        The Main CM Brain always runs in non-streaming mode. The Voice Agent
        (fast brain) handles all speech generation independently.

        Parameters
        ----------
        system_prompt : str
            The system prompt for the LLM.
        messages : str | dict | list
            The user message(s) representing current state.
        response_model : Type[BaseModel]
            Pydantic model for structured output (e.g., thoughts, call_guidance).
        _tools : dict[str, Callable] | None
            Optional tools for the LLM to call (send_sms, make_call, etc.).

        Returns
        -------
        SingleShotResult
            Contains structured_output (parsed response_model) and any tool execution result.
        """
        client = new_llm_client(self.model, reasoning_effort="low")
        client.set_system_message(system_prompt)

        # Preprocess messages: keep only the latest state snapshot
        processed_messages = self._preprocess_messages(messages)

        # Use tool_choice="required" to ensure the model always takes an action.
        # Even if there's nothing to do, it should call the "wait" tool.
        result = await single_shot_tool_decision(
            client,
            processed_messages,
            _tools or {},
            tool_choice="required" if _tools else "auto",
            response_format=response_model,
        )

        return result

    def _preprocess_messages(
        self,
        messages: str | dict | list,
    ) -> str | dict | list:
        """Keep only the latest state snapshot from message history.

        ConversationManager renders a full state snapshot each turn. We keep only the
        latest snapshot when calling the model, while preserving any system messages
        and user interjections.
        """
        if isinstance(messages, str):
            return messages
        if isinstance(messages, dict):
            return messages
        if not isinstance(messages, list):
            return messages

        try:
            # Find all state snapshot messages
            state_indices = [
                i
                for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("_cm_state_snapshot") is True
            ]
            if not state_indices:
                return messages

            # Keep only the latest state snapshot and non-state messages
            last_state = messages[state_indices[-1]]
            kept: list[dict] = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "system":
                    kept.append(m)
                elif role == "user" and not m.get("_cm_state_snapshot"):
                    kept.append(m)

            kept.append(last_state)
            return kept
        except Exception:
            return messages
