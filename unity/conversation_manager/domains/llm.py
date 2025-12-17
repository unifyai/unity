from collections.abc import Callable

from unity.common.llm_client import new_llm_client
from unity.common.async_tool_loop import start_async_tool_loop


class LLM:
    def __init__(self, model: str, event_broker=None):
        self.model = model
        self.event_broker = event_broker

    async def run(
        self,
        system_prompt: str,
        messages: str,
        response_model,
        *,
        _on_handle_created: Callable[[object], None] | None = None,
        _on_handle_finished: Callable[[object], None] | None = None,
        _interrupt_llm_with_interjections: bool = True,
    ):
        """Run the Main CM Brain and return structured output.

        The Main CM Brain always runs in non-streaming mode. The Voice Agent
        (fast brain) handles all speech generation independently.
        """
        client = new_llm_client(self.model, reasoning_effort="low")
        client.set_system_message(system_prompt)

        def _preprocess_msgs(msgs: list[dict]) -> list[dict]:
            """
            Keep the engineered state representation *transient*.

            ConversationManager renders a full state snapshot each turn. We keep only the
            latest snapshot when calling the model, while preserving any system messages
            (runtime context, response format hint) and any user interjections.
            """
            try:
                state_indices = [
                    i
                    for i, m in enumerate(msgs)
                    if isinstance(m, dict) and m.get("_cm_state_snapshot") is True
                ]
                if not state_indices:
                    return msgs

                last_state = msgs[state_indices[-1]]
                kept: list[dict] = []
                for m in msgs:
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
                return msgs

        # Use the async tool loop even for single-step structured output.
        # This provides a consistent execution model and enables incremental
        # rollout of interjections/tooling later.
        handle = start_async_tool_loop(
            client,
            messages,
            {},
            loop_id="ConversationManager._run_llm",
            response_format=response_model,
            preprocess_msgs=_preprocess_msgs if isinstance(messages, list) else None,
            interrupt_llm_with_interjections=_interrupt_llm_with_interjections,
            log_steps=False,
        )
        try:
            if _on_handle_created is not None:
                _on_handle_created(handle)
        except Exception:
            pass
        try:
            return await handle.result()
        finally:
            try:
                if _on_handle_finished is not None:
                    _on_handle_finished(handle)
            except Exception:
                pass
