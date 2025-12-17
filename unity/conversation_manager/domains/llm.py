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
    ):
        """Run the Main CM Brain and return structured output.

        The Main CM Brain always runs in non-streaming mode. The Voice Agent
        (fast brain) handles all speech generation independently.
        """
        client = new_llm_client(self.model, reasoning_effort="low")
        client.set_system_message(system_prompt)

        # Use the async tool loop even for single-step structured output.
        # This provides a consistent execution model and enables incremental
        # rollout of interjections/tooling later.
        handle = start_async_tool_loop(
            client,
            messages,
            {},
            loop_id="ConversationManager._run_llm",
            response_format=response_model,
            log_steps=False,
        )
        return await handle.result()
