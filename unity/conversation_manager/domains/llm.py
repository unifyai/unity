from unity.common.llm_client import new_llm_client


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
        """Run the LLM with the given prompt and return structured output.

        The Main CM Brain always runs in non-streaming mode. The Voice Agent
        (fast brain) handles all speech generation independently.
        """
        client = new_llm_client(self.model, reasoning_effort="low")
        client.set_response_format(response_model)
        return await client.generate(
            system_message=system_prompt,
            messages=messages if isinstance(messages, list) else None,
            user_message=messages if isinstance(messages, str) else None,
        )
