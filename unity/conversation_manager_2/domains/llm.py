import os
from openai import AsyncOpenAI

is_reasoning = lambda name: "gpt-5" in name
class LLM:
    def __init__(self, model: str, event_broker = None):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.event_broker = event_broker
    
    async def run(self, system_prompt: str, messages: str, response_model, stream_to_call: bool=False):
        if not stream_to_call:
            res = await self._run_non_stream(system_prompt, messages, response_model)
            return res
        else:
            ...
    
    async def _run_non_stream(self, system_prompt, messages, response_model):
        out = await self.client.responses.parse(
        model=self.model,
        instructions=system_prompt,
        input=messages,
        text_format=response_model,
    )
        out = out.output[0].content[0].text
        return out
    
    async def _run_stream():
        ...