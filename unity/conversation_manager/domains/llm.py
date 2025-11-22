import os
import json
from openai import AsyncOpenAI

from pydantic_core import from_json
from typing import Awaitable, Callable, Optional

is_reasoning = lambda name: "gpt-5" in name


class LLM:
    def __init__(self, model: str, event_broker=None):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.event_broker = event_broker

    async def run(
        self,
        system_prompt: str,
        messages: str,
        response_model,
        stream_to_call: bool = False,
        call_type: str = None,
        before_stream_start: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        if not stream_to_call:
            return await self._run_non_stream(system_prompt, messages, response_model)
        else:
            if not call_type:
                raise Exception("call type must be specified if using stream_to_call")
            return await self._run_stream(
                system_prompt,
                messages,
                response_model,
                call_type,
                before_stream_start,
            )

    async def _run_non_stream(self, system_prompt, messages, response_model):
        out = await self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=messages,
            text_format=response_model,
        )
        out = out.output[0].content[0].text
        return out

    async def _run_stream(
        self,
        system_prompt,
        messages,
        response_model,
        call_type,
        before_stream_start: Optional[Callable[[], Awaitable[None]]],
    ):
        last_phone_utterance = ""
        out = ""
        async with self.client.responses.stream(
            model=self.model,
            instructions=system_prompt,
            # input=self.chat_history + input_message,
            input=messages,
            text_format=response_model,
        ) as stream:
            done = False
            first_chunk = False
            async for event in stream:
                if event.type == "response.output_text.delta":
                    out += event.delta
                    parsed_out = from_json(out, allow_partial="trailing-strings")
                    if "actions" in parsed_out and not done:
                        await self.event_broker.publish(
                            f"app:{call_type}:response_gen",
                            json.dumps({"type": "end_gen"}),
                        )
                        done = True
                    elif parsed_out.get("phone_utterance"):
                        if len(last_phone_utterance) != len(
                            parsed_out["phone_utterance"],
                        ):
                            if not first_chunk:
                                if before_stream_start:
                                    await before_stream_start()
                                await self.event_broker.publish(
                                    f"app:{call_type}:response_gen",
                                    json.dumps({"type": "start_gen"}),
                                )
                                first_chunk = True
                            await self.event_broker.publish(
                                f"app:{call_type}:response_gen",
                                json.dumps(
                                    {
                                        "type": "gen_chunk",
                                        "chunk": parsed_out["phone_utterance"][
                                            len(last_phone_utterance) :
                                        ],
                                    },
                                ),
                            )
                            last_phone_utterance = parsed_out["phone_utterance"]
        return out
