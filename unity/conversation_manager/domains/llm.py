import json
from typing import Awaitable, Callable, Optional

from pydantic_core import from_json

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
        stream_to_call: bool = False,
        call_type: str = None,
        before_stream_start: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        if not stream_to_call:
            return await self._run_non_stream(system_prompt, messages, response_model)
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
        client = new_llm_client(self.model, reasoning_effort="low")
        client.set_response_format(response_model)
        return await client.generate(
            system_message=system_prompt,
            messages=messages if isinstance(messages, list) else None,
            user_message=messages if isinstance(messages, str) else None,
        )

    async def _run_stream(
        self,
        system_prompt,
        messages,
        response_model,
        call_type,
        before_stream_start: Optional[Callable[[], Awaitable[None]]],
    ):
        client = new_llm_client(self.model, reasoning_effort="low")
        client.set_response_format(self._to_streaming_format(response_model))

        out = ""
        last_utterance_len = 0
        done = False
        started = False

        stream = await client.generate(
            system_message=system_prompt,
            messages=messages if isinstance(messages, list) else None,
            user_message=messages if isinstance(messages, str) else None,
            stream=True,
        )

        async for chunk in stream:
            out += chunk
            try:
                parsed = from_json(out, allow_partial="trailing-strings")
            except Exception:
                continue

            if not isinstance(parsed, dict):
                continue

            if "actions" in parsed and not done:
                await self.event_broker.publish(
                    f"app:{call_type}:response_gen",
                    json.dumps({"type": "end_gen"}),
                )
                done = True

            utterance = parsed.get("phone_utterance", "")
            if len(utterance) > last_utterance_len:
                if not started:
                    if before_stream_start:
                        await before_stream_start()
                    await self.event_broker.publish(
                        f"app:{call_type}:response_gen",
                        json.dumps({"type": "start_gen"}),
                    )
                    started = True

                await self.event_broker.publish(
                    f"app:{call_type}:response_gen",
                    json.dumps(
                        {
                            "type": "gen_chunk",
                            "chunk": utterance[last_utterance_len:],
                        },
                    ),
                )
                last_utterance_len = len(utterance)

        return out

    def _to_streaming_format(self, response_model) -> dict:
        """Convert Pydantic model to json_schema format for streaming.

        OpenAI's strict mode requires:
        1. All properties must be in the `required` array
        2. `additionalProperties` must be false

        Pydantic excludes fields with default values from `required`, so we
        post-process the schema to ensure strict mode compliance.
        """
        schema = response_model.model_json_schema()
        self._make_strict_mode_compatible(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": False,
            },
        }

    def _make_strict_mode_compatible(self, schema: dict) -> None:
        """Recursively make a JSON schema compatible with OpenAI's strict mode.

        For strict mode, all properties must be in `required` and
        `additionalProperties` must be false.
        """
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())

        # Process nested definitions
        for def_schema in schema.get("$defs", {}).values():
            self._make_strict_mode_compatible(def_schema)
