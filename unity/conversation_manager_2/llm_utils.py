from pydantic_core import from_json


from openai import AsyncOpenAI


async def stream_llm_call(
    client: AsyncOpenAI,
    system_prompt,
    messages,
    model="gpt-4.1",
    response_model=None,
    streamed_field=None,
):
    last_phone_utterance = ""
    out = ""
    async with client.responses.stream(
        model=model,
        instructions=system_prompt,
        # input=self.chat_history + input_message,
        input=messages,
        text_format=response_model,
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                # print(event.delta)
                out += event.delta
                parsed_out = from_json(out, allow_partial="trailing-strings")
                if parsed_out.get(streamed_field):
                    if len(last_phone_utterance) != len(
                        parsed_out[streamed_field],
                    ):
                        yield {
                            "type": "chunk",
                            "content": parsed_out[streamed_field][
                                len(last_phone_utterance) :
                            ],
                        }
                    last_phone_utterance = parsed_out[streamed_field]
        yield {"type": "end_streamed_field"}
        yield {
            "type": "output",
            "content": out,
        }


async def llm_call(
    client: AsyncOpenAI,
    system_prompt,
    messages,
    model="gpt-4.1",
    response_model=None,
):
    out = await client.responses.parse(
        model="gpt-4.1",
        instructions=system_prompt,
        # input=self.chat_history + input_message,
        input=messages,
        text_format=response_model,
    )
    out = out.output[0].content[0].text
    return out
