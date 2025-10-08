from __future__ import annotations

import json
import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop


@pytest.mark.asyncio
async def test_all_llm_kwargs_are_forwarded_verbatim(monkeypatch):
    """Regression test: ensure all LLM-provided kwargs reach the tool.

    Prior behaviour filtered kwargs against the tool signature, which
    dropped legitimate fields when the tool accepted only **kwargs.
    """

    received: dict[str, str] = {}

    @unify.traced
    def accept_any(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal received
        received = dict(kwargs)
        return "ok"

    # give the tool stable names for readability
    accept_any.__name__ = "accept_any"
    accept_any.__qualname__ = "accept_any"

    client = unify.AsyncUnify("o4-mini@openai")

    # Monkeypatch the client's generate to emit a single tool call with kwargs,
    # then fall back to the real generation for the final turn.
    orig_generate = client.generate
    step = {"n": 0}

    async def _driver(**kwargs):
        if step["n"] == 0:
            step["n"] += 1
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_KWARGS",
                        "type": "function",
                        "function": {
                            "name": "accept_any",
                            "arguments": json.dumps(
                                {
                                    "first_name": "Luca",
                                    "surname": "Renaldi",
                                    "team": "Northbridge FC",
                                    "position": "Forward",
                                },
                            ),
                        },
                    },
                ],
            }
            client.append_messages([msg])
            return msg
        return await orig_generate(**kwargs)

    monkeypatch.setattr(client, "generate", _driver, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="Start",
        tools={"accept_any": accept_any},
    )

    final = await handle.result()
    assert final.strip(), "Loop did not complete"

    # The tool must have received the exact fields emitted by the LLM
    assert received.get("first_name") == "Luca"
    assert received.get("surname") == "Renaldi"
    assert received.get("team") == "Northbridge FC"
    assert received.get("position") == "Forward"
