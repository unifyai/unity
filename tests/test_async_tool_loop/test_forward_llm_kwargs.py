from __future__ import annotations

import json
import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop


class _StubUnify:
    """Tiny stub that produces a single tool call with predefined arguments
    followed by a final assistant turn with plain text.
    """

    def __init__(self):
        self.messages: list[dict] = []
        self._step = 0

    def append_messages(self, msgs):
        self.messages.extend(msgs)

    async def generate(self, **_):  # noqa: D401 – minimal stub
        if self._step == 0:
            self._step += 1
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
        else:
            self._step += 1
            msg = {
                "role": "assistant",
                "content": "done",
                "tool_calls": [],
            }
        self.messages.append(msg)
        return msg

    @property
    def system_message(self) -> str:  # noqa: D401 – unused in stub
        return ""


@pytest.mark.asyncio
async def test_all_llm_kwargs_are_forwarded_verbatim():
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

    client = _StubUnify()

    handle = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="Start",
        tools={"accept_any": accept_any},
        log_steps=False,
    )

    final = await handle.result()
    assert final.strip(), "Loop did not complete"

    # The tool must have received the exact fields emitted by the LLM
    assert received.get("first_name") == "Luca"
    assert received.get("surname") == "Renaldi"
    assert received.get("team") == "Northbridge FC"
    assert received.get("position") == "Forward"
