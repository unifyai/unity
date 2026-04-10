from __future__ import annotations

import pytest

from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.llm_client import new_llm_client

pytestmark = pytest.mark.llm_call


@pytest.mark.asyncio
async def test_all_llm_kwargs_are_forwarded_verbatim(llm_config, monkeypatch):
    """Regression test: ensure all LLM-provided kwargs reach the tool.

    Prior behaviour filtered kwargs against the tool signature, which
    dropped legitimate fields when the tool accepted only **kwargs.
    """

    received: dict[str, str] = {}

    def accept_any(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal received
        received = dict(kwargs)
        return "ok"

    # give the tool stable names for readability
    accept_any.__name__ = "accept_any"
    accept_any.__qualname__ = "accept_any"

    client = new_llm_client(**llm_config)
    # Instruct the real model to call `accept_any` once with the provided fields
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `accept_any` "
        'with the following arguments exactly: { "first_name": "Luca", "surname": "Renaldi", "team": "Northbridge FC", "position": "Forward" }. '
        "After the tool returns, provide a short final reply.",
    )

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
    # LLM may drop the space; normalize before comparing
    assert received.get("team", "").replace(" ", "") == "NorthbridgeFC"
    assert received.get("position") == "Forward"
