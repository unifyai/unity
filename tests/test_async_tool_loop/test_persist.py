from __future__ import annotations

import asyncio
import os
import pytest
import unify

from unity.common.llm_helpers import start_async_tool_use_loop
from tests.helpers import _handle_project, SETTINGS


MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


def new_client() -> unify.AsyncUnify:
    """Utility to get a fresh client with env-controlled caching / tracing."""
    return unify.AsyncUnify(
        MODEL_NAME,
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )


@pytest.mark.asyncio
@_handle_project
async def test_persist_requires_explicit_stop_and_returns_last_interjection():
    """
    With persist=True and no tools, the loop should not resolve until stop() is
    called. Interjections after the first assistant turn should still be
    honoured, and the final result must reflect the latest interjection.
    """

    client = new_client()
    client.set_system_message(
        "Please always respond with 'You said: {my_latest_message}', with the placeholder containing whatever I said most recently, and do not include the quotation marks in your response.",
    )

    handle = start_async_tool_use_loop(
        client=client,
        message="first",
        tools={},  # no tools – pure LLM reply
        persist=True,
        max_steps=20,
        timeout=300,
    )

    # Result must not resolve until we explicitly call stop.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(handle.result()), timeout=1.0)

    # Interject late – should become the latest message the assistant mirrors.
    await handle.interject("second")

    # Graceful stop – loop should now return the final answer.
    handle.stop("closing")
    final = await handle.result()

    assert final == "You said: second"

    # Ensure we indeed saw the interjection as a system message in the transcript
    assert any(
        m.get("role") == "system" and "user: **second**" in (m.get("content") or "")
        for m in client.messages
    )


@pytest.mark.asyncio
@_handle_project
async def test_non_persist_finishes_without_stop():
    """
    Default behaviour (persist=False) should complete and return immediately
    on the final tool-less assistant reply without requiring stop().
    """

    client = new_client()
    client.set_system_message("Reply exactly with the word OK. Do not call any tools.")

    handle = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={},
        # persist defaults to False
        max_steps=10,
        timeout=120,
    )

    final = await handle.result()
    assert final.strip().upper().startswith("OK")
