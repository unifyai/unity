from __future__ import annotations

import os
import json
import pytest
import unify
from pydantic import BaseModel, Field

from unity.common.llm_helpers import start_async_tool_use_loop
from tests.helpers import _handle_project


class SimpleGreeting(BaseModel):
    """Minimal schema used to validate structured-output support."""

    greeting: str = Field(..., description="A friendly greeting message.")
    lucky_number: int = Field(..., description="Any integer chosen by the model.")


@pytest.mark.asyncio
@_handle_project
async def test_structured_output_response_format() -> None:
    """The async-tool loop should honour *response_format* and return JSON that
    validates against the supplied Pydantic schema.

    Flow:
      1. Run the loop **without** any tools – the model immediately replies.
      2. The wrapper triggers an *extra* formatting step that enforces the
         `SimpleGreeting` schema via ``client.set_response_format``.
      3. The final returned string must parse successfully with
         ``SimpleGreeting.model_validate_json``.
    """

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )

    client.set_system_message(
        "When asked, respond with a JSON object that contains exactly two keys: "
        "'greeting' (a friendly greeting) and 'lucky_number' (an integer). Do not "
        "include any extra keys or commentary.",
    )

    handle = start_async_tool_use_loop(  # type: ignore[attr-defined]
        client,
        message="Please introduce yourself and pick a lucky number.",
        tools={},  # no tools needed
        response_format=SimpleGreeting,
        log_steps=False,
        max_steps=8,
        timeout=120,
    )

    # The call should finish quickly and return JSON conforming to the schema.
    final_reply = await handle.result()

    # Validate – will raise if the JSON structure is wrong.
    parsed = SimpleGreeting.model_validate_json(final_reply)

    # Light sanity checks on the parsed content.
    assert parsed.greeting.strip(), "Greeting must be non-empty"
    assert isinstance(parsed.lucky_number, int)


# ----------------------------------------------------------------------------
# Ensure trivial response formats are satisfied in the *first* LLM
#  reply so that no follow-up re-formatting step is scheduled.
# ----------------------------------------------------------------------------


class SimpleEcho(BaseModel):
    """Extremely simple schema – just a single string field."""

    text: str = Field(..., description="Echo text sent by the user.")


@pytest.mark.asyncio
@_handle_project
async def test_no_additional_formatting_roundtrip() -> None:  # noqa: D401
    """Verify that the loop skips the re-formatting turn when the first
    assistant reply already conforms to *response_format*.

    We check this by asserting that the synthetic follow-up prompt asking the
    model to re-emit its answer ("Please output your previous answer again…")
    is **absent** from the final chat transcript.
    """

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )

    # Prompt the model so that it can satisfy the schema in one go.
    client.set_system_message(
        "Return a JSON object with a single key 'text' containing any greeting "
        "string. Do NOT add any extra keys, numbers, or commentary.",
    )

    handle = start_async_tool_use_loop(  # type: ignore[attr-defined]
        client,
        message="Say hi!",
        tools={},
        response_format=SimpleEcho,
        log_steps=False,
        max_steps=6,
        timeout=60,
    )

    final_reply = await handle.result()

    # Validation should succeed directly.
    parsed = SimpleEcho.model_validate_json(final_reply)
    assert parsed.text.strip(), "Text must be non-empty"

    # Ensure *no* follow-up formatting prompt was injected.
    assert not any(
        m.get("role") == "user"
        and "Please output your previous answer again" in m.get("content", "")
        for m in client.messages
    ), "Unexpected additional formatting round triggered."
