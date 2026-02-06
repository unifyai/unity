"""
Tests for the `persist=True` mode in the async tool loop.

When `persist=True`, the loop does not terminate when the LLM produces content
without tool calls. Instead, it blocks waiting for the next interjection. This
enables a single persistent loop that can process multiple events over time.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_condition,
    _wait_for_tool_request,
)

from unity.common.async_tool_loop import start_async_tool_loop


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS                                                       #
# --------------------------------------------------------------------------- #
async def echo(text: str) -> str:
    """Echo back the input text."""
    await asyncio.sleep(0.05)
    return f"echoed: {text}"


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


# --------------------------------------------------------------------------- #
#  RESPONSE FORMAT MODELS                                                     #
# --------------------------------------------------------------------------- #
class Greeting(BaseModel):
    """Simple structured response for persist-mode tests."""

    message: str = Field(..., description="A greeting message.")
    number: int = Field(..., description="Any integer chosen by the model.")


# --------------------------------------------------------------------------- #
#  BASIC PERSIST MODE                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_waits_for_interjection(llm_config):
    """
    In persist mode, after the LLM produces content without tool calls,
    the loop blocks waiting for an interjection instead of returning.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="Say hello and nothing else.",
        tools={},
        persist=True,
        timeout=60,
    )

    # Wait for the LLM to produce initial content
    async def _has_assistant_response() -> bool:
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in (client.messages or [])
        )

    await _wait_for_condition(_has_assistant_response, poll=0.05, timeout=30.0)

    # The loop should NOT be done - it should be waiting for an interjection
    await asyncio.sleep(0.2)  # Small delay to ensure loop has reached wait state
    assert not handle.done(), "Persist loop should not terminate after first response"

    # Now interject to give it something to process
    await handle.interject("Now say goodbye.")

    # Wait for a second assistant response
    async def _has_two_responses() -> bool:
        assistant_msgs = [
            m
            for m in (client.messages or [])
            if m.get("role") == "assistant" and m.get("content")
        ]
        return len(assistant_msgs) >= 2

    await _wait_for_condition(_has_two_responses, poll=0.05, timeout=30.0)

    # Still should not be done - waiting for more
    await asyncio.sleep(0.2)
    assert not handle.done(), "Persist loop should continue waiting after interjection"

    # Stop the loop explicitly
    await handle.stop()
    result = await handle.result()
    assert result == "processed stopped early, no result"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_processes_multiple_interjections(llm_config):
    """
    The persist loop should process multiple interjections sequentially.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="Reply with 'ready' and wait for my instructions.",
        tools={"add": add},
        persist=True,
        timeout=60,
    )

    # Wait for initial response
    async def _has_response() -> bool:
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in (client.messages or [])
        )

    await _wait_for_condition(_has_response, poll=0.05, timeout=30.0)

    # First interjection - ask it to add
    await handle.interject(
        "Use the add tool to compute 2 + 3, then tell me the result.",
    )

    # Wait for the tool to be called and response generated
    await _wait_for_tool_request(client, "add")

    async def _has_tool_result() -> bool:
        return any(m.get("role") == "tool" for m in (client.messages or []))

    await _wait_for_condition(_has_tool_result, poll=0.05, timeout=30.0)

    # Wait for assistant to process tool result
    async def _has_result_response() -> bool:
        msgs = client.messages or []
        tool_idx = next(
            (i for i, m in enumerate(msgs) if m.get("role") == "tool"),
            -1,
        )
        if tool_idx < 0:
            return False
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in msgs[tool_idx + 1 :]
        )

    await _wait_for_condition(_has_result_response, poll=0.05, timeout=30.0)

    # Verify the result is in the response
    assistant_msgs = [
        m.get("content", "")
        for m in (client.messages or [])
        if m.get("role") == "assistant" and m.get("content")
    ]
    assert any("5" in msg for msg in assistant_msgs), "Should have computed 2+3=5"

    # Second interjection
    await handle.interject("Now add 10 and 20.")

    # Wait for second tool call
    async def _has_second_tool_call() -> bool:
        tool_calls = [
            m
            for m in (client.messages or [])
            if m.get("role") == "tool" and m.get("name") == "add"
        ]
        return len(tool_calls) >= 2

    await _wait_for_condition(_has_second_tool_call, poll=0.05, timeout=30.0)

    # Stop and verify
    await handle.stop()
    await handle.result()

    # Verify both additions were performed
    tool_msgs = [m for m in (client.messages or []) if m.get("role") == "tool"]
    assert len(tool_msgs) >= 2, "Should have made at least 2 tool calls"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_terminates_on_stop(llm_config):
    """
    A persist loop should terminate gracefully when handle.stop() is called.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="Say 'waiting' and wait.",
        tools={},
        persist=True,
        timeout=60,
    )

    # Wait for initial response
    async def _has_response() -> bool:
        return any(m.get("role") == "assistant" for m in (client.messages or []))

    await _wait_for_condition(_has_response, poll=0.05, timeout=30.0)

    # Stop the loop
    await handle.stop()

    # Should terminate with the standard stop message
    result = await handle.result()
    assert result == "processed stopped early, no result"
    assert handle.done()


# --------------------------------------------------------------------------- #
#  NON-PERSIST MODE REGRESSION TEST                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_non_persist_mode_terminates_normally(llm_config):
    """
    Without persist=True, the loop should terminate as normal when the LLM
    produces content without tool calls (regression test).
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="Say hello and nothing else.",
        tools={},
        persist=False,  # Explicitly false (the default)
        timeout=60,
    )

    # The loop should terminate with the assistant's response
    result = await handle.result()

    assert result.strip(), "Should have a non-empty response"
    assert handle.done(), "Loop should be done"

    # Verify only one assistant message (no waiting for interjections)
    assistant_msgs = [
        m for m in (client.messages or []) if m.get("role") == "assistant"
    ]
    assert len(assistant_msgs) == 1, "Should have exactly one assistant response"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_with_tool_calls(llm_config):
    """
    Persist mode should handle tool calls normally and only wait after
    the LLM produces content without tool calls.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="Use the add tool to compute 1 + 1, then tell me the result.",
        tools={"add": add},
        persist=True,
        timeout=60,
    )

    # Wait for the tool to be called
    await _wait_for_tool_request(client, "add")

    # Wait for the assistant to respond with the result
    async def _has_result_response() -> bool:
        msgs = client.messages or []
        tool_idx = next(
            (i for i, m in enumerate(msgs) if m.get("role") == "tool"),
            -1,
        )
        if tool_idx < 0:
            return False
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in msgs[tool_idx + 1 :]
        )

    await _wait_for_condition(_has_result_response, poll=0.05, timeout=30.0)

    # Verify the result contains "2"
    assistant_msgs = [
        m.get("content", "")
        for m in (client.messages or [])
        if m.get("role") == "assistant" and m.get("content")
    ]
    assert any("2" in msg for msg in assistant_msgs), "Should have computed 1+1=2"

    # Loop should still be alive, waiting for next interjection
    await asyncio.sleep(0.2)
    assert not handle.done(), "Persist loop should wait after tool call completes"

    # Clean up
    await handle.stop()
    await handle.result()


# --------------------------------------------------------------------------- #
#  PERSIST MODE: send_response DOES NOT TERMINATE THE LOOP                    #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_does_not_terminate_on_send_response(llm_config):
    """In persist mode with ``response_format``, calling ``send_response``
    should NOT terminate the loop.

    When ``response_format`` is set the loop injects a ``send_response`` tool
    (named ``final_response`` in non-persist mode) and forces
    ``tool_choice=required``, so the LLM *must* call it.  In non-persist mode,
    ``final_response`` returns the structured payload and exits the loop.

    In **persist** mode the loop treats the ``send_response`` payload as the
    response for the *current* turn, then continues waiting for the next
    interjection — just like it does for a plain text-only response.
    """
    client = new_llm_client(**llm_config)

    client.set_system_message(
        "When asked, respond with a JSON object containing exactly two keys: "
        "'message' (a greeting) and 'number' (an integer). Do not include any "
        "extra keys or commentary.",
    )

    handle = start_async_tool_loop(
        client,
        message="Say hello and pick a number.",
        tools={},
        persist=True,
        response_format=Greeting,
        timeout=60,
    )

    # Wait for the LLM to call send_response (it must, because
    # tool_choice=required and send_response is the only tool available).
    await _wait_for_tool_request(client, "send_response")

    # Give the loop time to process the send_response call
    await asyncio.sleep(0.5)

    # In persist mode the loop should still be alive, waiting for the next
    # interjection.
    assert not handle.done(), (
        "Persist loop should NOT terminate after send_response — "
        "it should continue waiting for interjections."
    )

    # Interject and get a second response.
    await handle.interject("Now greet me in French and pick a different number.")

    # Wait for the LLM to produce a second send_response call
    async def _has_second_send_response() -> bool:
        count = sum(
            1
            for m in (client.messages or [])
            if m.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "send_response"
                for tc in (m.get("tool_calls") or [])
            )
        )
        return count >= 2

    await _wait_for_condition(_has_second_send_response, poll=0.05, timeout=30.0)

    # Still alive
    await asyncio.sleep(0.2)
    assert (
        not handle.done()
    ), "Persist loop should survive multiple send_response calls."

    # Explicit stop is the only way to end a persist loop
    await handle.stop()
    result = await handle.result()
    assert result == "processed stopped early, no result"
