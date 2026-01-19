from __future__ import annotations

import asyncio
import logging
import pytest

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.test_async_tool_loop.async_helpers import (
    make_gated_async_tool,
    _wait_for_tool_result,
    _wait_for_next_assistant_response_event,
)

# ────────────────────────────────────────────────────────────────────────────
# Dummy tools – one finishes almost instantly, the other a little later
# ────────────────────────────────────────────────────────────────────────────


async def fast_task() -> str:
    """Return very quickly."""
    await asyncio.sleep(0.05)
    return "FAST_RESULT"


async def slow_task() -> str:
    """Return after the fast task but (usually) before the LLM finishes thinking."""
    await asyncio.sleep(0.15)
    return "SLOW_RESULT"


async def very_slow_task() -> str:
    """
    Take long enough that the LLM has time to finish a thought
    after the fast task is done.
    """
    await asyncio.sleep(5.0)
    return "VERY_SLOW_RESULT"


# ────────────────────────────────────────────────────────────────────────────
# Test
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_wait_called_and_pruned_when_other_tool_is_very_slow(
    model,
    caplog,
) -> None:
    """
    When two tools are requested in one turn and only the fast one completes,
    the model will choose the `wait` helper to no-op until the very slow tool
    finishes. We assert that:

    - `wait` was indeed called (via log capture), and
    - `wait` does not appear in the final transcript (pruned from messages).

    This test uses a gated tool for very_slow_task to ensure deterministic timing:
    1. fast_task completes
    2. LLM gets a chance to respond (should call `wait`)
    3. Only then is very_slow_task's gate released
    4. LLM produces final answer
    """

    system_prompt = (
        "You have two tools: 'fast_task' and 'very_slow_task'. "
        "When asked to run them, always call BOTH tools in a single assistant turn.\n\n"
        "CRITICAL RULE: After calling tools, if any tool is still pending (shows '_placeholder': 'pending'), "
        "you MUST call the `wait` tool and produce NO other output. Do not explain, do not ask questions, "
        "do not produce any text - just call `wait`. Only after ALL tool results are available "
        "(no pending placeholders remain) should you produce your final text response summarizing the results."
    )

    client = new_llm_client(
        model=model,
        system_message=system_prompt,
    )

    # Create a gated tool for very_slow_task to control timing deterministically
    very_slow_gate, gated_very_slow_task = make_gated_async_tool(
        return_value="VERY_SLOW_RESULT",
    )

    tools = {"fast_task": fast_task, "very_slow_task": gated_very_slow_task}

    handle = start_async_tool_loop(
        client,
        message="Please run fast_task and slow_task, triggering them both **immediately** (at the same time)",
        tools=tools,
        interrupt_llm_with_interjections=True,
    )

    caplog.set_level(logging.INFO)
    caplog.clear()

    # Wait for fast_task result to be processed
    await _wait_for_tool_result(client, "fast_task", min_results=1)

    # Wait for the LLM to actually respond after seeing the partial result.
    # This uses polling to detect when the next assistant message appears
    # (which should be the `wait` call after fast_task completed).
    # This avoids race conditions with fixed delays that might not be long
    # enough for uncached LLM responses.
    await _wait_for_next_assistant_response_event(client, timeout=120.0)

    # Now release the gate so very_slow_task can complete
    very_slow_gate.set()

    # Wait for the loop to complete
    await handle.result()

    # ── Assertions ───────────────────────────────────────────────────────

    # The loop may insert a small assistant→tool status pair (check_status_*) to
    # preserve ordering when earlier placeholders are no longer at the tail.
    is_status_assistant = lambda m: (
        m.get("role") == "assistant"
        and bool(m.get("tool_calls"))
        and any(
            tc.get("function", {}).get("name", "").startswith("check_status_")
            for tc in m["tool_calls"]
        )
    )
    is_status_tool = lambda m: (
        m.get("role") == "tool" and str(m.get("name", "")).startswith("check_status_")
    )

    non_stub_assistants = [
        m
        for m in client.messages
        if m.get("role") == "assistant" and not is_status_assistant(m)
    ]
    non_stub_tools = [
        m for m in client.messages if m.get("role") == "tool" and not is_status_tool(m)
    ]

    # 1) Assert that `wait` was called (via loop logger)
    wait_logged = any(
        "Assistant chose `wait` – no-op; not persisting to transcript."
        in r.getMessage()
        for r in caplog.records
    )
    assert wait_logged, (
        "Expected LLM to call `wait` while very_slow_task was pending, but no "
        "`wait` log was found. This may indicate the LLM did not follow the prompt "
        "instructions to call `wait` when partial results are available."
    )

    # 2) Assert that `wait` is not persisted in the transcript
    #    - no assistant tool_call with function name 'wait'
    assert all(
        all(
            tc.get("function", {}).get("name") != "wait"
            for tc in (m.get("tool_calls") or [])
        )
        for m in client.messages
        if m.get("role") == "assistant"
    )
    #    - no tool message named 'wait'
    assert all(
        m.get("name") != "wait" for m in client.messages if m.get("role") == "tool"
    )

    # Basic health checks (non‑strict): initial + final assistants and at least two tools
    assert len(non_stub_assistants) >= 2
    assert len(non_stub_tools) >= 2

    # Tool names include fast & very_slow; placeholder duplicates don't hurt
    tool_names = {m["name"] for m in client.messages if m["role"] == "tool"}
    assert {"fast_task", "very_slow_task"}.issubset(tool_names)

    # Initial assistant turn requested BOTH tools – search robustly (index can vary)
    assistant_tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "assistant"
        and m.get("tool_calls")
        and not is_status_assistant(m)
    ]
    assert any(
        {"fast_task", "very_slow_task"}.issubset(
            {tc.get("function", {}).get("name") for tc in (m.get("tool_calls") or [])},
        )
        for m in assistant_tool_msgs
    )


@pytest.mark.asyncio
@_handle_project
async def test_llm_step_is_preempted_by_late_tool_completion(model) -> None:
    """
    The model is instructed to call both tools in a single assistant turn. The fast
    task completes first, then the slow task completes while the model may still be
    thinking. The loop should pre-empt in-flight reasoning, deliver the late tool
    result, and run the model again to produce the final answer.

    Expected role shapes (excluding any synthetic check_status_* status stubs):

        0 user
        1 assistant (tool_calls fast & slow)
        2 tool  (fast_task result)
        3 tool  (slow_task result)
        4 assistant (final answer)

    The test asserts two assistant turns (initial + final) and two tool messages.
    """

    system_prompt = (
        "You have access to two tools called 'fast_task' and 'slow_task'. "
        "Always invoke *both* tools in the same assistant turn and wait for "
        "their results before replying to the user. Do not send any other "
        "assistant messages in between."
    )

    client = new_llm_client(
        model=model,
        system_message=system_prompt,
    )

    tools = {"fast_task": fast_task, "slow_task": slow_task}

    handle = start_async_tool_loop(
        client,
        message="Please run fast_task and slow_task, triggering them both **immediately** (at the same time)",
        tools=tools,
        interrupt_llm_with_interjections=True,
    )

    await handle.result()

    # ── Assertions ───────────────────────────────────────────────────────
    # Some real clients persist the system header as a first message.
    # Ignore any leading system messages when asserting the core skeleton.
    roles = [m["role"] for m in client.messages if m.get("role") != "system"]

    # Basic skeleton (excluding any system headers):
    #   user
    #   assistant(tool_calls fast & slow)
    #   tool  (fast_task result)
    #   assistant starts replying
    #   tool  (slow_task result)
    #   earlier assistant call is stopped
    #   assistant injests both results (final)
    assert roles[0] == "user"
    assert roles[1] == "assistant"
    # Exclude status stubs (check_status_*) from strict counts.
    is_status_assistant = lambda m: (
        m.get("role") == "assistant"
        and bool(m.get("tool_calls"))
        and any(
            tc.get("function", {}).get("name", "").startswith("check_status_")
            for tc in m["tool_calls"]
        )
    )
    is_status_tool = lambda m: (
        m.get("role") == "tool" and str(m.get("name", "")).startswith("check_status_")
    )
    non_stub_assistants = [
        m
        for m in client.messages
        if m.get("role") == "assistant" and not is_status_assistant(m)
    ]
    non_stub_tools = [
        m for m in client.messages if m.get("role") == "tool" and not is_status_tool(m)
    ]
    assert len(non_stub_assistants) == 2  # initial + final
    assert len(non_stub_tools) == 2  # fast + slow

    # The two tool results must correspond to the two tool names
    tool_names = {m["name"] for m in client.messages if m["role"] == "tool"}
    assert {"fast_task", "slow_task"}.issubset(tool_names)

    # Initial assistant turn must have requested *both* tools – search robustly
    assistant_tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "assistant"
        and m.get("tool_calls")
        and not is_status_assistant(m)
    ]
    assert any(
        {"fast_task", "slow_task"}.issubset(
            {tc.get("function", {}).get("name") for tc in (m.get("tool_calls") or [])},
        )
        for m in assistant_tool_msgs
    )
