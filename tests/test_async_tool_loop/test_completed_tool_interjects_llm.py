from __future__ import annotations

import asyncio
import pytest
import unify

from unity.common.llm_helpers import start_async_tool_use_loop
from tests.helpers import _handle_project

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
async def test_llm_keeps_intermediate_reasoning_when_other_tool_is_very_slow() -> None:
    """
    fast_task completes → loop triggers LLM reasoning.
    LLM finishes *before* very_slow_task returns, producing an intermediate
    assistant message.  Once very_slow_task finishes the loop runs the model
    again and a final assistant reply is produced.

    Expected shape of roles:

        0 user
        1 assistant  (tool_calls: fast & very_slow)
        2 tool       (fast_task)
        3 assistant  (intermediate reasoning)
        4 tool       (very_slow_task)
        5 assistant  (final answer)
    """

    system_prompt = (
        "You can call two tools: 'fast_task' and 'very_slow_task'. "
        "Always call *both* in the same assistant turn. "
        "If you receive only one result, think aloud and say you are still "
        "waiting for the other. After you have both results give a final answer."
    )

    client = unify.AsyncUnify(
        endpoint="gpt-4o@openai",
        system_message=system_prompt,
    )

    tools = {"fast_task": fast_task, "very_slow_task": very_slow_task}

    handle = start_async_tool_use_loop(
        client,
        message="Please run fast_task and slow_task, triggering them both **immediately** (at the same time)",
        tools=tools,
        interrupt_llm_with_interjections=True,
    )

    await handle.result()

    roles = [m["role"] for m in client.messages]

    # ── Assertions ───────────────────────────────────────────────────────

    # Counts with placeholder: three assistants, three tools
    assert roles.count("assistant") == 3
    assert roles.count("tool") == 3

    # Ensure at least one assistant message lies *between* the two tool results
    tool_indices = [i for i, r in enumerate(roles) if r == "tool"]
    assert len(tool_indices) == 3
    fast_tool, pending_slow_tool, completed_slow_tool = sorted(tool_indices)

    # There must be an assistant index strictly between them
    assert any(r == "assistant" for r in roles[fast_tool + 1 : completed_slow_tool])

    # Tool names include fast & very_slow; placeholder duplicates don't hurt
    tool_names = {m["name"] for m in client.messages if m["role"] == "tool"}
    assert {"fast_task", "very_slow_task"}.issubset(tool_names)

    # Initial assistant turn requested BOTH tools
    tool_calls = client.messages[1]["tool_calls"]
    fn_names = {tc["function"]["name"] for tc in tool_calls}
    assert fn_names == {"fast_task", "very_slow_task"}


@pytest.mark.asyncio
@_handle_project
async def test_llm_step_is_preempted_by_late_tool_completion() -> None:
    """
    The model is instructed to call **both** tools in a single assistant turn.
    ``fast_task`` completes first, triggering an LLM reasoning step; while the
    model is still thinking ``slow_task`` finishes.  The patched loop must
    cancel the in-flight `generate`, deliver the new result, and only then run
    a fresh LLM step.

    In the final transcript we therefore expect:

        user → assistant(tool_calls) → tool → tool → assistant

    i.e. exactly **two** assistant messages (initial selection & final answer)
    and exactly **two** tool messages (one per tool).
    """

    system_prompt = (
        "You have access to two tools called 'fast_task' and 'slow_task'. "
        "Always invoke *both* tools in the same assistant turn and wait for "
        "their results before replying to the user. Do not send any other "
        "assistant messages in between."
    )

    client = unify.AsyncUnify(
        endpoint="gpt-4o@openai",
        system_message=system_prompt,
    )

    tools = {"fast_task": fast_task, "slow_task": slow_task}

    handle = start_async_tool_use_loop(
        client,
        message="Please run fast_task and slow_task, triggering them both **immediately** (at the same time)",
        tools=tools,
        interrupt_llm_with_interjections=True,
    )

    await handle.result()

    # ── Assertions ───────────────────────────────────────────────────────
    roles = [m["role"] for m in client.messages]

    # Basic skeleton:
    #   user
    #   assistant(tool_calls fast & slow)
    #   tool  (fast_task result)
    #   assistant starts replying
    #   tool  (slow_task result)
    #   earlier assistant call is cancelled
    #   assistant injests both results (final)
    assert roles[0] == "user"
    assert roles[1] == "assistant"
    assert roles.count("assistant") == 2  # initial + final
    assert roles.count("tool") == 2  # fast + slow

    # The two tool results must correspond to the two tool names
    tool_names = {m["name"] for m in client.messages if m["role"] == "tool"}
    assert {"fast_task", "slow_task"}.issubset(tool_names)

    # Initial assistant turn must have requested *both* tools
    tool_calls = client.messages[1]["tool_calls"]
    fn_names = {call["function"]["name"] for call in tool_calls}
    assert fn_names == {"fast_task", "slow_task"}
