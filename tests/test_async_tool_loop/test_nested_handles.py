import pytest
import time
import unify

from unity.common.llm_helpers import start_async_tool_use_loop, AsyncToolLoopHandle


# ─────────────────────────────────────────────────────────────────────────────
#  Tools for the *inner* loop
# ─────────────────────────────────────────────────────────────────────────────


def inner_tool() -> str:  # noqa: D401 – simple value
    """Returns the literal string 'inner‑result'."""
    time.sleep(1)
    return "inner-result"


# ─────────────────────────────────────────────────────────────────────────────
#  Tool for the *outer* loop – spawns the nested loop and returns its handle
# ─────────────────────────────────────────────────────────────────────────────


async def outer_tool() -> AsyncToolLoopHandle:
    """Launch an **inner** async‑tool‑use loop and return its *handle*."""

    # brand‑new LLM client dedicated to the nested conversation
    inner_client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    inner_client.set_system_message(
        "You are running inside an automated test. "
        "ONLY do the following steps:\n"
        "1️⃣  Call `inner_tool` (no arguments).\n"
        "2️⃣  Wait for its response.\n"
        "3️⃣  Reply with exactly the single word 'done'.",
    )

    # Kick off the nested loop – **no interjectable_tools specified** on
    # purpose: the outer loop must deduce that from the returned handle.
    return start_async_tool_use_loop(
        client=inner_client,
        message="start",
        tools={"inner_tool": inner_tool},
        parent_chat_context=None,
        log_steps=False,
        max_steps=10,
        timeout=120,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  The actual test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nested_async_tool_loop():
    """Full end‑to‑end check – no mocks, real network call to OpenAI."""

    # Outer client that drives the *first* loop
    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  Continue running this tool call, when given the option.\n"
        "3️⃣  Once it is *completed*, respond with exactly 'all done'.",
    )

    handle = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        log_steps=False,
        max_steps=10,
        timeout=240,
    )

    # Wait for the outer loop to finish.
    final_reply = await handle.result()

    # The assistant must answer as instructed.
    assert final_reply.strip().lower() == "all done"

    # System message
    assert client.messages[0] == {
        "role": "system",
        "content": "You are running inside an automated test. Perform the steps exactly:\n1\ufe0f\u20e3  Call `outer_tool` with no arguments.\n2\ufe0f\u20e3  Continue running this tool call, when given the option.\n3\ufe0f\u20e3  Once it is *completed*, respond with exactly 'all done'.",
    }

    # User message
    assert client.messages[1] == {
        "role": "user",
        "content": "start",
    }

    # Assistant tool selection
    tool_selection_msg = client.messages[2]
    assert tool_selection_msg["role"] == "assistant"
    assert len(tool_selection_msg["tool_calls"]) == 1
    assert tool_selection_msg["tool_calls"][0]["function"] == {
        "arguments": "{}",
        "name": "outer_tool",
    }

    # Tool response
    tool_response = client.messages[3]
    assert tool_response["role"] == "tool"
    assert tool_response["name"] == "outer_tool"
    assert "done" in tool_response["content"].lower()

    # Assistant final response
    assert client.messages[4] == {
        "content": "all done",
        "refusal": None,
        "role": "assistant",
        "annotations": [],
        "audio": None,
        "function_call": None,
        "tool_calls": None,
    }
    assert len(client.messages) == 5
