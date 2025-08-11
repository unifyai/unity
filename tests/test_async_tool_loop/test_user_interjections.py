"""
End-to-end test for *interjectable* tools.

Flow we enforce:

1. Assistant calls `long_running(topic="cats")`.
2. Test code sees that call, then injects the user turn
   “Actually, please switch to dogs instead.”
3. Assistant calls the auto-generated helper `interject_<id>`
   which drops “dogs” on the tool’s private queue.
4. `long_running` returns a value that reflects the steer.
5. Assistant outputs one final plain-text answer.

We assert that the tool logs the steer and the final assistant reply
mentions “dogs”.
"""

from __future__ import annotations

import asyncio
import os
from typing import List

import pytest
import unify
from unity.common.llm_helpers import start_async_tool_use_loop
from tests.helpers import _handle_project
import json

MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


def new_client() -> unify.AsyncUnify:
    """Fresh client with caching enabled so the run becomes deterministic."""
    return unify.AsyncUnify(
        MODEL_NAME,
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )


@pytest.mark.asyncio
@_handle_project
async def test_interjectable_tool_roundtrip() -> None:
    client = new_client()

    # ── 1.  Dummy long-running tool ───────────────────────────────────
    exec_log: List[str] = []

    async def long_running(
        topic: str,
        *,
        interject_queue: asyncio.Queue[str],
    ) -> str:
        """
        Wait up to 2 s for a steer; echo whichever topic we end up with.
        """
        try:
            steer = await asyncio.wait_for(interject_queue.get(), timeout=2.0)
            exec_log.append(f"steered→{steer}")
            return f"Topic switched to: {steer}"
        except asyncio.TimeoutError:
            exec_log.append("no-steer")
            return f"Final topic: {topic}"

    long_running.__name__ = "long_running"
    long_running.__qualname__ = "long_running"

    # ── 2.  Kick off the async-tool loop ──────────────────────────────
    handle = start_async_tool_use_loop(
        client=client,
        message=(
            "Follow STRICTLY these steps:\n"
            '1️⃣  Call `long_running` with `{ "topic": "cats" }`.\n'
            "2️⃣  WAIT for my next instruction.\n"
            "3️⃣  When I say “Actually, please switch to X instead.” "
            'call the helper `interject_<id>` with `{ "content": "X" }`.\n'
            "4️⃣  After the tool finishes, reply with ONE sentence "
            "mentioning the final topic.\n"
            "Do NOT add extra text between steps."
        ),
        tools={"long_running": long_running},
    )

    # ── 3.  Wait until the model has really scheduled the first call ──
    cats_call_seen = False
    for _ in range(40):  # up to ~4 s
        await asyncio.sleep(0.1)
        for m in client.messages:
            if (
                m.get("role") == "assistant"
                and m.get("tool_calls")
                and '"topic":"cats"' in m["tool_calls"][0]["function"]["arguments"]
            ):
                cats_call_seen = True
                break
        if cats_call_seen:
            break
    assert cats_call_seen, "LLM never called long_running with cats."

    # ── 4.  Now inject the steer while the tool is still running ──────
    await handle.interject("Actually, please switch to dogs instead.")

    # ── 5.  Wait for completion ───────────────────────────────────────
    final_answer: str = await handle.result()

    # ── 6.  Assertions ────────────────────────────────────────────────
    assert exec_log == ["steered→dogs"], "Tool must receive the 'dogs' steer."
    assert "dogs" in final_answer.lower(), "Assistant reply must mention dogs."

    # There will be two assistant turns (one with the tool call, one final)
    assistant_msgs = [m for m in client.messages if m["role"] == "assistant"]
    assert assistant_msgs[-1]["tool_calls"] is None, (
        "The last assistant message should be plain-text – "
        "all tool calls should precede it."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verifies *chat-context propagation* down to a tool
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation() -> None:
    """
    The outer loop is given a *parent_chat_context* containing a single
    “root” message.  We expect this to be nested under the messages of the
    current loop and forwarded automatically to any tool that declares a
    ``parent_chat_context`` parameter.
    """
    client = new_client()

    root_ctx = [{"role": "user", "content": "root-level message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(
        *,
        parent_chat_context: list[dict] | None = None,
    ) -> str:
        captured_ctx.append(parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    # Kick off the loop – we *require* the model to call `record_context`
    handle = start_async_tool_use_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
    )

    final_ans = await handle.result()
    assert "done" in final_ans.lower(), "Assistant should finish with 'done'."

    # System header must exist in the loop’s messages
    assert client.messages[0]["role"] == "system"
    assert client.messages[0].get("_ctx_header") is True

    # Exactly one invocation and one propagated JSON context
    assert len(captured_ctx) == 1, "Tool must be called once."
    combined = captured_ctx[0]

    # Shape: root_ctx[0] has a children[] array that contains the new prompt
    assert combined[0]["content"] == "root-level message"
    assert "children" in combined[0], "Nested children list missing."
    child_msgs = combined[0]["children"]
    assert child_msgs and child_msgs[0]["content"].startswith(
        "Please call the function",
    ), "Current loop messages not included as children."


@pytest.mark.asyncio
@_handle_project
async def test_immediate_interjection_after_toolcall_has_tool_reply() -> None:
    """
    Validates that when a user interjection arrives immediately after an
    assistant message that contains tool_calls, there is *already* a
    corresponding tool message placed directly after that assistant turn.

    Pre-fix (no immediate placeholders), this could violate the API
    invariant and cause a 400 because the interjection would be appended
    before any tool reply. Post-fix, immediate placeholders ensure the
    ordering is valid.
    """
    client = new_client()

    # A small synchronous tool that takes a short time to complete
    import time as _time

    def slow_tool(x: int) -> str:
        _time.sleep(0.2)
        return f"ok:{x}"

    slow_tool.__name__ = "slow_tool"
    slow_tool.__qualname__ = "slow_tool"

    handle = start_async_tool_use_loop(
        client=client,
        message=(
            "Follow these steps strictly:\n"
            '1) Call slow_tool with { "x": 1 }.\n'
            "2) WAIT for my next instruction.\n"
            "3) After my next instruction, reply with ONE word: done."
        ),
        tools={"slow_tool": slow_tool},
    )

    # Wait for the assistant to emit the tool call we expect
    call_id: str | None = None
    assistant_idx: int | None = None
    for _ in range(50):  # up to ~5s
        await asyncio.sleep(0.1)
        for i, m in enumerate(client.messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc = m["tool_calls"][0]
                if tc["function"]["name"] == "slow_tool" and '"x":1' in tc["function"][
                    "arguments"
                ].replace(" ", ""):
                    call_id = tc["id"]
                    assistant_idx = i
                    break
        if call_id is not None:
            break
    assert (
        call_id is not None and assistant_idx is not None
    ), "Assistant did not call slow_tool as instructed."

    # Immediately inject the user follow-up to race with scheduling
    await handle.interject("finish")

    # Wait until the interjection user message appears, then assert the very next
    # message after the assistant is already a tool reply for that call.
    # This would FAIL pre-fix because the user interjection would be appended first.
    saw_interjection = False
    for _ in range(50):  # up to ~5s
        await asyncio.sleep(0.1)
        if any(
            m.get("role") == "user" and m.get("content") == "finish"
            for m in client.messages
        ):
            saw_interjection = True
            assert (assistant_idx + 1) < len(
                client.messages,
            ), "No message after assistant turn yet."
            next_msg = client.messages[assistant_idx + 1]
            assert (
                next_msg.get("role") == "tool"
                and next_msg.get("tool_call_id") == call_id
            ), "Expected a tool message directly after assistant tool_calls when interjection arrives."
            break
    assert saw_interjection, "Interjection user message never appeared in transcript."

    # Ensure the loop finishes cleanly and the final answer is produced
    final_ans = await handle.result()
    assert isinstance(final_ans, str) and len(final_ans) > 0
    assert "done" in final_ans.lower(), "Assistant should finish with 'done'."


@pytest.mark.asyncio
@_handle_project
async def test_backfills_missing_tool_reply_for_prior_assistant_turn() -> None:
    """
    Pre-seed the transcript with an assistant message that contains a tool_call
    but has no following tool reply. The loop must *backfill* a tool message
    directly after that assistant turn before proceeding (e.g., before any new
    user message is sent or another assistant turn occurs).

    This would have failed pre-fix (no invariant repair), because a subsequent
    user message could be appended before a tool reply, violating the API
    ordering invariant. With the backfill logic, a tool placeholder is inserted
    immediately after the assistant to restore ordering, and the corresponding
    tool task is scheduled.
    """
    client = new_client()

    # A quick tool that matches the pre-seeded tool_call
    def slow_tool(x: int) -> str:
        return f"ok:{x}"

    slow_tool.__name__ = "slow_tool"
    slow_tool.__qualname__ = "slow_tool"

    # Pre-populate an assistant tool_call without a following tool reply
    call_id = "call_TESTBACKFILL"
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "slow_tool", "arguments": '{"x": 1}'},
            },
        ],
    }
    client.append_messages([assistant_msg])

    # Start the loop – it should detect the missing tool reply and backfill it
    handle = start_async_tool_use_loop(
        client=client,
        message="Please proceed.",
        tools={"slow_tool": slow_tool},
    )

    # Allow the loop a brief moment to perform the backfill insertion
    assistant_idx: int | None = None
    for _ in range(50):  # up to ~2.5s
        await asyncio.sleep(0.05)
        # Find our pre-seeded assistant turn
        for i, m in enumerate(client.messages):
            if (
                m.get("role") == "assistant"
                and m.get("tool_calls")
                and any(tc.get("id") == call_id for tc in m["tool_calls"])
            ):
                assistant_idx = i
                break
        if assistant_idx is not None and len(client.messages) > assistant_idx + 1:
            break

    assert assistant_idx is not None, "Pre-seeded assistant tool_call not present."
    assert (assistant_idx + 1) < len(
        client.messages,
    ), "No message after the assistant turn yet."

    # The next message must be a tool reply for the same call_id, inserted by backfill
    next_msg = client.messages[assistant_idx + 1]
    assert (
        next_msg.get("role") == "tool" and next_msg.get("tool_call_id") == call_id
    ), "Expected backfilled tool message directly after assistant tool_calls."

    # Stop the loop cleanly to avoid dangling tasks
    handle.stop()
    with pytest.raises(asyncio.CancelledError):
        await handle.result()
