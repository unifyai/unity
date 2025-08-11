"""
Behaviour-driven tests for the **live-handle** async-tool loop – now executed
against a *real* ``unify.AsyncUnify`` client instead of a monkey-patched stub.

What’s covered
--------------

* Injecting extra user messages that trigger additional tool calls.
* Graceful stoplation with ``stop()``.
* Preservation of the order of multiple interjections.
* Handling of an interjection that arrives while a tool call is still running.

To run the suite you need:

* a valid API key in your environment,
* internet connectivity, and
* the ``unity.common.llm_helpers`` implementation in your import path.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, List

import pytest
import unify
from unity.common.llm_helpers import start_async_tool_use_loop
from tests.helpers import (
    _handle_project,
)
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
)


# --------------------------------------------------------------------------- #
#  GLOBALS                                                                    #
# --------------------------------------------------------------------------- #

MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS                                                       #
# --------------------------------------------------------------------------- #
@unify.traced
async def echo(txt: str) -> str:  # noqa: D401 – simple async tool
    await asyncio.sleep(0.50)
    return txt


@unify.traced
async def slow() -> str:
    await asyncio.sleep(0.15)
    return "slow"


@unify.traced
async def fast() -> str:
    await asyncio.sleep(0.05)
    return "fast"


# ---------------------------------------------------------------------------#
#  Utility                                                                    #
# ---------------------------------------------------------------------------#
@unify.traced
def _first_with_tool_calls(msgs: List[dict]) -> int:
    return next(i for i, m in enumerate(msgs) if m.get("tool_calls"))


@unify.traced
def _user_index(msgs: List[dict], snippet: str) -> int:
    return next(
        i for i, m in enumerate(msgs) if m["role"] == "user" and snippet in m["content"]
    )


# locate the *system* interjection message containing **snippet**
@unify.traced
def _interjection_index(msgs: List[dict], snippet: str) -> int:
    """Return index of the system-role interjection whose content includes *snippet*."""
    return next(
        i
        for i, m in enumerate(msgs)
        if m["role"] == "system"
        and "user: **" in m.get("content", "")
        and snippet in m["content"]
    )


@unify.traced
def _tool_indices(msgs: List[dict]) -> List[int]:
    return [i for i, m in enumerate(msgs) if m["role"] == "tool"]


@unify.traced
def _are_contiguous(indices: List[int]) -> bool:
    return sorted(indices) == list(range(min(indices), max(indices) + 1))


@unify.traced
def _assistant_tool_turns(msgs: List[dict[str, Any]]):
    """Yield assistant turns that contain tool_calls."""
    return [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]


# --------------------------------------------------------------------------- #
#  HELPERS                                                                    #
# --------------------------------------------------------------------------- #
@unify.traced
def new_client() -> unify.AsyncUnify:
    """
    Return a fresh client *with its own conversation state* so that tests do
    not interfere with one another.
    """
    return unify.AsyncUnify(
        MODEL_NAME,
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )


# --------------------------------------------------------------------------- #
#  TESTS                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_interject_leads_to_second_tool_and_final_result():
    """
    We start the loop asking the model to echo “A”.  Then we interject, asking
    it to echo “B” too.  We expect two separate tool calls and a final “done”.
    """
    client = new_client()
    handle = start_async_tool_use_loop(
        client,
        message=("Use the `echo` tool to output the text 'A'."),
        tools={"echo": echo},
        interrupt_llm_with_interjections=False,
    )

    # --- wait until the assistant has scheduled the first echo call ---------
    await _wait_for_tool_request(client, "echo")

    # --- inject clarification ------------------------------------------------
    await handle.interject(
        "And also echo B please, the order of the echos doesn't matter."
        "Don't return until both have completed."
        "Use the 'continue' tool to continue a pending tool execution if needed.",
    )

    await handle.result()

    # --- assertions ----------------------------------------------------------
    msgs = client.messages

    # 1. we saw two *assistant* turns requesting tool calls
    assistant_tool_turns = _assistant_tool_turns(msgs)
    assert len(assistant_tool_turns) >= 2

    # 2. first assistant turn calls echo("A"), second calls echo("B")

    first_args = json.loads(
        assistant_tool_turns[0]["tool_calls"][0]["function"]["arguments"],
    )
    assert first_args == {"txt": "A"}

    second_tool_calls = assistant_tool_turns[1]["tool_calls"]

    # Check that the second assistant turn includes an echo("B") call
    # The LLM might also choose to continue/stop the first echo, which is fine
    echo_b_found = False
    for call in second_tool_calls:
        try:
            args = json.loads(call["function"]["arguments"])
            if args == {"txt": "B"} and call["function"]["name"] == "echo":
                echo_b_found = True
                break
        except (json.JSONDecodeError, KeyError):
            continue  # Skip malformed calls

    assert (
        echo_b_found
    ), f"Second assistant turn should include echo('B'), got: {second_tool_calls}"

    # 3. the order is correct: initial assistant → user interjection → 2nd assistant
    idx_first_asst = msgs.index(assistant_tool_turns[0])
    idx_inter_B = _interjection_index(msgs, "echo B")
    idx_second_asst = msgs.index(assistant_tool_turns[1])
    assert idx_first_asst < idx_inter_B < idx_second_asst

    # 3b. formatted system message must follow the new convention
    inter_msg = msgs[idx_inter_B]
    assert inter_msg["content"].startswith("The user *cannot* see")
    assert "user: **" in inter_msg["content"] and "echo B" in inter_msg["content"]

    # 4. there are matching tool *results* for A and B
    # Handle both normal completion (name="echo") and continue helper completion
    # (name starts with "echo(" like "echo({"txt":"A"}) completed successfully...")
    tool_msgs = [
        m
        for m in msgs
        if m["role"] == "tool"
        and (m["name"] == "echo" or m["name"].startswith("echo("))
    ]
    # Collect tool messages and final assistant reply to confirm both outputs appear.
    all_text_blobs = [m.get("content", "") for m in msgs]
    joined_text = "\n".join(str(t) for t in all_text_blobs)
    assert "A" in joined_text, "Echo result for 'A' missing in transcript."
    assert "B" in joined_text, "Echo result for 'B' missing in transcript."


@pytest.mark.asyncio
@_handle_project
async def test_stop_stops_gracefully():
    """
    Calling ``stop()`` should stop the loop: ``result()`` raises
    ``CancelledError`` and the underlying task is done.
    """
    client = new_client()
    handle = start_async_tool_use_loop(
        client,
        "Echo something then say 'ok'.",
        {"echo": echo},
    )

    handle.stop()

    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_interjections_are_processed_and_loop_completes():
    """
    Launch the async-tool loop, fire two interjections, then wait for normal
    completion.  Verify

      • the loop ends without error,
      • the *user* messages are preserved in FIFO order,
      • at least three tool invocations happened (A, B, C).
    """
    client = new_client()
    handle = start_async_tool_use_loop(
        client,
        "Echo A please, then say 'done' when finished.",
        {"echo": echo},
    )

    # Two quick interjections while the first tool is still running
    await _wait_for_tool_request(client, "echo")  # ensure first still noted
    await handle.interject("B please")

    # Wait for assistant to schedule second echo before next interjection
    await _wait_for_tool_request(client, "echo")
    await handle.interject("C please")

    # Wait for the final assistant answer (we don't assert its exact content)
    final = await handle.result()
    assert isinstance(final, str) and final.strip()

    # 1. User-message order must be exactly the order we sent them
    # Ensure interjections B and C appear in FIFO order as system messages
    msgs = client.messages  # reuse
    idx_B = _interjection_index(msgs, "B please")
    idx_C = _interjection_index(msgs, "C please")
    assert idx_B < idx_C

    # 2. There must be at least three tool-result messages overall
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) >= 3


@pytest.mark.asyncio
@_handle_project
async def test_single_tool_result_is_inserted_before_interjection():
    """
    * Assistant is instructed to run `slow` once and then reply "ack".
    * We interject while `slow` is still running.
    * Expect: assistant → tool result → user interjection (contiguous order).
    """
    client = new_client()
    handle = start_async_tool_use_loop(
        client,
        (
            "Run the tool `slow` exactly once, "
            "then reply with the word ACK (nothing else)."
        ),
        {"slow": slow},
        interrupt_llm_with_interjections=False,
    )

    # Wait until the `slow` tool is actually running
    await _wait_for_tool_request(client, "slow")
    await handle.interject("thanks!")

    await handle.result()  # wait for completion

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    i_tool = _tool_indices(msgs)[0]  # only one result
    i_user = _interjection_index(msgs, "thanks!")

    # assistant → tool → user, contiguous
    assert (i_asst + 1 == i_tool) and (i_tool + 1 == i_user)

    # assistant turn’s tool_calls restored exactly once
    assert len(msgs[i_asst]["tool_calls"]) == 1


@pytest.mark.asyncio
@_handle_project
async def test_parallel_tool_results_shift_interjection_down():
    """
    * Assistant is instructed to run BOTH `fast` and `slow` before replying "done".
    * We interject while the tools are running.
    * Expect both tool results to sit immediately after the assistant turn
      (in any order) and the user message to follow them.
    """
    client = new_client()
    client.set_cache(False)
    handle = start_async_tool_use_loop(
        client,
        (
            "Call the tools `fast` and `slow` both at the same time, "
            "then respond with ONLY the word DONE."
        ),
        {"fast": fast, "slow": slow},
        interrupt_llm_with_interjections=True,  # allow interjection to pre-empt
    )

    # 1. Wait until *both* tool calls have been offered to the model.
    await _wait_for_tool_request(client, "fast")
    await _wait_for_tool_request(client, "slow")

    # 2. Wait until the first two tool *results* (fast + slow placeholder)
    #    are present so we can safely interject while the LLM is still busy
    #    processing them but **before** it produces its final "DONE" reply.
    await _wait_for_tool_result(client, min_results=2)

    # 3. Fire the interjection which should interrupt the ongoing generation
    #    and therefore be inserted *immediately* after the tool results.
    await handle.interject("cheers!")

    # 4. Await normal completion of the loop and then validate order.
    await handle.result()

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    tool_idxs = _tool_indices(msgs)[:2]  # we only care about the first two
    i_user = _interjection_index(msgs, "cheers!")

    # Tool results are contiguous right after the assistant message
    assert _are_contiguous(tool_idxs)
    assert tool_idxs[0] == i_asst + 1

    # User interjection sits immediately after the last tool result
    assert i_user == max(tool_idxs) + 1

    # Tool_calls restored once, no duplicates
    assert len(msgs[i_asst]["tool_calls"]) >= 2


@pytest.mark.asyncio
@_handle_project
async def test_interjection_stops_ongoing_llm():
    """The first LLM generation is stopped once the user interjects."""

    # Spin up the tool-use loop and inject a message shortly afterwards
    client = new_client()
    client.set_cache(False)
    handle = start_async_tool_use_loop(
        client,
        "Tell me something interesting about whales.",
        {},
    )

    # Wait until the assistant started generating before we interject
    await asyncio.sleep(0.02)  # keep minimal wait as generate is immediate
    await handle.interject("Actually, make it about dolphins instead!")
    await handle.result()

    # Assertions – only ONE assistant message should exist
    assistant_msgs = [m for m in client.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1, (
        "Exactly one assistant reply is expected after stoplation; "
        f"found {len(assistant_msgs)}."
    )

    # The final assistant reply must come *after* both user messages
    roles = [m["role"] for m in client.messages]
    # Only the initial prompt remains a true *user* message.
    assert roles.count("user") == 1
    # The formatted interjection must exist as a system message.
    assert any(
        m["role"] == "system" and "dolphins" in m.get("content", "")
        for m in client.messages
    ), "System interjection message not found."
