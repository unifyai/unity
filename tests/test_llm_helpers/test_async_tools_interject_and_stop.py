"""
Behaviour-driven tests for the **live-handle** async-tool loop – now executed
against a *real* ``unify.AsyncUnify`` client instead of a monkey-patched stub.

What’s covered
--------------

* Injecting extra user messages that trigger additional tool calls.
* Graceful cancellation with ``stop()``.
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


# --------------------------------------------------------------------------- #
#  GLOBALS                                                                    #
# --------------------------------------------------------------------------- #

MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS                                                       #
# --------------------------------------------------------------------------- #
async def echo(txt: str) -> str:  # noqa: D401 – simple async tool
    await asyncio.sleep(0.05)
    return txt


async def slow(txt: str = "Z", delay: float = 0.25) -> str:
    await asyncio.sleep(delay)
    return txt


async def fast() -> str:          # ~100 ms
    await asyncio.sleep(0.10)
    return "fast"



# ---------------------------------------------------------------------------#
#  Utility                                                                    #
# ---------------------------------------------------------------------------#
def _first_with_tool_calls(msgs: List[dict]) -> int:
    return next(i for i, m in enumerate(msgs) if m.get("tool_calls"))


def _user_index(msgs: List[dict], snippet: str) -> int:
    return next(i for i, m in enumerate(msgs)
                if m["role"] == "user" and snippet in m["content"])


def _tool_indices(msgs: List[dict]) -> List[int]:
    return [i for i, m in enumerate(msgs) if m["role"] == "tool"]


def _are_contiguous(indices: List[int]) -> bool:
    return sorted(indices) == list(range(min(indices), max(indices) + 1))

def _assistant_tool_turns(msgs: List[dict[str, Any]]):
    """Yield assistant turns that contain tool_calls."""
    return [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]


# --------------------------------------------------------------------------- #
#  FIXTURES                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client():
    """Provide a new client for every test function."""
    return unify.AsyncUnify(MODEL_NAME)


# --------------------------------------------------------------------------- #
#  TESTS                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_interject_leads_to_second_tool_and_final_result(client):
    """
    We start the loop asking the model to echo “A”.  Then we interject, asking
    it to echo “B” too.  We expect two separate tool calls and a final “done”.
    """
    handle = start_async_tool_use_loop(
        client,
        message=(
            "Use the `echo` tool to output the text 'A'. "
            "If the user later asks for another echo, call the tool again with "
            "that text and finally reply exactly 'done'."
        ),
        tools={"echo": echo},
    )

    # --- inject clarification ------------------------------------------------
    await asyncio.sleep(0.02)
    await handle.interject("And echo B please")

    final = await handle.result()
    assert final.strip().lower().startswith("done")

    # --- assertions ----------------------------------------------------------
    msgs = client.messages

    # 1. we saw two *assistant* turns requesting tool calls
    assistant_tool_turns = _assistant_tool_turns(msgs)
    assert len(assistant_tool_turns) >= 2

    # 2. first assistant turn calls echo("A"), second calls echo("B")
    first_args = json.loads(assistant_tool_turns[0]["tool_calls"][0]["function"]["arguments"])
    second_args = json.loads(assistant_tool_turns[1]["tool_calls"][0]["function"]["arguments"])
    assert first_args == {"txt": "A"}
    assert second_args == {"txt": "B"}

    # 3. the order is correct: initial assistant → user interjection → 2nd assistant
    idx_first_asst = msgs.index(assistant_tool_turns[0])
    idx_user_B = next(i for i, m in enumerate(msgs) if m["role"] == "user" and "echo B" in m["content"])
    idx_second_asst = msgs.index(assistant_tool_turns[1])
    assert idx_first_asst < idx_user_B < idx_second_asst

    # 4. there are matching tool *results* for A and B
    tool_msgs = [m for m in msgs if m["role"] == "tool" and m["name"] == "echo"]
    assert any("A" in m["content"] for m in tool_msgs)
    assert any("B" in m["content"] for m in tool_msgs)


@pytest.mark.asyncio
async def test_stop_cancels_gracefully(client):
    """
    Calling ``stop()`` should cancel the loop: ``result()`` raises
    ``CancelledError`` and the underlying task is done.
    """
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
async def test_interjections_are_processed_and_loop_completes(client):
    """
    Launch the async-tool loop, fire two interjections, then wait for normal
    completion.  Verify

      • the loop ends without error,
      • the *user* messages are preserved in FIFO order,
      • at least three tool invocations happened (A, B, C).
    """
    handle = start_async_tool_use_loop(
        client,
        "Echo A please, then say 'done' when finished.",
        {"echo": echo},
    )

    # Two quick interjections while the first tool is still running
    await asyncio.sleep(0.01)
    await handle.interject("B please")
    await asyncio.sleep(0.01)
    await handle.interject("C please")

    # Wait for the final assistant answer (we don't assert its exact content)
    final = await handle.result()
    assert isinstance(final, str) and final.strip()

    # 1. User-message order must be exactly the order we sent them
    seen_users = [m["content"] for m in client.messages if m["role"] == "user"]
    assert seen_users[:3] == [
        "Echo A please, then say 'done' when finished.",
        "B please",
        "C please",
    ]

    # 2. There must be at least three tool-result messages overall
    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) >= 3


@pytest.mark.asyncio
async def test_single_tool_result_is_inserted_before_interjection(client):
    """
    * Assistant is instructed to run `slow` once and then reply "ack".
    * We interject while `slow` is still running.
    * Expect: assistant → tool result → user interjection (contiguous order).
    """
    handle = start_async_tool_use_loop(
        client,
        (
            "Run the tool `slow` exactly once, "
            "then reply with the word ACK (nothing else)."
        ),
        {"slow": slow},
    )

    await asyncio.sleep(0.05)        # tool should still be running
    await handle.interject("thanks!")

    await handle.result()            # wait for completion

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    i_tool = _tool_indices(msgs)[0]          # only one result
    i_user = _user_index(msgs, "thanks!")

    # assistant → tool → user, contiguous
    assert (i_asst + 1 == i_tool) and (i_tool + 1 == i_user)

    # assistant turn’s tool_calls restored exactly once
    assert len(msgs[i_asst]["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_parallel_tool_results_shift_interjection_down(client):
    """
    * Assistant is instructed to run BOTH `fast` and `slow` before replying "done".
    * We interject while the tools are running.
    * Expect both tool results to sit immediately after the assistant turn
      (in any order) and the user message to follow them.
    """
    handle = start_async_tool_use_loop(
        client,
        (
            "Call the tools `fast` and `slow`, each exactly once, "
            "then respond with ONLY the word DONE."
        ),
        {"fast": fast, "slow": slow},
    )

    await asyncio.sleep(0.15)       # `fast` likely done, `slow` still running
    await handle.interject("cheers!")

    await handle.result()

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    tool_idxs = _tool_indices(msgs)[:2]      # we only care about the first two
    i_user   = _user_index(msgs, "cheers!")

    # Tool results are contiguous right after the assistant message
    assert _are_contiguous(tool_idxs)
    assert tool_idxs[0] == i_asst + 1

    # User interjection sits immediately after the last tool result
    assert i_user == max(tool_idxs) + 1

    # Tool_calls restored once, no duplicates
    assert len(msgs[i_asst]["tool_calls"]) >= 2
