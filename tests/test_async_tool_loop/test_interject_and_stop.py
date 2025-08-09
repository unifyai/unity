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


class _StrictProtocolUnify:
    """Stub client that enforces provider protocol at append-time.

    If a user message is appended while the latest assistant turn has tool_calls
    but there is no tool message responding to those calls immediately after it,
    raise an error – mirroring the provider's validation.
    """

    def __init__(self):
        self.messages: List[dict] = []
        self._step = 0

    def append_messages(self, msgs: List[dict]):
        for m in msgs:
            # Enforce: after an assistant with tool_calls, the next message must be a tool
            if m.get("role") == "user":
                # find the last assistant with tool_calls
                last_ai_with_calls_idx = None
                for i in range(len(self.messages) - 1, -1, -1):
                    mm = self.messages[i]
                    if mm.get("role") == "assistant" and mm.get("tool_calls"):
                        last_ai_with_calls_idx = i
                        break
                if last_ai_with_calls_idx is not None:
                    # Must be immediately followed by a tool response already present
                    if last_ai_with_calls_idx == len(self.messages) - 1:
                        raise RuntimeError(
                            "Protocol violation: user turn appended before tool message "
                            "responding to the latest assistant tool_calls.",
                        )
                    nxt = self.messages[last_ai_with_calls_idx + 1]
                    if nxt.get("role") != "tool":
                        raise RuntimeError(
                            "Protocol violation: user turn appended without an immediate "
                            "tool response following the assistant tool_calls.",
                        )
                    # If it is a tool, optionally verify id linkage when present
                    if mm := self.messages[last_ai_with_calls_idx]:
                        if mm.get("tool_calls"):
                            expected_id = mm["tool_calls"][0]["id"]
                            if nxt.get("tool_call_id") != expected_id:
                                raise RuntimeError(
                                    "Protocol violation: immediate tool does not match the "
                                    "assistant tool_call id.",
                                )
            self.messages.append(m)

    async def generate(self, *_, **__):
        # First call: ask to run `slow`
        if self._step == 0:
            self._step += 1
            msg = {
                "role": "assistant",
                "content": "run slow",
                "tool_calls": [
                    {
                        "id": "call_slow_1",
                        "type": "function",
                        "function": {"name": "slow", "arguments": "{}"},
                    },
                ],
            }
        else:
            # Subsequent calls: plain assistant content
            self._step += 1
            msg = {"role": "assistant", "content": "DONE", "tool_calls": []}
        self.messages.append(msg)
        return msg

    @property
    def system_message(self) -> str:
        return ""


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
    idx_user_B = next(
        i
        for i, m in enumerate(msgs)
        if m["role"] == "user" and "echo B" in m["content"]
    )
    idx_second_asst = msgs.index(assistant_tool_turns[1])
    assert idx_first_asst < idx_user_B < idx_second_asst

    # 4. there are matching tool *results* for A and B
    # Handle both normal completion (name="echo") and continue helper completion
    # (name starts with "echo(" like "echo({"txt":"A"}) completed successfully...")
    tool_msgs = [
        m
        for m in msgs
        if m["role"] == "tool"
        and (m["name"] == "echo" or m["name"].startswith("echo("))
    ]
    assert any("A" in m["content"] for m in tool_msgs)
    assert any("B" in m["content"] for m in tool_msgs)


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
    i_user = _user_index(msgs, "thanks!")

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
    i_user = _user_index(msgs, "cheers!")

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
    assert (
        roles.count("user") == 2
    ), "Both the original user turn and the interjection must be present."
    assert (
        roles[-1] == "assistant"
    ), "The conversation should end with the assistant's single reply."


@pytest.mark.asyncio
@_handle_project
async def test_interjection_before_any_tool_result_inserts_placeholder():
    """
    Regression test for race where an interjection arrives **after** an assistant
    emits tool_calls but **before** any tool message exists.

    Expectation: a placeholder tool message is inserted immediately after the
    assistant tool_calls turn, so the subsequent user interjection never violates
    the function-calling protocol.
    """
    client = new_client()
    client.set_cache(False)

    handle = start_async_tool_use_loop(
        client,
        ("Run the tool `slow` exactly once, then reply only with the word DONE."),
        {"slow": slow},
        interrupt_llm_with_interjections=True,  # maximize chance of pre-emption
    )

    # Wait until the assistant has requested the slow tool
    await _wait_for_tool_request(client, "slow")

    # Immediately interject BEFORE any tool result can be produced
    interjection_text = "(test) just checking ordering"
    await handle.interject(interjection_text)

    # Complete the loop normally (should not raise provider errors)
    await handle.result()

    # Assertions: ensure a tool message sits between assistant tool_calls and user
    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)

    # The very next message must be a tool message for the scheduled call (placeholder or real)
    assert msgs[i_asst + 1]["role"] == "tool"
    assert msgs[i_asst + 1]["name"] == "slow"

    # And the interjection must appear AFTER that tool message
    i_user = _user_index(msgs, interjection_text)
    assert i_user > i_asst + 1

    # Sanity: the placeholder/result tool message must respond to the assistant tool_call id
    asst_call_id = msgs[i_asst]["tool_calls"][0]["id"]
    assert msgs[i_asst + 1]["tool_call_id"] == asst_call_id


@pytest.mark.asyncio
@_handle_project
async def test_preemptive_interjection_violates_protocol():
    """
    Uses a strict stub that enforces protocol on append_messages. The test
    interjects immediately after the assistant schedules a tool.

    - Pre-patch: user interjection would be appended before any tool message is
      inserted, so append_messages raises.
    - Post-patch: immediate placeholder is inserted at scheduling; user append
      is allowed and the loop finishes.
    """
    client = _StrictProtocolUnify()

    handle = start_async_tool_use_loop(
        client,
        "Call the tool `slow` exactly once, then say DONE.",
        {"slow": slow},
        interrupt_llm_with_interjections=True,
    )

    # Wait until the assistant has requested the slow tool
    await _wait_for_tool_request(client, "slow")

    # Interject immediately – append_messages of stub will enforce ordering
    await handle.interject("(strict) now interjecting")

    # Should complete without raising under the patched implementation
    final = await handle.result()
    assert isinstance(final, str) and final.strip()
