"""
Interjection behaviours for the async tool loop (live-handle path).

Covers:
- Injecting extra user messages that trigger additional tool calls.
- Preservation and placement of interjections relative to tool results.
- Pre-empting/interrupting LLM turns with interjections.
- Graceful stop via handle.stop().
- Immediate placeholder insertion and backfill of missing tool replies.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, List

import pytest
import unify
from unity.common.async_tool_loop import start_async_tool_use_loop
from tests.helpers import _handle_project, SETTINGS
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
def new_client() -> unify.AsyncUnify:
    """
    Return a fresh client with its own conversation state so tests do not
    interfere with one another.
    """
    return unify.AsyncUnify(
        MODEL_NAME,
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )


@unify.traced
def _first_with_tool_calls(msgs: List[dict]) -> int:
    return next(i for i, m in enumerate(msgs) if m.get("tool_calls"))


@unify.traced
def _interjection_index(msgs: List[dict], snippet: str) -> int:
    """Return index of the system-role interjection whose content includes snippet."""
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
#  TESTS                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_interject_leads_to_second_tool_and_final_result():
    """
    Start with echo("A"), then interject to also echo("B"). Expect two tool
    calls and a final plain-text result.
    """
    client = new_client()
    handle = start_async_tool_use_loop(
        client,
        message=("Use the `echo` tool to output the text 'A'."),
        tools={"echo": echo},
        interrupt_llm_with_interjections=False,
    )

    await _wait_for_tool_request(client, "echo")

    await handle.interject(
        "And also echo B please, the order of the echos doesn't matter."
        "Don't return until both have completed."
        "Use the 'continue' tool to continue a pending tool execution if needed.",
    )

    await handle.result()

    msgs = client.messages

    assistant_tool_turns = _assistant_tool_turns(msgs)
    assert len(assistant_tool_turns) >= 2

    first_args = json.loads(
        assistant_tool_turns[0]["tool_calls"][0]["function"]["arguments"],
    )
    assert first_args == {"txt": "A"}

    second_tool_calls = assistant_tool_turns[1]["tool_calls"]

    echo_b_found = False
    for call in second_tool_calls:
        try:
            args = json.loads(call["function"]["arguments"])
            if args == {"txt": "B"} and call["function"]["name"] == "echo":
                echo_b_found = True
                break
        except (json.JSONDecodeError, KeyError):
            continue

    assert (
        echo_b_found
    ), f"Second assistant turn should include echo('B'), got: {second_tool_calls}"

    idx_first_asst = msgs.index(assistant_tool_turns[0])
    idx_inter_B = _interjection_index(msgs, "echo B")
    idx_second_asst = msgs.index(assistant_tool_turns[1])
    assert idx_first_asst < idx_inter_B < idx_second_asst

    inter_msg = msgs[idx_inter_B]
    assert inter_msg["content"].startswith("The user *cannot* see")
    assert "user: **" in inter_msg["content"] and "echo B" in inter_msg["content"]


@pytest.mark.asyncio
@_handle_project
async def test_stop_stops_gracefully():
    """handle.stop() cancels the loop and result() returns a standard notice string."""
    client = new_client()
    handle = start_async_tool_use_loop(
        client,
        "Echo something then say 'ok'.",
        {"echo": echo},
    )

    handle.stop()

    final = await handle.result()
    assert final == "processed stopped early, no result"


@pytest.mark.asyncio
@_handle_project
async def test_backfills_missing_tool_reply_for_helper_call() -> None:
    """
    Pre-seed transcript with an assistant helper tool_call (e.g. wait).
    New behaviour: helper `wait` is pruned (no backfilled tool reply, no chat clutter).
    The pre-seeded assistant helper turn should be removed, and no tool reply should appear.
    """
    client = new_client()

    helper_call_id = "call_TEST_HELPER"
    helper_name = "wait"
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": helper_call_id,
                "type": "function",
                "function": {"name": helper_name, "arguments": "{}"},
            },
        ],
    }
    client.append_messages([assistant_msg])

    handle = start_async_tool_use_loop(
        client=client,
        message="Please proceed.",
        tools={},  # helpers are acknowledged during backfill without execution
    )

    # Allow the loop to process backfill/pruning
    for _ in range(60):
        await asyncio.sleep(0.05)
        if client.messages:
            break

    # The pre-seeded helper assistant turn should be pruned
    assert assistant_msg not in client.messages

    # No assistant message should contain the helper tool_call id
    assert not any(
        m.get("role") == "assistant"
        and m.get("tool_calls")
        and any(tc.get("id") == helper_call_id for tc in m["tool_calls"])
        for m in client.messages
    )

    # No tool reply should reference the helper call id
    assert not any(
        m.get("role") == "tool" and m.get("tool_call_id") == helper_call_id
        for m in client.messages
    )

    # Cleanly stop the loop
    handle.stop()
    final2 = await handle.result()
    assert final2 == "processed stopped early, no result"

    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_interjections_are_processed_and_loop_completes():
    """
    Fire two interjections (B, then C) and validate FIFO order and sufficient tool work.
    """
    client = new_client()
    handle = start_async_tool_use_loop(
        client,
        "Echo A please, then say 'done' when finished.",
        {"echo": echo},
    )

    await _wait_for_tool_request(client, "echo")
    await handle.interject("B please")

    await _wait_for_tool_request(client, "echo")
    await handle.interject("C please")

    final = await handle.result()
    assert isinstance(final, str) and final.strip()

    msgs = client.messages
    idx_B = _interjection_index(msgs, "B please")
    idx_C = _interjection_index(msgs, "C please")
    assert idx_B < idx_C

    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) >= 3


@pytest.mark.asyncio
@_handle_project
async def test_single_tool_result_is_inserted_before_interjection():
    """
    Run `slow` once then reply "ACK". Interject while running.
    Expect: assistant → tool result → interjection.
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

    await _wait_for_tool_request(client, "slow")
    await handle.interject("thanks!")

    await handle.result()

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    i_tool = _tool_indices(msgs)[0]
    i_user = _interjection_index(msgs, "thanks!")

    assert (i_asst + 1 == i_tool) and (i_tool + 1 == i_user)
    assert len(msgs[i_asst]["tool_calls"]) == 1


@pytest.mark.asyncio
@_handle_project
async def test_parallel_tool_results_shift_interjection_down():
    """
    Run both `fast` and `slow`, interject while they are running.
    Expect both tool results right after the assistant turn, interjection follows.
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
        interrupt_llm_with_interjections=True,
    )

    await _wait_for_tool_request(client, "fast")
    await _wait_for_tool_request(client, "slow")

    await _wait_for_tool_result(client, min_results=2)

    await handle.interject("cheers!")
    await handle.result()

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    tool_idxs = _tool_indices(msgs)[:2]
    i_user = _interjection_index(msgs, "cheers!")

    assert _are_contiguous(tool_idxs)
    assert tool_idxs[0] == i_asst + 1
    assert i_user == max(tool_idxs) + 1
    assert len(msgs[i_asst]["tool_calls"]) >= 2


@pytest.mark.asyncio
@_handle_project
async def test_interjection_stops_ongoing_llm():
    """The first LLM generation is stopped once the user interjects."""
    client = new_client()
    client.set_cache(False)
    handle = start_async_tool_use_loop(
        client,
        "Tell me something interesting about whales.",
        {},
    )

    await asyncio.sleep(0.02)
    await handle.interject("Actually, make it about dolphins instead!")
    await handle.result()

    assistant_msgs = [m for m in client.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1

    roles = [m["role"] for m in client.messages]
    assert roles.count("user") == 1
    assert any(
        m["role"] == "system" and "dolphins" in m.get("content", "")
        for m in client.messages
    )


# ─────────────────────────────────────────────────────────────────────────────
# Additional interjection behaviours: steerable tool roundtrip + invariants
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interjectable_tool_roundtrip() -> None:
    client = new_client()

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

    # Wait for long_running(topic="cats") to be scheduled
    cats_call_seen = False
    for _ in range(40):
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

    await handle.interject("Actually, please switch to dogs instead.")

    final_answer: str = await handle.result()

    assert exec_log == ["steered→dogs"], "Tool must receive the 'dogs' steer."
    assert "dogs" in final_answer.lower(), "Assistant reply must mention dogs."

    assistant_msgs = [m for m in client.messages if m["role"] == "assistant"]
    assert assistant_msgs[-1]["tool_calls"] is None


@pytest.mark.asyncio
@_handle_project
async def test_immediate_interjection_after_toolcall_has_tool_reply() -> None:
    """
    When an interjection arrives immediately after an assistant tool_calls turn,
    a tool placeholder must already be present to maintain API ordering.
    """
    client = new_client()

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

    call_id: str | None = None
    assistant_idx: int | None = None
    for _ in range(50):
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
    assert call_id is not None and assistant_idx is not None

    await handle.interject("finish")

    saw_interjection = False
    for _ in range(50):
        await asyncio.sleep(0.1)
        if any(
            m.get("role") == "system" and "user: **finish**" in (m.get("content") or "")
            for m in client.messages
        ):
            saw_interjection = True
            assert (assistant_idx + 1) < len(client.messages)
            next_msg = client.messages[assistant_idx + 1]
            assert (
                next_msg.get("role") == "tool"
                and next_msg.get("tool_call_id") == call_id
            )
            break
    assert saw_interjection

    final_ans = await handle.result()
    assert isinstance(final_ans, str) and len(final_ans) > 0
    assert "done" in final_ans.lower()


@pytest.mark.asyncio
@_handle_project
async def test_backfills_missing_tool_reply_for_prior_assistant_turn() -> None:
    """
    Pre-seed transcript with assistant tool_call but no tool reply.
    The loop must backfill a tool message directly after that assistant turn.
    """
    client = new_client()

    def slow_tool(x: int) -> str:
        return f"ok:{x}"

    slow_tool.__name__ = "slow_tool"
    slow_tool.__qualname__ = "slow_tool"

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

    handle = start_async_tool_use_loop(
        client=client,
        message="Please proceed.",
        tools={"slow_tool": slow_tool},
    )

    assistant_idx: int | None = None
    for _ in range(50):
        await asyncio.sleep(0.05)
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

    assert assistant_idx is not None
    assert (assistant_idx + 1) < len(client.messages)

    next_msg = client.messages[assistant_idx + 1]
    assert next_msg.get("role") == "tool" and next_msg.get("tool_call_id") == call_id

    handle.stop()
    final2 = await handle.result()
    assert final2 == "processed stopped early, no result"
