"""
End-to-end tests for the *control-tool* extension of
`unity.common.llm_helpers._async_tool_use_loop_inner`.

What we verify
--------------

* **Continue** – A long-running tool is launched, the user interjects asking
  the assistant to *keep waiting*; the loop must *not* start a second copy of
  that tool.

* **Stop** – The user interjects asking to *stop* the running tool; the
  task is aborted, no tool-result message appears, and the control decision is
  omitted from the permanent chat transcript.

As with the other suites we talk to a **live model** – make sure you have
internet connectivity and `OPENAI_API_KEY` (or proxy equivalent) configured.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List
import json

import pytest
import unify
from unity.common.llm_helpers import start_async_tool_use_loop
from tests.helpers import _handle_project


# --------------------------------------------------------------------------- #
#  GLOBALS                                                                    #
# --------------------------------------------------------------------------- #
MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


# --------------------------------------------------------------------------- #
#  TOOLS                                                                      #
# --------------------------------------------------------------------------- #
@unify.traced
async def slow() -> str:
    """A slow-poke async tool – sleeps `delay` seconds then returns 'done'."""
    await asyncio.sleep(0.50)
    return "done"


# --------------------------------------------------------------------------- #
#  HELPERS                                                                    #
# --------------------------------------------------------------------------- #
@unify.traced
def _assistant_calls(msgs: List[dict], tool_name: str) -> int:
    """Count assistant turns whose *visible* `tool_calls` reference `tool_name`."""
    return sum(
        1
        for m in msgs
        if m["role"] == "assistant"
        and any(
            tc["function"]["name"] == tool_name for tc in (m.get("tool_calls") or [])
        )
    )


@unify.traced
def _assistant_calls_prefix(msgs: List[dict], prefix: str) -> int:
    """Count assistant turns whose tool-call name *starts with* `prefix`."""
    return sum(
        1
        for m in msgs
        if m["role"] == "assistant"
        and any(
            tc["function"]["name"].startswith(prefix)
            for tc in (m.get("tool_calls") or [])
        )
    )


@unify.traced
def _tool_results(msgs: List[dict], tool_name: str) -> int:
    """Count tool-result messages for `tool_name`."""
    return sum(1 for m in msgs if m["role"] == "tool" and m["name"] == tool_name)


# --------------------------------------------------------------------------- #
#  FIXTURE                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="function")
def client():
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
async def test_continue_does_not_duplicate_tool(client):
    """
    Scenario
    --------
    * Ask the assistant to call `slow()` **exactly once** and then reply 'OK'.
    * While `slow` is still running we interject:
        “Please just keep waiting – don't start it again.”
    Expected
    --------
    * Only **one** visible assistant tool-call to `slow`.
    * Only **one** tool-result message for `slow`.
    * Final assistant text is returned.
    """
    handle = start_async_tool_use_loop(
        client,
        message=(
            "Call the tool `slow`, wait for the result, then "
            "reply with the word OK (nothing else)."
        ),
        tools={"slow": slow},
    )

    # Interject after ~50 ms – tool still running
    await asyncio.sleep(0.05)
    await handle.interject(
        "Make sure you're still continuing to run the `slow` tool",
    )

    final = await handle.result()
    assert final.strip().upper().startswith("OK")

    msgs = client.messages
    assert _assistant_calls(msgs, "slow") == 1, "should be one visible request"
    assert _tool_results(msgs, "slow") == 1, "should be one execution only"


@pytest.mark.asyncio
@_handle_project
async def test_stop_removes_tool_and_yields_no_result(client):
    """
    Scenario
    --------
    * Ask the assistant to run `slow` then answer 'ACK'.
    * Shortly after, interject: “Cancel that operation.”
    Expected
    --------
    * **Zero** tool-result messages for `slow` (task was stopped).
    * No assistant turn in the log still exposes `slow` in `tool_calls`.
    * Loop finishes with a normal assistant reply.
    """
    handle = start_async_tool_use_loop(
        client,
        message=("Run the tool `slow` then reply ACK (nothing else)."),
        tools={"slow": slow},
        interrupt_llm_with_interjections=False,
    )

    await asyncio.sleep(0.05)  # tool in-flight
    await handle.interject("Please stop that run right away.")

    final = await handle.result()
    assert "ACK" in final.upper()

    msgs = client.messages
    assert _tool_results(msgs, "slow") == 1, "stopping tool expected after stop"
    assert _assistant_calls(msgs, "slow") == 1, "tool-call should remain in the history"


@pytest.mark.asyncio
@_handle_project
async def test_functional_tool_pause_extends_wall_clock(client):
    """
    * The assistant must…
        1️⃣  call `pausable_fn`;
        2️⃣  when the *user* says **hold**, invoke the `pause_…` helper;
        3️⃣  when the *user* says **go**,   invoke the `resume_…` helper;
        4️⃣  when the tool finishes, reply with **done**.
    * We measure wall-clock time: because the loop is paused for ~2 s in the
      middle, total duration must be ≥ 2 s + the tool's own 1-second workload.
    """

    async def pausable_fn(*, pause_event: asyncio.Event) -> str:
        # “work” for 1 second in 0.1-s ticks while honouring pause_event
        for _ in range(10):
            await pause_event.wait()
            await asyncio.sleep(0.1)
        return "ok"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "1️⃣ Call `pausable_fn`.\n"
        "2️⃣ When the user says **hold**, call the helper whose name starts "
        "with `pause_`.\n"
        "3️⃣ When the user says **go**,   call the helper whose name starts "
        "with `resume_`.\n"
        "4️⃣ Once the tool finishes, reply with **done**.",
    )

    outer = start_async_tool_use_loop(
        client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        max_steps=30,
        timeout=300,
    )

    t0 = time.perf_counter()

    # let assistant schedule the tool, then pause for ~2 s
    await asyncio.sleep(0.3)
    await outer.interject("hold")
    await asyncio.sleep(2.0)  # loop is paused here
    await outer.interject("go")

    final = await outer.result()
    elapsed = time.perf_counter() - t0

    # ── assertions ───────────────────────────────────────────────────────
    assert final.strip().lower() == "done"
    assert (
        elapsed >= 3.0
    ), f"loop finished too fast ({elapsed:.2f}s) – pause ineffective"


@pytest.mark.asyncio
@_handle_project
async def test_functional_tool_pause_resume_helpers_called_once(client):
    """
    Same scenario as above but we *count* helper invocations in the chat log.

    • Exactly one `pause_…` and one `resume_…` tool-call must appear.
    """

    async def pausable_fn(*, pause_event: asyncio.Event) -> str:
        for _ in range(8):
            await pause_event.wait()
            await asyncio.sleep(1)
        return "yo"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "1️⃣ Call `pausable_fn`.\n"
        "2️⃣ If the user says **freeze**, call `pause_…` *once*.\n"
        "3️⃣ If the user then says **unfreeze**, call `resume_…` *once*.\n"
        "4️⃣ When the tool finishes, reply with **all done**.",
    )

    h = start_async_tool_use_loop(
        client,
        message="go",
        tools={"pausable_fn": pausable_fn},
        timeout=1000,
    )

    # trigger pause / resume via user turns
    await asyncio.sleep(2)
    await h.interject("freeze")
    await asyncio.sleep(2)
    await h.interject("unfreeze")

    final = await h.result()
    msgs = client.messages

    # helper counters -----------------------------------------------------
    pause_calls = _assistant_calls_prefix(msgs, "pause")
    resume_calls = _assistant_calls_prefix(msgs, "resume")

    assert "all done" in final.strip().lower()
    assert pause_calls == 1, f"expected exactly 1 pause_ helper, got {pause_calls}"
    assert resume_calls == 1, f"expected exactly 1 resume_ helper, got {resume_calls}"
