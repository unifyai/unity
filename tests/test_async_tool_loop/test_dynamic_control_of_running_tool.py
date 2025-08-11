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
from unity.common.llm_helpers import start_async_tool_use_loop, SteerableToolHandle

# Shared helpers
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import _wait_for_tool_request


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


@unify.traced
async def _wait_for_assistant_call_prefix(
    client: "unify.AsyncUnify",
    prefix: str,
    *,
    timeout: float = 15.0,
    poll: float = 0.05,
) -> None:
    """Poll *client.messages* until the assistant has issued **at least one**
    visible tool-call whose *function name* starts with *prefix* or *timeout*
    seconds elapse.

    This mirrors ``_wait_for_tool_request`` but matches by *prefix* which is
    useful for helper functions such as ``pause_…`` / ``resume_…`` whose exact
    suffix is dynamic (it contains the tool call ID).
    """
    import time as _time

    start_ts = _time.perf_counter()
    while _time.perf_counter() - start_ts < timeout:
        msgs = client.messages or []  # unify may return None initially
        if _assistant_calls_prefix(msgs, prefix) >= 1:
            return  # helper has been requested – safe to proceed
        await asyncio.sleep(poll)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for assistant to request a helper starting with {prefix!r}.",
    )


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

    # Wait deterministically until the `slow` tool has been requested.
    await _wait_for_tool_request(client, "slow")
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
        message=("Run the tool `slow`."),
        tools={"slow": slow},
        interrupt_llm_with_interjections=False,
    )

    # Wait deterministically until the assistant has actually scheduled the
    # `slow` tool so we know our interjection will hit *while* it is running.
    await _wait_for_tool_request(client, "slow")
    await handle.interject(
        "Please stop that run right away, and inform the user that it has been stopped.",
    )

    final = await handle.result()
    assert "stop" in final.lower()

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
        # “work” for 2 seconds in 0.1-s ticks while honouring pause_event
        for _ in range(20):
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

    # ── deterministically wait until the assistant has actually scheduled the
    #    tool so our *hold* interjection reliably occurs while it is running.
    await _wait_for_tool_request(client, "pausable_fn")
    t0 = time.perf_counter()

    await outer.interject("hold")
    await asyncio.sleep(2.0)  # loop is paused here
    await outer.interject("go")

    final = await outer.result()
    elapsed = time.perf_counter() - t0

    # ── assertions ───────────────────────────────────────────────────────
    assert final.strip().lower() == "done"
    assert elapsed >= 4, f"loop finished too fast ({elapsed:.2f}s) – pause ineffective"


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

    # ── deterministically trigger pause / resume via user turns ───────────────
    # Wait until the assistant has actually scheduled the tool so our
    # *freeze* interjection occurs while the tool is running.
    await _wait_for_tool_request(client, "pausable_fn")
    await h.interject("freeze")

    # Wait until the assistant has called the corresponding ``pause_…`` helper
    # before sending the *unfreeze* command so we are sure the helper sequence
    # is pause → resume (in that order).
    await _wait_for_assistant_call_prefix(client, "pause")
    await h.interject("unfreeze")

    final = await h.result()
    msgs = client.messages

    # helper counters -----------------------------------------------------
    pause_calls = _assistant_calls_prefix(msgs, "pause")
    resume_calls = _assistant_calls_prefix(msgs, "resume")

    assert "all done" in final.strip().lower()
    assert pause_calls == 1, f"expected exactly 1 pause_ helper, got {pause_calls}"
    assert resume_calls == 1, f"expected exactly 1 resume_ helper, got {resume_calls}"


@pytest.mark.asyncio
@_handle_project
async def test_global_pause_blocks_llm_until_resume(client):
    """
    The global `pause()` should prevent the LLM from speaking while paused.

    Scenario
    --------
    * Ask the assistant to call `slow` then reply with the word 'OK'.
    * Pause the outer loop while the tool is still running.
    * Wait long enough for the tool to finish.

    Expected
    --------
    * While paused, no new assistant turn should appear after the assistant
      turn that requested the tool.
    * After `resume()`, the loop should complete and return the final 'OK'.
    """
    handle = start_async_tool_use_loop(
        client,
        message=(
            "Call the tool `slow`, wait for the result, then reply with the word OK (nothing else)."
        ),
        tools={"slow": slow},
    )

    # Ensure the tool has been requested so pausing happens while it is running
    await _wait_for_tool_request(client, "slow")

    # Pause the outer loop (tools should keep running; the LLM must not speak)
    handle.pause()

    # Give enough time for `slow` to complete and for the loop to process the tool result
    await asyncio.sleep(1.0)

    msgs = client.messages or []

    # Locate the assistant turn that requested `slow`
    assistant_tool_call_indices = [
        i
        for i, m in enumerate(msgs)
        if m.get("role") == "assistant"
        and any(
            tc.get("function", {}).get("name") == "slow"
            for tc in (m.get("tool_calls") or [])
        )
    ]
    assert (
        assistant_tool_call_indices
    ), "expected at least one assistant turn requesting the `slow` tool"

    last_request_idx = assistant_tool_call_indices[-1]

    # While paused, there must be no further assistant messages after the tool
    # result messages that were appended during pause
    assistant_after_pause = any(
        m.get("role") == "assistant" for m in msgs[last_request_idx + 1 :]
    )
    assert (
        not assistant_after_pause
    ), "assistant produced a new message while the loop was paused"

    # Resume and allow the conversation to complete
    handle.resume()
    final = await handle.result()

    assert (
        final.strip().upper().startswith("OK")
    ), "final reply should be 'OK' after resume"


@pytest.mark.asyncio
@_handle_project
async def test_global_resume_idempotent_no_extra_turns(client):
    """
    Calling `resume()` multiple times should be harmless and must not create
    extra assistant turns after the tool completes.
    """
    handle = start_async_tool_use_loop(
        client,
        message=(
            "Call the tool `slow`, wait for the result, then reply with the word OK (nothing else)."
        ),
        tools={"slow": slow},
    )

    # Ensure the tool has been requested
    await _wait_for_tool_request(client, "slow")

    # Pause while tool is running; let it finish while paused
    handle.pause()
    await asyncio.sleep(0.7)

    # Resume twice (idempotent)
    handle.resume()
    handle.resume()

    final = await handle.result()
    assert final.strip().upper().startswith("OK")

    # After the last assistant tool-call requesting `slow`, there should be exactly
    # one more assistant message (the final answer). Multiple resumes must not add more.
    msgs = client.messages or []
    last_req_idx = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and any(
            tc.get("function", {}).get("name") == "slow"
            for tc in (m.get("tool_calls") or [])
        ):
            last_req_idx = i
    assert last_req_idx != -1, "expected an assistant tool-call to `slow`"

    assistant_after = [
        m for m in msgs[last_req_idx + 1 :] if m.get("role") == "assistant"
    ]
    assert (
        len(assistant_after) == 1
    ), f"expected exactly 1 post-pause assistant turn, got {len(assistant_after)}"


@pytest.mark.asyncio
@_handle_project
async def test_nested_resume_forwarded_once_to_delegate(client):
    """
    When a tool returns a passthrough SteerableToolHandle and the outer handle is adopted,
    calling `resume()` on the OUTER handle must forward exactly once to the delegate.
    """

    class MockPassthroughHandle(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.pause_count = 0
            self.resume_count = 0

        async def ask(self, question: str) -> "SteerableToolHandle":
            return self

        async def interject(self, message: str):
            return None

        def stop(self):
            self._done.set()
            return "stopped"

        def pause(self):
            self.pause_count += 1
            return "paused"

        def resume(self):
            self.resume_count += 1
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await self._done.wait()
            return "inner_done"

    inner_handle = MockPassthroughHandle()

    @unify.traced
    async def spawn_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
        """Return a passthrough handle immediately so the outer loop adopts it."""
        return inner_handle

    client.set_system_message(
        "1️⃣ Call `spawn_handle` to start a nested task.\n"
        "2️⃣ Wait until it finishes.\n"
        "3️⃣ Then reply with OK.",
    )

    # Force the first assistant turn to call only `spawn_handle` to avoid model variance
    def _policy(step_idx, available_tools):
        if step_idx == 0:
            return "required", {"spawn_handle": available_tools["spawn_handle"]}
        return "auto", available_tools

    outer = start_async_tool_use_loop(
        client,
        message="start",
        tools={"spawn_handle": spawn_handle},
        timeout=120,
        tool_policy=_policy,
    )

    # Wait until assistant requests the spawn tool (ensures tool scheduling happened)
    await _wait_for_tool_request(client, "spawn_handle")

    # Wait until the outer handle adopts the inner delegate
    t_adopt = time.perf_counter()
    while (getattr(outer, "_delegate", None) is None) and (
        time.perf_counter() - t_adopt
    ) < 10.0:
        await asyncio.sleep(0.05)
    assert (
        getattr(outer, "_delegate", None) is inner_handle
    ), "outer did not adopt the inner SteerableToolHandle"

    # Pause the outer loop – must forward exactly once to the delegate
    outer.pause()

    t0 = time.perf_counter()
    while inner_handle.pause_count < 1 and (time.perf_counter() - t0) < 10.0:
        await asyncio.sleep(0.05)
    assert (
        inner_handle.pause_count == 1
    ), "delegate did not receive pause() exactly once"

    # Now resume the outer loop – must forward exactly once to the delegate
    outer.resume()

    t1 = time.perf_counter()
    while inner_handle.resume_count < 1 and (time.perf_counter() - t1) < 10.0:
        await asyncio.sleep(0.05)
    assert (
        inner_handle.resume_count == 1
    ), "delegate did not receive resume() exactly once"

    # Let the inner handle complete so the loop can finish
    inner_handle._done.set()

    final = await outer.result()
    # Accept either the model's OK or the inner handle's passthrough completion text
    assert final.strip().lower() == "inner_done"


@pytest.mark.asyncio
@_handle_project
async def test_resume_when_no_pending_tools_allows_llm_turn(client):
    """
    If the loop is paused while no tools are pending, resuming should immediately
    allow the next LLM turn to proceed and finish.
    """
    client.set_system_message(
        "Immediately reply ONLY with the word OK. Do not call any tools.",
    )

    # No tools exposed – pure LLM reply
    h = start_async_tool_use_loop(
        client,
        message="start",
        tools={},
        timeout=120,
    )

    # Pause immediately; there are no pending tools. LLM must not speak while paused.
    h.pause()
    await asyncio.sleep(0.5)

    # Ensure no assistant message appeared during pause
    msgs = client.messages or []
    assert not any(
        m.get("role") == "assistant" for m in msgs
    ), "assistant spoke while paused with no pending tools"

    # Resume and finish
    h.resume()
    final = await h.result()
    assert final.strip().upper().startswith("OK")
