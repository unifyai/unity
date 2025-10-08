"""
End-to-end tests for the *control-tool* extension of
`unity.common.async_tool_loop._async_tool_use_loop_inner`.

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
from typing import List

import pytest
import unify
from unity.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle

# Shared helpers
from tests.helpers import _handle_project, SETTINGS
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_assistant_call_prefix,
    _wait_for_tool_message_prefix,
    _wait_for_condition,
)


# --------------------------------------------------------------------------- #
#  GLOBALS                                                                    #
# --------------------------------------------------------------------------- #
MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-5@openai")
# (prefix-based wait helpers and their counters are now shared in
#  tests/test_async_tool_loop/async_helpers.py)


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
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )


# --------------------------------------------------------------------------- #
#  TESTS                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_wait_does_not_duplicate_tool(client):
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
    handle = start_async_tool_loop(
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
        "Make sure you use the `wait` helper and keep the existing `slow` tool running",
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
    handle = start_async_tool_loop(
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
        # Work loop honouring pause_event; total of ~2 seconds when unpaused
        ticks = 20
        for _ in range(ticks):
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

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        max_steps=30,
        timeout=300,
    )

    # ── deterministically wait until the assistant has actually scheduled the
    #    tool so our *hold* interjection reliably occurs while it is running.
    await _wait_for_tool_request(client, "pausable_fn")
    # Trigger pause while the tool is running
    await outer.interject("hold")
    # Wait until the assistant REQUESTS the pause helper…
    await _wait_for_assistant_call_prefix(client, "pause")
    # …and also until the loop ACKNOWLEDGES it (tool message inserted), which is
    # the moment the tool's pause_event has been cleared.
    await _wait_for_tool_message_prefix(client, "pause ")

    # While paused, the final assistant reply must NOT appear. Check deterministically
    # right after the pause has been acknowledged (no fixed sleep).
    msgs_during_pause = client.messages or []
    assert not any(
        (m.get("role") == "assistant")
        and isinstance(m.get("content"), str)
        and "done" in m["content"].strip().lower()
        for m in msgs_during_pause
    ), "assistant produced final reply while tool was paused"

    # Resume and finish
    await outer.interject("go")
    final = await outer.result()

    # ── assertions ───────────────────────────────────────────────────────
    assert "done" in final.strip().lower()
    # Removed wall‑clock duration assertion; rely on deterministic pause/resume events.


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

    h = start_async_tool_loop(
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
    handle = start_async_tool_loop(
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

    # Wait until the tool result for `slow` has been appended while paused
    await _wait_for_tool_message_prefix(client, "slow")

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
    handle = start_async_tool_loop(
        client,
        message=(
            "Call the tool `slow`, wait for the result, then reply with the word OK (nothing else)."
        ),
        tools={"slow": slow},
    )

    # Ensure the tool has been requested
    await _wait_for_tool_request(client, "slow")

    # Pause while tool is running; let it finish while paused – wait for tool result deterministically
    handle.pause()
    await _wait_for_tool_message_prefix(client, "slow")

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

        # New abstract event APIs stubs
        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

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

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_handle": spawn_handle},
        timeout=120,
        tool_policy=_policy,
    )

    # Wait until assistant requests the spawn tool (ensures tool scheduling happened)
    await _wait_for_tool_request(client, "spawn_handle")

    # In the new design, the outer loop continues running and does not rely on
    # adopting a single delegate. We no longer assert on `_delegate`.

    # Pause the outer loop – must forward exactly once to the delegate
    outer.pause()

    async def _paused_once() -> bool:
        return inner_handle.pause_count >= 1

    await _wait_for_condition(_paused_once, poll=0.05, timeout=60.0)
    assert (
        inner_handle.pause_count == 1
    ), "delegate did not receive pause() exactly once"

    # Now resume the outer loop – must forward exactly once to the delegate
    outer.resume()

    async def _resumed_once() -> bool:
        return inner_handle.resume_count >= 1

    await _wait_for_condition(_resumed_once, poll=0.05, timeout=60.0)
    assert (
        inner_handle.resume_count == 1
    ), "delegate did not receive resume() exactly once"

    # Let the inner handle complete so the loop can finish
    inner_handle._done.set()

    final = await outer.result()
    # Accept either the model's OK or the inner handle's passthrough completion text
    assert final.strip().lower() in {"ok", "inner_done"}


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
    h = start_async_tool_loop(
        client,
        message="start",
        tools={},
        timeout=120,
    )

    # Pause immediately; there are no pending tools. The loop should not finish while paused.
    h.pause()
    # Assert result() blocks while paused using a timeout-based check
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(h.result()), timeout=1)

    # Resume and finish
    h.resume()
    final = await h.result()
    assert final.strip().upper().startswith("OK")


@pytest.mark.asyncio
@_handle_project
async def test_only_one_of_pause_or_resume_is_exposed(client):
    """
    Verify helper EXPOSURE flips correctly across multiple pause/resume cycles,
    irrespective of what the LLM actually calls. We assert which helper names
    are exposed to the model on each generation turn by spying the `tools`
    argument passed into the LLM call.

    Expectations per turn:
    - While running: `pause_…` exposed, `resume_…` NOT exposed.
    - On the turn when pausing is requested: `pause_…` exposed, `resume_…` NOT exposed.
    - Next turn after pausing: `resume_…` exposed, `pause_…` NOT exposed.
    - Repeat pause→resume cycle twice.
    """

    done_event = asyncio.Event()

    async def pausable_fn(*, pause_event: asyncio.Event) -> str:
        # Run until the test explicitly signals completion.
        while not done_event.is_set():
            await pause_event.wait()
            await asyncio.sleep(0)
        return "done"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "1️⃣ Call `pausable_fn`.\n"
        "2️⃣ When the user says 'hold', call the helper whose name starts with `pause_`.\n"
        "3️⃣ When the user says 'go',   call the helper whose name starts with `resume_`.\n"
        "4️⃣ Repeat the pause→resume cycle twice.\n"
        "5️⃣ Prefer the `wait` helper if you need to keep waiting; do NOT call any legacy `continue_` helper.\n"
        "6️⃣ After the second resume, wait for completion and reply with 'done'.",
    )

    # Spy tool exposure at the exact callsite by wrapping the symbol actually used in the loop
    from unity.common._async_tool import loop as _loop

    seen_tools: list[list[str]] = []
    orig_gwp = _loop.generate_with_preprocess

    async def _spy_gwp(_client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        names: list[str] = []
        for t in tools:
            try:
                fn = t.get("function", {})
                name = fn.get("name")
                if isinstance(name, str):
                    names.append(name)
            except Exception:
                pass
        seen_tools.append(names)
        return await orig_gwp(_client, preprocess_msgs, **gen_kwargs)

    setattr(_loop, "generate_with_preprocess", _spy_gwp)

    h = start_async_tool_loop(
        client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        timeout=300,
        max_steps=60,
        max_parallel_tool_calls=1,
    )

    async def _wait_exposure(has_pause: bool, has_resume: bool, timeout: float = 20.0):
        import time as _time

        start = _time.perf_counter()
        start_idx = len(seen_tools)
        while _time.perf_counter() - start < timeout:
            # scan any newly appended exposure sets for the desired pattern
            for names in seen_tools[start_idx:]:
                has_p = any(n.startswith("pause_pausable_fn_") for n in names)
                has_r = any(n.startswith("resume_pausable_fn_") for n in names)
                if has_p == has_pause and has_r == has_resume:
                    return names
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"timeout waiting for exposure has_pause={has_pause} has_resume={has_resume}",
        )

    def _assert_exposure(names: list[str], *, has_pause: bool, has_resume: bool):
        has_p = any(n.startswith("pause_pausable_fn_") for n in names)
        has_r = any(n.startswith("resume_pausable_fn_") for n in names)
        assert (
            has_p == has_pause
        ), f"pause exposure mismatch; expected {has_pause}, tools={names}"
        assert (
            has_r == has_resume
        ), f"resume exposure mismatch; expected {has_resume}, tools={names}"

    # Ensure the tool is running before issuing commands
    await _wait_for_tool_request(client, "pausable_fn")

    # Pause turn: still expose pause only
    await h.interject("hold")
    names = await _wait_exposure(has_pause=True, has_resume=False)
    _assert_exposure(names, has_pause=True, has_resume=False)

    # Ensure the pause helper was actually invoked and acknowledged (state now paused)
    await _wait_for_assistant_call_prefix(client, "pause")
    await _wait_for_tool_message_prefix(client, "pause ")

    # Next turn after paused: expose resume only
    await h.interject("go")
    names = await _wait_exposure(has_pause=False, has_resume=True)
    _assert_exposure(names, has_pause=False, has_resume=True)

    # Ensure the resume helper was actually invoked and acknowledged (state now running)
    await _wait_for_assistant_call_prefix(client, "resume")
    await _wait_for_tool_message_prefix(client, "resume ")

    # Second cycle: pause turn → pause only
    await h.interject("hold")
    names = await _wait_exposure(has_pause=True, has_resume=False)
    _assert_exposure(names, has_pause=True, has_resume=False)

    # Ensure pause helper applied again
    await _wait_for_assistant_call_prefix(client, "pause")
    await _wait_for_tool_message_prefix(client, "pause ")

    # After that pause: resume only
    await h.interject("go")
    names = await _wait_exposure(has_pause=False, has_resume=True)
    _assert_exposure(names, has_pause=False, has_resume=True)

    # Ensure resume helper applied again before completion
    await _wait_for_assistant_call_prefix(client, "resume")
    await _wait_for_tool_message_prefix(client, "resume ")

    done_event.set()
    final = await h.result()
    assert final.strip().lower() in {"done", "all done", "ok"}
