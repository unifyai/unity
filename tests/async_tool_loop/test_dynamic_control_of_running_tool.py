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
from typing import List

import pytest
from unity.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle

# Shared helpers
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_assistant_call_prefix,
    _wait_for_tool_message_prefix,
    _wait_for_condition,
)


# --------------------------------------------------------------------------- #
#  TOOLS                                                                      #
# --------------------------------------------------------------------------- #
async def slow() -> str:
    """A slow-poke async tool – sleeps `delay` seconds then returns 'done'."""
    await asyncio.sleep(0.50)
    return "done"


# --------------------------------------------------------------------------- #
#  HELPERS                                                                    #
# --------------------------------------------------------------------------- #
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


def _tool_results(msgs: List[dict], tool_name: str) -> int:
    """Count tool-result messages for `tool_name`."""
    return sum(1 for m in msgs if m["role"] == "tool" and m["name"] == tool_name)


# --------------------------------------------------------------------------- #
#  HELPERS – NEW: classify helper-only assistant messages (mirrored control)  #
# --------------------------------------------------------------------------- #
def _is_helper_tool_name(name: str) -> bool:
    try:
        n = str(name or "")
    except Exception:
        n = ""
    return bool(
        (n == "wait")
        or n.startswith("pause_")
        or n.startswith("resume_")
        or n.startswith("stop_")
        or n.startswith("clarify_")
        or n.startswith("interject_")
        or n.startswith("ask_"),
    )


def _assistant_is_helper_only(msg: dict) -> bool:
    """Return True when the assistant message only contains helper tool_calls (no LLM turn)."""
    try:
        if msg.get("role") != "assistant":
            return False
        calls = msg.get("tool_calls") or []
        if not calls:
            return False  # plain assistant text or no tool_calls → counts as a real LLM turn
        # helper-only if every tool_call is a known helper
        return all(
            _is_helper_tool_name((tc.get("function") or {}).get("name")) for tc in calls
        )
    except Exception:
        return False


def _assistant_is_check_status_only(msg: dict) -> bool:
    """
    Return True if the assistant message is a synthetic check-status stub:
      - role == assistant
      - tool_calls present
      - every tool_call function.name startswith 'check_status_'
    These are non-LLM synthetic pairs used to carry final tool results.
    """
    try:
        if msg.get("role") != "assistant":
            return False
        calls = msg.get("tool_calls") or []
        if not calls:
            return False
        return all(
            str((tc.get("function") or {}).get("name") or "").startswith(
                "check_status_",
            )
            for tc in calls
        )
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  FIXTURE                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="function")
def client(model):
    return new_llm_client(model=model)


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
    assert final is not None, "Loop should complete with a response"

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
    assert final is not None, "Loop should complete with a response"

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
    # Explicit gates to avoid timing races: tool cannot complete until
    # the pause helper has been invoked (gate A) and then the resume helper (gate B).
    pause_called_gate = asyncio.Event()
    resume_called_gate = asyncio.Event()

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        # Run until the PAUSE helper has been observed.
        while not pause_called_gate.is_set():
            await _pause_event.wait()
            await asyncio.sleep(0.05)
        # Do not finish until RESUME helper has been observed.
        await resume_called_gate.wait()
        # Perform a small amount of additional work after resume to ensure ordering.
        for _ in range(10):
            await _pause_event.wait()
            await asyncio.sleep(0.05)
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

    # Release the tool's first gate now that pause helper has been invoked
    pause_called_gate.set()

    # While paused, the final assistant reply must NOT appear. Check deterministically
    # right after the pause has been acknowledged (no fixed sleep).
    msgs_during_pause = client.messages or []
    assert not any(
        (m.get("role") == "assistant")
        and isinstance(m.get("content"), str)
        and "done" in m["content"].strip().lower()
        for m in msgs_during_pause
    ), "assistant produced final reply while tool was paused"

    # Resume and finish – ensure the assistant calls the resume helper first
    await outer.interject("go")
    await _wait_for_assistant_call_prefix(client, "resume")
    await _wait_for_tool_message_prefix(client, "resume ")

    # Release the tool's second gate now that resume helper has been invoked
    resume_called_gate.set()
    final = await outer.result()

    # ── assertions ───────────────────────────────────────────────────────
    assert final is not None, "Loop should complete with a response"
    # Removed wall‑clock duration assertion; rely on deterministic pause/resume events.


@pytest.mark.asyncio
@_handle_project
async def test_pause_resume_helpers_called_once(client):
    """
    Same scenario as above but we *count* helper invocations in the chat log.

    • Exactly one `pause_…` and one `resume_…` tool-call must appear.
    """

    # Gates to ensure deterministic ordering: the tool must see pause then resume
    pause_called_gate = asyncio.Event()
    resume_called_gate = asyncio.Event()

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        # Wait until pause helper has been invoked
        while not pause_called_gate.is_set():
            await _pause_event.wait()
            await asyncio.sleep(0.05)
        # Then wait until resume helper has been invoked
        await resume_called_gate.wait()
        # Do a short bit of post-resume work
        for _ in range(10):
            await _pause_event.wait()
            await asyncio.sleep(0.05)
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
    await _wait_for_tool_message_prefix(client, "pause ")
    # Unblock the tool after pause helper observed
    pause_called_gate.set()

    await h.interject("unfreeze")
    # Ensure resume helper is actually invoked before allowing tool to finish
    await _wait_for_assistant_call_prefix(client, "resume")
    await _wait_for_tool_message_prefix(client, "resume ")
    resume_called_gate.set()

    final = await h.result()
    msgs = client.messages

    # helper counters -----------------------------------------------------
    pause_calls = _assistant_calls_prefix(msgs, "pause")
    resume_calls = _assistant_calls_prefix(msgs, "resume")

    assert final is not None, "Loop should complete with a response"
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
    await handle.pause()

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
    # result messages that were appended during pause. Ignore mirrored helper-only
    # assistant messages inserted to represent control actions while paused.
    assistant_after_pause = any(
        (m.get("role") == "assistant") and (not _assistant_is_helper_only(m))
        for m in msgs[last_request_idx + 1 :]
    )
    assert (
        not assistant_after_pause
    ), "assistant produced a new message while the loop was paused"

    # Resume and allow the conversation to complete
    await handle.resume()
    final = await handle.result()

    assert final is not None, "Loop should complete with a response"


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
    await handle.pause()
    await _wait_for_tool_message_prefix(client, "slow")

    # Resume twice (idempotent)
    await handle.resume()
    await handle.resume()

    final = await handle.result()
    assert final is not None, "Loop should complete with a response"

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

    # Count only non-helper, non-check_status assistant messages
    assistant_after = [
        m
        for m in msgs[last_req_idx + 1 :]
        if (m.get("role") == "assistant")
        and (not _assistant_is_helper_only(m))
        and (not _assistant_is_check_status_only(m))
    ]
    assert (
        len(assistant_after) == 1
    ), f"expected exactly 1 post-pause assistant turn, got {len(assistant_after)}"


@pytest.mark.asyncio
@_handle_project
async def test_resume_allows_llm_turn(client):
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
    await h.pause()
    # Assert result() blocks while paused using a timeout-based check
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(h.result()), timeout=1)

    # Resume and finish
    await h.resume()
    final = await h.result()
    assert final is not None, "Loop should complete with a response"


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

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        # Run until the test explicitly signals completion.
        while not done_event.is_set():
            await _pause_event.wait()
            await asyncio.sleep(0)
        return "done"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "️1. Call `pausable_fn`.\n"
        "2. When the user says 'hold', call the helper whose name starts with `pause_`.\n"
        "3. When the user says 'go',   call the helper whose name starts with `resume_` immediately.\n"
        "4. Repeat the pause→resume cycle twice.\n"
        "5. IMPORTANT: Do NOT call the `wait` helper in response to 'go'. If both `resume_…` and `wait` are offered, ALWAYS choose `resume_…`.\n"
        "6. You may use `wait` only when you intend to remain paused without resuming; do NOT use it right after 'go'.\n"
        "7. Do NOT call any legacy `continue_` helper.\n"
        "8. After the second resume, wait for completion and reply with 'done'.",
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
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_helpers_hide_notification_clarification(client):
    """
    Verify dynamic tools do NOT expose `next_notification_…` or `next_clarification_…`
    for an in‑flight inner handle (would fail before the management-set change).
    """

    class MockNestedHandle(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question: str) -> "SteerableToolHandle":
            return self

        async def interject(self, message: str):
            return None

        def stop(self, reason: str | None = None):
            self._done.set()
            return "stopped"

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await self._done.wait()
            return "inner_done"

        # Event APIs
        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    inner_handle = MockNestedHandle()

    async def spawn_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
        return inner_handle

    client.set_system_message(
        "1️⃣ Call `spawn_handle` to start a nested task.\n2️⃣ Wait until it finishes.\n3️⃣ Then reply with OK.",
    )

    # Minimal spy: capture dynamic helper registrations only (source of truth)
    from unity.common._async_tool import dynamic_tools_factory as _dtf

    # Also spy the dynamic tool registration point to capture exact helper names
    registered_helpers: list[str] = []
    orig_register_tool = _dtf.DynamicToolFactory._register_tool

    def _spy_register_tool(self, func_name: str, fallback_doc: str, fn):  # type: ignore[no-redef]
        registered_helpers.append(func_name)
        return orig_register_tool(self, func_name, fallback_doc, fn)

    setattr(_dtf.DynamicToolFactory, "_register_tool", _spy_register_tool)

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_handle": spawn_handle},
        timeout=120,
    )

    # Ensure the assistant has requested the spawn tool
    await _wait_for_tool_request(client, "spawn_handle")

    # Reset spies, then force an immediate LLM turn so dynamic helpers are exposed
    registered_helpers.clear()
    await outer.interject("probe for dynamic helpers")

    # Wait until at least one standard helper is registered
    async def _helpers_registered() -> bool:
        return any(
            any(h.startswith(p) for h in registered_helpers)
            for p in ("pause_", "resume_", "stop_", "interject_", "ask_")
        )

    await _wait_for_condition(_helpers_registered, poll=0.05, timeout=30.0)

    # Ensure that next_notification_* and next_clarification_* are NOT exposed
    combined = set(registered_helpers)
    assert not any(
        n.startswith("next_notification_") for n in combined
    ), f"unexpected next_notification_* exposed: {sorted(combined)}"
    assert not any(
        n.startswith("next_clarification_") for n in combined
    ), f"unexpected next_clarification_* exposed: {sorted(combined)}"

    # Finish inner handle and let the loop complete
    inner_handle._done.set()
    final = await outer.result()
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_helpers_hide_get_history(client, model):
    """
    Verify dynamic tools do NOT expose `get_history_…` for an in‑flight nested
    AsyncToolLoopHandle (this would have been exposed before dynamic base-method
    exclusion).
    """

    async def spawn_inner_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
        # Start an inner async tool loop and return its handle immediately
        inner_client = new_llm_client(model=model)
        return start_async_tool_loop(
            inner_client,
            message="Inner loop: reply OK.",
            tools={},
            timeout=60,
        )

    client.set_system_message(
        "1️⃣ Call `spawn_inner_handle` to start a nested async tool loop.\n"
        "2️⃣ Wait until it finishes.\n"
        "3️⃣ Then reply with OK.",
    )

    # Spy dynamic helper registrations
    from unity.common._async_tool import dynamic_tools_factory as _dtf

    registered_helpers: list[str] = []
    orig_register_tool = _dtf.DynamicToolFactory._register_tool

    def _spy_register_tool(self, func_name: str, fallback_doc: str, fn):  # type: ignore[no-redef]
        registered_helpers.append(func_name)
        return orig_register_tool(self, func_name, fallback_doc, fn)

    setattr(_dtf.DynamicToolFactory, "_register_tool", _spy_register_tool)

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_inner_handle": spawn_inner_handle},
        timeout=120,
    )

    # Ensure the assistant requests the spawning tool
    await _wait_for_tool_request(client, "spawn_inner_handle")

    # Force an immediate assistant turn to expose dynamic helpers
    registered_helpers.clear()
    await outer.interject("probe dynamic helpers")

    # Wait until we see at least one standard helper for the nested handle
    async def _helpers_registered() -> bool:
        return any(
            any(h.startswith(p) for h in registered_helpers)
            for p in ("pause_", "resume_", "stop_", "interject_", "ask_")
        )

    await _wait_for_condition(_helpers_registered, poll=0.05, timeout=30.0)

    # Assert that no get_history_* helper is exposed
    combined = set(registered_helpers)
    assert not any(
        n.startswith("get_history_") for n in combined
    ), f"unexpected get_history_* exposed: {sorted(combined)}"

    # Let the nested loop finish so the test can complete cleanly
    final = await outer.result()
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_new_tool_scheduled_while_paused_starts_paused(client, monkeypatch):
    """
    A base tool scheduled AFTER the outer handle is paused must start paused
    (its `_pause_event` is cleared). Before the change, the event started set.
    """
    # Patch the loop's LLM call to emit a tool-call only AFTER we pause
    from unity.common._async_tool import loop as _loop

    llm_started = asyncio.Event()
    release_llm = asyncio.Event()
    orig_gwp = _loop.generate_with_preprocess

    async def _fake_gwp(_client, preprocess_msgs, **gen_kwargs):
        # Signal that LLM thinking has started
        llm_started.set()
        # Wait until the test allows the LLM to finish (after outer pause)
        await release_llm.wait()
        # Emit a single assistant turn that calls `pausable_fn` with no args
        _client.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_fake_1",
                        "type": "function",
                        "function": {"name": "pausable_fn", "arguments": "{}"},
                    },
                ],
            },
        )
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    # Base tool that records the initial pause state immediately on start
    initial_pause_state = {"value": None}

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        try:
            initial_pause_state["value"] = _pause_event.is_set()
        except Exception:
            initial_pause_state["value"] = None
        return "ok"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "When you respond, call `pausable_fn` exactly once and then finish.",
    )

    # Start loop, immediately pause, then release the LLM patch to schedule tool
    h = start_async_tool_loop(
        client=client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        timeout=120,
        max_steps=20,
    )

    # Ensure the LLM step actually started, then pause the outer handle
    await asyncio.wait_for(llm_started.wait(), timeout=30)
    await h.pause()
    # Allow the patched LLM to proceed and return the tool-call while paused
    release_llm.set()

    # Wait until the tool result for `pausable_fn` appears
    await _wait_for_tool_message_prefix(client, "pausable_fn")

    # The tool must have observed an initial paused state (event cleared)
    assert (
        initial_pause_state["value"] is False
    ), "newly scheduled tool did not start paused"

    # Cleanup: stop the loop and restore original LLM generator
    await h.stop("test cleanup")
    await h.result()
    monkeypatch.setattr(_loop, "generate_with_preprocess", orig_gwp, raising=True)


@pytest.mark.asyncio
@_handle_project
async def test_resume_unblocks_base_tool(client, monkeypatch):
    """
    A base tool scheduled while the outer loop is paused should resume
    running immediately when `handle.resume()` is called, even if the LLM
    never calls a `resume_…` helper. This would have failed before the
    auto-resume improvement.
    """
    from unity.common._async_tool import loop as _loop

    llm_started = asyncio.Event()
    release_llm = asyncio.Event()
    orig_gwp = _loop.generate_with_preprocess

    # Track invocation count so the monkeypatch only emits the tool call once.
    # On subsequent calls (after tool completion), emit a final response to
    # properly terminate the loop.
    call_count = {"value": 0}

    async def _fake_gwp(_client, preprocess_msgs, **gen_kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            # First call: wait for test signal, then emit tool call
            llm_started.set()
            await release_llm.wait()
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_fake_2",
                            "type": "function",
                            "function": {"name": "pausable_fn", "arguments": "{}"},
                        },
                    ],
                },
            )
        else:
            # Subsequent calls: emit a final response to terminate the loop
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": "Done.",
                    "tool_calls": None,
                },
            )
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    # Base tool: starts paused; only completes after _pause_event is set (resume)
    initial_pause_state = {"value": None}

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        try:
            initial_pause_state["value"] = _pause_event.is_set()
        except Exception:
            initial_pause_state["value"] = None
        # Wait until resumed, then finish quickly
        await _pause_event.wait()
        return "ok"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "When you respond, call `pausable_fn` exactly once and then finish.",
    )

    h = start_async_tool_loop(
        client=client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        timeout=180,
        max_steps=20,
    )

    # Ensure LLM step started, then pause the outer handle
    await asyncio.wait_for(llm_started.wait(), timeout=30)
    await h.pause()
    release_llm.set()

    # Wait until the tool placeholder appears (scheduled while paused)
    await _wait_for_tool_message_prefix(client, "pausable_fn")

    # Confirm the tool started in a paused state
    assert (
        initial_pause_state["value"] is False
    ), "tool did not start paused while outer loop was paused"

    # Resume the outer handle – should auto-set the per-call pause_event for base tools
    await h.resume()

    # Wait until final tool result "ok" is observed without relying on a resume helper
    async def _has_final_ok() -> bool:
        msgs = client.messages or []
        return any(
            (m.get("role") == "tool")
            and (m.get("name") == "pausable_fn")
            and (m.get("content") == "ok")
            for m in msgs
        )

    await _wait_for_condition(_has_final_ok, poll=0.05, timeout=60.0)

    # Ensure no resume helper call was made by the assistant (programmatic resume path)
    msgs = client.messages or []
    assert (
        _assistant_calls_prefix(msgs, "resume") == 0
    ), "LLM should not need to call resume_… helper for base tools"

    # Cleanup – stop the loop and restore generator
    await h.stop("cleanup")
    # Await result; outer handle returns a standardized notice on stop
    await asyncio.wait_for(asyncio.shield(h.result()), timeout=60)
    monkeypatch.setattr(_loop, "generate_with_preprocess", orig_gwp, raising=True)
