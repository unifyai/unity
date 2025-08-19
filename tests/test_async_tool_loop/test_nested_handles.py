import pytest
import time
import json
import asyncio
import unify

from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    AsyncToolUseLoopHandle,
)
from tests.helpers import _get_unity_test_env_var


# ─────────────────────────────────────────────────────────────────────────────
#  Tools for the *inner* loop
# ─────────────────────────────────────────────────────────────────────────────


def inner_tool() -> str:  # noqa: D401 – simple value
    """Returns the literal string 'inner‑result'."""
    time.sleep(8)
    return "inner-result"


# ─────────────────────────────────────────────────────────────────────────────
#  Tool for the *outer* loop – spawns the nested loop and returns its handle
# ─────────────────────────────────────────────────────────────────────────────


async def outer_tool() -> AsyncToolUseLoopHandle:
    """Launch an **inner** async‑tool‑use loop and return its *handle*."""

    # brand‑new LLM client dedicated to the nested conversation
    inner_client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
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
#  Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nested_async_tool_loop():
    """Full end-to-end check – no mocks, real network call to OpenAI."""

    # Outer client that drives the *first* loop
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
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

    assert len(client.messages) == 5, "Expected a 5-message sequence"

    # 0. System message
    assert client.messages[0]["role"] == "system"
    assert client.messages[0]["content"] == (
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1\ufe0f\u20e3  Call `outer_tool` with no arguments.\n"
        "2\ufe0f\u20e3  Continue running this tool call, when given the option.\n"
        "3\ufe0f\u20e3  Once it is *completed*, respond with exactly 'all done'."
    )

    # 1. User message
    assert client.messages[1] == {"role": "user", "content": "start"}

    # 2. Assistant: initial tool selection
    initial_call = client.messages[2]
    assert initial_call["role"] == "assistant"
    assert (
        initial_call.get("tool_calls") is not None
    ), "Assistant should make a tool call"
    assert len(initial_call["tool_calls"]) == 1
    assert initial_call["tool_calls"][0]["function"] == {
        "arguments": "{}",
        "name": "outer_tool",
    }

    # 3. Tool: response for outer_tool.
    # Its content should reflect the final result of the nested loop ("done" as a JSON string).
    first_tool_resp = client.messages[3]
    assert first_tool_resp["role"] == "tool"
    assert first_tool_resp["name"] == "outer_tool"
    assert (
        first_tool_resp["content"] == '"done"'
    ), "The placeholder for outer_tool should be updated with the inner loop's final result."

    # 4. Assistant: final response "all done"
    final_assistant_msg = client.messages[4]
    assert final_assistant_msg["role"] == "assistant"
    assert final_assistant_msg["content"].strip().lower() == "all done"
    assert (
        final_assistant_msg.get("tool_calls") is None
    ), "Final assistant message should not have tool calls"


@pytest.mark.asyncio
async def test_stop_nested_loop_calls_stop(monkeypatch):
    """
    Launch `outer_tool`, then instruct the assistant to *stop* it via the
    dynamic helper.  The test passes only if that helper ends up calling
    `AsyncToolLoopHandle.stop()` exactly once.
    """

    # 1.  Instrument `AsyncToolLoopHandle.stop` so we can count invocations
    stop_called = {"count": 0}

    original_stop = AsyncToolUseLoopHandle.stop

    def patched_stop(self):
        stop_called["count"] += 1
        return original_stop(self)

    monkeypatch.setattr(
        AsyncToolUseLoopHandle,
        "stop",
        patched_stop,
        raising=True,
    )

    # 2.  Fire up the *outer* conversational loop
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  If the *user* later says **stop**, call the appropriate "
        "`_stop_…` helper to stop that running call.\n"
        "2️⃣b Immediately after that, call the corresponding `continue_…` helper to keep waiting if needed.\n"
        "3️⃣  Do not produce any other reply until the stop has taken effect; then reply exactly the single line 'outer stopped'.",
    )

    outer_handle = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        log_steps=False,
        max_steps=20,
        timeout=240,
    )

    # 3.  Interject: ask the assistant to stop the running tool call
    # Give the assistant a moment to schedule `outer_tool` so that the
    # dynamic `_stop_…` helper exists in the next turn.
    await asyncio.sleep(3)
    await outer_handle.interject("stop")

    # 4.  Wait for completion & check outcomes
    final_reply = await outer_handle.result()

    # A. The assistant must have followed the instructions.
    assert final_reply.strip().lower() == "outer stopped"

    # B. Our patched `stop()` *must* have been invoked once.
    assert (
        stop_called["count"] == 1
    ), "Nested AsyncToolLoopHandle.stop() was *not* invoked via stoplation"

    # C. Optional sanity – a tool message that confirms stoplation.
    assert any(
        m.get("role") == "tool"
        and "stop" in (m.get("name") or "")
        and "stopped successfully" in (m.get("content") or "").lower()
        for m in client.messages
    ), "No tool-message indicates the stoplation happened"


@pytest.mark.asyncio
async def test_interject_nested_handle(monkeypatch):
    """
    * Inner tool returns a handle (nested loop).
    * Assistant is instructed to interject with "dogs".
    * We monkey-patch `AsyncToolLoopHandle.interject` to count calls.
    """

    # 1.  Monkey-patch the public interject method so we can detect use
    interject_calls = {"count": 0, "payloads": []}

    orig_interject = AsyncToolUseLoopHandle.interject

    async def patched_interject(self, message: str):
        interject_calls["count"] += 1
        interject_calls["payloads"].append(message)
        await orig_interject(self, message)

    monkeypatch.setattr(
        AsyncToolUseLoopHandle,
        "interject",
        patched_interject,
        raising=True,
    )

    # 2.  Inner tool that waits for the steer via `interject_queue`
    async def slow_topic(
        *,
        interject_queue: asyncio.Queue[str],
    ) -> str:
        try:
            new = await asyncio.wait_for(interject_queue.get(), timeout=5)
            return f"topic={new}"
        except asyncio.TimeoutError:
            return "topic=cats"

    slow_topic.__name__ = "slow_topic"
    slow_topic.__qualname__ = "slow_topic"

    # 3.  Outer tool: launches nested loop and returns its handle
    async def outer_tool() -> AsyncToolUseLoopHandle:
        inner_client = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=_get_unity_test_env_var("UNIFY_CACHE"),
            traced=_get_unity_test_env_var("UNIFY_TRACED"),
        )
        inner_client.set_system_message(
            "1️⃣  Call `slow_topic`.\n"
            "2️⃣  Wait until the topic changes.\n"
            "3️⃣  Answer with exactly 'done'.",
        )
        return start_async_tool_use_loop(
            client=inner_client,
            message="start",
            tools={"slow_topic": slow_topic},
        )

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # 4.  Top-level loop – assistant must use `_interject_…`
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "1️⃣  Call `outer_tool`.\n"
        "2️⃣  When the *user* says 'switch to dogs', call the helper whose "
        'name starts with `_interject_` and pass `{ "content": "dogs" }`.\n'
        "2️⃣b Immediately call the corresponding `continue_…` helper to keep waiting.\n"
        "3️⃣  Do not produce any other reply until the work completes.\n"
        "4️⃣  Finally, reply with 'outer done'.",
    )

    top_handle = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=20,
        timeout=240,
    )

    # give assistant time to schedule outer_tool so helper exists
    await asyncio.sleep(5)
    await top_handle.interject("switch to dogs")

    await top_handle.result()

    # 5. Assertions
    msgs = client.messages

    # a) The assistant should have invoked `outer_tool` in its first tool call
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "outer_tool"

    # b) The tool should then return a message indicating the loop started
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["name"] == "outer_tool"
    assert msgs[3]["content"].startswith("Nested async tool loop started")

    # c) Find the user message "switch to dogs"
    interjection_msg = next(
        (
            m
            for m in msgs
            if m["role"] == "system"
            and "user: **switch to dogs**" in (m.get("content") or "")
        ),
        None,
    )
    assert (
        interjection_msg is not None
    ), "Interjection 'switch to dogs' system message not found"

    # d) Find the assistant message that calls the interject helper
    interject_call_msg = next(
        (
            m
            for m in msgs
            if m.get("tool_calls")
            and any(
                call["function"]["name"].startswith("interject_outer_tool_call_")
                for call in m["tool_calls"]
            )
        ),
        None,
    )
    assert (
        interject_call_msg is not None
    ), "Assistant call to interject helper not found"

    # Confirm correct arguments were passed in interject helper call
    interj_call = next(
        call
        for call in interject_call_msg["tool_calls"]
        if call["function"]["name"].startswith("interject_outer_tool_call_")
    )
    assert json.loads(interj_call["function"]["arguments"]) == {"content": "dogs"}

    # e) Find the tool response from the interject helper
    interject_response_msg = next(
        (
            m
            for m in msgs
            if m["role"] == "tool"
            and m["name"].startswith("interject outer_tool")
            and 'Guidance "dogs" forwarded to the running tool.' in m["content"]
        ),
        None,
    )
    assert (
        interject_response_msg is not None
    ), "Tool response from interject helper not found"

    # f) Assistant may either perform a status check, or the loop may update
    #    the existing placeholder tool message directly upon completion. Accept
    #    either outcome.
    status_check_msg = next(
        (
            m
            for m in msgs
            if m.get("tool_calls")
            and any(
                call["function"]["name"].startswith("check_status_call_")
                for call in m["tool_calls"]
            )
        ),
        None,
    )

    if status_check_msg is not None:
        # Tool response to status check should be '"done"'
        status_response_msg = next(
            (
                m
                for m in msgs
                if m["role"] == "tool"
                and m["name"].startswith("check_status_call_")
                and m["content"] == '"done"'
            ),
            None,
        )
        assert (
            status_response_msg is not None
        ), "Tool response with '\"done\"' not found"
    else:
        # Fallback: ensure there is some tool message that delivered '"done"'
        # as the completion result even without an explicit status check.
        fallback_done = next(
            (
                m
                for m in msgs
                if m.get("role") == "tool" and m.get("content") == '"done"'
            ),
            None,
        )
        assert (
            fallback_done is not None
        ), "Assistant neither performed a status check nor produced a final '\"done\"' tool response"

    # h) Assistant's final message should be "outer done"
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"].strip().lower() == "outer done"


@pytest.mark.asyncio
async def test_clarification_nested_handle():
    """
    Inner tool asks a question, outer loop surfaces it, assistant answers
    via `_clarify_<id>`, inner loop receives the answer, outer loop completes.
    """
    exec_log = []

    # ── inner tool that *requires* clarification ─────────────────────────
    async def ask_colour(
        *,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: asyncio.Queue[str],
    ) -> str:
        await clarification_up_q.put("Which colour?")
        colour = await clarification_down_q.get()
        exec_log.append(colour)
        return f"Chose {colour}"

    ask_colour.__name__ = "ask_colour"
    ask_colour.__qualname__ = "ask_colour"

    # ── outer tool launches a nested loop and *exposes the same queues* ──
    async def outer_tool() -> AsyncToolUseLoopHandle:
        up_q, down_q = asyncio.Queue(), asyncio.Queue()
        inner_client = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=_get_unity_test_env_var("UNIFY_CACHE"),
            traced=_get_unity_test_env_var("UNIFY_TRACED"),
        )
        inner_client.set_system_message(
            "1️⃣  Call `ask_colour`.\n"
            "2️⃣  Wait for the clarification answer.\n."
            "3️⃣  Reply with **only** with 'done'.",
        )

        async def _ask_colour_wrapped() -> str:  # type: ignore[valid-type]
            return await ask_colour(
                clarification_up_q=up_q,
                clarification_down_q=down_q,
            )

        _ask_colour_wrapped.__name__ = "ask_colour"
        _ask_colour_wrapped.__qualname__ = "ask_colour"

        handle = start_async_tool_use_loop(
            client=inner_client,
            message="go",
            tools={"ask_colour": _ask_colour_wrapped},
            max_steps=10,
            timeout=60,
        )

        # Expose the same queues on the returned *handle* so the **outer** loop
        # can surface the clarification request and later push the answer down.
        handle.clarification_up_q = up_q
        handle.clarification_down_q = down_q

        return handle

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # ── top-level loop – the assistant must answer the clar request ——––
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "Call `outer_tool`.  When the tool asks a question, answer **only** with 'blue' via the provided helper.\n"
        "Immediately call the corresponding `continue_…` helper to keep waiting and do not reply to the user yet.\n"
        "Finally say 'all done'.",
    )

    top_handle = start_async_tool_use_loop(
        client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=20,
        timeout=240,
    )

    final_reply = await top_handle.result()

    # Assertions ---------------------------------------------------------
    assert "all done" in final_reply.strip().lower()
    assert exec_log == ["blue"], "Inner loop must receive 'blue' from outer helper."


@pytest.mark.asyncio
async def test_handle_interject_method_appears_late():
    """
    Handle initially exposes no `.interject`, then adds it after 1 s.
    The outer loop should create `_interject_…` helper *only* after it
    becomes available, and the assistant must use it successfully.
    """

    interject_seen = {"called": False, "payload": None}

    # dummy handle that adds .interject later --------------------------
    class SlowHandle(AsyncToolUseLoopHandle):
        pass  # will monkey-patch .interject later

    async def dummy_tool() -> SlowHandle:
        handle = SlowHandle(
            task=asyncio.create_task(asyncio.sleep(6)),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

        # after 1 s expose `.interject`
        async def add_interject():
            await asyncio.sleep(1)

            async def _interject(self, msg: str):
                interject_seen["called"] = True
                interject_seen["payload"] = msg
                await asyncio.sleep(0)  # no-op

            setattr(handle, "interject", _interject.__get__(handle, SlowHandle))

        asyncio.create_task(add_interject())
        return handle

    dummy_tool.__name__ = "dummy_tool"
    dummy_tool.__qualname__ = "dummy_tool"

    # outer conversation ----------------------------------------------
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `dummy_tool`.\n"
        "2️⃣  *After* the tool starts and the user says **now**, you MUST call "
        "the helper whose name starts with `_interject_` *exactly once*, "
        'passing `{ "content": "ping" }`.\n'
        "3️⃣  Do **NOT** reply 'done' until after the helper returns.\n"
        "3️⃣b After calling the interject helper, call the corresponding `continue_…` helper to keep waiting if the tool is still running.\n"
        "4️⃣  Finally, respond with the single word **done**.",
    )

    outer = start_async_tool_use_loop(
        client,
        message="start",
        tools={"dummy_tool": dummy_tool},
        max_steps=20,
        timeout=240,
    )

    # wait long enough for the handle to grow `.interject`
    await asyncio.sleep(4)  # helper will exist now
    await outer.interject("now")

    final = await outer.result()

    assert final.strip().lower() == "done"
    assert interject_seen["called"], "handle.interject should have been invoked"
    assert interject_seen["payload"] == "ping"


@pytest.mark.asyncio
async def test_pause_nested_loop_calls_pause():
    """
    Launch a nested loop, tell the assistant to *pause* it via the helper,
    and verify that `AsyncToolLoopHandle.pause()` is invoked exactly once.
    """
    pause_called = {"count": 0}

    async def dummy_long_job() -> (
        AsyncToolUseLoopHandle
    ):  # returns quickly, but "long" enough to pause
        handle = AsyncToolUseLoopHandle(
            task=asyncio.create_task(asyncio.sleep(16)),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

        # expose `.pause` and `.resume`
        async def _pause(self):  # noqa: D401
            pause_called["count"] += 1

        async def _resume(self):  # noqa: D401
            pass  # no-op for this test

        setattr(handle, "pause", _pause.__get__(handle, AsyncToolUseLoopHandle))
        setattr(handle, "resume", _resume.__get__(handle, AsyncToolUseLoopHandle))
        return handle

    dummy_long_job.__name__ = "dummy_long_job"
    dummy_long_job.__qualname__ = "dummy_long_job"

    # outer conversation --------------------------------------------------
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "1️⃣  Call `dummy_long_job`.\n"
        "2️⃣  When the *user* says **pause**, call the helper whose name "
        "starts with `_pause_`.\n"
        "2️⃣b Immediately call the corresponding `continue_…` helper to keep waiting.\n"
        "3️⃣  Keep waiting for the job to finish and do not produce any other reply; then reply with 'paused done'.",
    )

    top = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"dummy_long_job": dummy_long_job},
        max_steps=20,
        timeout=240,
    )

    # helper exists next turn – now ask to pause
    await asyncio.sleep(8)
    await top.interject("pause")

    final = await top.result()

    # assertions ----------------------------------------------------------
    assert final.strip().lower() == "paused done"
    assert pause_called["count"] == 1, "handle.pause() should be called exactly once"


@pytest.mark.asyncio
async def test_resume_nested_loop_calls_resume():
    """
    Pause *and then* resume a running nested loop; ensure both helpers
    reach the corresponding `AsyncToolLoopHandle` methods once each.
    """
    counts = {"pause": 0, "resume": 0}

    async def dummy_job() -> AsyncToolUseLoopHandle:
        """Return a handle whose underlying coroutine can be paused / resumed."""

        # ── internal pausable sleeper ─────────────────────────────────────────
        async def _run(timer: float, gate: asyncio.Event):
            remaining = timer
            step = 0.1  # seconds per loop-tick
            while remaining > 0:
                await gate.wait()  # block if paused
                await asyncio.sleep(step)
                remaining -= step

        gate = asyncio.Event()
        gate.set()  # start in *running* state
        task = asyncio.create_task(_run(8, gate))

        handle = AsyncToolUseLoopHandle(
            task=task,
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

        # ── public pause / resume on the handle ──────────────────────────────
        async def _pause(self):
            if gate.is_set():  # already running → switch to paused
                gate.clear()
                counts["pause"] += 1

        async def _resume(self):
            if not gate.is_set():  # currently paused → resume
                gate.set()
                counts["resume"] += 1

        setattr(handle, "pause", _pause.__get__(handle, AsyncToolUseLoopHandle))
        setattr(handle, "resume", _resume.__get__(handle, AsyncToolUseLoopHandle))
        return handle

    dummy_job.__name__ = "dummy_job"
    dummy_job.__qualname__ = "dummy_job"

    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "1️⃣  Call `dummy_job`.\n"
        "2️⃣  When the *user* says **hold on**, call the `_pause_…` helper.\n"
        "3️⃣  When the *user* then says **continue**, call the `_resume_…` helper.\n"
        "3️⃣b Use the appropriate `continue_…` helper to keep waiting while the job runs.\n"
        "4️⃣  Finally reply **only** with 'all done' once the job completes.",
    )

    h = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"dummy_job": dummy_job},
        max_steps=30,
        timeout=300,
    )

    await asyncio.sleep(4)
    await h.interject("hold on")
    await asyncio.sleep(4)
    await h.interject("continue")

    final = await h.result()

    assert final.strip().lower() == "all done"
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


@pytest.mark.asyncio
async def test_handle_pause_and_resume_freeze_and_unfreeze_loop(monkeypatch):
    """
    • Call pause very early.
    • Wait three seconds while paused (tool finishes in the meantime).
    • Resume and ensure the outer loop *now* completes.
    • Verify pause/resume got invoked once each and that total duration
      exceeds the pause interval.
    """
    counts = {"pause": 0, "resume": 0}

    # ── 1.  Count invocations of the public API  ─────────────────────────
    original_pause = AsyncToolUseLoopHandle.pause
    original_resume = AsyncToolUseLoopHandle.resume

    def patched_pause(self):
        counts["pause"] += 1
        return original_pause(self)

    def patched_resume(self):
        counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(AsyncToolUseLoopHandle, "pause", patched_pause, raising=True)
    monkeypatch.setattr(AsyncToolUseLoopHandle, "resume", patched_resume, raising=True)

    # ── 2.  A very short tool (1 s) – proves that waiting is *because* of pause
    async def long_tool() -> AsyncToolUseLoopHandle:
        async def _run():
            await asyncio.sleep(1)  # completes quickly
            return "done-inside"

        return AsyncToolUseLoopHandle(
            task=asyncio.create_task(_run()),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
        )

    long_tool.__name__ = "long_tool"
    long_tool.__qualname__ = "long_tool"

    # ── 3.  Kick off outer loop ───────────────────────────────────────────
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "1️⃣ Call `long_tool`.\n"
        "2️⃣ Wait for completion (use a `continue_…` helper if exposed) and do not produce any other reply.\n"
        "3️⃣ Reply with exactly **finished**.",
    )

    outer_handle = start_async_tool_use_loop(
        client,
        message="start",
        tools={"long_tool": long_tool},
        max_steps=25,
        timeout=300,
    )

    # ── 4.  Pause soon after launch, wait 3 s, then resume ────────────────
    start_ts = time.perf_counter()

    await asyncio.sleep(0.5)  # allow assistant to schedule the tool
    outer_handle.pause()

    await asyncio.sleep(3.0)  # tool finishes while loop is paused
    outer_handle.resume()

    final_reply = await outer_handle.result()
    elapsed = time.perf_counter() - start_ts

    # ── 5.  Assertions ───────────────────────────────────────────────────
    assert final_reply.strip().lower() == "finished"

    # pause/resume each called exactly once
    assert counts == {"pause": 1, "resume": 1}

    # prove that pause stretched total runtime
    assert (
        elapsed >= 3.0
    ), f"loop finished too quickly ({elapsed:.2f}s) – pause gate failed"


@pytest.mark.asyncio
async def test_handle_result_blocks_until_resume():
    """
    `.result()` hangs while the loop is paused and unblocks immediately once
    `.resume()` is called.
    """

    async def noop_tool() -> str:
        await asyncio.sleep(0.2)
        return "ok"

    noop_tool.__name__ = "noop_tool"
    noop_tool.__qualname__ = "noop_tool"

    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "Call `noop_tool` then answer **only** with 'done'. Do not answer while the loop is paused or while tools are running; only answer after completion.",
    )

    h = start_async_tool_use_loop(
        client,
        message="go",
        tools={"noop_tool": noop_tool},
        timeout=120,
    )

    # pause almost immediately
    h.pause()

    with pytest.raises(asyncio.TimeoutError):
        # Shield protects the inner task from the stoplation that
        # `wait_for` sends when the 1-second timeout expires.
        await asyncio.wait_for(asyncio.shield(h.result()), timeout=1)

    # resume – now it should finish quickly
    h.resume()
    final = await asyncio.wait_for(h.result(), timeout=20)

    assert final.strip().lower() == "done"


@pytest.mark.asyncio
async def test_dynamic_handle_public_method():
    """
    The inner tool returns a handle exposing a **public `.ask()` method**.
    The outer loop must surface an `_ask_…` helper, use it exactly once when
    the user asks "progress?", and finally reply with 'all done' after the
    long-running task completes.
    """

    progress_calls = {"count": 0}

    # ── tool that returns a handle with `.ask` ──────────────────────────
    async def long_compute() -> AsyncToolUseLoopHandle:
        """
        • Runs a 3-second dummy job in the background.
        • Provides `.ask()` so external callers can query the elapsed time.
        """

        start_ts = time.perf_counter()

        async def _job():
            await asyncio.sleep(8)
            return "compute-done"

        handle = AsyncToolUseLoopHandle(
            task=asyncio.create_task(_job()),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

        # public helper – gets exposed automatically
        async def _ask(self):
            progress_calls["count"] += 1
            elapsed = time.perf_counter() - start_ts
            return f"{elapsed:.1f}s elapsed"

        # Bind and expose
        setattr(handle, "ask", _ask.__get__(handle, AsyncToolUseLoopHandle))
        return handle

    long_compute.__name__ = "long_compute"
    long_compute.__qualname__ = "long_compute"

    # ── outer conversation that uses `long_compute` ────────────────────
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "1️⃣  Call `long_compute`.\n"
        "2️⃣  When the *user* asks **progress?**, call the helper whose name "
        "starts with `ask_` exactly once.\n"
        "3️⃣  After calling the `ask_…` helper, do not reply to the user yet. "
        "Immediately call the helper whose name starts with `continue_` to keep waiting.\n"
        "4️⃣  Only once the computation finishes, answer **only** with 'all done'",
    )

    top = start_async_tool_use_loop(
        client,
        message="start",
        tools={"long_compute": long_compute},
        max_steps=25,
        timeout=300,
    )

    # Give the assistant a moment to launch the tool so `_ask_…` exists
    await asyncio.sleep(5)
    await top.interject("progress?")

    final_reply = await top.result()

    # ── Assertions ─────────────────────────────────────────────────────
    assert "all done" in final_reply.strip().lower()
    assert progress_calls["count"] == 1, ".ask should be invoked exactly once"

    # Optional: sanity-check that a tool-message from `_ask_…` is present
    assert any(
        m.get("role") == "tool" and "ask_" in (m.get("name") or "")
        for m in client.messages
    ), "No tool-message from the `ask_…` helper found"
