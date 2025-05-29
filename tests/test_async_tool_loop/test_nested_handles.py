import pytest
import time
import asyncio
import unify

from unity.common.llm_helpers import start_async_tool_use_loop, AsyncToolLoopHandle


# ─────────────────────────────────────────────────────────────────────────────
#  Tools for the *inner* loop
# ─────────────────────────────────────────────────────────────────────────────


def inner_tool() -> str:  # noqa: D401 – simple value
    """Returns the literal string 'inner‑result'."""
    time.sleep(1)
    return "inner-result"


# ─────────────────────────────────────────────────────────────────────────────
#  Tool for the *outer* loop – spawns the nested loop and returns its handle
# ─────────────────────────────────────────────────────────────────────────────


async def outer_tool() -> AsyncToolLoopHandle:
    """Launch an **inner** async‑tool‑use loop and return its *handle*."""

    # brand‑new LLM client dedicated to the nested conversation
    inner_client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
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
    """Full end‑to‑end check – no mocks, real network call to OpenAI."""

    # Outer client that drives the *first* loop
    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
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

    # System message
    assert client.messages[0] == {
        "role": "system",
        "content": "You are running inside an automated test. Perform the steps exactly:\n1\ufe0f\u20e3  Call `outer_tool` with no arguments.\n2\ufe0f\u20e3  Continue running this tool call, when given the option.\n3\ufe0f\u20e3  Once it is *completed*, respond with exactly 'all done'.",
    }

    # User message
    assert client.messages[1] == {
        "role": "user",
        "content": "start",
    }

    # Assistant tool selection
    tool_selection_msg = client.messages[2]
    assert tool_selection_msg["role"] == "assistant"
    assert len(tool_selection_msg["tool_calls"]) == 1
    assert tool_selection_msg["tool_calls"][0]["function"] == {
        "arguments": "{}",
        "name": "outer_tool",
    }

    # Tool response
    tool_response = client.messages[3]
    assert tool_response["role"] == "tool"
    assert tool_response["name"] == "outer_tool"
    assert "done" in tool_response["content"].lower()

    # Assistant final response
    assert client.messages[4] == {
        "content": "all done",
        "refusal": None,
        "role": "assistant",
        "annotations": [],
        "audio": None,
        "function_call": None,
        "tool_calls": None,
    }
    assert len(client.messages) == 5


@pytest.mark.asyncio
async def test_cancel_nested_loop_calls_stop(monkeypatch):
    """
    Launch `outer_tool`, then instruct the assistant to *cancel* it via the
    dynamic helper.  The test passes only if that helper ends up calling
    `AsyncToolLoopHandle.stop()` exactly once.
    """

    # 1.  Instrument `AsyncToolLoopHandle.stop` so we can count invocations
    stop_called = {"count": 0}

    original_stop = AsyncToolLoopHandle.stop

    def patched_stop(self):
        stop_called["count"] += 1
        return original_stop(self)

    monkeypatch.setattr(
        AsyncToolLoopHandle,
        "stop",
        patched_stop,
        raising=True,
    )

    # 2.  Fire up the *outer* conversational loop
    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  If the *user* later says **cancel**, call the appropriate "
        "`_cancel_…` helper to cancel that running call.\n"
        "3️⃣  Then reply with exactly the single line 'outer cancelled'.",
    )

    outer_handle = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        log_steps=False,
        max_steps=20,
        timeout=240,
    )

    # 3.  Interject: ask the assistant to cancel the running tool call
    # Give the assistant a moment to schedule `outer_tool` so that the
    # dynamic `_cancel_…` helper exists in the next turn.
    await asyncio.sleep(3)
    await outer_handle.interject("cancel")

    # 4.  Wait for completion & check outcomes
    final_reply = await outer_handle.result()

    # A. The assistant must have followed the instructions.
    assert final_reply.strip().lower() == "outer cancelled"

    # B. Our patched `stop()` *must* have been invoked once.
    assert (
        stop_called["count"] == 1
    ), "Nested AsyncToolLoopHandle.stop() was *not* invoked via cancellation"

    # C. Optional sanity – a tool message that confirms cancellation.
    assert any(
        m.get("role") == "tool"
        and "_cancel" in (m.get("name") or "")
        and "cancelled successfully" in (m.get("content") or "").lower()
        for m in client.messages
    ), "No tool-message indicates the cancellation happened"


@pytest.mark.asyncio
async def test_interject_nested_handle(monkeypatch):
    """
    * Inner tool returns a handle (nested loop).
    * Assistant is instructed to interject with "dogs".
    * We monkey-patch `AsyncToolLoopHandle.interject` to count calls.
    """

    # 1.  Monkey-patch the public interject method so we can detect use
    interject_calls = {"count": 0, "payloads": []}

    orig_interject = AsyncToolLoopHandle.interject

    async def patched_interject(self, message: str):
        interject_calls["count"] += 1
        interject_calls["payloads"].append(message)
        await orig_interject(self, message)

    monkeypatch.setattr(
        AsyncToolLoopHandle,
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

    # 3.  Outer tool: launches nested loop and returns its handle
    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
        inner_client.set_system_message(
            "1️⃣  Call `slow_topic`.\n"
            "2️⃣  Wait until the topic changes.\n"
            "3️⃣  Answer with exactly 'done'.",
        )
        return start_async_tool_use_loop(
            client=inner_client,
            message="start",
            tools={"slow_topic": slow_topic},
            max_steps=10,
            timeout=120,
        )

    # 4.  Top-level loop – assistant must use `_interject_…`
    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(
        "1️⃣  Call `outer_tool`.\n"
        "2️⃣  When the *user* says 'switch to dogs', call the helper whose "
        'name starts with `_interject_` and pass `{ "content": "dogs" }`.\n'
        "3️⃣  Finally, reply with 'outer done'.",
    )

    top_handle = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=20,
        timeout=240,
    )

    # give assistant time to schedule outer_tool so helper exists
    await asyncio.sleep(2)
    await top_handle.interject("switch to dogs")

    final_reply = await top_handle.result()

    # 5.  Assertions
    assert final_reply.strip().lower() == "outer done"
    assert interject_calls["count"] == 1, "handle.interject should be called once"
    assert "dogs" in interject_calls["payloads"][0].lower()


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

    # ── outer tool launches a nested loop and *exposes the same queues* ──
    async def outer_tool() -> AsyncToolLoopHandle:
        up_q, down_q = asyncio.Queue(), asyncio.Queue()
        inner_client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
        inner_client.set_system_message(
            "1️⃣  Call `ask_colour`.\n"
            "2️⃣  Wait for the clarification answer.\n"
            "3️⃣  Reply with exactly 'done'.",
        )
        handle = start_async_tool_use_loop(
            client=inner_client,
            message="go",
            tools={"ask_colour": ask_colour},
            max_steps=10,
            timeout=120,
        )
        # expose the queues so the *outer* loop sees them
        handle.clarification_up_q = up_q
        handle.clarification_down_q = down_q
        return handle

    # ── top-level loop – the assistant must answer the clar request ——––
    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(
        "Call `outer_tool`.  When the tool asks a question, answer 'blue' "
        "via the provided helper, then say 'all done'.",
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
    class SlowHandle(AsyncToolLoopHandle):
        pass  # will monkey-patch .interject later

    async def dummy_tool() -> SlowHandle:
        handle = SlowHandle(
            task=asyncio.create_task(asyncio.sleep(6)),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
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

    # outer conversation ----------------------------------------------
    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `dummy_tool`.\n"
        "2️⃣  *After* the tool starts and the user says **now**, you MUST call "
        "the helper whose name starts with `_interject_` *exactly once*, "
        'passing `{ "content": "ping" }`.\n'
        "3️⃣  Do **NOT** reply 'done' until after the helper returns.\n"
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
        AsyncToolLoopHandle
    ):  # returns quickly, but "long" enough to pause
        handle = AsyncToolLoopHandle(
            task=asyncio.create_task(asyncio.sleep(4)),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
        )

        # expose `.pause` and `.resume`
        async def _pause(self):  # noqa: D401
            pause_called["count"] += 1

        async def _resume(self):  # noqa: D401
            pass  # no-op for this test

        setattr(handle, "pause", _pause.__get__(handle, AsyncToolLoopHandle))
        setattr(handle, "resume", _resume.__get__(handle, AsyncToolLoopHandle))
        return handle

    # outer conversation --------------------------------------------------
    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(
        "1️⃣  Call `dummy_long_job`.\n"
        "2️⃣  When the *user* says **pause**, call the helper whose name "
        "starts with `_pause_`.\n"
        "3️⃣  Keep waiting for the job to finish, then reply with 'paused done'.",
    )

    top = start_async_tool_use_loop(
        client=client,
        message="start",
        tools={"dummy_long_job": dummy_long_job},
        max_steps=20,
        timeout=240,
    )

    # helper exists next turn – now ask to pause
    await asyncio.sleep(2)
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

    async def dummy_job() -> AsyncToolLoopHandle:
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

        handle = AsyncToolLoopHandle(
            task=task,
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
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

        setattr(handle, "pause", _pause.__get__(handle, AsyncToolLoopHandle))
        setattr(handle, "resume", _resume.__get__(handle, AsyncToolLoopHandle))
        return handle

        setattr(handle, "pause", _pause.__get__(handle, AsyncToolLoopHandle))
        setattr(handle, "resume", _resume.__get__(handle, AsyncToolLoopHandle))
        return handle

    client = unify.AsyncUnify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(
        "1️⃣  Call `dummy_job`.\n"
        "2️⃣  When the *user* says **hold on**, call the `_pause_…` helper.\n"
        "3️⃣  When the *user* then says **continue**, call the `_resume_…` helper.\n"
        "4️⃣  Finally reply with 'all done' once the job completes.",
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
