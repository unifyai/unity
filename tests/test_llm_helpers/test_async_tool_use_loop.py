"""
pytest tests for the *asynchronous* helper:

* async_tool_use_loop            – happy path, mixed sync/async tools,
                                    early-return concurrency, recovery after
                                    failure, abort after N failures
"""

from __future__ import annotations

import asyncio
import json
import time
import types
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
#  MODULE UNDER TEST                                                          #
# --------------------------------------------------------------------------- #
# Change import path if your helpers live elsewhere
import unity.common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
#  TEST DOUBLES                                                               #
# --------------------------------------------------------------------------- #
class FakeToolCall:
    """Mimics OpenAI's ToolCall object."""

    def __init__(self, name: str, args: dict, call_id: str = "1"):
        self.id = call_id
        self.function = types.SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        )


def make_response(message):
    """Wrap a Message-like object in the structure returned by AsyncUnify.generate."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
    )


class FakeAsyncClient:
    """
    Dumb stand-in for `unify.AsyncUnify`.

    * `generate` pops a pre-scripted response (async).
    * `append_messages` records the conversation for assertions.
    """

    def __init__(self, scripted_responses: list):
        self._responses = scripted_responses[:]
        self.messages: list[dict[str, Any]] = []

    async def generate(self, **_kwargs) -> Any:
        try:
            return self._responses.pop(0)
        except IndexError as exc:
            raise RuntimeError("FakeAsyncClient ran out of scripted responses") from exc

    def append_messages(self, msgs):
        self.messages.extend(msgs)


# --------------------------------------------------------------------------- #
#  HELPERS TO BUILD "MODEL" MESSAGES                                          #
# --------------------------------------------------------------------------- #
def msg_tool_call(name: str, args: dict, call_id: str = "1"):
    return types.SimpleNamespace(
        tool_calls=[FakeToolCall(name, args, call_id)],
        content="",
    )


def msg_final(content: str):
    return types.SimpleNamespace(tool_calls=None, content=content)


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS (sync + async)                                        #
# --------------------------------------------------------------------------- #
def add(x: int, y: int) -> int:  # synchronous
    return x + y


def divide(a: int, b: int) -> float:  # synchronous – may raise
    return a / b


async def fast_tool(res: str = "fast") -> str:  # completes quickly
    await asyncio.sleep(0.05)
    return res


async def slow_tool(res: str = "slow") -> str:  # noticeably slower
    await asyncio.sleep(0.3)
    return res


# --------------------------------------------------------------------------- #
#  HAPPY PATH – single synchronous tool                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_happy_path_single_sync_tool():
    scripted = [
        make_response(msg_tool_call("add", {"x": 2, "y": 3})),
        make_response(msg_final("5")),
    ]
    client = FakeAsyncClient(scripted)

    answer = await llmh.async_tool_use_loop(
        client,
        message="Add numbers",
        tools={"add": add},
        max_consecutive_failures=2,
    )

    assert answer.strip() == "5"
    # exactly one tool result fed back
    assert sum(m["role"] == "tool" for m in client.messages) == 1


# --------------------------------------------------------------------------- #
#  MIXED sync/async tools and *early* return to the LLM                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_concurrent_tools_early_generate():
    """
    One LLM turn triggers *both* `fast` and `slow`.
    The loop must return to the model after `fast` completes
    while `slow` is still running.
    """
    events: list[tuple[str, float]] = []

    async def fast():
        events.append(("fast_start", time.monotonic()))
        await asyncio.sleep(0.05)
        events.append(("fast_end", time.monotonic()))
        return "fast"

    async def slow():
        events.append(("slow_start", time.monotonic()))
        await asyncio.sleep(0.3)
        events.append(("slow_end", time.monotonic()))
        return "slow"

    def record_generate():
        events.append(("generate", time.monotonic()))

    # SINGLE model message with *two* tool calls
    first_turn = types.SimpleNamespace(
        tool_calls=[
            FakeToolCall("fast", {}, "1"),
            FakeToolCall("slow", {}, "2"),
        ],
        content="",
    )

    scripted = [
        make_response(first_turn),  # triggers both tools concurrently
        make_response(msg_final("done")),  # model answers after `fast` result only
        make_response(msg_final("ok")),  # model replies after slow (no tools)
    ]

    class InstrumentedClient(FakeAsyncClient):
        async def generate(self, **kwargs):
            record_generate()
            return await super().generate(**kwargs)

    client = InstrumentedClient(scripted)

    answer = await llmh.async_tool_use_loop(
        client,
        message="Run fast & slow",
        tools={"fast": fast, "slow": slow},
        max_consecutive_failures=2,
    )

    assert answer.strip() == "ok"

    # ── Timing assertions ────────────────────────────────────────────
    generate_times = [t for e, t in events if e == "generate"]
    fast_end = next(t for e, t in events if e == "fast_end")
    slow_end = next(t for e, t in events if e == "slow_end")

    # 0-th generate: initial model request (before any tool starts)
    # 1-st generate: after `fast` finishes but *before* `slow` ends
    assert generate_times[0] < fast_end
    assert generate_times[1] < slow_end


# --------------------------------------------------------------------------- #
#  RECOVERY AFTER A FAILURE & COUNTER RESET                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_recovers_after_failure():
    scripted = [
        make_response(msg_tool_call("divide", {"a": 4, "b": 0})),  # raises
        make_response(msg_tool_call("divide", {"a": 4, "b": 2})),  # succeeds
        make_response(msg_final("2.0")),
    ]
    client = FakeAsyncClient(scripted)

    answer = await llmh.async_tool_use_loop(
        client,
        message="Divide numbers",
        tools={"divide": divide},
        max_consecutive_failures=3,
    )

    assert answer.strip().startswith("2")

    tool_msgs = [m["content"] for m in client.messages if m["role"] == "tool"]
    # first feedback must contain traceback mentioning ZeroDivisionError
    assert any("ZeroDivisionError" in tb for tb in tool_msgs)


# --------------------------------------------------------------------------- #
#  ABORT AFTER MAX CONSECUTIVE FAILURES                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_aborts_after_too_many_failures():
    scripted = [
        make_response(msg_tool_call("divide", {"a": 1, "b": 0})),
        make_response(msg_tool_call("divide", {"a": 1, "b": 0})),
    ]
    client = FakeAsyncClient(scripted)

    with pytest.raises(RuntimeError):
        await llmh.async_tool_use_loop(
            client,
            message="Break me",
            tools={"divide": divide},
            max_consecutive_failures=2,
        )


# --------------------------------------------------------------------------- #
#  REALISTIC MIX – first async fast, then sync add                            #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_mixed_sync_async_tools():
    """Ensures sync tools are transparently run in the thread-pool."""
    scripted = [
        make_response(msg_tool_call("fast_tool", {})),
        make_response(msg_tool_call("add", {"x": 6, "y": 7})),
        make_response(msg_final("42")),
    ]
    client = FakeAsyncClient(scripted)

    answer = await llmh.async_tool_use_loop(
        client,
        message="Run async then sync",
        tools={"fast_tool": fast_tool, "add": add},
        max_consecutive_failures=2,
    )

    assert answer.strip() == "42"


# --------------------------------------------------------------------------- #
#  CANCEL – the tool is stopped by the calling code                           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_fast_cancel():
    """Loop exits promptly (<150 ms) when `cancel_event` is set."""
    cancel_event = asyncio.Event()
    flagged = {"cancelled": False}

    async def long_tool():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            flagged["cancelled"] = True
            raise

    model_turn = types.SimpleNamespace(
        tool_calls=[FakeToolCall("long", {}, "1")],
        content="",
    )
    scripted = [make_response(model_turn)]
    client = FakeAsyncClient(scripted)

    task = asyncio.create_task(
        llmh.async_tool_use_loop(
            client,
            message="run",
            tools={"long": long_tool},
            cancel_event=cancel_event,
        ),
    )

    await asyncio.sleep(0.05)  # give tool time to start
    t0 = time.monotonic()
    cancel_event.set()  # trigger stop
    with pytest.raises(asyncio.CancelledError):
        await task
    dt = time.monotonic() - t0

    assert dt < 0.15  # stop was fast
    assert flagged["cancelled"]  # tool got cancelled
