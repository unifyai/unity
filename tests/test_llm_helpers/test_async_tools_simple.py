"""
pytest tests for the llm helpers:
"""

from __future__ import annotations

import unify
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

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Add numbers",
        tools={"add": add},
        max_consecutive_failures=2,
    ).result()

    assert answer.strip() == "5"
    # exactly one tool result fed back
    assert sum(m["role"] == "tool" for m in client.messages) == 1


# --------------------------------------------------------------------------- #
#  CONCURRENT sync/async tools – waits for *all* results before next LLM turn #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_concurrent_tools_waits_for_all_results():
    """
    The loop launches `fast` and `slow` concurrently but must *not* call the
    model again until *both* have finished.  It should therefore:

      • make exactly two `generate()` calls
      • return the content of the second call
      • still run the tools truly concurrently
    """
    events: list[tuple[str, float]] = []

    async def fast():
        events.append(("fast_start", time.monotonic()))
        await asyncio.sleep(0.05)
        events.append(("fast_end", time.monotonic()))
        return "fast"

    async def slow():
        events.append(("slow_start", time.monotonic()))
        await asyncio.sleep(0.30)
        events.append(("slow_end", time.monotonic()))
        return "slow"

    def record_generate():
        events.append(("generate", time.monotonic()))

    # assistant first asks for two tools, then replies once with "ok"
    first_turn = types.SimpleNamespace(
        tool_calls=[
            FakeToolCall("fast", {}, "1"),
            FakeToolCall("slow", {}, "2"),
        ],
        content="",
    )

    scripted = [
        make_response(first_turn),
        make_response(msg_final("ok")),  # produced after *both* results
    ]

    class InstrumentedClient(FakeAsyncClient):
        async def generate(self, **kwargs):
            record_generate()
            return await super().generate(**kwargs)

    client = InstrumentedClient(scripted)

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Run fast & slow",
        tools={"fast": fast, "slow": slow},
    ).result()

    # 1. loop must return the assistant’s second reply
    assert answer.strip() == "ok"

    # 2. exactly two model calls (before tools, after all tools)
    generate_times = [t for e, t in events if e == "generate"]
    assert len(generate_times) == 2

    # 3. the second generate happens *after* the slow tool finishes
    slow_end = next(t for e, t in events if e == "slow_end")
    assert generate_times[1] > slow_end

    # 4. tools really overlapped
    fast_start = next(t for e, t in events if e == "fast_start")
    fast_end = next(t for e, t in events if e == "fast_end")
    slow_start = next(t for e, t in events if e == "slow_start")
    assert fast_start < slow_start < fast_end  # overlap window


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

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Divide numbers",
        tools={"divide": divide},
        max_consecutive_failures=3,
    ).result()

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
        await llmh.start_async_tool_use_loop(
            client,
            message="Break me",
            tools={"divide": divide},
            max_consecutive_failures=2,
        ).result()


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

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Run async then sync",
        tools={"fast_tool": fast_tool, "add": add},
        max_consecutive_failures=2,
    ).result()

    assert answer.strip() == "42"


def square(x: int) -> int:
    return x * x


@pytest.mark.asyncio
async def test_parallel_tool_calls():

    client = unify.AsyncUnify("gpt-4o@openai")

    # Run the loop – ask it to give back the history as well
    await llmh.start_async_tool_use_loop(
        client,
        "Square 2 and 3 please",
        {"square": square},
    ).result()

    # Find the first assistant turn that *requested* tool calls
    first_llm_turn = next(
        m for m in client.messages if m["role"] == "assistant" and m.get("tool_calls")
    )

    # Ensure it actually asked for >1 tools – i.e. parallel tool calls
    assert len(first_llm_turn["tool_calls"]) == 2
