"""
pytest tests for the async-tool loop helpers **using a real `unify.AsyncUnify`
client for every test** – no stubs, no scripted completions.

Running these tests will make real requests to the model you pass to
`unify.AsyncUnify` (by default we use **GPT-4o**).
Make sure you have:

* a valid OpenAI (or Unify-proxy) API key in your environment, and
* internet connectivity while the suite runs.

The tests still exercise exactly the same behaviours as before:

* single-tool “happy path”
* waiting for multiple concurrent tools to finish before the 2nd LLM call
* recovery after a tool error and counter reset
* aborting when too many consecutive tool failures occur
* a realistic mix of async + sync tools
* verifying that the first assistant turn contains _parallel_ tool calls

If any test starts to fail intermittently, tweak the user-message so that
the model’s behaviour stays deterministic enough for the assertions below.
"""

from __future__ import annotations

import asyncio
import os
import time
from tests.helpers import _handle_project

import pytest
import unify

# --------------------------------------------------------------------------- #
#  MODULE UNDER TEST                                                          #
# --------------------------------------------------------------------------- #
import unity.common.llm_helpers as llmh  # noqa: E402 – after site-imports


MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")  # override if you like


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS (sync + async)                                        #
# --------------------------------------------------------------------------- #
@unify.traced
def add(x: int, y: int) -> int:
    return x + y


@unify.traced
def divide(a: int, b: int) -> float:  # may raise
    return a / b


@unify.traced
def launch() -> None:
    raise Exception


@unify.traced
async def fast_tool(res: str = "fast") -> str:
    await asyncio.sleep(0.05)
    return res


@unify.traced
async def slow_tool(res: str = "slow") -> str:
    await asyncio.sleep(0.3)
    return res


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
        cache=os.environ.get("UNIFY_CACHE"),
        traced=os.environ.get("UNIFY_TRACED"),
    ).set_system_message(
        "Feel free to call multiple *different* tools per turn if appropriate.",
    )


@unify.traced
def count_tool_messages(client: unify.AsyncUnify) -> int:
    return sum(1 for m in client.messages if m["role"] == "tool")


# --------------------------------------------------------------------------- #
#  HAPPY PATH – single synchronous tool                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_happy_path_single_sync_tool():
    client = new_client()

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Add 2 and 3 using the `add` tool and answer with the result only.",
        tools={"add": add},
        max_consecutive_failures=2,
    ).result()

    assert answer.strip().startswith("5")
    assert count_tool_messages(client) >= 1


# --------------------------------------------------------------------------- #
#  CONCURRENT sync/async tools                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_concurrent_tools_waits_for_all_results():
    """
    The loop launches `fast` and `slow` concurrently but must *not* call the
    model again until *both* have finished.
    """
    events: list[tuple[str, float]] = []

    @unify.traced
    async def fast():
        events.append(("fast_start", time.monotonic()))
        await asyncio.sleep(0.05)
        events.append(("fast_end", time.monotonic()))
        return "fast"

    @unify.traced
    async def slow():
        events.append(("slow_start", time.monotonic()))
        await asyncio.sleep(0.30)
        events.append(("slow_end", time.monotonic()))
        return "slow"

    class InstrumentedClient(unify.AsyncUnify):  # type: ignore[misc]
        async def generate(self, **kwargs):  # noqa: D401
            events.append(("generate", time.monotonic()))
            return await super().generate(**kwargs)

    client = InstrumentedClient(MODEL_NAME)
    client.set_traced(True)

    _ = await llmh.start_async_tool_use_loop(
        client,
        message=(
            "Call *both* tools `fast` and `slow` in parallel, wait for the "
            "results, then reply with 'ok'."
        ),
        tools={"fast": fast, "slow": slow},
    ).result()

    # 1. there were at least two model calls (tool-request + final answer)
    generate_times = [t for e, t in events if e == "generate"]
    assert len(generate_times) >= 2

    # 2. the last LLM call happened AFTER the slow tool finished
    slow_end = next(t for e, t in events if e == "slow_end")
    assert generate_times[-1] > slow_end

    # 3. the two tools actually overlapped
    fast_start = next(t for e, t in events if e == "fast_start")
    fast_end = next(t for e, t in events if e == "fast_end")
    slow_start = next(t for e, t in events if e == "slow_start")
    assert fast_start < slow_start < fast_end

    # 4. the first assistant turn really requested BOTH tool calls
    first_llm_turn = next(
        m for m in client.messages if m["role"] == "assistant" and m.get("tool_calls")
    )
    assert len(first_llm_turn["tool_calls"]) == 2


# --------------------------------------------------------------------------- #
#  RECOVERY AFTER A FAILURE & COUNTER RESET                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_recovers_after_failure():
    client = new_client()

    answer = await llmh.start_async_tool_use_loop(
        client,
        message=(
            "First divide 4 by 0 using the `divide` tool – that will fail.   "
            "Then divide 4 by 2 with the same tool and give me just the result."
        ),
        tools={"divide": divide},
        max_consecutive_failures=3,
    ).result()

    assert "2" in answer.strip()

    tool_msgs = [m["content"] for m in client.messages if m["role"] == "tool"]
    assert any("ZeroDivisionError" in (tb or "") for tb in tool_msgs)


# --------------------------------------------------------------------------- #
#  ABORT AFTER MAX CONSECUTIVE FAILURES                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_aborts_after_too_many_failures():
    client = new_client()

    with pytest.raises(RuntimeError):
        await llmh.start_async_tool_use_loop(
            client,
            message=("Please run the launch tool."),
            tools={"launch": launch},
            max_consecutive_failures=1,  # abort after the very first failure
        ).result()


# --------------------------------------------------------------------------- #
#  REALISTIC MIX – first async, then sync                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mixed_sync_async_tools():
    client = new_client()

    answer = await llmh.start_async_tool_use_loop(
        client,
        message=(
            "Call the async tool `fast_tool` (which just returns a token),   "
            "then call the sync `add` tool with 6 and 7, and finally reply "
            "with the result only."
        ),
        tools={"fast_tool": fast_tool, "add": add},
        max_consecutive_failures=2,
    ).result()

    assert "13" in answer.strip()


@pytest.mark.asyncio
@_handle_project
async def test_duplicate_tool_calls_are_optionally_pruned() -> None:  # noqa: D401
    """Verify that duplicate tool calls are kept or pruned according to the flag."""

    log: list[str] = []

    async def echo(text: str) -> str:
        """Minimal echo tool used only to count invocations."""
        log.append(text)
        return text.upper()

    prompt = (
        "You have access to a function named `echo(text: str)`.\n"
        "For demonstration purposes **call `echo` twice** with exactly the same JSON "
        'arguments `{ "text": "hello" }` – do not merge the calls.  After both calls, '
        "answer with a single short sentence."
    )

    # ------------------------------------------------------------------ #
    # 1️⃣  duplicates SHOULD be executed when pruning is disabled
    # ------------------------------------------------------------------ #
    log.clear()
    client = new_client()
    await llmh.start_async_tool_use_loop(
        client=client,
        message=prompt,
        tools={"echo": echo},
        prune_tool_duplicates=False,
    ).result()
    assert log == [
        "hello",
        "hello",
    ], "With ignore_tool_duplicates=False the tool should be invoked twice."
    assert [m["role"] for m in client.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]

    # ------------------------------------------------------------------ #
    # 2️⃣  duplicates SHOULD be removed when pruning is enabled
    # ------------------------------------------------------------------ #
    log.clear()
    client = new_client()
    await llmh.start_async_tool_use_loop(
        client=client,
        message=prompt,
        tools={"echo": echo},
        prune_tool_duplicates=True,
    ).result()
    assert log == [
        "hello",
        "hello",
    ], "With ignore_tool_duplicates=True, two invocations are still expected."
    assert [m["role"] for m in client.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
