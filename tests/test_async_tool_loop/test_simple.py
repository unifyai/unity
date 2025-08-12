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
from tests.helpers import _handle_project, _get_unity_test_env_var
from tests.test_async_tool_loop.async_helpers import _wait_for_tool_request

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
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
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

    fast.__name__ = "fast"
    fast.__qualname__ = "fast"

    @unify.traced
    async def slow():
        events.append(("slow_start", time.monotonic()))
        await asyncio.sleep(0.30)
        events.append(("slow_end", time.monotonic()))
        return "slow"

    slow.__name__ = "slow"
    slow.__qualname__ = "slow"

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
            raise_on_limit=True,
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

    echo.__name__ = "echo"
    echo.__qualname__ = "echo"

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
    roles = [
        m["role"]
        for m in client.messages
        if not (
            m.get("role") == "assistant"
            and m.get("tool_calls")
            and any(
                (tc.get("function", {}) or {})
                .get("name", "")
                .startswith("check_status_")
                for tc in m["tool_calls"]
            )
        )
        and not (
            m.get("role") == "tool"
            and str(m.get("name", "")).startswith("check_status_")
        )
    ]
    assert roles == [
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
    roles = [
        m["role"]
        for m in client.messages
        if not (
            m.get("role") == "assistant"
            and m.get("tool_calls")
            and any(
                (tc.get("function", {}) or {})
                .get("name", "")
                .startswith("check_status_")
                for tc in m["tool_calls"]
            )
        )
        and not (
            m.get("role") == "tool"
            and str(m.get("name", "")).startswith("check_status_")
        )
    ]
    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


# --------------------------------------------------------------------------- #
#  NO-TOOLS FLOWS                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_no_tools_with_system_message() -> None:
    """
    Verify that the loop completes correctly when **no** tools are available
    and the conversation starts with a system prompt:

        system → user → assistant
    """
    client = new_client()  # ← already includes the system message

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Just reply with a friendly greeting – no tools are available.",
        tools={},  # ← empty tool-kit
    ).result()

    # The assistant must answer directly and never insert any tool messages.
    assert answer.strip(), "Assistant reply should not be empty."
    assert count_tool_messages(client) == 0
    assert [m["role"] for m in client.messages] == [
        "system",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
@_handle_project
async def test_no_tools_without_system_message() -> None:
    """
    Same as above, but without a leading system message, giving the flow:

        user → assistant
    """
    client = unify.AsyncUnify(MODEL_NAME)
    client.set_traced(True)

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Say hello back to me – there are no tools at all.",
        tools={},  # ← still an empty tool-kit
    ).result()

    assert answer.strip(), "Assistant reply should not be empty."
    assert count_tool_messages(client) == 0
    assert [m["role"] for m in client.messages] == [
        "user",
        "assistant",
    ]


# --------------------------------------------------------------------------- #
#  CONCURRENCY LIMIT – max_concurrent                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_max_concurrent_limit_is_obeyed() -> None:  # noqa: D401
    """Ensure the per‑tool *runtime* concurrency cap is respected.

    We ask the model to run the same tool twice *in parallel*.  The limit
    (`max_concurrent=1`) should force the second call to wait until the
    first ends.  We tolerate the model making more than two invocations –
    what matters is that **no overlap > 1** happens.
    """

    events: list[tuple[str, float]] = []

    @unify.traced
    async def limited(label: str) -> str:
        events.append(("start", time.monotonic()))
        await asyncio.sleep(0.15)
        events.append(("end", time.monotonic()))
        return label.upper()

    limited.__name__ = "limited"
    limited.__qualname__ = "limited"

    tools = {"limited": llmh.ToolSpec(fn=limited, max_concurrent=1)}

    client = new_client()

    # Kick off the interactive loop *without* awaiting the final result yet so
    # that we can synchronise on the **first** tool request and ensure all
    # timing events are captured in the correct order.

    handle = llmh.start_async_tool_use_loop(
        client=client,
        message=(
            "Invoke `limited` twice *concurrently* – once with 'one', once "
            "with 'two' – then reply 'done'."
        ),
        tools=tools,
        prune_tool_duplicates=False,
    )

    # Block until the assistant has actually *requested* at least one call to
    # the `limited` tool.  This makes the test independent from model latency
    # and guarantees that the event-log below reflects the real execution
    # window of the tool.
    await _wait_for_tool_request(client, "limited")

    # Now wait for the whole loop to finish.
    await handle.result()

    starts = [t for e, t in events if e == "start"]
    ends = [t for e, t in events if e == "end"]

    # Sanity: any start must be paired with an end
    assert len(starts) == len(
        ends,
    ), "Mismatched start/end counts – tool never returned?"

    # The model may decide to retry the second invocation *after* the first one
    # completed.  What matters is that **no two calls** to the tool ever ran in
    # parallel.  Therefore, we only check that at least one invocation
    # happened and that the *peak* concurrency never exceeded 1.

    assert len(starts) >= 1, "The tool was never invoked at all – test setup failed."

    # Verify the core requirement: that the peak concurrency never exceeded 1.
    timeline = sorted(events, key=lambda p: p[1])
    running = peak = 0
    for kind, _ in timeline:
        running += 1 if kind == "start" else -1
        peak = max(peak, running)

    assert (
        peak == 1
    ), "More than one instance ran concurrently despite max_concurrent=1."
