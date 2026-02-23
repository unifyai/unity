"""
pytest tests for the async-tool loop helpers **using a real `unillm.AsyncUnify`
client for every test** – no stubs, no scripted completions.

Running these tests will make real requests to the model you pass to
`unillm.AsyncUnify` (by default we use **GPT-5**).
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
import time
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _is_synthetic_check_status_stub,
    _is_synthetic_check_status_tool_msg,
)

import pytest
import unillm

# --------------------------------------------------------------------------- #
#  MODULE UNDER TEST                                                          #
# --------------------------------------------------------------------------- #
from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.tool_spec import ToolSpec


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS (sync + async)                                        #
# --------------------------------------------------------------------------- #
def add(x: int, y: int) -> int:
    return x + y


def divide(a: int, b: int) -> float:  # may raise
    return a / b


def launch() -> None:
    raise Exception


async def fast_tool(res: str = "fast") -> str:
    await asyncio.sleep(0.05)
    return res


async def slow_tool(res: str = "slow") -> str:
    await asyncio.sleep(0.3)
    return res


def count_tool_messages(client) -> int:
    return sum(1 for m in client.messages if m["role"] == "tool")


# --------------------------------------------------------------------------- #
#  HAPPY PATH – single synchronous tool                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_happy_path_single_sync_tool(llm_config):
    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
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
async def test_concurrent_tools_waits_for_all_results(llm_config):
    """
    The loop launches `fast` and `slow` concurrently but must *not* call the
    model again until *both* have finished.
    """
    events: list[tuple[str, float]] = []

    async def fast():
        events.append(("fast_start", time.monotonic()))
        await asyncio.sleep(0.05)
        events.append(("fast_end", time.monotonic()))
        return "fast"

    fast.__name__ = "fast"
    fast.__qualname__ = "fast"

    async def slow():
        events.append(("slow_start", time.monotonic()))
        await asyncio.sleep(0.30)
        events.append(("slow_end", time.monotonic()))
        return "slow"

    slow.__name__ = "slow"
    slow.__qualname__ = "slow"

    class InstrumentedClient(unillm.AsyncUnify):
        async def generate(self, **kwargs):  # noqa: D401
            events.append(("generate", time.monotonic()))
            return await super().generate(**kwargs)

    # Manually constructing to support inheritance, but mirroring new_llm_client defaults
    client = InstrumentedClient(
        llm_config["model"],
        **{k: v for k, v in llm_config.items() if k != "model"},
    )

    _ = await start_async_tool_loop(
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
async def test_recovers_after_failure(llm_config):
    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
        client,
        message=(
            "Perform EXACTLY two tool calls and no more, in this order:\n"
            "1) Call `divide` with a=4 and b=0. This will fail with ZeroDivisionError. Do NOT retry or fix it.\n"
            "2) Then call `divide` with a=4 and b=2.\n"
            "After the second call returns, reply with the result only (plain text `2`). Do not schedule any additional tool calls,"
            " and do not attempt to re-run the failing divide(4,0)."
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
async def test_aborts_after_too_many_failures(llm_config):
    client = new_llm_client(**llm_config)

    with pytest.raises(RuntimeError):
        await start_async_tool_loop(
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
async def test_mixed_sync_async_tools(llm_config):
    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
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


# --------------------------------------------------------------------------- #
#  PRETTY PRINTING – tool returns pure JSON string                            #
# --------------------------------------------------------------------------- #
def emit_json() -> str:
    # Compact JSON string (no spaces/newlines). The loop should pretty‑print it.
    return '{"foo":1,"bar":[2,3],"baz":{"ok":true}}'


@pytest.mark.asyncio
@_handle_project
async def test_pretty_prints_json_string_tool_result(llm_config):
    client = new_llm_client(**llm_config)

    # Ask the model to call the tool once, then reply. Result should be pretty‑printed in the transcript.
    _ = await start_async_tool_loop(
        client,
        message=(
            "In your FIRST assistant message, request EXACTLY ONE tool call to `emit_json` "
            "with empty arguments {}; after receiving the tool reply, answer with the single word 'ok'."
        ),
        tools={"emit_json": emit_json},
        max_consecutive_failures=2,
        time_awareness=False,
    ).result()

    # Find the tool message for emit_json
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "emit_json"
    ]
    assert tool_msgs, "Expected a tool reply from emit_json"
    content = tool_msgs[0].get("content") or ""

    # It should now be a pretty‑printed JSON object string (not a quoted JSON string)
    assert isinstance(content, str)
    assert content.strip().startswith(
        "{",
    ), "Content should start with a JSON object, not a quoted string"
    assert not content.strip().startswith(
        '"{',
    ), "Content should not be a double‑encoded JSON string"
    assert "\n" in content, "Pretty‑printed JSON should contain newlines"
    assert (
        '"foo": 1' in content
    ), "Pretty‑printed JSON should include spaces after colons"


@pytest.mark.asyncio
@_handle_project
async def test_duplicate_tool_calls_are_optionally_pruned(
    llm_config,
) -> None:  # noqa: D401
    """Verify that duplicate tool calls are kept or pruned according to the flag."""

    log: list[str] = []

    async def echo(text: str) -> str:
        """Minimal echo tool used only to count invocations."""
        log.append(text)
        return text.upper()

    echo.__name__ = "echo"
    echo.__qualname__ = "echo"

    # Seed a transcript with an assistant message containing TWO identical parallel
    # tool calls. This removes dependency on model behavior for making parallel calls.
    seeded = [
        {
            "role": "user",
            "content": "Call echo twice with 'hello', then reply with a short sentence.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_dup_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hello"}'},
                },
                {
                    "id": "call_dup_2",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hello"}'},
                },
            ],
        },
    ]

    # ------------------------------------------------------------------ #
    # 1️⃣  duplicates SHOULD be executed when pruning is disabled
    # ------------------------------------------------------------------ #
    log.clear()
    client = new_llm_client(**llm_config)
    await start_async_tool_loop(
        client=client,
        message=seeded,
        tools={"echo": echo},
        prune_tool_duplicates=False,
    ).result()
    # Both duplicate calls should execute
    assert log == [
        "hello",
        "hello",
    ], "With prune_tool_duplicates=False, both duplicate tool calls should be invoked."
    # Verify transcript structure: both tool results appear after the seeded assistant turn
    tool_results = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "echo"
    ]
    assert len(tool_results) == 2, "Expected 2 tool results when pruning is disabled"

    # ------------------------------------------------------------------ #
    # 2️⃣  duplicates SHOULD be pruned when pruning is enabled
    # ------------------------------------------------------------------ #
    log.clear()
    client = new_llm_client(**llm_config)
    await start_async_tool_loop(
        client=client,
        message=seeded,
        tools={"echo": echo},
        prune_tool_duplicates=True,
    ).result()
    # Only one call should execute from the seeded turn (duplicate pruned)
    assert log[0] == "hello", "First tool call should execute"
    # The seeded turn should only produce 1 tool result due to pruning
    # (model may make additional calls in subsequent turns to compensate)

    # Find the first REAL assistant turn with tool calls (exclude synthetic check_status stubs)
    first_assistant_idx = None
    for i, m in enumerate(client.messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            if not _is_synthetic_check_status_stub(m):
                first_assistant_idx = i
                break
    assert first_assistant_idx is not None, "Expected a real assistant tool call"

    # Count REAL tool results (excluding synthetic check_status) that appear before the next REAL assistant turn
    tool_results_after_first = []
    for m in client.messages[first_assistant_idx + 1 :]:
        if m.get("role") == "tool" and m.get("name") == "echo":
            # Skip synthetic check_status tool messages
            if _is_synthetic_check_status_tool_msg(m):
                continue
            tool_results_after_first.append(m)
        elif m.get("role") == "assistant" and not _is_synthetic_check_status_stub(m):
            # Stop at the next real assistant turn
            break
    assert (
        len(tool_results_after_first) == 1
    ), "With prune_tool_duplicates=True, only 1 tool result should follow the seeded assistant turn"


# --------------------------------------------------------------------------- #
#  NO-TOOLS FLOWS                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_no_tools_with_system_message(llm_config) -> None:
    """
    Verify that the loop completes correctly when **no** tools are available
    and the conversation starts with a system prompt.

    The assistant must answer directly without any tool calls.
    """
    client = new_llm_client(**llm_config).set_system_message(
        "You are a helpful assistant.",
    )

    answer = await start_async_tool_loop(
        client,
        message="Just reply with a friendly greeting – no tools are available.",
        tools={},  # ← empty tool-kit
    ).result()

    # The assistant must answer directly and never insert any tool messages.
    assert answer.strip(), "Assistant reply should not be empty."
    assert count_tool_messages(client) == 0


@pytest.mark.asyncio
@_handle_project
async def test_no_tools_without_explicit_system_message(llm_config) -> None:
    """
    No tools, no explicit system message provided by the caller.

    The assistant must answer directly without any tool calls.
    """
    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
        client,
        message="Say hello back to me – there are no tools at all.",
        tools={},  # ← still an empty tool-kit
    ).result()

    assert answer.strip(), "Assistant reply should not be empty."
    assert count_tool_messages(client) == 0


# --------------------------------------------------------------------------- #
#  CONCURRENCY LIMIT – max_concurrent                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_max_concurrent_limit_is_obeyed(llm_config) -> None:  # noqa: D401
    """Ensure the per‑tool *runtime* concurrency cap is respected.

    We ask the model to run the same tool twice *in parallel*.  The limit
    (`max_concurrent=1`) should force the second call to wait until the
    first ends.  We tolerate the model making more than two invocations –
    what matters is that **no overlap > 1** happens.
    """

    events: list[tuple[str, float]] = []

    async def limited(label: str) -> str:
        events.append(("start", time.monotonic()))
        await asyncio.sleep(0.15)
        events.append(("end", time.monotonic()))
        return label.upper()

    limited.__name__ = "limited"
    limited.__qualname__ = "limited"

    tools = {"limited": ToolSpec(fn=limited, max_concurrent=1)}

    client = new_llm_client(**llm_config)

    # Kick off the interactive loop *without* awaiting the final result yet so
    # that we can synchronise on the **first** tool request and ensure all
    # timing events are captured in the correct order.

    handle = start_async_tool_loop(
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
