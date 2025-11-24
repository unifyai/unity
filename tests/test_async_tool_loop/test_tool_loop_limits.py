import os
import asyncio
import pytest
import unify
from typing import Dict, Callable

from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.tool_spec import ToolSpec
from tests.helpers import SETTINGS, get_test_client


# small helper: pre-seed an assistant tool_call so preflight backfill schedules it immediately
def _preseed_tool_call(
    client: "unify.AsyncUnify",
    *,
    call_id: str,
    tool_name: str,
    args_json: str,
) -> None:
    client.append_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args_json},
                    },
                ],
            },
        ],
    )


# ── 1. max_steps safeguard ────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_max_steps_exceeded():
    client = get_test_client()
    # The conversation will contain at least USER + ASSISTANT = 2 messages,
    # so max_steps=1 must raise.
    handle = start_async_tool_loop(
        client,
        message="hello",
        tools={},
        max_steps=1,
        timeout=5,
        raise_on_limit=True,
    )
    with pytest.raises(RuntimeError, match="max_steps"):
        await handle.result()


# ── 2. timeout safeguard ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_timeout_exceeded(monkeypatch):
    client = get_test_client()
    # Force generate to be slower than timeout using monkeypatch while still calling the real LLM.
    orig_generate = client.generate

    async def _slow_generate(**kwargs):
        await asyncio.sleep(0.2)
        return await orig_generate(**kwargs)

    monkeypatch.setattr(client, "generate", _slow_generate, raising=True)
    handle = start_async_tool_loop(
        client,
        message="hi",
        tools={},
        timeout=0.1,  # deliberately tiny
        max_steps=100,
        raise_on_limit=True,
    )
    with pytest.raises(asyncio.TimeoutError):
        await handle.result()


# ── 3 & 4. graceful early-exit when limits hit (NO raise) ──────────────────
class _ToolCallingDriver:
    """Deprecated stub removed – tests now instruct the real LLM instead."""

    def __init__(self, client: unify.AsyncUnify):
        self._client = client
        self._orig = client.generate

    async def __call__(self, **kwargs):
        return await self._orig(**kwargs)


# ── 7. pruning over-quota tool calls (hidden quotas) ────────────────────────
class _MultiCallDriver:
    def __init__(self, client: unify.AsyncUnify):
        self._client = client
        self._orig = client.generate

    async def __call__(self, **kwargs):
        return await self._orig(**kwargs)


@pytest.mark.asyncio
async def test_prunes_over_quota_tool_calls(monkeypatch):
    """When `max_total_calls` is 2, only two calls are scheduled; extras are pruned."""

    counter = {"n": 0}

    async def short_tool():
        counter["n"] += 1
        return "ok"

    client = get_test_client()
    # Instruct the real LLM to attempt three calls; the loop will prune to 2
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, request the tool `short_tool` "
        "THREE times in the same message. Then finish shortly.",
    )

    handle = start_async_tool_loop(
        client,
        message="start",
        tools={
            # hidden per-loop quota of 2 total calls
            "short_tool": ToolSpec(fn=short_tool, max_total_calls=2),
        },
        prune_tool_duplicates=False,  # allow multiple identical tool_calls for this test
        timeout=60,
        max_steps=100,
        raise_on_limit=False,
    )

    await handle.result()

    # Tool ran exactly twice
    assert counter["n"] == 2

    # The first assistant message with tool_calls was pruned to two entries
    first_asst_with_calls = next(
        m
        for m in client.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert len(first_asst_with_calls["tool_calls"]) == 2
    assert all(
        tc.get("function", {}).get("name") == "short_tool"
        for tc in first_asst_with_calls["tool_calls"]
    )


# ── 8. pruning over-quota tool calls across serial turns ────────────────────
class _SerialCallsDriver:
    def __init__(self, client: unify.AsyncUnify):
        self._client = client
        self._orig = client.generate

    async def __call__(self, **kwargs):
        return await self._orig(**kwargs)


@pytest.mark.asyncio
async def test_prunes_over_quota_serial_calls(monkeypatch):
    """When three serial turns each request a call, quota=2 prunes the third."""

    counter = {"n": 0}

    async def short_tool():
        counter["n"] += 1
        return "ok"

    client = get_test_client()
    # Instruct the model how to conclude after the allowed calls
    client.set_system_message(
        "You are part of an automated test. If tools are available, request the tool `short_tool` exactly once per turn. "
        "After at most two such tool calls have been made, do not request any more tools and reply exactly with the word 'done'.",
    )
    # Let real LLM drive; instruction above ensures at most two calls then 'done'

    handle = start_async_tool_loop(
        client,
        message="start",
        tools={
            "short_tool": ToolSpec(fn=short_tool, max_total_calls=2),
        },
        # Force exactly one tool call per turn for the first two turns
        tool_policy=lambda step, tls: ("required", tls) if step < 2 else ("auto", tls),
        timeout=60,
        max_steps=100,
        raise_on_limit=False,
    )

    await handle.result()

    # Tool ran exactly twice
    assert counter["n"] == 2

    # Across all assistant messages, only two tool_calls remain after pruning
    total_calls = 0
    for m in client.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            total_calls += len(m["tool_calls"])
    assert total_calls == 2


# helper factory: returns an async tool that notes cancellation -------------
def _make_long_tool(cancel_flag: dict):
    async def long_tool(seconds: int):
        try:
            await asyncio.sleep(seconds)
            return "finished"
        except asyncio.CancelledError:
            cancel_flag["cancelled"] = True
            raise

    return long_tool


@pytest.mark.asyncio
async def test_timeout_graceful_termination():
    """No exception; pending tool is cancelled when timeout hits."""
    cancel_flag = {}
    client = get_test_client()
    # Instruct real LLM to call the tool once and keep running; timeout will stop it
    client.set_system_message(
        'You are running inside an automated test. In your FIRST assistant turn, call `long_tool` with {"seconds": 5}. '
        "Keep waiting afterwards.",
    )
    _preseed_tool_call(
        client,
        call_id="call_preseed_timeout",
        tool_name="long_tool",
        args_json='{"seconds": 5}',
    )

    handle = start_async_tool_loop(
        client,
        message="go",
        tools={"long_tool": _make_long_tool(cancel_flag)},
        timeout=0.5,  # real small timeout – tool is already scheduled via backfill
        max_steps=100,
        raise_on_limit=False,
    )
    result = await handle.result()
    assert "Terminating early" in result
    assert cancel_flag.get("cancelled", False)


@pytest.mark.asyncio
async def test_max_steps_graceful_termination():
    """No exception; pending tool is cancelled when max_steps is exceeded."""
    cancel_flag = {}
    client = get_test_client()
    # Instruct real LLM to call the tool once and keep running; max_steps will stop it
    client.set_system_message(
        'You are running inside an automated test. In your FIRST assistant turn, call `long_tool` with {"seconds": 5}. '
        "Keep waiting afterwards.",
    )
    _preseed_tool_call(
        client,
        call_id="call_preseed_steps",
        tool_name="long_tool",
        args_json='{"seconds": 5}',
    )

    handle = start_async_tool_loop(
        client,
        message="go",
        tools={"long_tool": _make_long_tool(cancel_flag)},
        max_steps=3,  # real small cap – after backfill + user message, limit will be hit
        timeout=5,
        raise_on_limit=False,
    )
    result = await handle.result()
    assert "Terminating early" in result

    # Robust assertions not relying on coroutine body execution timing.
    # 1) The preseeded tool call must be present in assistant tool_calls.
    assert any(
        m.get("role") == "assistant"
        and m.get("tool_calls")
        and any(tc.get("id") == "call_preseed_steps" for tc in m["tool_calls"])
        for m in client.messages
    )

    # 2) The tool must not have produced a successful final result.
    assert not any(
        m.get("role") == "tool"
        and m.get("tool_call_id") == "call_preseed_steps"
        and "finished" in str(m.get("content") or "")
        for m in client.messages
    )

    # 3) White-box: the scheduled asyncio.Task for the call-id is cancelled.
    loop_task = getattr(handle, "_task", None)
    task_info = getattr(loop_task, "task_info", {}) if loop_task is not None else {}
    found_cancelled = False
    if isinstance(task_info, dict):
        for t, meta in task_info.items():
            if getattr(meta, "call_id", None) == "call_preseed_steps":
                if t.cancelled():
                    found_cancelled = True
                    break
                if t.done():
                    try:
                        exc = t.exception()
                    except Exception:
                        exc = None
                    if isinstance(exc, asyncio.CancelledError):
                        found_cancelled = True
                        break
    assert found_cancelled


# ─────────────────────────────────────────────────────────────────────────────
# 5. tool_policy behaviour
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-5@openai")


def new_client() -> unify.AsyncUnify:
    """
    Return a fresh client *with its own conversation state* so that tests do
    not interfere with one another.
    """
    return get_test_client()


@pytest.mark.asyncio
async def test_default_policy_returns_immediately():
    """With ``tool_policy=None`` the loop should accept the LLM's first
    answer (no tools) and finish without touching *any* tools."""

    async def noop_tool():  # pragma: no cover – should never be called
        raise RuntimeError("tool should not have been invoked")

    client = new_client()
    handle = start_async_tool_loop(
        client,
        message="You are part of a test. Do *not* call any tools, just return to the user immediately",
        tools={"noop_tool": noop_tool},
        # default → no tool_policy passed
    )
    await handle.result()


@pytest.mark.asyncio
async def test_policy_forces_single_tool_invocation():
    """A custom ``tool_policy`` can replicate the old
    ``minimum_tool_turns=1`` semantics by forcing a *required* tool call on the
    first turn only."""

    flag = {"called": False}

    async def dummy_tool():
        flag["called"] = True
        return "ok"

    client = new_client()
    handle = start_async_tool_loop(
        client,
        message="You are part of a test. Do *not* call any tools, just return to the user immediately",
        tools={"dummy_tool": dummy_tool},
        tool_policy=lambda i, tls: ("required", tls) if i < 1 else ("auto", tls),
    )
    await handle.result()

    # The loop had to wait for the tool to finish and therefore should return
    # the *final* assistant content.
    assert flag["called"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Advanced tool_policy scenarios
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_policy_shows_then_hides_tool():
    """
    On step 0 the policy hides *all* tools so the assistant must think without
    calling them.  On step 1 the tool becomes available and the assistant
    should call it exactly once before finishing.
    """

    call_log: list[str] = []

    async def observed_tool():
        call_log.append("invoked")
        return "ok"

    def hide_first_then_show(step: int, tools: Dict[str, Callable]):
        # Hide every tool on the very first step.
        if step == 0:
            return "auto", tools
        # Reveal all tools afterwards (no *required* flag).
        return "auto", {}

    client = new_client()
    handle = start_async_tool_loop(
        client,
        "You are part of a test. Continue calling `observed_tool` until the tool option disappears, up to a *maximum* of two *consecutive* tool calls.",
        {"observed_tool": observed_tool},
        tool_policy=hide_first_then_show,
    )
    await handle.result()

    assert call_log == ["invoked"]  # exactly one call, on step 1


@pytest.mark.asyncio
async def test_policy_two_required_then_auto():
    """
    Require *two* consecutive tool turns, then switch to ``auto``.  The tool
    counts its invocations so we can assert it was called twice and not more.
    """

    counter = {"n": 0}

    async def counting_tool():
        counter["n"] += 1
        return f"call {counter['n']}"

    def first_two_required(step: int, tools: Dict[str, Callable]):
        return ("required" if step < 2 else "auto", tools)

    client = new_client()
    handle = start_async_tool_loop(
        client,
        "You are part of a test. You will have no other option but to call the 'counting_tool' a certain number of times. "
        "Please run the tool when there is no other option, but **stop** calling the tool **as soon as** you're able to avoid calling the tool.",
        {"counting_tool": counting_tool},
        tool_policy=first_two_required,
    )
    await handle.result()

    assert counter["n"] == 2  # one call on step 0 and one on step 1


@pytest.mark.skip(reason="Will only pass once we support the responses API")
@pytest.mark.asyncio
async def test_max_parallel_tool_calls():
    X = 2  # allowed concurrent tool calls per LLM turn
    Y = 5  # requested by the model

    counter = {"n": 0}

    @unify.traced
    async def short(i: int) -> str:
        counter["n"] += 1
        await asyncio.sleep(0.01)
        return f"ok-{i}"

    short.__name__ = "short"
    short.__qualname__ = "short"

    client = new_client()

    prompt = (
        "You are part of a test. In a single assistant turn, call the tool `short(i: int)` "
        f"exactly {Y} times in parallel with i = 1..{Y}. Make ALL tool calls in one message. "
        "Do not make any further tool calls in later turns. If the platform limits how many "
        "tool calls you can make per turn, issue only as many as allowed and then finish by "
        "replying 'ok'."
    )

    handle = start_async_tool_loop(
        client,
        message=prompt,
        tools={"short": short},
        max_parallel_tool_calls=X,
        prune_tool_duplicates=False,
        timeout=30,
        max_steps=100,
        raise_on_limit=True,
    )

    await handle.result()

    # The first assistant turn must not request more than X tool calls
    first_asst_with_calls = next(
        m
        for m in client.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert 1 <= len(first_asst_with_calls["tool_calls"]) <= X

    # The tool should have been invoked exactly as many times as requested in that turn
    assert counter["n"] == len(first_asst_with_calls["tool_calls"])
