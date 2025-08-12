import os
import asyncio
import pytest
import unify
from typing import Dict, Callable

from unity.common.llm_helpers import start_async_tool_use_loop
from tests.helpers import _get_unity_test_env_var


class DummyAsyncUnify:
    """Minimal stub to satisfy the loop in unit-tests."""

    def __init__(self, *, delay: float = 0.0):
        self.messages = []
        self._delay = delay

    def append_messages(self, msgs):
        self.messages.extend(msgs)

    async def generate(self, *a, **_):
        if self._delay:
            await asyncio.sleep(self._delay)
        # The real LLM JSON always includes 'tool_calls'; we mirror that.
        msg = {"role": "assistant", "content": "done", "tool_calls": []}
        self.messages.append(msg)
        return msg

    @property
    def system_message(self) -> str:
        return ""


# ── 1. max_steps safeguard ────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_max_steps_exceeded():
    client = DummyAsyncUnify()
    # The conversation will contain at least USER + ASSISTANT = 2 messages,
    # so max_steps=1 must raise.
    handle = start_async_tool_use_loop(
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
async def test_timeout_exceeded():
    client = DummyAsyncUnify(delay=0.2)  # ensure > timeout
    handle = start_async_tool_use_loop(
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
class _ToolCallingUnify(DummyAsyncUnify):
    """
    On its first call the stub requests execution of `long_tool`, creating a
    *pending* tool task in the outer loop.  Subsequent calls are inert.
    """

    def __init__(self):
        super().__init__()
        self._step = 0

    async def generate(self, *a, **_):
        if self._step == 0:
            self._step += 1
            msg = {
                "role": "assistant",
                "content": "running tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "long_tool",
                            "arguments": '{"seconds": 5}',
                        },
                    },
                ],
            }
        else:
            self._step += 1
            msg = {"role": "assistant", "content": "noop", "tool_calls": []}
        self.messages.append(msg)
        return msg


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
    client = _ToolCallingUnify()
    handle = start_async_tool_use_loop(
        client,
        message="go",
        tools={"long_tool": _make_long_tool(cancel_flag)},
        timeout=0.05,  # tiny → timeout reached quickly
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
    client = _ToolCallingUnify()
    handle = start_async_tool_use_loop(
        client,
        message="go",
        tools={"long_tool": _make_long_tool(cancel_flag)},
        max_steps=4,  # USER + ASSISTANT + TOOL-placeholder = 3
        timeout=5,
        raise_on_limit=False,
    )
    result = await handle.result()
    assert "Terminating early" in result
    assert cancel_flag.get("cancelled", False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. tool_policy behaviour
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("UNIFY_MODEL", "o4-mini@openai")


def new_client() -> unify.AsyncUnify:
    """
    Return a fresh client *with its own conversation state* so that tests do
    not interfere with one another.
    """
    return unify.AsyncUnify(
        MODEL_NAME,
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )


@pytest.mark.asyncio
async def test_default_policy_returns_immediately():
    """With ``tool_policy=None`` the loop should accept the LLM's first
    answer (no tools) and finish without touching *any* tools."""

    async def noop_tool():  # pragma: no cover – should never be called
        raise RuntimeError("tool should not have been invoked")

    client = new_client()
    handle = start_async_tool_use_loop(
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
    handle = start_async_tool_use_loop(
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
    handle = start_async_tool_use_loop(
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
    handle = start_async_tool_use_loop(
        client,
        "You are part of a test. Use the tool whenever required but stop when no longer forced.",
        {"counting_tool": counting_tool},
        tool_policy=first_two_required,
    )
    await handle.result()

    assert counter["n"] == 2  # one call on step 0 and one on step 1
