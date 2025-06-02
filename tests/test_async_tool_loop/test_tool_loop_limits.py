import asyncio
import pytest

from unity.common.llm_helpers import start_async_tool_use_loop


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
        max_steps=3,  # USER + ASSISTANT + TOOL-placeholder = 3
        timeout=5,
        raise_on_limit=False,
    )
    result = await handle.result()
    assert "Terminating early" in result
    assert cancel_flag.get("cancelled", False)
