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
    )
    with pytest.raises(asyncio.TimeoutError):
        await handle.result()
