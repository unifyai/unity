"""Integration tests for context compression wiring in the async tool loop."""

from __future__ import annotations

import asyncio
import json

import pytest

import unity.common._async_tool.loop as _loop_mod
import unity.common._async_tool.context_compression as _cc_mod
import unity.common.async_tool_loop as _atl_mod
from unity.common._async_tool.context_compression import (
    CompressedMessage,
    CompressedMessages,
)
from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.llm_client import new_llm_client
from tests.helpers import _handle_project
from tests.async_helpers import (
    _wait_for_condition,
    _wait_for_tool_request,
    make_gated_async_tool,
)

pytestmark = pytest.mark.llm_call

_SYS = (
    "You are in a test. Follow the steps exactly:\n"
    "1. Call `add` with a=2, b=3.\n"
    "2. Call `add` with a=10, b=20.\n"
    "3. Report both results."
)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_threshold_trigger():
    _fire = False

    def trigger():
        nonlocal _fire
        _fire = True

    def reset():
        nonlocal _fire
        _fire = False

    def check(n_tokens, threshold, max_input_tokens):
        return _fire

    return trigger, reset, check


async def _mock_compress(messages, endpoint, **kwargs):
    return CompressedMessages(
        messages=[
            CompressedMessage(
                content=json.dumps(
                    {
                        "role": m.get("role", "user"),
                        "content": f"[c] {str(m.get('content', ''))[:40]}",
                    },
                ),
            )
            for m in messages
        ],
    )


def _make_add(trigger):
    _triggered = False

    def add(a: int, b: int) -> str:
        """Add two numbers."""
        nonlocal _triggered
        if not _triggered:
            _triggered = True
            trigger()
        return str(a + b)

    return add


def _msg_contains(client, snippet):
    async def _check():
        for m in client.messages or []:
            if snippet in str(m.get("content", "")):
                return True
        return False

    return _check


async def _wait_compression(handle, timeout=60):
    async def _done():
        return handle._compression.count >= 1

    await _wait_for_condition(_done, poll=0.1, timeout=timeout)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_handle_state_preserved_after_compression(llm_config, monkeypatch):
    """Events and queues on the handle must be the same objects after compression."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )

    orig_queue = handle._queue
    orig_pause = handle._pause_event
    orig_cancel = handle._cancel_event
    orig_stop = handle._stop_event

    result = await handle.result()

    assert handle._compression.count >= 1
    assert handle._queue is orig_queue
    assert handle._pause_event is orig_pause
    assert handle._cancel_event is orig_cancel
    assert handle._stop_event is orig_stop
    assert result is not None


@pytest.mark.asyncio
@_handle_project
async def test_nested_inner_compression_outer_unaffected(llm_config, monkeypatch):
    """Inner loop compresses; outer handle finishes normally."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

    inner_refs: dict = {}
    add = _make_add(trigger)

    async def spawn_inner():
        """Spawn an inner async tool loop and return its result."""
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(_SYS)
        h = start_async_tool_loop(
            inner_client,
            message="Add the numbers",
            tools={"add": add},
            timeout=120,
            max_parallel_tool_calls=1,
        )
        inner_refs["handle"] = h
        return await h.result()

    outer_client = new_llm_client(**llm_config)
    outer_client.set_system_message(
        "You are in a test. Call `spawn_inner`. Report what it returns.",
    )

    handle = start_async_tool_loop(
        outer_client,
        message="start",
        tools={"spawn_inner": spawn_inner},
        timeout=240,
    )

    result = await handle.result()
    assert result is not None

    inner_h = inner_refs.get("handle")
    assert inner_h is not None
    assert inner_h._compression.count >= 1


@pytest.mark.asyncio
@_handle_project
async def test_compression_blocked_while_tool_in_flight(llm_config, monkeypatch):
    """compress_context must not appear while another tool is still running."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

    add = _make_add(trigger)
    gate, raw_gated = make_gated_async_tool(return_value="gated-done")

    async def gated():
        """A long-running tool."""
        return await raw_gated()

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are in a test. Follow the steps exactly:\n"
        "1. Call `add` with a=2, b=3.\n"
        "2. Call `gated` with no arguments.\n"
        "3. Report both results.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="Go",
        tools={"add": add, "gated": gated},
        timeout=120,
        max_parallel_tool_calls=1,
    )

    result_task = asyncio.create_task(handle.result())

    await _wait_for_tool_request(client, "gated")
    await handle.interject("status check")

    await _wait_for_condition(
        _msg_contains(client, "cannot start new tools"),
        poll=0.1,
        timeout=60,
    )

    gate.set()

    result = await result_task
    assert result is not None
    assert handle._compression.count >= 1


@pytest.mark.asyncio
@_handle_project
async def test_no_new_tools_when_threshold_triggered(llm_config, monkeypatch):
    """When threshold fires with no pending tools, only compress_context is callable."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )

    result = await handle.result()
    assert handle._compression.count >= 1

    archived = handle._compression.raw_archives[0]
    threshold_idx = next(
        (
            i
            for i, m in enumerate(archived)
            if "must call" in str(m.get("content", ""))
            and "compress_context" in str(m.get("content", ""))
        ),
        None,
    )
    assert threshold_idx is not None, "Expected 'must call compress_context' in archive"

    for m in archived[threshold_idx:]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                name = tc.get("function", {}).get("name", "")
                assert (
                    name == "compress_context"
                ), f"Expected only compress_context after threshold, got {name}"

    assert result is not None


@pytest.mark.asyncio
@_handle_project
async def test_pause_carries_over_during_compression(llm_config, monkeypatch):
    """Pause set during the compression window is respected by the new loop."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    handle_ref: dict = {}

    async def _compress_with_pause(messages, endpoint, **kwargs):
        h = handle_ref.get("handle")
        if h:
            h._pause_event.clear()
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_with_pause)

    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )
    handle_ref["handle"] = handle

    result_task = asyncio.create_task(handle.result())

    await _wait_compression(handle)

    assert not handle._pause_event.is_set(), "New loop should start paused"

    await handle.resume()
    result = await result_task
    assert result is not None


@pytest.mark.asyncio
@_handle_project
async def test_stop_carries_over_during_compression(llm_config, monkeypatch):
    """Stop set during the compression window causes the new loop to exit."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    handle_ref: dict = {}

    async def _compress_with_stop(messages, endpoint, **kwargs):
        h = handle_ref.get("handle")
        if h:
            h._stop_event.set()
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_with_stop)

    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )
    handle_ref["handle"] = handle

    result = await handle.result()
    assert result is not None
    assert handle._compression.count >= 1


@pytest.mark.asyncio
@_handle_project
async def test_interjection_carries_over_during_compression(llm_config, monkeypatch):
    """Interjection queued during compression is received by the new loop."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    handle_ref: dict = {}
    interjection_text = "URGENT: also compute 10 + 20"

    async def _compress_with_interjection(messages, endpoint, **kwargs):
        h = handle_ref.get("handle")
        if h:
            await h._queue.put(interjection_text)
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_with_interjection)

    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are in a test. Follow the steps exactly:\n"
        "1. Call `add` with a=2, b=3.\n"
        "2. Call `add` with a=10, b=20.\n"
        "3. Report both results.\n"
        "If you receive additional instructions, follow them.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )
    handle_ref["handle"] = handle

    result_task = asyncio.create_task(handle.result())

    await _wait_compression(handle)

    await _wait_for_condition(
        _msg_contains(client, interjection_text),
        poll=0.1,
        timeout=60,
    )

    result = await result_task
    assert result is not None


@pytest.mark.asyncio
@_handle_project
async def test_enable_compression_false(llm_config, monkeypatch):
    """enable_compression=False suppresses all compression machinery."""
    monkeypatch.setattr(_loop_mod, "context_over_threshold", lambda *a, **kw: True)

    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
        enable_compression=False,
    )

    result = await handle.result()
    assert result is not None
    assert handle._compression.count == 0


@pytest.mark.asyncio
@_handle_project
async def test_compression_failure_returns_gracefully(llm_config, monkeypatch):
    """When compress_and_rebuild raises, result() returns gracefully."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_atl_mod, "compress_and_rebuild", _boom)

    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )

    result = await handle.result()
    assert isinstance(result, str)
    assert handle._compression.count == 0
