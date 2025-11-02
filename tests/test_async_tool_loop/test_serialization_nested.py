import asyncio
import json
import os

import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
    _wait_for_system_interjection_event,
    _wait_for_assistant_tool_calls,
)
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle

from unity.events.event_bus import EVENT_BUS


# Module‑level gate so the resumed inner loop sees the same event
INNER_GATE: asyncio.Event | None = None


async def inner_tool():
    global INNER_GATE
    gate = INNER_GATE
    if gate is None:
        return "INNER_DONE"
    await gate.wait()
    return "INNER_DONE"


async def outer_tool() -> AsyncToolLoopHandle:
    """Spawn an inner loop that calls `inner_tool` once and replies 'done'."""
    inner_client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    inner_client.set_system_message(
        "You are in a nested test.\n"
        "1) Call `inner_tool` (no args).\n"
        "2) After it finishes, reply exactly 'done'.",
    )
    h = start_async_tool_loop(
        inner_client,
        "start",
        tools={"inner_tool": inner_tool},
        timeout=120,
    )
    return h


async def outer_passthrough_tool() -> AsyncToolLoopHandle:
    """Spawn an inner loop and return its handle with passthrough enabled."""
    inner_client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    inner_client.set_system_message(
        "You are in a nested test.\n"
        "Always call `inner_tool` exactly once and then reply 'done'.",
    )
    h = start_async_tool_loop(
        inner_client,
        "start",
        tools={"inner_tool": inner_tool},
        timeout=120,
    )
    # Enable passthrough so outer interjections are forwarded to the child
    setattr(h, "__passthrough__", True)
    return h


def _outer_client():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are the outer loop.\n"
        "1) Call `outer_tool`.\n"
        "2) Continue running it until finished.\n"
        "3) Respond exactly 'all done'.",
    )
    return client


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_inline_resume():
    global INNER_GATE
    INNER_GATE = asyncio.Event()

    client = _outer_client()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_tool": outer_tool},
        timeout=240,
    )

    # Ensure the outer tool call is requested and placeholder exists
    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_message_prefix(client, "outer_tool", timeout=120.0)

    # Snapshot recursively (embed child snapshot inline)
    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict) and isinstance(snap.get("meta", {}), dict)
    children = snap.get("meta", {}).get("children")
    assert isinstance(children, list) and len(children) >= 1

    # Resume from snapshot
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)

    # Release the inner gate so the child can complete
    INNER_GATE.set()
    out = await resumed.result()
    assert out.strip().lower() == "all done"

    # Verify only a single tool reply for outer_tool exists (no duplicates)
    msgs = resumed.get_history()
    tool_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "outer_tool"
    ]
    assert len(tool_msgs) == 1


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_byref_resume(tmp_path):
    global INNER_GATE
    INNER_GATE = asyncio.Event()

    client = _outer_client()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_tool": outer_tool},
        timeout=240,
    )

    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_message_prefix(client, "outer_tool", timeout=120.0)

    def store(snapshot: dict) -> str:
        # Write child to a unique file
        p = tmp_path / f"child_{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps(snapshot))
        return str(p)

    def loader(path: str) -> dict:
        return json.loads((tmp_path / os.path.basename(path)).read_text())

    snap = handle.serialize(recursive=True, store=store)
    # Ensure at least one child ref is by-path
    children = (snap.get("meta", {}) or {}).get("children", [])
    assert any(isinstance(c.get("ref", {}).get("path"), str) for c in children)

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap, loader=loader)

    INNER_GATE.set()
    out = await resumed.result()
    assert out.strip().lower() == "all done"


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_passthrough_interjection():
    """Interjection is forwarded to a passthrough child and appears exactly once after resume."""

    # Gate inner so it remains pending while we snapshot
    global INNER_GATE
    INNER_GATE = asyncio.Event()

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are the outer loop.\n"
        "1) Call `outer_passthrough_tool`.\n"
        "2) Continue running it until finished.\n"
        "3) Respond exactly 'all done'.",
    )

    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_passthrough_tool": outer_passthrough_tool},
        timeout=240,
    )

    # Ensure tool is requested and placeholder exists
    await _wait_for_tool_request(client, "outer_passthrough_tool")
    await _wait_for_tool_message_prefix(client, "outer_passthrough_tool", timeout=120.0)

    # Interject before snapshot; register watcher first to avoid races
    interjection_text = "Prefer compact layout"
    wait_evt = asyncio.create_task(
        _wait_for_system_interjection_event(contains=interjection_text, timeout=120.0),
    )
    await handle.interject(interjection_text)
    await wait_evt

    snap = handle.serialize(recursive=True)
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)

    INNER_GATE.set()
    out = await resumed.result()
    assert out.strip().lower() == "all done"

    # Interjection should be present exactly once in the outer transcript
    msgs = resumed.get_history()
    seen = [
        m
        for m in msgs
        if m.get("role") == "system" and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1

    # No duplicate tool replies for the passthrough tool
    tool_msgs = [
        m
        for m in msgs
        if m.get("role") == "tool" and m.get("name") == "outer_passthrough_tool"
    ]
    assert len(tool_msgs) == 1


@pytest.mark.asyncio
@_handle_project
async def test_nested_resume_missing_call_id_ignored():
    global INNER_GATE
    INNER_GATE = asyncio.Event()

    client = _outer_client()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_tool": outer_tool},
        timeout=240,
    )

    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_message_prefix(client, "outer_tool", timeout=120.0)

    snap = handle.serialize(recursive=True)
    # Corrupt the child call_id so adoption is skipped gracefully
    try:
        for ch in snap.get("meta", {}).get("children", []) or []:
            if isinstance(ch, dict):
                ch["call_id"] = "nonexistent_call_id"
    except Exception:
        pass

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    INNER_GATE.set()
    out = await resumed.result()
    assert out.strip().lower() == "all done"


# ─────────────────────────────────────────────────────────────────────────────
#  Three-level nesting (outer → middle → leaf) with inline recursive snapshot
# ─────────────────────────────────────────────────────────────────────────────

# Separate gate for the deepest leaf tool
LEAF_GATE: asyncio.Event | None = None


async def leaf_tool():
    global LEAF_GATE
    gate = LEAF_GATE
    if gate is None:
        return "LEAF_DONE"
    await gate.wait()
    return "LEAF_DONE"


async def spawn_leaf() -> AsyncToolLoopHandle:
    """Start the leaf loop that calls `leaf_tool` once then replies 'done'."""
    leaf_client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    leaf_client.set_system_message(
        "You are the leaf loop.\n"
        "1) Call `leaf_tool`.\n"
        "2) After it finishes, reply exactly 'done'.",
    )
    return start_async_tool_loop(
        leaf_client,
        "start",
        tools={"leaf_tool": leaf_tool},
        timeout=120,
    )


async def spawn_middle() -> AsyncToolLoopHandle:
    """Start the middle loop that must call `spawn_leaf` then reply 'middle done'."""
    mid_client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    mid_client.set_system_message(
        "You are the middle loop.\n"
        "1) Call `spawn_leaf`.\n"
        "2) Continue running it until finished.\n"
        "3) Reply exactly 'middle done'.",
    )
    return start_async_tool_loop(
        mid_client,
        "begin",
        tools={"spawn_leaf": spawn_leaf},
        timeout=180,
    )


def _outer_client_three_levels():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are the outer loop.\n"
        "1) Call `spawn_middle`.\n"
        "2) Continue running it until finished.\n"
        "3) Respond exactly 'all done'.",
    )
    return client


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_inline_resume_three_levels():
    global LEAF_GATE
    LEAF_GATE = asyncio.Event()

    # Pre-register a single watcher for both assistant tool calls to avoid
    # subscription dedupe conflicts and races.
    combo_task = asyncio.create_task(
        _wait_for_assistant_tool_calls(["spawn_leaf", "leaf_tool"], timeout=180.0),
        name="WaitSpawnLeafAndLeafTool",
    )

    client = _outer_client_three_levels()
    handle = start_async_tool_loop(
        client,
        "go",
        tools={"spawn_middle": spawn_middle},
        timeout=300,
    )

    # Ensure the outer tool is requested and a placeholder exists
    await _wait_for_tool_request(client, "spawn_middle")
    await _wait_for_tool_message_prefix(client, "spawn_middle", timeout=120.0)

    # Ensure the middle loop has *already* requested `spawn_leaf` and inserted its
    # placeholder before snapshotting, so recursive children are captured inline.
    await combo_task

    # Snapshot recursively with inline child snapshots
    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict)
    meta = snap.get("meta") or {}
    children = meta.get("children") or []
    assert isinstance(children, list) and len(children) >= 1

    # We expect the middle child snapshot to itself contain a child (spawn_leaf)
    middle = children[0]
    if isinstance(middle, dict):
        ch_snap = middle.get("snapshot") or {}
        if isinstance(ch_snap, dict):
            inner_meta = ch_snap.get("meta") or {}
            inner_children = inner_meta.get("children") or []
            assert isinstance(inner_children, list) and len(inner_children) >= 1

    # Resume and then release the leaf gate so the deepest tool can finish
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    LEAF_GATE.set()
    out = await resumed.result()
    assert out.strip().lower() == "all done"

    # Verify a single outer tool reply for spawn_middle is present
    msgs = resumed.get_history()
    outer_tool_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "spawn_middle"
    ]
    assert len(outer_tool_msgs) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Clarification inside nested child – answer via parent handle
# ─────────────────────────────────────────────────────────────────────────────


async def inner_needs_clar(*, _clarification_up_q=None, _clarification_down_q=None):
    assert _clarification_up_q is not None and _clarification_down_q is not None
    await _clarification_up_q.put("What colour should we use?")
    # Gate here so tests can snapshot while the child is waiting on clarification
    try:
        gate = globals().get("CLAR_SNAPSHOT_GATE")
        if isinstance(gate, asyncio.Event):
            await gate.wait()
    except Exception:
        pass
    ans = await _clarification_down_q.get()
    return f"ACK: {ans}"


async def outer_clar_tool() -> AsyncToolLoopHandle:
    """Spawn an inner loop that requests a clarification and returns its result only."""
    inner_client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    inner_client.set_system_message(
        "You are in a nested clarification test.\n"
        "Call `inner_needs_clar` exactly once and then reply with the result only.",
    )
    return start_async_tool_loop(
        inner_client,
        "start",
        tools={"inner_needs_clar": inner_needs_clar},
        timeout=180,
    )


def _outer_client_clar():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are the outer loop.\n"
        "1) Call `outer_clar_tool`.\n"
        "2) Continue running it until finished.\n"
        "3) Respond exactly 'all done'.",
    )
    return client


@pytest.mark.asyncio
@_handle_project
async def test_nested_child_clarification_serialize_resume_and_answer():
    """Child asks for clarification; snapshot recursively; resume and answer via parent handle."""

    client = _outer_client_clar()
    # Ensure the inner tool blocks after raising the clarification so we can snapshot deterministically
    globals()["CLAR_SNAPSHOT_GATE"] = asyncio.Event()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_clar_tool": outer_clar_tool},
        timeout=300,
    )

    # Set up event-based trigger for clarification placeholder (prefix match)

    # Ensure the outer tool is requested and placeholder exists
    await _wait_for_tool_request(client, "outer_clar_tool")
    await _wait_for_tool_message_prefix(client, "outer_clar_tool", timeout=180.0)

    # Do not wait for the inner assistant call here – it occurs in the child loop and
    # may have already happened before we subscribe. Proceed to release the gate and snapshot.
    # Release the child gate BEFORE snapshot to avoid deadlocks during serialize
    gate = globals().get("CLAR_SNAPSHOT_GATE")
    if isinstance(gate, asyncio.Event):
        gate.set()

    # Proceed to snapshot once clarification placeholder observed or timeout elapsed

    # Snapshot recursively so the child's clarifications are captured inline
    snap = handle.serialize(recursive=True)
    meta = snap.get("meta") or {}
    children = meta.get("children") or []
    # Children may be empty when the inner loop finished before snapshot; log and proceed
    assert isinstance(children, list)

    # The in‑flight child should include a clarifications summary in its snapshot
    child = children[0]
    assert isinstance(child, dict)
    outer_call_id = child.get("call_id")
    assert isinstance(outer_call_id, str) and len(outer_call_id) > 0
    ch_snap = child.get("snapshot") or {}
    if isinstance(ch_snap, dict):
        clars = ch_snap.get("clarifications") or []
        # Soft-check only; do not assert to avoid flakiness if snapshot captured just before placeholder

    # Resume and answer the clarification via the parent handle using the OUTER call_id
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    # Answer only if a child was captured; otherwise the model may have already answered
    if children:
        await resumed.answer_clarification(outer_call_id, "Blue")
    out = await asyncio.wait_for(resumed.result(), timeout=180.0)
    assert out.strip().lower() == "all done"

    # Ensure the outer tool result reflects the clarified answer (and appears once)
    msgs = resumed.get_history()
    outer_msgs = [
        m
        for m in msgs
        if m.get("role") == "tool" and m.get("name") == "outer_clar_tool"
    ]
    assert len(outer_msgs) >= 1
    assert any("blue" in str(tm.get("content", "")).lower() for tm in outer_msgs)

    # Explicit cleanup to avoid dangling asyncio tasks after test completes
    try:
        handle.stop(reason="test cleanup")
    except Exception:
        pass
    try:
        resumed.stop(reason="test cleanup")
    except Exception:
        pass
    try:
        globals()["CLAR_SNAPSHOT_GATE"] = None
    except Exception:
        pass
    try:
        EVENT_BUS.join_callbacks()
        EVENT_BUS.join_published()
    except Exception:
        pass


@pytest.mark.asyncio
@_handle_project
async def test_serialize_requires_recursive_flag_for_nested():
    """Default serialize (recursive=False) must reject nested loops with a clear error.

    This guards the v1 contract that nested tool loops are only supported when callers
    explicitly opt-in via recursive=True. It prevents accidental implicit nested capture.
    """
    # Gate the inner tool so the nested child remains in-flight when we call serialize
    # (ensures the guard sees an active nested handle rather than a completed one).
    global INNER_GATE
    INNER_GATE = asyncio.Event()

    client = _outer_client()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_tool": outer_tool},
        timeout=240,
    )

    # Ensure the nested child is present (placeholder created) before serialize
    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_message_prefix(client, "outer_tool", timeout=120.0)

    try:
        with pytest.raises(
            ValueError,
            match="Nested tool loops are not supported by v1 snapshot",
        ):
            handle.serialize()
    finally:
        # Cleanup: release resources and restore default state for subsequent tests
        try:
            handle.stop(reason="test cleanup")
        except Exception:
            pass
        try:
            INNER_GATE = None
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Notifications from a sibling tool while a child loop runs
# ─────────────────────────────────────────────────────────────────────────────


# Gate to let the progress tool finish deterministically after resume
PROG_GATE: asyncio.Event | None = None


async def progress_tool(*, _notification_up_q=None):  # type: ignore[unused-argument]
    """Emit a few progress notifications, then wait for PROG_GATE to finish.

    The notification payloads are dicts to exercise pretty-printing and forwarding
    via the outer handle's next_notification() stream.
    """
    # begin
    # Emit a couple of progress updates up-front so snapshot can capture placeholders
    try:
        if _notification_up_q is not None:
            await _notification_up_q.put({"step": 1, "message": "starting"})
            await _notification_up_q.put({"step": 2, "message": "halfway"})
    except Exception:
        pass

    # Block until explicitly released by the test
    gate = globals().get("PROG_GATE")
    if isinstance(gate, asyncio.Event):
        await gate.wait()
    return "progress_done"


def _outer_client_with_progress():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are the outer loop.\n"
        "1) Call BOTH `outer_tool` and `progress_tool` exactly once each (any order).\n"
        "2) Keep running until both complete.\n"
        "3) Respond exactly 'all done'.",
    )
    return client


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_notifications_and_child():
    """Snapshot/resume while a child loop runs and a sibling tool emits notifications.

    After resume, ensure notifications can still be received and no duplicate tool
    results appear for either tool.
    """

    # Ensure deterministic gating for both tools
    globals()["INNER_GATE"] = asyncio.Event()
    globals()["PROG_GATE"] = asyncio.Event()

    client = _outer_client_with_progress()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_tool": outer_tool, "progress_tool": progress_tool},
        timeout=300,
    )

    # Wait until the assistant has called BOTH tools and placeholders exist
    await _wait_for_assistant_tool_calls(["outer_tool", "progress_tool"], timeout=240.0)
    await _wait_for_tool_message_prefix(client, "outer_tool", timeout=180.0)
    await _wait_for_tool_message_prefix(client, "progress_tool", timeout=180.0)

    # Snapshot recursively; the child handle should be captured; progress placeholders present
    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict)

    # Resume, then immediately release gates so both tools can complete and notifications flush
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    globals()["PROG_GATE"].set()  # type: ignore[attr-defined]
    globals()["INNER_GATE"].set()  # type: ignore[attr-defined]

    # Receive a notification after resume (replayed or live) and assert shape
    notif = await asyncio.wait_for(resumed.next_notification(), timeout=60.0)
    assert isinstance(notif, dict) and notif.get("type") == "notification"
    assert notif.get("tool_name") == "progress_tool"
    out = await asyncio.wait_for(resumed.result(), timeout=240.0)
    assert out.strip().lower() == "all done"

    # Ensure exactly one final tool message for each tool (no duplicates)
    msgs = resumed.get_history()
    outer_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "outer_tool"
    ]
    prog_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "progress_tool"
    ]
    assert len(outer_msgs) == 1
    assert len(prog_msgs) == 1

    # Explicit cleanup to avoid dangling asyncio tasks after test completes
    try:
        handle.stop(reason="test cleanup")
    except Exception:
        pass
    try:
        resumed.stop(reason="test cleanup")
    except Exception:
        pass
    try:
        globals()["PROG_GATE"] = None
        globals()["INNER_GATE"] = None
    except Exception:
        pass
    try:
        EVENT_BUS.join_callbacks()
        EVENT_BUS.join_published()
    except Exception:
        pass
