import asyncio
import json
import os

import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
    _wait_for_assistant_tool_calls,
)
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle


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
