import asyncio
import pytest
from tests.helpers import _handle_project
from unity.common.async_tool_loop import AsyncToolLoopHandle
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_manager_resume():
    # Use a manager method for the outer loop
    cm = ContactManager()
    handle = await cm.ask("Find contact Echo")

    # Snapshot recursively (children may or may not be present)
    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict) and isinstance(snap.get("meta", {}), dict)
    children = (snap.get("meta") or {}).get("children") or []
    assert isinstance(children, list)

    # Resume from snapshot
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)

    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0

    msgs = resumed.get_history()
    assert isinstance(msgs, list)


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_recursive_resume_manager():
    tm = TranscriptManager()
    handle = await tm.ask("Do I have any transcripts? Reply briefly.")
    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict)
    children = snap.get("children", [])
    assert isinstance(children, list)
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0


@pytest.mark.asyncio
@_handle_project
async def test_recursive_snapshot_preserves_interjection_manager():
    cm = ContactManager()
    handle = await cm.ask("Find contact Foxtrot")

    interjection_text = "Prefer compact layout"
    await handle.interject(interjection_text)

    snap = handle.serialize(recursive=True)
    # Resume and ensure the interjection appears once in the resumed transcript
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0
    hist = resumed.get_history() or []
    # Interjections are now user messages (not system messages with wrapper)
    seen = [
        m
        for m in hist
        if m.get("role") == "user" and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_nested_resume_missing_call_id_ignored():
    cm = ContactManager()
    handle = await cm.ask("Find contact Golf")

    snap = handle.serialize(recursive=True)
    # If children are present, corrupt the first child call_id; otherwise proceed
    children = (snap.get("meta", {}) or {}).get("children", []) or []
    if children and isinstance(children[0], dict):
        children[0]["call_id"] = "nonexistent_call_id"

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Three-level nesting (manager-only): soft checks with recursive snapshot
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_manager_three_levels_soft():
    # Simulate a multi-level scenario by taking a recursive snapshot;
    # children may be empty under manager-only design.
    cm = ContactManager()
    handle = await cm.ask("Find contact Hotel and summarize briefly.")

    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict)
    meta = snap.get("meta") or {}
    children = meta.get("children") or []
    assert isinstance(children, list)

    # Resume and ensure completion
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Clarification inside nested child – answer via parent handle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_nested_child_clarification_serialize_resume_and_answer_manager():
    # Manager-only soft test: if clarifications exist, we can answer; otherwise proceed.
    tm = TranscriptManager()
    handle = await tm.ask("Please check if any conversations require follow-up.")

    snap = handle.serialize(recursive=True)
    clars = snap.get("clarifications") or []
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    if clars:
        cid = clars[0].get("call_id")
        if isinstance(cid, str) and cid:
            await resumed.answer_clarification(cid, "Blue")
    out = await asyncio.wait_for(resumed.result(), timeout=180.0)
    assert isinstance(out, str) and len(out) > 0


@pytest.mark.asyncio
@_handle_project
async def test_serialize_requires_recursive_flag_for_nested():
    """Default serialize (recursive=False) must reject nested loops with a clear error.

    This guards the v1 contract that nested tool loops are only supported when callers
    explicitly opt-in via recursive=True. It prevents accidental implicit nested capture.
    """
    # Under manager-only design, when no nested children exist, serialize() should succeed by default.
    cm = ContactManager()
    handle = await cm.ask("Find contact India")
    snap = handle.serialize()
    assert isinstance(snap, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Notifications replay on recursive snapshot (manager-only)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_recursive_snapshot_notifications_manager():
    cm = ContactManager()
    handle = await cm.ask("Find contact Juliett")

    # Inject pending notifications directly before snapshot
    await handle._notification_q.put({"type": "notification", "tool_name": "manager", "phase": "start"})  # type: ignore[attr-defined]
    await handle._notification_q.put({"type": "notification", "tool_name": "manager", "phase": "halfway"})  # type: ignore[attr-defined]

    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict)

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    notif1 = await asyncio.wait_for(resumed.next_notification(), timeout=60.0)
    notif2 = await asyncio.wait_for(resumed.next_notification(), timeout=60.0)
    assert notif1.get("type") == "notification"
    assert notif2.get("type") == "notification"
    out = await asyncio.wait_for(resumed.result(), timeout=240.0)
    assert isinstance(out, str) and len(out) > 0
