from __future__ import annotations

import asyncio
import pytest

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_system_interjection_event,
)
from unity.common.async_tool_loop import (
    AsyncToolLoopHandle,
    _parse_entrypoint_from_loop_id_label,
)
from unity.common.loop_snapshot import (
    LoopSnapshot,
    validate_snapshot,
)
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager


# No inline helper tools in manager-only snapshot design


# ----------------------------------------------------------------------------
# Basic serialization shape and meta
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_handle_serialize_minimal():
    cm = ContactManager()
    handle = await cm.ask("Find contact Alice")

    snap = handle.serialize()

    assert isinstance(snap, dict)
    assert snap.get("version") == 1
    assert snap.get("loop_id", "").startswith("ContactManager.ask")
    root = snap.get("root") or {}
    assert root.get("tool") == "ContactManager.ask"
    # assistant may be empty depending on timing, but it must be a list
    assert isinstance(snap.get("assistant"), list)
    assert isinstance(snap.get("tools"), list)


@pytest.mark.asyncio
@_handle_project
async def test_snapshot_contains_meta_and_semantic_namespace():
    cm = ContactManager()
    handle = await cm.ask("Find contact Alice")

    snap = handle.serialize()

    meta = snap.get("meta") or {}
    assert isinstance(meta, dict)
    assert isinstance(meta.get("run_id"), str) and len(meta["run_id"]) > 0
    assert isinstance(meta.get("snapshot_at"), str) and len(meta["snapshot_at"]) > 0
    # loop_created_at may be present; if present it must be a string
    if meta.get("loop_created_at") is not None:
        assert isinstance(meta["loop_created_at"], str)

    ctx = meta.get("assistant_context") or {}
    assert isinstance(ctx, dict)
    assert "read" in ctx and "write" in ctx

    # For manager entrypoints, semantic_cache_namespace should be present
    assert meta.get("semantic_cache_namespace") == "ContactManager.ask"


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_migrates_missing_version():
    cm = ContactManager()
    handle = await cm.ask("Find contact Bob")
    snap = handle.serialize()

    # Simulate an older snapshot: drop version
    snap.pop("version", None)

    # Should migrate and still be able to resume
    resumed = AsyncToolLoopHandle.deserialize(snap)
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0


# ----------------------------------------------------------------------------
# Snapshot schema roundtrip
# ----------------------------------------------------------------------------


def _sample_snapshot_dict():
    snap = LoopSnapshot(
        loop_id="ContactManager.ask(abcdef)",
        root={"tool": "ContactManager.ask", "handle": "AsyncToolLoopHandle"},
        system_message="You are helpful.",
        initial_user_message="Find contact Alice",
        assistant=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_contacts",
                            "arguments": "{}",
                        },
                    },
                ],
            },
        ],
        tools=[
            {
                "id": "call_1",
                "name": "search_contacts",
                "content": "[]",
            },
        ],
    ).model_dump()
    return snap


def test_loop_snapshot_roundtrip():
    data = _sample_snapshot_dict()
    validated = validate_snapshot(data)
    assert validated.version == 1
    assert isinstance(validated.root, dict)
    assert (validated.root or {}).get("tool") == "ContactManager.ask"
    assert isinstance(validated.assistant, list)
    assert isinstance(validated.tools, list)


def test_loop_snapshot_unsupported_version():
    data = _sample_snapshot_dict()
    data["version"] = 999
    with pytest.raises(ValueError):
        validate_snapshot(data)


# ----------------------------------------------------------------------------
# Deserialize: managers
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_contact_manager_resume():
    cm = ContactManager()

    # Start a loop and immediately snapshot (may still be in-flight)
    handle = await cm.ask("Find contact Alice")
    snap = handle.serialize()

    # Resume from snapshot and ensure we obtain a final answer
    resumed = AsyncToolLoopHandle.deserialize(snap)
    answer = await resumed.result()
    assert isinstance(answer, str) and len(answer) > 0


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_transcript_manager_resume():
    tm = TranscriptManager()

    # Seed a minimal prompt that will exercise search tools
    handle = await tm.ask(
        "Do I have any transcripts? Reply briefly.",
    )
    snap = handle.serialize()

    resumed = AsyncToolLoopHandle.deserialize(snap)
    answer = await resumed.result()
    assert isinstance(answer, str) and len(answer) > 0


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_unknown_manager_class_raises():
    snap = {
        "version": 1,
        "loop_id": "NoSuchManager.ask(xxx)",
        "root": {"tool": "NoSuchManager.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Hello",
        "assistant": [],
        "tools": [],
    }

    with pytest.raises(ValueError, match="Manager class not found"):
        AsyncToolLoopHandle.deserialize(snap)


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_unknown_manager_method_raises():
    snap = {
        "version": 1,
        "loop_id": "ContactManager.nope(xxx)",
        "root": {"tool": "ContactManager.nope", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Hello",
        "assistant": [],
        "tools": [],
    }

    with pytest.raises(ValueError, match="No tools registered for ContactManager.nope"):
        AsyncToolLoopHandle.deserialize(snap)


# ----------------------------------------------------------------------------
# Inline and flat-loop tests removed for manager-only snapshot design
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interjection_captured_and_preserved():
    cm = ContactManager()
    handle = await cm.ask("Find contact Charlie")

    interjection_text = "Prefer compact layout"
    await handle.interject(interjection_text)
    # Wait until the interjection is materialised as a system message in the transcript
    await _wait_for_system_interjection_event(contains=interjection_text, timeout=120.0)

    snap = handle.serialize()

    # Resume and ensure the resumed loop still completes successfully
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    out = await asyncio.wait_for(resumed.result(), timeout=180)
    assert isinstance(out, str) and len(out) > 0


@pytest.mark.asyncio
@_handle_project
async def test_pause_resume_survives_serialization():
    tm = TranscriptManager()
    handle = await tm.ask("List my transcripts briefly.")

    # Pause before snapshot
    await handle.pause()
    snap = handle.serialize()

    # Resume after deserialization and complete
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    await resumed.resume()
    out = await asyncio.wait_for(resumed.result(), timeout=180)
    assert isinstance(out, str) and len(out) > 0


@pytest.mark.asyncio
@_handle_project
async def test_notifications_replayed_after_resume():
    cm = ContactManager()
    handle = await cm.ask("Find contact Delta")

    # Inject pending notifications directly onto the handle prior to snapshot
    # (mirrors behaviour of tools emitting _notification_up_q events)
    await handle._notification_q.put({"type": "notification", "tool_name": "test", "step": 1})  # type: ignore[attr-defined]
    await handle._notification_q.put({"type": "notification", "tool_name": "test", "step": 2})  # type: ignore[attr-defined]

    snap = handle.serialize()
    notifs = snap.get("notifications") or []
    assert isinstance(notifs, list) and len(notifs) >= 2

    # After resume, notifications are re-injected and immediately consumable
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    evt1 = await asyncio.wait_for(resumed.next_notification(), timeout=60)
    evt2 = await asyncio.wait_for(resumed.next_notification(), timeout=60)
    assert evt1.get("type") == "notification"
    assert evt2.get("type") == "notification"


# ----------------------------------------------------------------------------
# loop_id parsing – derive entrypoint from labels (serialize/deserialize)
# ----------------------------------------------------------------------------


def test_parse_entrypoint_from_loop_id_label_variants():
    # Simple "Class.method"
    assert _parse_entrypoint_from_loop_id_label("ContactManager.ask") == (
        "ContactManager",
        "ask",
    )
    # Trailing unique id in parentheses
    assert _parse_entrypoint_from_loop_id_label("ContactManager.ask(x2ab)") == (
        "ContactManager",
        "ask",
    )
    # Nested lineage; keep last segment only
    assert _parse_entrypoint_from_loop_id_label(
        "ContactManager.update->ContactManager.ask(x2ab)",
    ) == ("ContactManager", "ask")
    # Multiple segments; still last wins
    assert _parse_entrypoint_from_loop_id_label(
        "A.update->B.exec->ContactManager.ask(zzz)",
    ) == ("ContactManager", "ask")


@pytest.mark.asyncio
@_handle_project
async def test_serialize_entrypoint_parsed_from_loop_id_label_nested():
    cm = ContactManager()
    handle = await cm.ask("Find contact Kilo")
    # Force a nested lineage style label; serialize should use only the last segment
    inner = getattr(handle, "__wrapped__", handle)
    setattr(inner, "_log_label", "ContactManager.update->ContactManager.ask(custom123)")
    snap = handle.serialize()
    root = snap.get("root") or {}
    assert root.get("tool") == "ContactManager.ask"


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_prefers_loop_id_over_entrypoint_fields():
    cm = ContactManager()
    handle = await cm.ask("Find contact Lima")
    snap = handle.serialize()
    # Corrupt root.tool but set an authoritative loop_id; deserializer must use loop_id.
    if isinstance(snap.get("root"), dict):
        snap["root"]["tool"] = "NoSuchManager.nope"
    snap["loop_id"] = "ContactManager.update->ContactManager.ask(custom456)"
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    out = await asyncio.wait_for(resumed.result(), timeout=180.0)
    assert isinstance(out, str) and len(out) > 0
