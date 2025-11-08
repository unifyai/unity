from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.common.async_tool_loop import AsyncToolLoopHandle
from unity.common.loop_snapshot import (
    LoopSnapshot,
    EntryPointManagerMethod,
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
    assert snap.get("entrypoint", {}).get("class_name") == "ContactManager"
    assert snap.get("entrypoint", {}).get("method_name") == "ask"
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
async def test_deserialize_migrates_missing_version_and_type():
    cm = ContactManager()
    handle = await cm.ask("Find contact Bob")
    snap = handle.serialize()

    # Simulate an older snapshot: drop version and entrypoint.type
    snap.pop("version", None)
    ep = dict(snap.get("entrypoint") or {})
    ep.pop("type", None)
    snap["entrypoint"] = ep

    # Should migrate and still be able to resume
    resumed = AsyncToolLoopHandle.deserialize(snap)
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0


# ----------------------------------------------------------------------------
# Snapshot schema roundtrip
# ----------------------------------------------------------------------------


def _sample_snapshot_dict():
    snap = LoopSnapshot(
        entrypoint=EntryPointManagerMethod(
            class_name="ContactManager",
            method_name="ask",
        ),
        loop_id="ContactManager.ask(abcdef)",
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
    assert validated.entrypoint.class_name == "ContactManager"
    assert validated.entrypoint.method_name == "ask"
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
        "entrypoint": {
            "type": "manager_method",
            "class_name": "NoSuchManager",
            "method_name": "ask",
        },
        "loop_id": "NoSuchManager.ask(xxx)",
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
        "entrypoint": {
            "type": "manager_method",
            "class_name": "ContactManager",
            "method_name": "nope",
        },
        "loop_id": "ContactManager.nope(xxx)",
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
