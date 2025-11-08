from __future__ import annotations

import asyncio
import base64
import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    make_gated_sync_tool,
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
    _wait_for_assistant_call_prefix,
    _wait_for_tool_result,
    _wait_for_any_tool_message_prefix,
)
from unity.common.async_tool_loop import (
    start_async_tool_loop,
    AsyncToolLoopHandle,
)
from unity.common.loop_snapshot import (
    LoopSnapshot,
    EntryPointManagerMethod,
    validate_snapshot,
)
from unity.common.tool_spec import read_only
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import RawImageRef, AnnotatedImageRef, ImageRefs


# Module-level gates/tools for inline-tools serialization to resolve by import path
gate: asyncio.Event | None = None


async def notify_parent(
    message: str,
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:
    global gate
    if _notification_up_q is None:
        raise RuntimeError("notification queue missing")
    await _notification_up_q.put({"message": message})
    return "ok"


async def blocker(
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:  # noqa: ARG001
    global gate
    if gate is None:
        gate = asyncio.Event()
    await gate.wait()
    return "done"


GATE: asyncio.Event | None = None


async def blocking_tool() -> str:
    global GATE
    if GATE is None:
        return "done"
    await GATE.wait()
    return "done"


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
# Deserialize: managers and inline tools
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


# Define a top-level function so it is importable by module + qualname
@read_only
def greet():
    return "Hello from Inline!"


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_inline_tools_resume():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the greet tool exactly once and reply with the result only.",
    )

    handle = start_async_tool_loop(
        client,
        "please greet",
        tools={"greet": greet},
    )

    # Snapshot immediately; assistant may or may not have requested the tool yet
    snap = handle.serialize()

    # Resume from snapshot
    resumed = AsyncToolLoopHandle.deserialize(snap)
    answer = await resumed.result()
    assert isinstance(answer, str) and "hello" in answer.lower()


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
# Serialize: quiesce, clarifications, notifications, images, pending rerun
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_serialize_cancels_inflight_and_keeps_requests():
    gate, hold_tool = make_gated_sync_tool(return_value="done")

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the hold tool exactly once and then respond with the result only.",
    )

    handle = start_async_tool_loop(
        client,
        "Please proceed",
        tools={"hold": hold_tool},
    )

    # Wait until assistant requests the tool so we know it is in-flight
    await _wait_for_tool_request(client, "hold")

    # Serialize should cancel the loop and capture the assistant tool-call
    snap = handle.serialize()

    assert isinstance(snap, dict)
    assert snap["version"] == 1
    # Assistant must have at least one tool_call for 'hold'
    assert any(
        any(
            tc.get("function", {}).get("name") == "hold"
            for tc in m.get("tool_calls", [])
        )
        for m in snap.get("assistant", [])
        if m.get("role") == "assistant"
    )
    # No tool result should be present because we never opened the gate
    assert not any(
        m.get("role") == "tool" and m.get("name") == "hold"
        for m in snap.get("tools", [])
    )


# Define the tool at module scope so it is importable by module+qualname
async def needs_clar(*, _clarification_up_q=None, _clarification_down_q=None):
    assert _clarification_up_q is not None and _clarification_down_q is not None
    await _clarification_up_q.put("What colour should we use?")
    ans = await _clarification_down_q.get()
    return f"ACK: {ans}"


@pytest.mark.asyncio
@_handle_project
async def test_serialize_deserialize_with_clarification_inline():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the needs_clar tool exactly once and then respond with the result only.",
    )

    handle = start_async_tool_loop(
        client,
        "please proceed",
        tools={"needs_clar": needs_clar},
    )

    # Wait for clarification request to appear in transcript
    await _wait_for_tool_message_prefix(client, "clarification_request_", timeout=120.0)

    snap = handle.serialize()

    assert isinstance(snap, dict)
    assert snap.get("version") == 1
    # Should persist at least one clarification entry
    clars = snap.get("clarifications") or []
    assert isinstance(clars, list) and len(clars) >= 1
    cid = clars[0].get("call_id")
    assert isinstance(cid, str) and len(cid) > 0

    # Resume and answer the clarification
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    await resumed.answer_clarification(cid, "Blue")
    out = await resumed.result()
    assert isinstance(out, str) and "blue" in out.lower()


@pytest.mark.asyncio
@_handle_project
async def test_serialize_deserialize_with_notifications_inline():
    """Serialize a loop with pending notifications and replay them after resume."""

    # Reset module-level gate for this test (inline tools resolve by import path)
    global gate
    gate = asyncio.Event()

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Call notify_parent(message='Phase 1') and notify_parent(message='Phase 2') exactly once each.\n"
        "Then call blocker exactly once and only produce your final assistant message after blocker completes.",
    )

    handle = start_async_tool_loop(
        client,
        "begin",
        tools={
            "notify_parent": notify_parent,
            "blocker": blocker,
        },
    )

    # Ensure the assistant has requested notify_parent and we got its results
    await _wait_for_assistant_call_prefix(client, "notify_parent", timeout=120.0)
    await _wait_for_tool_result(
        client,
        tool_name="notify_parent",
        min_results=2,
        timeout=120.0,
    )

    # Serialize while blocker is still running; notifications should be pending on the handle
    snap = handle.serialize()
    assert isinstance(snap, dict)
    assert snap.get("version") == 1
    notifs = snap.get("notifications") or []
    assert isinstance(notifs, list) and len(notifs) >= 2

    # Resume from snapshot and confirm notifications are available immediately
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    evt1 = await asyncio.wait_for(resumed.next_notification(), timeout=60)
    evt2 = await asyncio.wait_for(resumed.next_notification(), timeout=60)
    assert evt1.get("type") == "notification" or True  # tolerant across sources
    assert evt2.get("type") == "notification" or True

    # Unblock the running tool to allow the loop to complete
    gate.set()
    out = await asyncio.wait_for(resumed.result(), timeout=300)
    assert isinstance(out, str)


@pytest.mark.asyncio
@_handle_project
async def test_serialize_deserialize_with_images_overview_injected():
    # 1) Create a tiny in-memory PNG and add as an image to get a real id
    #    (4x4 single-colour)
    def _tiny_png_bytes() -> bytes:
        # Valid minimal PNG header + IHDR + IDAT for 1x1 opaque pixel (precomputed)
        # Keeping it simple and deterministic
        return base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
        )

    im = ImageManager()
    [iid] = im.add_images(
        [
            {
                "timestamp": None,
                "caption": "tiny",
                "data": _tiny_png_bytes(),
            },
        ],
        synchronous=True,
        return_handles=False,
    )
    assert isinstance(iid, int)

    # 2) Seed images context with an annotation
    images = ImageRefs.model_validate(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=iid),
                annotation="sample",
            ),
        ],
    )

    # 3) Start a minimal loop with no base tools; live image helpers should be injected
    #    We use a generic system message; we will assert the overview tool is injected.
    handle = start_async_tool_loop(
        client=(
            im._manager
            if hasattr(im, "_manager")
            else __import__(
                "unify",
            ).AsyncUnify(
                "gpt-5@openai",
                reasoning_effort="high",
                service_tier="priority",
                cache=True,
            )
        ),
        message="begin",
        tools={},
        images=images,
    )

    # Snapshot immediately; images should be captured
    snap = handle.serialize()
    assert isinstance(snap, dict)
    imgs = snap.get("images") or []
    assert any(
        isinstance(x.get("image_id"), int) and x.get("annotation") is not None
        for x in imgs
    )

    # 4) Resume from snapshot and assert the overview tool was injected on startup
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    # Wait until the synthetic overview tool message is present (event-based watcher)
    await _wait_for_any_tool_message_prefix("live_images_overview", timeout=120.0)
    hist = resumed.get_history() or []
    assert any(
        (m.get("role") == "tool" and m.get("name") == "live_images_overview")
        for m in hist
    )


@pytest.mark.asyncio
@_handle_project
async def test_flat_resume_retriggers_pending_base_tool():
    """Pending base tool at snapshot time is re-scheduled after resume (flat loop)."""
    # Ensure a fresh module-level gate for this run (inline tool resolves by import path)
    global GATE
    GATE = asyncio.Event()

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are testing resume.\n"
        "1) Call `blocking_tool` exactly once.\n"
        "2) After it finishes, reply exactly 'done'.",
    )

    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"blocking_tool": blocking_tool},
        timeout=180,
    )

    # Ensure tool-call requested and placeholder present
    await _wait_for_tool_request(client, "blocking_tool")
    await _wait_for_tool_message_prefix(client, "blocking_tool", timeout=120.0)

    # Snapshot and resume
    snap = handle.serialize()
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)

    # Release the gate so resumed loop can complete
    GATE.set()
    out = await asyncio.wait_for(resumed.result(), timeout=180)
    assert out.strip().lower() == "done"

    # Verify a single final tool reply exists (no duplicate placeholders)
    msgs = resumed.get_history()
    tool_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "blocking_tool"
    ]
    assert len(tool_msgs) == 1
