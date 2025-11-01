from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager
from unity.common.async_tool_loop import AsyncToolLoopHandle


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
