from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager


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
    # assistant_steps may be empty depending on timing, but it must be a list
    assert isinstance(snap.get("assistant_steps"), list)
    assert isinstance(snap.get("tool_results"), list)
