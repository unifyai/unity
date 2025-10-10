from __future__ import annotations


import pytest

from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager


@pytest.mark.asyncio
@_handle_project
async def test_contact_manager_ask_guard_triggers_when_enabled(monkeypatch):
    """
    When UNITY_READONLY_ASK_GUARD is enabled, ContactManager.ask should be guarded:
    mutation intent triggers an early stop and returns the early response.
    """

    # Ensure the env flag is on for this test only
    monkeypatch.setenv("UNITY_READONLY_ASK_GUARD", "true")

    cm = ContactManager()

    # Mutation-intent phrasing to trigger the guard classifier
    handle = await cm.ask(
        "Please change Bob Johnson's email to foo@bar.com; make the update now.",
    )
    result = await handle.result()
    assert isinstance(result, str) and result.strip() != ""
    # Expect guidance towards using update; exact phrasing may vary but should include 'update'
    assert "update" in result.lower()
