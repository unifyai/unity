"""Tests for All/Contacts context mirroring and private field injection."""

from __future__ import annotations

import os

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_context
from unity.contact_manager.contact_manager import ContactManager


def _get_raw_log_by_contact_id(ctx: str, contact_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"contact_id == {contact_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_contacts_entry():
    """Creating a contact should mirror to All/<Ctx>."""
    cm = ContactManager()

    # Create a contact
    result = cm._create_contact(first_name="TestUser", surname="AllCtx")
    contact_id = result["details"]["contact_id"]

    # Verify it exists in the manager's context
    contacts = cm.filter_contacts(filter=f"contact_id == {contact_id}")["contacts"]
    assert len(contacts) == 1, "Contact should exist in manager's context"

    # Derive the All/<Ctx> context from the manager's context
    all_ctx = _derive_all_context(cm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it was mirrored to All/<Ctx>
    all_logs = unify.get_logs(
        context=all_ctx,
        filter=f"contact_id == {contact_id}",
    )
    assert len(all_logs) >= 1, f"Contact should be mirrored to {all_ctx}"


@_handle_project
def test_assistant_field_injected():
    """Logs should have _assistant field set to assistant name."""
    from unittest.mock import patch

    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_name",
        return_value=test_assistant_name,
    ):
        cm = ContactManager()
        result = cm._create_contact(first_name="AssistantTest", surname="Injection")
        contact_id = result["details"]["contact_id"]

        log = _get_raw_log_by_contact_id(cm._ctx, contact_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant") == test_assistant_name
        ), f"_assistant should be '{test_assistant_name}', got {entries.get('_assistant')}"


@_handle_project
def test_assistant_id_field_injected():
    """Logs should have _assistant_id field set to assistant's agent_id."""
    from unittest.mock import patch

    test_assistant_id = "test-agent-789"

    with patch(
        "unity.common.log_utils._get_assistant_id",
        return_value=test_assistant_id,
    ):
        cm = ContactManager()
        result = cm._create_contact(first_name="AssistantIdTest", surname="Injection")
        contact_id = result["details"]["contact_id"]

        log = _get_raw_log_by_contact_id(cm._ctx, contact_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant_id") == test_assistant_id
        ), f"_assistant_id should be '{test_assistant_id}', got {entries.get('_assistant_id')}"


@_handle_project
def test_user_id_field_injected():
    """Logs should have _user_id field when USER_ID env is set."""
    from unittest.mock import patch

    test_user_id = "test-user-456"

    with patch.dict(os.environ, {"USER_ID": test_user_id}):
        cm = ContactManager()
        result = cm._create_contact(first_name="UserIdTest", surname="Injection")
        contact_id = result["details"]["contact_id"]

        log = _get_raw_log_by_contact_id(cm._ctx, contact_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_context_created_on_provision():
    """All/<Ctx> context should be created when ContactManager provisions storage."""
    # ContactManager provisions storage via ContextRegistry.get_context() in __init__
    cm = ContactManager()

    # Derive the expected All/<Ctx> context
    all_ctx = _derive_all_context(cm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify All/<Ctx> exists
    contexts = unify.get_contexts()
    assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter_contacts():
    """Private fields should be excluded when reading contacts via public API."""
    cm = ContactManager()

    result = cm._create_contact(first_name="PrivateTest", surname="Exclusion")
    contact_id = result["details"]["contact_id"]

    # Get contact via public filter_contacts API
    contacts = cm.filter_contacts(filter=f"contact_id == {contact_id}")["contacts"]
    assert len(contacts) == 1

    contact = contacts[0]
    # Private fields should NOT be in the Contact model (they're excluded on read)
    # The Contact pydantic model doesn't have _assistant, _assistant_id, _user_id fields
    assert not hasattr(contact, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(contact, "_assistant_id"), "_assistant_id should not be exposed"
    assert not hasattr(contact, "_user_id"), "_user_id should not be exposed"


@_handle_project
def test_deleting_contact_removes_from_all_ctx():
    """Deleting a contact should also remove it from All/<Ctx>."""
    cm = ContactManager()

    # Create a contact
    result = cm._create_contact(first_name="DeleteTest", surname="Contact")
    contact_id = result["details"]["contact_id"]

    # Derive the All/<Ctx> context
    all_ctx = _derive_all_context(cm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it exists in All/<Ctx> before deletion
    all_logs_before = unify.get_logs(
        context=all_ctx,
        filter=f"contact_id == {contact_id}",
    )
    assert (
        len(all_logs_before) >= 1
    ), "Contact should exist in All/<Ctx> before deletion"

    # Delete the contact
    cm._delete_contact(contact_id=contact_id)

    # Verify it's removed from All/<Ctx> after deletion
    all_logs_after = unify.get_logs(
        context=all_ctx,
        filter=f"contact_id == {contact_id}",
    )
    assert (
        len(all_logs_after) == 0
    ), "Contact should be removed from All/<Ctx> after deletion"
