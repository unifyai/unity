"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
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
def test_log_creates_all_contacts_entries():
    """Creating a contact should mirror to both aggregation contexts."""
    cm = ContactManager()

    # Create a contact
    result = cm._create_contact(first_name="TestUser", surname="AllCtx")
    contact_id = result["details"]["contact_id"]

    # Verify it exists in the manager's context
    contacts = cm.filter_contacts(filter=f"contact_id == {contact_id}")["contacts"]
    assert len(contacts) == 1, "Contact should exist in manager's context"

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(cm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"contact_id == {contact_id}",
        )
        assert len(all_logs) >= 1, f"Contact should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user context (ID)."""
    from unittest.mock import patch

    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_context",
        return_value=test_user_name,
    ):
        cm = ContactManager()
        result = cm._create_contact(first_name="UserTest", surname="Injection")
        contact_id = result["details"]["contact_id"]

        log = _get_raw_log_by_contact_id(cm._ctx, contact_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user") == test_user_name
        ), f"_user should be '{test_user_name}', got {entries.get('_user')}"


@_handle_project
def test_assistant_field_injected():
    """Logs should have _assistant field set to assistant context (ID)."""
    from unittest.mock import patch

    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_context",
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
    """Logs should have _user_id field set to user's ID."""
    from unittest.mock import patch

    test_user_id = "test-user-456"

    with patch(
        "unity.common.log_utils._get_user_id",
        return_value=test_user_id,
    ):
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
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when ContactManager provisions storage."""
    # ContactManager provisions storage via ContextRegistry.get_context() in __init__
    cm = ContactManager()

    # Derive the expected aggregation contexts
    all_ctxs = _derive_all_contexts(cm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
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
    # The Contact pydantic model doesn't have _user, _user_id, _assistant, _assistant_id fields
    assert not hasattr(contact, "_user"), "_user should not be exposed"
    assert not hasattr(contact, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(contact, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(contact, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_deleting_contact_removes_from_all_ctxs():
    """Deleting a contact should also remove it from all aggregation contexts."""
    cm = ContactManager()

    # Create a contact
    result = cm._create_contact(first_name="DeleteTest", surname="Contact")
    contact_id = result["details"]["contact_id"]

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(cm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"contact_id == {contact_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Contact should exist in {all_ctx} before deletion"

    # Delete the contact
    cm._delete_contact(contact_id=contact_id)

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"contact_id == {contact_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Contact should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating a contact should be immediately visible in all aggregation contexts."""
    cm = ContactManager()

    # Create a contact with initial values
    result = cm._create_contact(first_name="UpdateSync", surname="Original")
    contact_id = result["details"]["contact_id"]

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(cm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial surname in all contexts
    for ctx in [cm._ctx, *all_ctxs]:
        log = _get_raw_log_by_contact_id(ctx, contact_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert log.entries.get("surname") == "Original", f"Initial surname in {ctx}"

    # Update the contact's surname
    cm.update_contact(contact_id=contact_id, surname="Updated")

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [cm._ctx, *all_ctxs]:
        log = _get_raw_log_by_contact_id(ctx, contact_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("surname") == "Updated", (
            f"Updated surname should be visible in {ctx}. "
            f"Expected 'Updated', got '{log.entries.get('surname')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    cm = ContactManager()

    # Create a contact
    result = cm._create_contact(first_name="SameLogId", surname="Before")
    contact_id = result["details"]["contact_id"]

    # Get the original log ID
    original_log = _get_raw_log_by_contact_id(cm._ctx, contact_id)
    original_log_id = original_log.id

    # Update the contact
    cm.update_contact(contact_id=contact_id, surname="After")

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_contact_id(cm._ctx, contact_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(cm._ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_contact_id(all_ctx, contact_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
