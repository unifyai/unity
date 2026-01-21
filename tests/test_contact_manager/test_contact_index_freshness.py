"""
tests/test_contact_manager/test_contact_index_freshness.py
==========================================================

Tests verifying that ContactIndex always returns fresh data from ContactManager.

These tests verify that when contacts are created or updated via ContactManager,
the ContactIndex.get_contact() method returns the up-to-date data.

This is critical because:
1. The Actor might create new contacts during task execution
2. Contact details might be updated via ContactManager.update()
3. ContactIndex delegates all reads to ContactManager (source of truth)

Note: These tests require the REAL ContactManager (not simulated) because
they test database-backed data freshness behavior.
"""

from unity.contact_manager.contact_manager import ContactManager
from unity.conversation_manager.domains.contact_index import ContactIndex
from tests.helpers import _handle_project


@_handle_project
def test_contact_created_via_contact_manager_is_visible():
    """Contacts created via ContactManager should be immediately visible via ContactIndex."""
    cm = ContactManager()
    contact_index = ContactIndex()
    contact_index.set_contact_manager(cm)

    # Create a unique email to avoid conflicts
    unique_email = "actor.created.freshness@example.com"

    # Create contact directly via ContactManager (simulating Actor behavior)
    result = cm._create_contact(
        first_name="ActorCreated",
        surname="Contact",
        email_address=unique_email,
        phone_number="+15005550001",
    )

    # Get the assigned contact_id from the result
    new_contact_id = result["details"]["contact_id"]

    # ContactIndex.get_contact() should find it via ContactManager
    contact = contact_index.get_contact(contact_id=new_contact_id)
    assert contact is not None
    assert contact["first_name"] == "ActorCreated"

    # Also verify search by email works
    contact_by_email = contact_index.get_contact(email=unique_email)
    assert contact_by_email is not None
    assert contact_by_email["first_name"] == "ActorCreated"


@_handle_project
def test_contact_updated_via_contact_manager_reflects_changes():
    """Updates to contacts via ContactManager should be reflected in ContactIndex."""
    cm = ContactManager()
    contact_index = ContactIndex()
    contact_index.set_contact_manager(cm)

    # Use system contact 0 (assistant) which always exists in ContactManager
    contact_id = 0

    # Get original data
    original = contact_index.get_contact(contact_id=contact_id)
    assert original is not None
    original_bio = original.get("bio")

    # Update the contact directly via ContactManager (simulating Actor behavior)
    new_bio = "Updated bio for freshness test"
    cm.update_contact(
        contact_id=contact_id,
        bio=new_bio,
    )

    # ContactIndex.get_contact() should return the updated data
    updated = contact_index.get_contact(contact_id=contact_id)
    assert updated is not None
    assert updated["bio"] == new_bio

    # Restore original
    cm.update_contact(
        contact_id=contact_id,
        bio=original_bio or "",
    )


@_handle_project
def test_updates_immediately_visible_without_cache_refresh():
    """
    ContactIndex.get_contact() always fetches fresh data from ContactManager.

    There is no local cache to become stale - every get_contact() call
    queries ContactManager's DataStore-backed cache directly.
    """
    cm = ContactManager()
    contact_index = ContactIndex()
    contact_index.set_contact_manager(cm)

    # Use system contact 0
    contact_id = 0

    # Get initial data
    initial = contact_index.get_contact(contact_id=contact_id)
    assert initial is not None
    original_bio = initial.get("bio")

    # Update via ContactManager
    test_bio = "Test bio for immediate visibility"
    cm.update_contact(contact_id=contact_id, bio=test_bio)

    # Immediately fetch again - should see the update
    updated = contact_index.get_contact(contact_id=contact_id)
    assert updated is not None
    assert updated["bio"] == test_bio

    # Update again
    test_bio_2 = "Second test bio"
    cm.update_contact(contact_id=contact_id, bio=test_bio_2)

    # Should see the second update immediately
    updated_2 = contact_index.get_contact(contact_id=contact_id)
    assert updated_2 is not None
    assert updated_2["bio"] == test_bio_2

    # Restore original
    cm.update_contact(contact_id=contact_id, bio=original_bio or "")


def test_contact_manager_not_set_returns_none():
    """When ContactManager is not set, get_contact() returns None."""
    contact_index = ContactIndex()
    # Do NOT set a ContactManager

    # Should return None since there's no ContactManager to query
    contact = contact_index.get_contact(contact_id=0)
    assert contact is None

    contact_by_email = contact_index.get_contact(email="test@example.com")
    assert contact_by_email is None

    contact_by_phone = contact_index.get_contact(phone_number="+15555550000")
    assert contact_by_phone is None
