"""
tests/test_contact_manager/test_contact_index_freshness.py
==========================================================

Tests verifying that ContactIndex always returns fresh data from ContactManager.

These tests verify that when contacts are created or updated via ContactManager,
the ContactIndex.get_contact() method returns the up-to-date data.

This is critical because:
1. The Actor might create new contacts during task execution
2. Contact details might be updated via ContactManager.update()
3. The ContactIndex local cache can become stale

The fix ensures ContactIndex always queries ContactManager (which has an
auto-syncing cache backed by the database) for fresh data.

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
    unique_email = f"actor.created.freshness@example.com"

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
def test_stale_local_cache_is_bypassed():
    """Even if local cache has old data, get_contact returns fresh data."""
    cm = ContactManager()
    contact_index = ContactIndex()
    contact_index.set_contact_manager(cm)

    # Use system contact 0 which exists in both local cache and ContactManager
    contact_id = 0

    # Get fresh data first to know what's in ContactManager
    fresh_data = contact_index.get_contact(contact_id=contact_id)
    assert fresh_data is not None

    # Manually populate local cache with stale data
    from unity.conversation_manager.domains.contact_index import Contact

    contact_index.contacts[contact_id] = Contact(
        contact_id=contact_id,
        first_name="Stale",  # Must match name pattern (no underscores)
        bio="STALE_BIO_DATA",
    )

    # Update via ContactManager to have different "fresh" data
    fresh_bio = "FreshBio_for_test"
    cm.update_contact(
        contact_id=contact_id,
        bio=fresh_bio,
    )

    # get_contact should return fresh data, not stale cache
    result = contact_index.get_contact(contact_id=contact_id)
    assert result is not None
    # The bio should be from ContactManager, not the stale cache
    assert result["bio"] == fresh_bio
    # The first name should also be from ContactManager, not "Stale"
    assert result["first_name"] != "Stale"


@_handle_project
def test_contact_manager_not_set_falls_back_to_local_cache():
    """When ContactManager is not set, should fall back to local cache."""
    from unity.conversation_manager.domains.contact_index import Contact

    contact_index = ContactIndex()
    # Do NOT set a ContactManager - test fallback behavior

    # Populate local cache with test data
    test_contact_id = 999
    contact_index.contacts[test_contact_id] = Contact(
        contact_id=test_contact_id,
        first_name="LocalCacheTest",
        surname="Contact",
        email_address="local.cache@test.com",
    )

    # Should find it in local cache
    contact = contact_index.get_contact(contact_id=test_contact_id)
    assert contact is not None
    assert contact["first_name"] == "LocalCacheTest"

    # Search by email should also work
    contact_by_email = contact_index.get_contact(email="local.cache@test.com")
    assert contact_by_email is not None
    assert contact_by_email["first_name"] == "LocalCacheTest"
