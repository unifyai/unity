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
    """When ContactManager is not set and no fallback, get_contact() returns None."""
    contact_index = ContactIndex()
    # Do NOT set a ContactManager, do NOT set any fallback contacts

    # Should return None since there's no ContactManager and no fallback
    contact = contact_index.get_contact(contact_id=0)
    assert contact is None

    contact_by_email = contact_index.get_contact(email="test@example.com")
    assert contact_by_email is None

    contact_by_phone = contact_index.get_contact(phone_number="+15555550000")
    assert contact_by_phone is None


# =============================================================================
# Fallback Contacts Tests
# =============================================================================


class TestFallbackContacts:
    """Tests for the fallback contacts mechanism.

    The fallback mechanism caches contacts from inbound messages so they're
    available before ContactManager is initialized. When ContactManager is
    not set, get_contact() uses the fallback cache. Once ContactManager is
    set, it becomes the sole source of truth and the fallback is not used.
    """

    def test_is_contact_manager_initialized_false_by_default(self):
        """is_contact_manager_initialized is False when no ContactManager is set."""
        contact_index = ContactIndex()
        assert contact_index.is_contact_manager_initialized is False

    def test_is_contact_manager_initialized_true_after_set(self):
        """is_contact_manager_initialized is True after set_contact_manager()."""
        from unittest.mock import MagicMock

        contact_index = ContactIndex()
        mock_cm = MagicMock()
        contact_index.set_contact_manager(mock_cm)
        assert contact_index.is_contact_manager_initialized is True

    def test_set_fallback_contacts_caches_by_contact_id(self):
        """set_fallback_contacts() caches contacts indexed by contact_id."""
        contact_index = ContactIndex()

        contacts = [
            {"contact_id": 1, "first_name": "Boss", "email_address": "boss@test.com"},
            {"contact_id": 2, "first_name": "Contact", "phone_number": "+15555550001"},
        ]
        contact_index.set_fallback_contacts(contacts)

        # Check internal cache
        assert 1 in contact_index._fallback_contacts
        assert 2 in contact_index._fallback_contacts
        assert contact_index._fallback_contacts[1]["first_name"] == "Boss"
        assert contact_index._fallback_contacts[2]["first_name"] == "Contact"

    def test_get_contact_returns_fallback_when_no_contact_manager(self):
        """get_contact() returns from fallback cache when ContactManager is not set."""
        contact_index = ContactIndex()

        contacts = [
            {"contact_id": 1, "first_name": "Boss", "email_address": "boss@test.com"},
            {"contact_id": 2, "first_name": "Contact", "phone_number": "+15555550001"},
        ]
        contact_index.set_fallback_contacts(contacts)

        # Query by contact_id
        contact = contact_index.get_contact(contact_id=1)
        assert contact is not None
        assert contact["first_name"] == "Boss"

        # Query by phone_number
        contact_by_phone = contact_index.get_contact(phone_number="+15555550001")
        assert contact_by_phone is not None
        assert contact_by_phone["first_name"] == "Contact"

        # Query by email
        contact_by_email = contact_index.get_contact(email="boss@test.com")
        assert contact_by_email is not None
        assert contact_by_email["first_name"] == "Boss"

    def test_get_contact_uses_fallback_when_contact_manager_not_set(self):
        """get_contact() uses fallback cache when ContactManager is not set."""
        contact_index = ContactIndex()

        # Set fallback contacts (no ContactManager set)
        contacts = [
            {
                "contact_id": 99,
                "first_name": "Fallback",
                "phone_number": "+15555550099",
            },
        ]
        contact_index.set_fallback_contacts(contacts)

        # ContactManager is NOT set, so fallback should be used
        contact = contact_index.get_contact(contact_id=99)
        assert contact is not None
        assert contact["first_name"] == "Fallback"

    def test_get_contact_uses_contact_manager_when_set(self):
        """get_contact() uses ContactManager when it's set (ignores fallback)."""
        from unittest.mock import MagicMock

        contact_index = ContactIndex()

        # Set fallback contacts
        contacts = [
            {"contact_id": 99, "first_name": "Fallback"},
        ]
        contact_index.set_fallback_contacts(contacts)

        # Set up a mock ContactManager that returns a different contact
        mock_cm = MagicMock()
        mock_cm.get_contact_info.return_value = {
            99: {"contact_id": 99, "first_name": "FromManager"},
        }
        contact_index.set_contact_manager(mock_cm)

        # Should use ContactManager now, not fallback
        contact = contact_index.get_contact(contact_id=99)
        assert contact is not None
        assert contact["first_name"] == "FromManager"

        # ContactManager SHOULD have been called
        mock_cm.get_contact_info.assert_called_once_with(99)

    def test_get_contact_falls_through_to_contact_manager(self):
        """get_contact() queries ContactManager if not found in fallback."""
        from unittest.mock import MagicMock

        contact_index = ContactIndex()

        # Set up a mock ContactManager
        mock_cm = MagicMock()
        mock_cm.get_contact_info.return_value = {
            100: {"contact_id": 100, "first_name": "FromManager"},
        }
        contact_index.set_contact_manager(mock_cm)

        # Query for contact that's NOT in fallback (empty)
        contact = contact_index.get_contact(contact_id=100)
        assert contact is not None
        assert contact["first_name"] == "FromManager"

        # ContactManager SHOULD have been called
        mock_cm.get_contact_info.assert_called_once_with(100)

    def test_fallback_not_cleared_when_contact_manager_set(self):
        """Fallback cache is NOT cleared when set_contact_manager() is called."""
        from unittest.mock import MagicMock

        contact_index = ContactIndex()

        # Set fallback contacts BEFORE ContactManager
        contacts = [
            {"contact_id": 1, "first_name": "Boss"},
        ]
        contact_index.set_fallback_contacts(contacts)

        # Now set ContactManager
        mock_cm = MagicMock()
        contact_index.set_contact_manager(mock_cm)

        # Fallback should still contain the contact
        assert 1 in contact_index._fallback_contacts
        assert contact_index._fallback_contacts[1]["first_name"] == "Boss"

    def test_fallback_not_used_after_contact_manager_set(self):
        """Once ContactManager is set, fallback is not used for lookups."""
        from unittest.mock import MagicMock

        contact_index = ContactIndex()

        # Set fallback contacts BEFORE ContactManager
        contacts = [
            {"contact_id": 5, "first_name": "Fallback"},
        ]
        contact_index.set_fallback_contacts(contacts)

        # Now set ContactManager
        mock_cm = MagicMock()
        mock_cm.get_contact_info.return_value = {
            5: {"contact_id": 5, "first_name": "FromManager"},
        }
        contact_index.set_contact_manager(mock_cm)

        # get_contact should use ContactManager now, not fallback
        contact = contact_index.get_contact(contact_id=5)
        assert contact is not None
        assert contact["first_name"] == "FromManager"

        # ContactManager SHOULD have been called
        mock_cm.get_contact_info.assert_called_once_with(5)

    def test_fallback_ignores_contacts_without_contact_id(self):
        """set_fallback_contacts() ignores contacts missing contact_id."""
        contact_index = ContactIndex()

        contacts = [
            {"first_name": "NoId"},  # Missing contact_id
            {"contact_id": 1, "first_name": "HasId"},
        ]
        contact_index.set_fallback_contacts(contacts)

        # Only the contact with contact_id should be cached
        assert len(contact_index._fallback_contacts) == 1
        assert 1 in contact_index._fallback_contacts
