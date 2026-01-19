"""
tests/test_conversation_manager/test_comms_manager.py
=====================================================

Tests for CommsManager utilities and contact schema consistency.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock


# =============================================================================
# Contact Schema Tests
# =============================================================================

# Required keys for contact dictionaries throughout the system
REQUIRED_CONTACT_KEYS = {
    "contact_id",
    "first_name",
    "surname",
    "phone_number",
    "email_address",  # NOT "email" - must be "email_address" for consistency
}


def test_get_local_contact_has_correct_keys():
    """
    Verify that _get_local_contact() returns a contact dict with the
    correct field names. Specifically, the email field must be 'email_address',
    not 'email', to match the expected contact schema used throughout the system.
    """
    # Mock SESSION_DETAILS to avoid needing real session context
    mock_user = MagicMock()
    mock_user.name = "Test User"
    mock_user.number = "+15555551234"
    mock_user.email = "test@example.com"

    mock_session = MagicMock()
    mock_session.user = mock_user

    with patch(
        "unity.conversation_manager.comms_manager.SESSION_DETAILS",
        mock_session,
    ):
        from unity.conversation_manager.comms_manager import _get_local_contact

        contact = _get_local_contact()

    # Verify all required keys are present
    assert set(contact.keys()) == REQUIRED_CONTACT_KEYS, (
        f"Contact dict has unexpected keys. "
        f"Expected: {REQUIRED_CONTACT_KEYS}, Got: {set(contact.keys())}"
    )

    # Explicitly verify 'email_address' is used, not 'email'
    assert (
        "email_address" in contact
    ), "Contact must use 'email_address' key, not 'email'"
    assert (
        "email" not in contact
    ), "Contact should NOT have 'email' key - use 'email_address' instead"

    # Verify the value is correctly mapped
    assert contact["email_address"] == "test@example.com"
