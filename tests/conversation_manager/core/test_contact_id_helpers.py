"""Unit tests for the per-body contact-id helpers in comms_manager.

- ``_boss_contact_from_contacts`` resolves the boss contact out of an
  Adapters-supplied contact list via ``SESSION_DETAILS.user.contact_id``,
  returning ``None`` when the body's ``ContactMembership`` overlay has
  not materialised yet.
- ``_contact_ids_from_event`` extracts ``(self, boss)`` ids from the two
  bootstrap envelopes (JSON Secret + Pub/Sub ``assistant_update``),
  tolerating missing and malformed values.
"""

from __future__ import annotations

import pytest

from unity.conversation_manager.comms_manager import (
    _boss_contact_from_contacts,
    _contact_ids_from_event,
)


@pytest.fixture
def mock_session_details():
    """Patch ``comms_manager.SESSION_DETAILS`` for helper-level tests."""

    from unittest.mock import patch

    with patch("unity.conversation_manager.comms_manager.SESSION_DETAILS") as mock:
        mock.user.contact_id = 1
        mock.assistant.contact_id = 0
        yield mock


class TestBossContactFromContacts:
    """Resolution uses ``SESSION_DETAILS.user.contact_id``, not hard-coded ``1``."""

    def test_returns_contact_matching_session_boss_id(
        self,
        mock_session_details,
    ):
        mock_session_details.user.contact_id = 42
        contacts = [
            {"contact_id": 0, "first_name": "Self"},
            {"contact_id": 42, "first_name": "Boss"},
            {"contact_id": 7, "first_name": "Someone else"},
        ]

        resolved = _boss_contact_from_contacts(contacts)

        assert resolved is not None
        assert resolved["contact_id"] == 42
        assert resolved["first_name"] == "Boss"

    def test_hive_boss_id_is_not_one(self, mock_session_details):
        """Hive bodies carry arbitrary boss ids; the helper routes through
        the resolved ``SESSION_DETAILS.user.contact_id`` rather than ``1``."""

        mock_session_details.user.contact_id = 9999
        contacts = [
            {"contact_id": 1, "first_name": "Unrelated"},
            {"contact_id": 9999, "first_name": "HiveBoss"},
        ]

        resolved = _boss_contact_from_contacts(contacts)

        assert resolved is not None
        assert resolved["first_name"] == "HiveBoss"

    def test_returns_none_when_boss_id_unresolved(self, mock_session_details):
        """Freshly-provisioned body (overlay not yet materialised) must
        degrade to ``None`` — callers log + skip rather than crashing."""

        mock_session_details.user.contact_id = None
        contacts = [
            {"contact_id": 1, "first_name": "Boss"},
            {"contact_id": 0, "first_name": "Self"},
        ]

        assert _boss_contact_from_contacts(contacts) is None

    def test_returns_none_when_contacts_list_missing_boss_row(
        self,
        mock_session_details,
    ):
        mock_session_details.user.contact_id = 42
        contacts = [
            {"contact_id": 0, "first_name": "Self"},
            {"contact_id": 7, "first_name": "Someone else"},
        ]

        assert _boss_contact_from_contacts(contacts) is None

    def test_returns_none_on_empty_contacts_list(self, mock_session_details):
        mock_session_details.user.contact_id = 1

        assert _boss_contact_from_contacts([]) is None


class TestContactIdsFromEvent:
    """Bootstrap envelope allowlist (JSON Secret + Pub/Sub `assistant_update`)."""

    def test_reads_integers(self):
        event = {"self_contact_id": 0, "boss_contact_id": 1}

        assert _contact_ids_from_event(event) == (0, 1)

    def test_hive_scoped_integers(self):
        event = {"self_contact_id": 123, "boss_contact_id": 456}

        assert _contact_ids_from_event(event) == (123, 456)

    def test_missing_fields_map_to_none(self):
        assert _contact_ids_from_event({}) == (None, None)

    def test_none_fields_map_to_none(self):
        event = {"self_contact_id": None, "boss_contact_id": None}

        assert _contact_ids_from_event(event) == (None, None)

    def test_stringified_integers_coerce(self):
        """Orchestra may serialize ids over JSON as strings on some
        payload paths; the helper must round-trip them to ``int``."""

        event = {"self_contact_id": "42", "boss_contact_id": "17"}

        assert _contact_ids_from_event(event) == (42, 17)

    def test_malformed_value_maps_to_none(self):
        event = {"self_contact_id": "not-a-number", "boss_contact_id": "also-bad"}

        assert _contact_ids_from_event(event) == (None, None)
