"""
tests/conversation_manager/core/test_implicit_contact_detail_creation.py
========================================================================

Tests for implicit contact-detail attachment on comms tools.

When a contact already exists in active_conversations but is missing a
detail (phone_number, email_address), the caller may provide it inline.
The detail is saved to the contact and the communication proceeds.

Overwriting an existing detail with a *different* value is rejected with
an error directing the caller to use ``act`` instead.  Providing the
*same* value that is already on file is treated as a no-op (happy path).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.contact_manager.simulated import SimulatedContactManager
from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.notifications import NotificationBar

# =============================================================================
# Fixtures
# =============================================================================


def _setup_mock_contacts(
    contact_index: ContactIndex,
    contacts: list[dict],
) -> SimulatedContactManager:
    """Populate a SimulatedContactManager and wire it into the ContactIndex."""
    contact_manager = SimulatedContactManager()
    for contact_data in contacts:
        contact_manager.update_contact(
            contact_id=contact_data["contact_id"],
            first_name=contact_data.get("first_name"),
            surname=contact_data.get("surname"),
            email_address=contact_data.get("email_address"),
            phone_number=contact_data.get("phone_number"),
            should_respond=contact_data.get("should_respond", True),
        )
    contact_index.set_contact_manager(contact_manager)
    return contact_manager


@pytest.fixture
def mock_cm():
    """Minimal mock ConversationManager."""
    cm = MagicMock()
    cm.mode = "text"
    cm.contact_index = ContactIndex()
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.notifications_bar = NotificationBar()
    cm.chat_history = []
    cm.assistant_number = "+15555550000"
    cm.assistant_email = "assistant@test.com"
    cm.assistant_email_provider = "google_workspace"
    cm.assistant_whatsapp_number = ""
    cm.assistant_discord_bot_id = ""
    cm.call_manager.has_active_call = False
    cm.call_manager.has_active_google_meet = False
    cm.call_manager.has_active_teams_meet = False
    cm.call_manager.has_gmeet_presenting = False
    cm.call_manager.has_teams_presenting = False
    cm.call_manager._meet_joining = False
    cm.call_manager._whatsapp_call_joining = False
    cm.assistant_has_teams = False
    cm.contact_manager = _setup_mock_contacts(cm.contact_index, [])
    return cm


@pytest.fixture
def brain_action_tools(mock_cm):
    """ConversationManagerBrainActionTools with mocked event broker."""
    with patch(
        "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        mock_broker.return_value.publish = AsyncMock()
        tools = ConversationManagerBrainActionTools(mock_cm)
        yield tools


# =============================================================================
# send_sms – implicit phone_number attachment
# =============================================================================


class TestSendSmsImplicitPhoneNumber:
    """Implicit phone_number attachment for send_sms."""

    @pytest.mark.asyncio
    async def test_attaches_phone_when_missing(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Contact has no phone → inline phone_number is saved and SMS proceeds."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@example.com",
            "should_respond": True,
            # No phone_number
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
            phone_number="+15559990000",
        )

        assert result["status"] == "ok"
        # Verify the phone was persisted
        updated = mock_cm.contact_index.get_contact(5)
        assert updated["phone_number"] == "+15559990000"

    @pytest.mark.asyncio
    async def test_same_phone_is_happy_path(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Providing the same phone that's already on file is fine."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15559990000",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
            phone_number="+15559990000",
        )

        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_different_phone_returns_error(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Providing a different phone than what's on file is rejected."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15551111111",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
            phone_number="+15552222222",
        )

        assert result["status"] == "error"
        assert "act" in result["error"].lower()
        assert "+15551111111" in result["error"]

    @pytest.mark.asyncio
    async def test_no_phone_no_inline_returns_error(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Missing phone and no inline value → error with hint."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@example.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
        )

        assert result["status"] == "error"
        assert "phone" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_existing_phone_no_inline_is_happy_path(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Contact already has phone, no inline provided → proceeds normally."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15559990000",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
        )

        assert result["status"] == "ok"


# =============================================================================
# make_call – implicit phone_number attachment
# =============================================================================


class TestMakeCallImplicitPhoneNumber:
    """Implicit phone_number attachment for make_call."""

    @pytest.mark.asyncio
    async def test_attaches_phone_when_missing(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Contact has no phone → inline phone_number is saved and call proceeds."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@example.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.make_call(
            contact_id=5,
            context="Calling to discuss project timeline",
            phone_number="+15559990000",
        )

        assert result["status"] == "ok"
        updated = mock_cm.contact_index.get_contact(5)
        assert updated["phone_number"] == "+15559990000"

    @pytest.mark.asyncio
    async def test_same_phone_is_happy_path(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Providing the same phone that's already on file is fine."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15559990000",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.make_call(
            contact_id=5,
            context="Calling to discuss project timeline",
            phone_number="+15559990000",
        )

        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_different_phone_returns_error(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Providing a different phone than what's on file is rejected."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15551111111",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.make_call(
            contact_id=5,
            context="Calling to discuss project timeline",
            phone_number="+15552222222",
        )

        assert result["status"] == "error"
        assert "act" in result["error"].lower()
        assert "+15551111111" in result["error"]

    @pytest.mark.asyncio
    async def test_no_phone_no_inline_returns_error(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Missing phone and no inline value → error."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@example.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.make_call(
            contact_id=5,
            context="Calling to discuss project timeline",
        )

        assert result["status"] == "error"
        assert "phone" in result["error"].lower()


# =============================================================================
# send_email – implicit email_address attachment (Option A: mixed-type lists)
# =============================================================================


class TestSendEmailImplicitEmailAddress:
    """Implicit email_address attachment for send_email."""

    @pytest.mark.asyncio
    async def test_attaches_email_when_missing(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Contact has no email → inline email_address is saved and email sends."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15559990000",
            "should_respond": True,
            # No email_address
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        with patch(
            "unity.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-123"}

            result = await brain_action_tools.send_email(
                to=[{"contact_id": 5, "email_address": "alice@example.com"}],
                subject="Test",
                body="Hello",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            assert mock_send.call_args.kwargs["to"] == ["alice@example.com"]

        # Verify the email was persisted
        updated = mock_cm.contact_index.get_contact(5)
        assert updated["email_address"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_same_email_is_happy_path(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Providing the same email already on file is fine."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@example.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        with patch(
            "unity.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-123"}

            result = await brain_action_tools.send_email(
                to=[{"contact_id": 5, "email_address": "alice@example.com"}],
                subject="Test",
                body="Hello",
            )

            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_different_email_returns_error(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Providing a different email than what's on file is rejected."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@old.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.send_email(
            to=[{"contact_id": 5, "email_address": "alice@new.com"}],
            subject="Test",
            body="Hello",
        )

        assert result["status"] == "error"
        assert "act" in result["error"].lower()
        assert "alice@old.com" in result["error"]

    @pytest.mark.asyncio
    async def test_int_contact_id_still_works(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Plain int contact_id (existing path) still works."""
        contact = {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@example.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        with patch(
            "unity.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-123"}

            result = await brain_action_tools.send_email(
                to=[5],
                subject="Test",
                body="Hello",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            assert mock_send.call_args.kwargs["to"] == ["alice@example.com"]

    @pytest.mark.asyncio
    async def test_mixed_list_int_and_dict(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """to list can mix plain ints and dicts."""
        contacts = [
            {
                "contact_id": 5,
                "first_name": "Alice",
                "surname": "Smith",
                "email_address": "alice@example.com",
                "should_respond": True,
            },
            {
                "contact_id": 6,
                "first_name": "Bob",
                "surname": "Jones",
                "phone_number": "+15551111111",
                "should_respond": True,
                # No email
            },
        ]
        _setup_mock_contacts(mock_cm.contact_index, contacts)

        with patch(
            "unity.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-123"}

            result = await brain_action_tools.send_email(
                to=[
                    5,  # Alice – has email
                    {
                        "contact_id": 6,
                        "email_address": "bob@example.com",
                    },  # Bob – attach
                ],
                subject="Test",
                body="Hello",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            sent_to = mock_send.call_args.kwargs["to"]
            assert "alice@example.com" in sent_to
            assert "bob@example.com" in sent_to

        # Verify Bob's email was persisted
        bob = mock_cm.contact_index.get_contact(6)
        assert bob["email_address"] == "bob@example.com"

    @pytest.mark.asyncio
    async def test_error_in_one_recipient_aborts_whole_send(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """If any recipient has a detail conflict, entire send is aborted."""
        contacts = [
            {
                "contact_id": 5,
                "first_name": "Alice",
                "surname": "Smith",
                "email_address": "alice@example.com",
                "should_respond": True,
            },
            {
                "contact_id": 6,
                "first_name": "Bob",
                "surname": "Jones",
                "email_address": "bob@old.com",
                "should_respond": True,
            },
        ]
        _setup_mock_contacts(mock_cm.contact_index, contacts)

        result = await brain_action_tools.send_email(
            to=[
                5,  # Alice – fine
                {"contact_id": 6, "email_address": "bob@new.com"},  # conflict
            ],
            subject="Test",
            body="Hello",
        )

        assert result["status"] == "error"
        assert "act" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cc_and_bcc_support_inline_email(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """cc and bcc also accept the dict form for inline email attachment."""
        contacts = [
            {
                "contact_id": 5,
                "first_name": "Alice",
                "surname": "Smith",
                "email_address": "alice@example.com",
                "should_respond": True,
            },
            {
                "contact_id": 6,
                "first_name": "Bob",
                "surname": "Jones",
                "phone_number": "+15551111111",
                "should_respond": True,
                # No email
            },
            {
                "contact_id": 7,
                "first_name": "Charlie",
                "surname": "Brown",
                "phone_number": "+15552222222",
                "should_respond": True,
                # No email
            },
        ]
        _setup_mock_contacts(mock_cm.contact_index, contacts)

        with patch(
            "unity.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-123"}

            result = await brain_action_tools.send_email(
                to=[5],
                cc=[{"contact_id": 6, "email_address": "bob@example.com"}],
                bcc=[{"contact_id": 7, "email_address": "charlie@example.com"}],
                subject="Test",
                body="Hello",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            assert mock_send.call_args.kwargs["cc"] == ["bob@example.com"]
            assert mock_send.call_args.kwargs["bcc"] == ["charlie@example.com"]
