"""
tests/demo_meta/test_demo_meta.py
=================================

Unit tests for demo assistant metadata fetching and prospect details application.

These are symbolic tests that verify:
- DemoProspectDetails data class behavior
- fetch_demo_meta correctly parses Orchestra responses
- apply_prospect_to_boss_contact correctly updates the contact manager
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx

from unity.demo_meta import (
    DemoProspectDetails,
    fetch_demo_meta,
    apply_prospect_to_boss_contact,
    DEMO_META_FETCH_TIMEOUT,
)


class TestDemoProspectDetails:
    """Tests for the DemoProspectDetails data class."""

    def test_has_any_details_all_none(self):
        """has_any_details returns False when all fields are None."""
        prospect = DemoProspectDetails()
        assert prospect.has_any_details() is False

    def test_has_any_details_first_name_only(self):
        """has_any_details returns True when only first_name is set."""
        prospect = DemoProspectDetails(first_name="John")
        assert prospect.has_any_details() is True

    def test_has_any_details_surname_only(self):
        """has_any_details returns True when only surname is set."""
        prospect = DemoProspectDetails(surname="Doe")
        assert prospect.has_any_details() is True

    def test_has_any_details_email_only(self):
        """has_any_details returns True when only email is set."""
        prospect = DemoProspectDetails(email="john@example.com")
        assert prospect.has_any_details() is True

    def test_has_any_details_phone_only(self):
        """has_any_details returns True when only phone is set."""
        prospect = DemoProspectDetails(phone="+15555551234")
        assert prospect.has_any_details() is True

    def test_has_any_details_all_set(self):
        """has_any_details returns True when all fields are set."""
        prospect = DemoProspectDetails(
            first_name="John",
            surname="Doe",
            email="john@example.com",
            phone="+15555551234",
        )
        assert prospect.has_any_details() is True


class TestFetchDemoMeta:
    """Tests for the fetch_demo_meta function."""

    @pytest.mark.asyncio
    async def test_fetch_demo_meta_success(self):
        """fetch_demo_meta returns DemoProspectDetails on successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 42,
            "demo_assistant_id": 123,
            "label": "Test Demo",
            "prospect_first_name": "Jane",
            "prospect_surname": "Smith",
            "prospect_email": "jane@example.com",
            "prospect_phone": "+15555559999",
        }

        with patch.dict("os.environ", {"ORCHESTRA_ADMIN_KEY": "test-key"}):
            with patch("unity.demo_meta.httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = await fetch_demo_meta(42)

        assert result is not None
        assert result.first_name == "Jane"
        assert result.surname == "Smith"
        assert result.email == "jane@example.com"
        assert result.phone == "+15555559999"

    @pytest.mark.asyncio
    async def test_fetch_demo_meta_partial_data(self):
        """fetch_demo_meta handles partial prospect data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 42,
            "prospect_first_name": "John",
            # Other prospect fields are null/missing
        }

        with patch.dict("os.environ", {"ORCHESTRA_ADMIN_KEY": "test-key"}):
            with patch("unity.demo_meta.httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = await fetch_demo_meta(42)

        assert result is not None
        assert result.first_name == "John"
        assert result.surname is None
        assert result.email is None
        assert result.phone is None

    @pytest.mark.asyncio
    async def test_fetch_demo_meta_404_returns_none(self):
        """fetch_demo_meta returns None when demo not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.dict("os.environ", {"ORCHESTRA_ADMIN_KEY": "test-key"}):
            with patch("unity.demo_meta.httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = await fetch_demo_meta(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_demo_meta_no_api_key_returns_none(self):
        """fetch_demo_meta returns None when ORCHESTRA_ADMIN_KEY is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Ensure ORCHESTRA_ADMIN_KEY is not in environment
            import os

            os.environ.pop("ORCHESTRA_ADMIN_KEY", None)

            result = await fetch_demo_meta(42)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_demo_meta_timeout_returns_none(self):
        """fetch_demo_meta returns None on timeout."""
        with patch.dict("os.environ", {"ORCHESTRA_ADMIN_KEY": "test-key"}):
            with patch("unity.demo_meta.httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get.side_effect = httpx.TimeoutException("timeout")
                mock_client_class.return_value = mock_client

                result = await fetch_demo_meta(42)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_demo_meta_server_error_returns_none(self):
        """fetch_demo_meta returns None on server error."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.dict("os.environ", {"ORCHESTRA_ADMIN_KEY": "test-key"}):
            with patch("unity.demo_meta.httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = await fetch_demo_meta(42)

        assert result is None


class TestApplyProspectToBossContact:
    """Tests for the apply_prospect_to_boss_contact function."""

    def test_apply_full_prospect_details(self):
        """apply_prospect_to_boss_contact updates contact with all details."""
        mock_contact_manager = MagicMock()
        prospect = DemoProspectDetails(
            first_name="Jane",
            surname="Smith",
            email="jane@example.com",
            phone="+15555559999",
        )

        result = apply_prospect_to_boss_contact(mock_contact_manager, prospect)

        assert result is True
        mock_contact_manager.update_contact.assert_called_once_with(
            contact_id=1,
            first_name="Jane",
            surname="Smith",
            email_address="jane@example.com",
            phone_number="+15555559999",
        )

    def test_apply_partial_prospect_details(self):
        """apply_prospect_to_boss_contact only sets non-None fields."""
        mock_contact_manager = MagicMock()
        prospect = DemoProspectDetails(
            first_name="John",
            email="john@example.com",
        )

        result = apply_prospect_to_boss_contact(mock_contact_manager, prospect)

        assert result is True
        mock_contact_manager.update_contact.assert_called_once_with(
            contact_id=1,
            first_name="John",
            email_address="john@example.com",
        )

    def test_apply_no_prospect_details(self):
        """apply_prospect_to_boss_contact returns False when no details available."""
        mock_contact_manager = MagicMock()
        prospect = DemoProspectDetails()

        result = apply_prospect_to_boss_contact(mock_contact_manager, prospect)

        assert result is False
        mock_contact_manager.update_contact.assert_not_called()

    def test_apply_prospect_handles_exception(self):
        """apply_prospect_to_boss_contact returns False on exception."""
        mock_contact_manager = MagicMock()
        mock_contact_manager.update_contact.side_effect = Exception("DB error")
        prospect = DemoProspectDetails(first_name="Jane")

        result = apply_prospect_to_boss_contact(mock_contact_manager, prospect)

        assert result is False

    def test_apply_prospect_only_phone(self):
        """apply_prospect_to_boss_contact works with phone only."""
        mock_contact_manager = MagicMock()
        prospect = DemoProspectDetails(phone="+15555551234")

        result = apply_prospect_to_boss_contact(mock_contact_manager, prospect)

        assert result is True
        mock_contact_manager.update_contact.assert_called_once_with(
            contact_id=1,
            phone_number="+15555551234",
        )


class TestUpdateSessionContactsDemoModeProtection:
    """Tests for update_session_contacts behavior in demo mode.

    In demo mode:
    - contact_id=0 (assistant) is updated with assistant details
    - contact_id=1 (boss/prospect) is NOT updated (preserved from Orchestra meta)
    - contact_id=2 (demoer) IS updated with user_* details
    """

    @pytest.mark.asyncio
    async def test_update_session_contacts_creates_demoer_in_demo_mode(self):
        """In demo mode, update_session_contacts should create demoer contact (id=2)."""
        from unittest.mock import MagicMock, AsyncMock
        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )
        from unity.settings import SETTINGS

        # Save original and enable demo mode
        original_demo_mode = SETTINGS.DEMO_MODE
        SETTINGS.DEMO_MODE = True

        try:
            # Create mock ConversationManager with mock ContactManager
            mock_cm = MagicMock()
            mock_contact_manager = MagicMock()
            mock_cm.contact_manager = mock_contact_manager

            await update_session_contacts(
                cm=mock_cm,
                assistant_name="Lucy Test",
                assistant_number="+15555550000",
                assistant_email="lucy@test.com",
                user_name="Demoer Person",  # This is demoer, NOT prospect
                user_number="+15555551111",
                user_email="demoer@unify.ai",
            )

            # Verify contact updates
            calls = mock_contact_manager.update_contact.call_args_list
            contact_ids_updated = [call.kwargs.get("contact_id") for call in calls]

            assert (
                0 in contact_ids_updated
            ), "Assistant contact (id=0) should be updated"
            assert 1 not in contact_ids_updated, (
                "Boss contact (id=1) should NOT be updated in demo mode "
                "(prospect details come from Orchestra meta)"
            )
            assert (
                2 in contact_ids_updated
            ), "Demoer contact (id=2) should be updated with user_* details"

            # Verify demoer contact has correct details
            demoer_call = next(
                call for call in calls if call.kwargs.get("contact_id") == 2
            )
            assert demoer_call.kwargs.get("first_name") == "Demoer"
            assert demoer_call.kwargs.get("surname") == "Person"
            assert demoer_call.kwargs.get("phone_number") == "+15555551111"
            assert demoer_call.kwargs.get("email_address") == "demoer@unify.ai"

        finally:
            SETTINGS.DEMO_MODE = original_demo_mode

    @pytest.mark.asyncio
    async def test_update_session_contacts_updates_boss_in_normal_mode(self):
        """In normal mode, update_session_contacts should update both contacts."""
        from unittest.mock import MagicMock
        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )
        from unity.settings import SETTINGS

        # Save original and disable demo mode
        original_demo_mode = SETTINGS.DEMO_MODE
        SETTINGS.DEMO_MODE = False

        try:
            mock_cm = MagicMock()
            mock_contact_manager = MagicMock()
            mock_cm.contact_manager = mock_contact_manager

            await update_session_contacts(
                cm=mock_cm,
                assistant_name="Lucy Test",
                assistant_number="+15555550000",
                assistant_email="lucy@test.com",
                user_name="Real Boss",
                user_number="+15555551111",
                user_email="boss@company.com",
            )

            # Verify both contacts were updated
            calls = mock_contact_manager.update_contact.call_args_list
            contact_ids_updated = [call.kwargs.get("contact_id") for call in calls]

            assert 0 in contact_ids_updated, "Assistant contact should be updated"
            assert (
                1 in contact_ids_updated
            ), "Boss contact should be updated in normal mode"

        finally:
            SETTINGS.DEMO_MODE = original_demo_mode
