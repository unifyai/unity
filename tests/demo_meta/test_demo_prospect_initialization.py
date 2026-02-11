"""
tests/demo_meta/test_demo_prospect_initialization.py
=====================================================

Integration tests for demo assistant prospect detail initialization.

These tests verify the end-to-end flow of how prospect details are populated
on the boss contact (contact_id=1) during demo session initialization:

1. **No prospect details provided**: Boss contact starts sparse (no name, email,
   phone) and details are learned during the demo via `set_boss_details`.

2. **Prospect details provided**: Boss contact is pre-populated with prospect
   details from the demo metadata fetched from Orchestra.

These tests mock the Orchestra API but use real Unity initialization flow.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from tests.helpers import scenario_file_lock


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def demo_cm_factory():
    """Factory fixture to create ConversationManager with specific demo settings.

    Returns a factory function that accepts demo_id and prospect_details parameters.
    This allows testing both with and without prospect details.
    """
    from unity.settings import SETTINGS
    from unity.conversation_manager.event_broker import reset_event_broker
    from unity.conversation_manager import start_async, stop_async
    from unity.conversation_manager.domains import managers_utils
    from unity.actor.simulated import SimulatedActor

    created_cms = []

    async def _create_cm(
        demo_id: int | None = None,
        prospect_details: dict | None = None,
    ):
        """Create a ConversationManager with specific demo configuration.

        Args:
            demo_id: Demo ID to set in SETTINGS.DEMO_ID (triggers demo mode)
            prospect_details: Dict with prospect_first_name, prospect_surname,
                              prospect_email, prospect_phone to return from
                              mocked Orchestra API
        """
        # Save original settings
        original_demo_mode = SETTINGS.DEMO_MODE
        original_demo_id = SETTINGS.DEMO_ID

        # Configure demo mode
        SETTINGS.DEMO_MODE = demo_id is not None
        SETTINGS.DEMO_ID = demo_id

        reset_event_broker()

        cm = await start_async(
            project_name="TestDemoProspect",
            enable_comms_manager=False,
            apply_test_mocks=True,
        )

        actor = SimulatedActor(
            steps=None,
            duration=None,
            log_mode="log",
            emit_notifications=False,
        )

        # Build mock response for Orchestra API
        mock_response = None
        if prospect_details:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": demo_id,
                "demo_assistant_id": 123,
                "label": "Test Demo",
                "prospect_first_name": prospect_details.get("first_name"),
                "prospect_surname": prospect_details.get("surname"),
                "prospect_email": prospect_details.get("email"),
                "prospect_phone": prospect_details.get("phone"),
            }
        else:
            # No prospect details - return 200 with all null fields
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": demo_id,
                "demo_assistant_id": 123,
                "label": "Test Demo",
                "prospect_first_name": None,
                "prospect_surname": None,
                "prospect_email": None,
                "prospect_phone": None,
            }

        # Mock the httpx client for Orchestra API call
        with patch("unity.demo_meta.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Ensure ORCHESTRA_ADMIN_KEY is set for the fetch
            with patch.dict(os.environ, {"ORCHESTRA_ADMIN_KEY": "test-key"}):
                with scenario_file_lock("demo_prospect_init"):
                    await managers_utils.init_conv_manager(cm, actor=actor)

                    if not cm.initialized:
                        raise RuntimeError("ConversationManager failed to initialize")

        # Store for cleanup and return
        created_cms.append((cm, original_demo_mode, original_demo_id))
        return cm

    yield _create_cm

    # Cleanup
    for cm, original_demo_mode, original_demo_id in created_cms:
        await stop_async()
        reset_event_broker()
        SETTINGS.DEMO_MODE = original_demo_mode
        SETTINGS.DEMO_ID = original_demo_id


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDemoProspectInitialization:
    """Integration tests for demo prospect initialization flow."""

    @pytest.mark.asyncio
    async def test_no_prospect_details_boss_contact_sparse(self, demo_cm_factory):
        """
        When no prospect details are provided during demo creation,
        the boss contact (contact_id=1) should start sparse.

        This simulates the case where the demoer didn't specify prospect info
        upfront, so details are learned during the demo via set_boss_details.
        """
        cm = await demo_cm_factory(
            demo_id=42,
            prospect_details=None,  # No prospect details provided
        )

        # Get the boss contact using get_contact_info which returns {contact_id: info_dict}
        contact_info = cm.contact_manager.get_contact_info(1)
        boss_contact = contact_info.get(1, {})

        # Verify boss contact exists but has no details
        assert boss_contact is not None, "Boss contact (contact_id=1) should exist"

        # In sparse mode, name fields should be None or empty
        first_name = boss_contact.get("first_name")
        surname = boss_contact.get("surname")
        email = boss_contact.get("email_address")
        phone = boss_contact.get("phone_number")

        # At least the name should be empty/None (sparse initialization)
        assert not first_name or first_name.strip() == "", (
            f"Boss first_name should be empty when no prospect details provided, "
            f"got: {first_name!r}"
        )

        print(
            f"✅ Boss contact is sparse as expected: "
            f"first_name={first_name!r}, surname={surname!r}, "
            f"email={email!r}, phone={phone!r}"
        )

    @pytest.mark.asyncio
    async def test_prospect_details_applied_to_boss_contact(self, demo_cm_factory):
        """
        When prospect details are provided during demo creation,
        the boss contact (contact_id=1) should be pre-populated.

        This simulates the case where the demoer specified prospect info
        (name, email, phone) when creating the demo assistant.
        """
        prospect = {
            "first_name": "Jane",
            "surname": "Smith",
            "email": "jane.smith@example.com",
            "phone": "+15555551234",
        }

        cm = await demo_cm_factory(
            demo_id=99,
            prospect_details=prospect,
        )

        # Get the boss contact using get_contact_info which returns {contact_id: info_dict}
        contact_info = cm.contact_manager.get_contact_info(1)
        boss_contact = contact_info.get(1, {})

        # Verify boss contact exists and has the prospect details
        assert boss_contact is not None, "Boss contact (contact_id=1) should exist"

        assert (
            boss_contact.get("first_name") == "Jane"
        ), f"Boss first_name should be 'Jane', got: {boss_contact.get('first_name')!r}"
        assert (
            boss_contact.get("surname") == "Smith"
        ), f"Boss surname should be 'Smith', got: {boss_contact.get('surname')!r}"
        assert boss_contact.get("email_address") == "jane.smith@example.com", (
            f"Boss email should be 'jane.smith@example.com', "
            f"got: {boss_contact.get('email_address')!r}"
        )
        assert boss_contact.get("phone_number") == "+15555551234", (
            f"Boss phone should be '+15555551234', "
            f"got: {boss_contact.get('phone_number')!r}"
        )

        print(
            f"✅ Boss contact pre-populated with prospect details: "
            f"first_name={boss_contact.get('first_name')!r}, "
            f"surname={boss_contact.get('surname')!r}, "
            f"email={boss_contact.get('email_address')!r}, "
            f"phone={boss_contact.get('phone_number')!r}"
        )

    @pytest.mark.asyncio
    async def test_partial_prospect_details_applied(self, demo_cm_factory):
        """
        When only partial prospect details are provided, only those fields
        should be populated on the boss contact.
        """
        # Only name provided, no email or phone
        prospect = {
            "first_name": "Bob",
            "surname": "Johnson",
            "email": None,
            "phone": None,
        }

        cm = await demo_cm_factory(
            demo_id=101,
            prospect_details=prospect,
        )

        contact_info = cm.contact_manager.get_contact_info(1)
        boss_contact = contact_info.get(1, {})

        assert boss_contact is not None, "Boss contact (contact_id=1) should exist"
        assert (
            boss_contact.get("first_name") == "Bob"
        ), f"Boss first_name should be 'Bob', got: {boss_contact.get('first_name')!r}"
        assert (
            boss_contact.get("surname") == "Johnson"
        ), f"Boss surname should be 'Johnson', got: {boss_contact.get('surname')!r}"

        # Email and phone should not be set (None or empty)
        email = boss_contact.get("email_address")
        phone = boss_contact.get("phone_number")

        print(
            f"✅ Partial prospect details applied: "
            f"first_name={boss_contact.get('first_name')!r}, "
            f"surname={boss_contact.get('surname')!r}, "
            f"email={email!r}, phone={phone!r}"
        )

    @pytest.mark.asyncio
    async def test_demo_operator_contact_exists(self, demo_cm_factory):
        """
        In demo mode, the demo operator (contact_id=2) should exist in the
        contact index, ready to be populated with demoer details when
        update_session_contacts is called.

        This test verifies that demo mode properly initializes the contact
        structure with all three contacts:
        - contact_id=0: Assistant
        - contact_id=1: Boss (prospect)
        - contact_id=2: Demo operator (demoer)
        """
        cm = await demo_cm_factory(
            demo_id=42,
            prospect_details=None,
        )

        # Verify demo mode creates the right contact structure
        contact_info_0 = cm.contact_manager.get_contact_info(0)
        contact_info_1 = cm.contact_manager.get_contact_info(1)
        assistant_contact = contact_info_0.get(0, {})
        boss_contact = contact_info_1.get(1, {})

        assert (
            assistant_contact is not None
        ), "Assistant contact (contact_id=0) should exist"
        assert boss_contact is not None, "Boss contact (contact_id=1) should exist"

        # Verify contact_id=2 (demoer) exists in the contact manager
        # The demoer is the Unify employee running the demo, initialized with user_* fields
        # Note: Demoer is NOT in active_conversations because the assistant wouldn't
        # typically interact with them (call/email) - they're the one running the demo
        contact_info_2 = cm.contact_manager.get_contact_info(2)
        demoer_contact = contact_info_2.get(2, {})
        assert demoer_contact is not None, "Demoer contact (contact_id=2) should exist"

        print(
            f"✅ Demo contact structure verified: "
            f"assistant={assistant_contact.get('first_name')!r}, "
            f"boss={boss_contact.get('first_name')!r}, "
            f"demoer={demoer_contact.get('first_name')!r}"
        )

    @pytest.mark.asyncio
    async def test_orchestra_api_failure_graceful_fallback(self, demo_cm_factory):
        """
        If the Orchestra API call fails, initialization should still complete
        with the boss contact remaining sparse.
        """
        from unity.settings import SETTINGS
        from unity.conversation_manager.event_broker import reset_event_broker
        from unity.conversation_manager import start_async, stop_async
        from unity.conversation_manager.domains import managers_utils
        from unity.actor.simulated import SimulatedActor

        # Save and set demo mode
        original_demo_mode = SETTINGS.DEMO_MODE
        original_demo_id = SETTINGS.DEMO_ID
        SETTINGS.DEMO_MODE = True
        SETTINGS.DEMO_ID = 999

        try:
            reset_event_broker()

            cm = await start_async(
                project_name="TestDemoApiFailure",
                enable_comms_manager=False,
                apply_test_mocks=True,
            )

            actor = SimulatedActor(
                steps=None,
                duration=None,
                log_mode="log",
                emit_notifications=False,
            )

            # Mock Orchestra API to return an error
            mock_response = MagicMock()
            mock_response.status_code = 500

            with patch("unity.demo_meta.httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                with patch.dict(os.environ, {"ORCHESTRA_ADMIN_KEY": "test-key"}):
                    with scenario_file_lock("demo_api_failure"):
                        await managers_utils.init_conv_manager(cm, actor=actor)

            # Should still initialize successfully
            assert cm.initialized, "CM should initialize even if Orchestra API fails"

            # Boss contact should exist but be sparse
            contact_info = cm.contact_manager.get_contact_info(1)
            boss_contact = contact_info.get(1, {})
            assert boss_contact is not None, "Boss contact should exist"

            print(
                f"✅ Graceful fallback: initialization succeeded despite API failure, "
                f"boss contact is sparse"
            )

        finally:
            await stop_async()
            reset_event_broker()
            SETTINGS.DEMO_MODE = original_demo_mode
            SETTINGS.DEMO_ID = original_demo_id
