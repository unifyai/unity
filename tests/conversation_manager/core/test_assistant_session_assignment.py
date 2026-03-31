from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from unity.conversation_manager.comms_manager import CommsManager


@pytest.mark.asyncio
async def test_poll_for_assignment_bootstraps_from_assistant_session():
    event_broker = MagicMock()
    event_broker.publish = AsyncMock()

    with (
        patch("unity.conversation_manager.comms_manager.SESSION_DETAILS") as session,
        patch(
            "unity.conversation_manager.comms_manager.wait_for_assistant_session_name",
            return_value="assistant-session-42",
        ),
        patch(
            "unity.conversation_manager.comms_manager.read_assistant_session",
            return_value={
                "spec": {"startupSecretRef": "assistant-session-bootstrap-42"},
            },
        ),
        patch(
            "unity.conversation_manager.comms_manager.read_session_bootstrap_secret",
            return_value={
                "api_key": "user-key",
                "medium": "startup",
                "assistant_id": "42",
                "user_id": "7",
                "assistant_first_name": "Ada",
                "assistant_surname": "Lovelace",
                "assistant_age": "28",
                "assistant_nationality": "UK",
                "assistant_timezone": "UTC",
                "assistant_about": "Researcher",
                "assistant_number": "",
                "assistant_email": "ada@example.com",
                "user_first_name": "Grace",
                "user_surname": "Hopper",
                "user_number": "",
                "user_email": "grace@example.com",
                "voice_provider": "",
                "voice_id": "",
                "desktop_mode": "ubuntu",
                "team_ids": [],
                "org_id": None,
                "demo_id": None,
            },
        ),
        patch(
            "unity.conversation_manager.comms_manager.mark_job_container_ready",
        ) as mark_ready,
    ):
        session.assistant.agent_id = None
        session.assistant.email = "ada@example.com"
        session.user.first_name = "Grace"
        session.user.surname = "Hopper"
        session.user.number = ""
        session.user.email = "grace@example.com"

        cm = CommsManager(event_broker)
        cm.subscribe_to_topic = MagicMock()

        with patch(
            "unity.conversation_manager.comms_manager.SETTINGS",
        ) as settings:
            settings.conversation.JOB_NAME = "unity-2026-03-30-u1234"
            settings.ENV_SUFFIX = ""
            await cm._poll_for_assignment()

        assert session.assistant.agent_id == 42
        cm.subscribe_to_topic.assert_called_once_with("unity-42-sub", max_messages=10)
        event_broker.publish.assert_awaited()
        publish_channel, payload = event_broker.publish.await_args.args
        assert publish_channel == "app:comms:startup"
        assert "assistant_id" in payload
        mark_ready.assert_called_once_with("unity-2026-03-30-u1234")
