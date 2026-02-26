"""
tests/conversation_manager/core/test_comms_utils.py
=============================================================

Unit tests for comms_utils functions, particularly attachment handling.

Marked ``real_comms_functions`` to opt out of the universal comms stub
in the parent conftest.  These tests call the real ``comms_utils``
implementations with mocked dependencies (aiohttp, SESSION_DETAILS, etc.).

Every test that mocks aiohttp asserts the **URL and HTTP method** used for
the request, not just the response.  This catches wrong-service bugs (like
posting to COMMS_URL when the endpoint lives on the adapters service).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains import comms_utils

pytestmark = pytest.mark.real_comms_functions

COMMS_URL = "http://comms.test:8080"
ADAPTERS_URL = "http://adapters.test:8081"


def _mock_aiohttp_session(response_json=None, raise_on_status=None):
    """Build a mock aiohttp session + response pair."""
    mock_response = MagicMock()
    if raise_on_status:
        mock_response.raise_for_status = MagicMock(side_effect=raise_on_status)
    else:
        mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=response_json or {})

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
    )
    mock_session.get = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


class TestUploadUnifyAttachment:
    """Tests for upload_unify_attachment function."""

    @pytest.mark.asyncio
    async def test_upload_success(self):
        """Successfully uploads file and returns attachment details."""
        mock_session = _mock_aiohttp_session(
            response_json={
                "id": "test-uuid-123",
                "filename": "document.pdf",
                "url": "https://storage.googleapis.com/signed-url",
            },
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.agent_id = 42
            mock_settings.conversation.ADAPTERS_URL = ADAPTERS_URL

            result = await comms_utils.upload_unify_attachment(
                file_content=b"test file content",
                filename="document.pdf",
            )

            assert result["id"] == "test-uuid-123"
            assert result["filename"] == "document.pdf"
            assert "storage.googleapis.com" in result["url"]

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{ADAPTERS_URL}/unify/attachment"

    @pytest.mark.asyncio
    async def test_upload_with_custom_assistant_id(self):
        """Uses provided assistant_id instead of session default."""
        mock_session = _mock_aiohttp_session(
            response_json={"id": "uuid", "filename": "file.txt", "url": "https://url"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.conversation.ADAPTERS_URL = ADAPTERS_URL

            result = await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
                assistant_id="custom-assistant-id",
            )

            assert result["id"] == "uuid"
            mock_session.post.assert_called_once()
            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{ADAPTERS_URL}/unify/attachment"

    @pytest.mark.asyncio
    async def test_upload_failure_returns_error(self):
        """Returns error dict when upload fails."""
        mock_session = _mock_aiohttp_session(
            raise_on_status=Exception("Connection refused"),
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.agent_id = 42
            mock_settings.conversation.ADAPTERS_URL = ADAPTERS_URL

            result = await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
            )

            assert result.get("success") is False
            assert "error" in result

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{ADAPTERS_URL}/unify/attachment"

    @pytest.mark.asyncio
    async def test_upload_posts_to_adapters_not_comms_app(self):
        """The /unify/attachment endpoint is on the adapters service.

        COMMS_URL points to the comms-app (phone, gmail, infra routes).
        The attachment upload endpoint lives on the separate adapters service.
        Using COMMS_URL results in a 404 because the comms-app has no /unify/* routes.
        """
        mock_session = _mock_aiohttp_session(
            response_json={"id": "uuid", "filename": "file.txt", "url": "https://url"},
        )

        comms_app_url = (
            "https://unity-comms-app-staging-262420637606.us-central1.run.app"
        )
        adapters_url = "https://unity-adapters-staging-ky4ja5fxna-uc.a.run.app"

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.agent_id = 42
            mock_settings.conversation.COMMS_URL = comms_app_url
            mock_settings.conversation.ADAPTERS_URL = adapters_url

            await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
            )

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url.startswith(adapters_url), (
                f"upload_unify_attachment must post to ADAPTERS_URL "
                f"({adapters_url}), not COMMS_URL ({comms_app_url}). "
                f"Got: {posted_url}"
            )
            assert posted_url == f"{adapters_url}/unify/attachment"


class TestSendSms:
    """Tests for send_sms_message_via_number."""

    @pytest.mark.asyncio
    async def test_send_sms_url_and_body(self):
        """Posts to /phone/send-text on COMMS_URL with correct JSON body."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "sid": "SM123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.number = "+15551234567"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            result = await comms_utils.send_sms_message_via_number(
                to_number="+15559876543",
                content="Hello from tests",
            )

            assert result["success"] is True

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{COMMS_URL}/phone/send-text"

            payload = mock_session.post.call_args.kwargs["json"]
            assert payload["From"] == "+15551234567"
            assert payload["To"] == "+15559876543"
            assert payload["Body"] == "Hello from tests"

    @pytest.mark.asyncio
    async def test_send_sms_no_from_number(self):
        """Returns failure when assistant has no phone number."""
        with patch(
            "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
        ) as mock_session_details:
            mock_session_details.assistant.number = None

            result = await comms_utils.send_sms_message_via_number(
                to_number="+15559876543",
                content="Hello",
            )

            assert result["success"] is False


class TestStartCall:
    """Tests for start_call."""

    @pytest.mark.asyncio
    async def test_start_call_url_and_body(self):
        """Posts to /phone/send-call on COMMS_URL with correct JSON body."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "call_sid": "CA123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.number = "+15551234567"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            result = await comms_utils.start_call(to_number="+15559876543")

            assert result["success"] is True

            posted_url = mock_session.post.call_args[0][0]
            assert posted_url == f"{COMMS_URL}/phone/send-call"

            payload = mock_session.post.call_args.kwargs["json"]
            assert payload["From"] == "+15551234567"
            assert payload["To"] == "+15559876543"

    @pytest.mark.asyncio
    async def test_start_call_no_from_number(self):
        """Returns failure when assistant has no phone number."""
        with patch(
            "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
        ) as mock_session_details:
            mock_session_details.assistant.number = None

            result = await comms_utils.start_call(to_number="+15559876543")

            assert result["success"] is False


class TestSendUnifyMessage:
    """Tests for send_unify_message function with attachments."""

    @pytest.mark.asyncio
    async def test_send_without_attachment(self):
        """Sends message without attachment via Pub/Sub."""
        with (
            patch(
                "unity.conversation_manager.domains.comms_utils._get_publisher",
            ) as mock_get_publisher,
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.STAGING = False
            mock_session.assistant.agent_id = 42

            mock_publisher = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = "message-id-123"
            mock_publisher.publish.return_value = mock_future
            mock_publisher.topic_path.return_value = "projects/test/topics/unity-test"
            mock_get_publisher.return_value = mock_publisher

            result = await comms_utils.send_unify_message(
                content="Hello world",
                contact_id=1,
            )

            assert result["success"] is True
            mock_publisher.publish.assert_called_once()

            call_args = mock_publisher.publish.call_args
            assert call_args[0][0] == "projects/test/topics/unity-test"

    @pytest.mark.asyncio
    async def test_send_with_attachment(self):
        """Sends message with attachment included in Pub/Sub event."""
        with (
            patch(
                "unity.conversation_manager.domains.comms_utils._get_publisher",
            ) as mock_get_publisher,
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.STAGING = False
            mock_session.assistant.agent_id = 42

            mock_publisher = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = "message-id-123"
            mock_publisher.publish.return_value = mock_future
            mock_publisher.topic_path.return_value = "projects/test/topics/unity-test"
            mock_get_publisher.return_value = mock_publisher

            attachment = {
                "id": "att-uuid",
                "filename": "report.pdf",
                "url": "https://storage.googleapis.com/signed-url",
            }

            result = await comms_utils.send_unify_message(
                content="Here's the report",
                contact_id=1,
                attachment=attachment,
            )

            assert result["success"] is True

            call_args = mock_publisher.publish.call_args
            import json

            published_data = json.loads(call_args[0][1].decode("utf-8"))
            assert published_data["event"]["attachments"] == [attachment]


class TestSendEmailViaAddress:
    """Tests for send_email_via_address function with attachments."""

    @pytest.mark.asyncio
    async def test_send_with_attachment(self):
        """Sends email with attachment to /gmail/send on COMMS_URL."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "id": "email-123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.email = "assistant@test.com"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            attachment = {
                "filename": "report.pdf",
                "content_base64": "UERGIGNvbnRlbnQ=",
            }

            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                subject="Report",
                body="Please see attached.",
                attachment=attachment,
            )

            assert result["success"] is True

            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"{COMMS_URL}/gmail/send"
            assert call_args.kwargs["json"]["attachment"] == attachment

    @pytest.mark.asyncio
    async def test_send_with_cc_and_bcc(self):
        """Sends email with cc and bcc to /gmail/send on COMMS_URL."""
        mock_session = _mock_aiohttp_session(
            response_json={"success": True, "id": "email-123"},
        )

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.email = "assistant@test.com"
            mock_settings.conversation.COMMS_URL = COMMS_URL

            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                cc=["cc1@example.com", "cc2@example.com"],
                bcc=["bcc@example.com"],
                subject="Team Update",
                body="Here's the update.",
            )

            assert result["success"] is True

            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"{COMMS_URL}/gmail/send"

            payload = call_args.kwargs["json"]
            assert payload["to"] == ["user@example.com"]
            assert payload["cc"] == ["cc1@example.com", "cc2@example.com"]
            assert payload["bcc"] == ["bcc@example.com"]
