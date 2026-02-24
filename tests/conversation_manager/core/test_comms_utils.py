"""
tests/conversation_manager/core/test_comms_utils.py
=============================================================

Unit tests for comms_utils functions, particularly attachment handling.

Marked ``real_comms_functions`` to opt out of the universal comms stub
in the parent conftest.  These tests call the real ``comms_utils``
implementations with mocked dependencies (aiohttp, SESSION_DETAILS, etc.).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains import comms_utils

pytestmark = pytest.mark.real_comms_functions


class TestUploadUnifyAttachment:
    """Tests for upload_unify_attachment function."""

    @pytest.mark.asyncio
    async def test_upload_success(self):
        """Successfully uploads file and returns attachment details."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "test-uuid-123",
                "filename": "document.pdf",
                "url": "https://storage.googleapis.com/signed-url",
            },
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.id = "test-assistant"
            mock_settings.conversation.COMMS_URL = "http://localhost:8080"

            result = await comms_utils.upload_unify_attachment(
                file_content=b"test file content",
                filename="document.pdf",
            )

            assert result["id"] == "test-uuid-123"
            assert result["filename"] == "document.pdf"
            assert "storage.googleapis.com" in result["url"]

    @pytest.mark.asyncio
    async def test_upload_with_custom_assistant_id(self):
        """Uses provided assistant_id instead of session default."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "uuid",
                "filename": "file.txt",
                "url": "https://url",
            },
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.conversation.COMMS_URL = "http://localhost:8080"

            result = await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
                assistant_id="custom-assistant-id",
            )

            assert result["id"] == "uuid"
            # Verify the session.post was called (custom assistant_id used)
            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_failure_returns_error(self):
        """Returns error dict when upload fails."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("Connection refused"),
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.conversation_manager.domains.comms_utils.SESSION_DETAILS",
            ) as mock_session_details,
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_session_details.assistant.id = "test-assistant"
            mock_settings.conversation.COMMS_URL = "http://localhost:8080"

            result = await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
            )

            assert result.get("success") is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_upload_posts_to_adapters_not_comms_app(self):
        """The /unify/attachment endpoint is on the adapters service.

        COMMS_URL points to the comms-app (phone, gmail, infra routes).
        The attachment upload endpoint lives on the separate adapters service.
        Using COMMS_URL results in a 404 because the comms-app has no /unify/* routes.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "uuid",
                "filename": "file.txt",
                "url": "https://url",
            },
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        comms_app_url = (
            "https://unity-comms-app-staging-262420637606.us-central1.run.app"
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
            mock_session_details.assistant.id = "test-assistant"
            mock_settings.conversation.COMMS_URL = comms_app_url

            await comms_utils.upload_unify_attachment(
                file_content=b"content",
                filename="file.txt",
            )

            posted_url = mock_session.post.call_args[0][0]
            assert not posted_url.startswith(comms_app_url), (
                f"upload_unify_attachment incorrectly posts to COMMS_URL "
                f"({comms_app_url}). The /unify/attachment endpoint lives on "
                f"the adapters service, not the comms-app. Got: {posted_url}"
            )


class TestSendUnifyMessage:
    """Tests for send_unify_message function with attachments."""

    @pytest.mark.asyncio
    async def test_send_without_attachment(self):
        """Sends message without attachment."""
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
            mock_session.assistant.id = "test-assistant"

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

    @pytest.mark.asyncio
    async def test_send_with_attachment(self):
        """Sends message with attachment included in event."""
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
            mock_session.assistant.id = "test-assistant"

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

            # Verify attachment was included in the published message
            call_args = mock_publisher.publish.call_args
            import json

            published_data = json.loads(call_args[0][1].decode("utf-8"))
            assert published_data["event"]["attachments"] == [attachment]


class TestSendEmailViaAddress:
    """Tests for send_email_via_address function with attachments."""

    @pytest.mark.asyncio
    async def test_send_with_attachment(self):
        """Sends email with attachment."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={"success": True, "id": "email-123"},
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

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
            mock_settings.conversation.COMMS_URL = "http://localhost:8080"

            attachment = {
                "filename": "report.pdf",
                "content_base64": "UERGIGNvbnRlbnQ=",  # "PDF content" in base64
            }

            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                subject="Report",
                body="Please see attached.",
                attachment=attachment,
            )

            assert result["success"] is True

            # Verify attachment was included in the request
            call_args = mock_session.post.call_args
            assert call_args.kwargs["json"]["attachment"] == attachment

    @pytest.mark.asyncio
    async def test_send_with_cc_and_bcc(self):
        """Sends email with cc and bcc recipients."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={"success": True, "id": "email-123"},
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

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
            mock_settings.conversation.COMMS_URL = "http://localhost:8080"

            result = await comms_utils.send_email_via_address(
                to=["user@example.com"],
                cc=["cc1@example.com", "cc2@example.com"],
                bcc=["bcc@example.com"],
                subject="Team Update",
                body="Here's the update.",
            )

            assert result["success"] is True

            # Verify cc and bcc were included in the request
            call_args = mock_session.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["to"] == ["user@example.com"]
            assert payload["cc"] == ["cc1@example.com", "cc2@example.com"]
            assert payload["bcc"] == ["bcc@example.com"]
