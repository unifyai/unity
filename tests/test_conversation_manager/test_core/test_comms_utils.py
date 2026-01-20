"""
tests/test_conversation_manager/test_core/test_comms_utils.py
=============================================================

Unit tests for comms_utils functions, particularly attachment handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains import comms_utils


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
                to_email="user@example.com",
                subject="Report",
                body="Please see attached.",
                attachment=attachment,
            )

            assert result["success"] is True

            # Verify attachment was included in the request
            call_args = mock_session.post.call_args
            assert call_args.kwargs["json"]["attachment"] == attachment
