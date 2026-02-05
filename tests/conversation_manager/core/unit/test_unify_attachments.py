"""
tests/conversation_manager/core/unit/test_unify_attachments.py
================================================================

Unit tests for Unify message attachment handling.

These tests verify:
- Attachment metadata is properly structured in events
- add_unify_message_attachments downloads from signed URLs
- Message model includes attachments field
- Attachments are logged in transcripts

RUNNING THESE TESTS:
    These are isolated unit tests that don't require unify API authentication.
    Run with --confcutdir to skip the main conftest.py session hooks:

    .venv/bin/python -m pytest tests/conversation_manager/core/unit/ \\
        --confcutdir=tests/conversation_manager/core/unit -v

Tests are behavior-focused and will fail (xfail) until features are implemented.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest

from unity.conversation_manager.domains import comms_utils
from unity.conversation_manager.events import UnifyMessageReceived, UnifyMessageSent


# =============================================================================
# Event Attachment Metadata Tests
# =============================================================================


class TestUnifyMessageReceivedAttachments:
    """Tests for UnifyMessageReceived event attachment handling."""

    def test_event_includes_attachment_dicts(self):
        """Attachments field contains full metadata dicts."""
        attachments = [
            {"id": "att-1", "filename": "report.pdf", "gs_url": "gs://bucket/path"},
            {"id": "att-2", "filename": "data.xlsx", "gs_url": "gs://bucket/path2"},
        ]
        event = UnifyMessageReceived(
            contact={"id": 1, "name": "Boss"},
            content="Here's the document",
            attachments=attachments,
        )

        assert len(event.attachments) == 2
        assert event.attachments[0]["filename"] == "report.pdf"
        assert event.attachments[1]["filename"] == "data.xlsx"
        assert event.content == "Here's the document"

    def test_event_with_empty_attachments(self):
        """Event works with no attachments."""
        event = UnifyMessageReceived(
            contact={"id": 1, "name": "Boss"},
            content="Just a message",
        )

        assert event.attachments == []

    def test_event_includes_full_attachment_metadata(self):
        """Events accept full attachment objects with all metadata.

        The event attachments field accepts list of dicts with
        id, filename, gs_url, content_type, size_bytes.
        """
        attachment_data = [
            {
                "id": "att-uuid-1",
                "filename": "report.pdf",
                "gs_url": "gs://unify-message-attachments/12345/att-uuid-1_report.pdf",
                "content_type": "application/pdf",
                "size_bytes": 1024,
            },
        ]

        event = UnifyMessageReceived(
            contact={"id": 1, "name": "Boss"},
            content="Here's the document",
            attachments=attachment_data,
        )

        # Attachments can be list of dicts
        assert isinstance(event.attachments[0], dict)
        assert event.attachments[0]["id"] == "att-uuid-1"
        assert event.attachments[0]["gs_url"].startswith("gs://")


class TestUnifyMessageSentAttachments:
    """Tests for UnifyMessageSent event attachment handling."""

    def test_event_includes_attachment_dicts(self):
        """Attachments field contains full metadata dicts."""
        attachments = [
            {"id": "att-1", "filename": "output.csv", "gs_url": "gs://bucket/path"},
        ]
        event = UnifyMessageSent(
            contact={"id": 1, "name": "Boss"},
            content="Sending you this file",
            attachments=attachments,
        )

        assert len(event.attachments) == 1
        assert event.attachments[0]["filename"] == "output.csv"

    def test_sent_event_includes_full_attachment_metadata(self):
        """Sent events also accept full attachment metadata."""
        attachment_data = [
            {
                "id": "att-uuid-2",
                "filename": "output.csv",
                "gs_url": "gs://unify-message-attachments/12345/att-uuid-2_output.csv",
                "content_type": "text/csv",
                "size_bytes": 512,
            },
        ]

        event = UnifyMessageSent(
            contact={"id": 1, "name": "Boss"},
            content="Here's the export",
            attachments=attachment_data,
        )

        assert isinstance(event.attachments[0], dict)
        assert event.attachments[0]["filename"] == "output.csv"


# =============================================================================
# add_unify_message_attachments Tests
# =============================================================================


class TestAddUnifyMessageAttachments:
    """Tests for add_unify_message_attachments function."""

    @pytest.mark.asyncio
    async def test_downloads_from_signed_url(self):
        """Downloads attachment content from the provided signed URL."""
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=b"PDF file content")

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_file_manager = MagicMock()
        mock_file_manager.save_file_to_downloads = MagicMock()

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.file_manager.managers.local.LocalFileManager",
                return_value=mock_file_manager,
            ),
        ):
            attachments = [
                {
                    "id": "att-1",
                    "filename": "report.pdf",
                    "url": "https://storage.googleapis.com/signed-url-here",
                },
            ]

            await comms_utils.add_unify_message_attachments(attachments)

            # Verify file was saved
            mock_file_manager.save_file_to_downloads.assert_called_once()
            call_args = mock_file_manager.save_file_to_downloads.call_args
            assert call_args[0][0] == "report.pdf"  # filename
            assert call_args[0][1] == b"PDF file content"  # content

    @pytest.mark.asyncio
    async def test_handles_empty_attachments(self):
        """No-op when attachments list is empty."""
        # Should not raise any errors
        await comms_utils.add_unify_message_attachments([])

    @pytest.mark.asyncio
    async def test_handles_missing_url(self):
        """Handles attachments without URL gracefully (writes empty placeholder)."""
        mock_file_manager = MagicMock()
        mock_file_manager.save_file_to_downloads = MagicMock()

        with patch(
            "unity.file_manager.managers.local.LocalFileManager",
            return_value=mock_file_manager,
        ):
            attachments = [
                {
                    "id": "att-1",
                    "filename": "placeholder.txt",
                    # No URL provided
                },
            ]

            await comms_utils.add_unify_message_attachments(attachments)

            # Should still save (empty content)
            mock_file_manager.save_file_to_downloads.assert_called_once()
            call_args = mock_file_manager.save_file_to_downloads.call_args
            assert call_args[0][0] == "placeholder.txt"
            assert call_args[0][1] == b""  # Empty content

    @pytest.mark.asyncio
    async def test_sanitizes_filename(self):
        """Sanitizes filename to prevent path traversal."""
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=b"content")

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_file_manager = MagicMock()
        mock_file_manager.save_file_to_downloads = MagicMock()

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.file_manager.managers.local.LocalFileManager",
                return_value=mock_file_manager,
            ),
        ):
            attachments = [
                {
                    "id": "att-1",
                    "filename": "../../../etc/passwd",  # Malicious path
                    "url": "https://example.com/file",
                },
            ]

            await comms_utils.add_unify_message_attachments(attachments)

            # Filename should be sanitized
            call_args = mock_file_manager.save_file_to_downloads.call_args
            saved_filename = call_args[0][0]
            assert ".." not in saved_filename
            assert "/" not in saved_filename

    @pytest.mark.asyncio
    async def test_generates_signed_url_from_gs_url(self):
        """When attachment has gs_url, generates signed URL for download.

        If attachment includes gs_url instead of url,
        the function calls Orchestra API to generate a signed URL.
        """
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=b"content")

        mock_signed_url_response = MagicMock()
        mock_signed_url_response.json = AsyncMock(
            return_value={"signed_url": "https://storage.googleapis.com/signed-url"},
        )
        mock_signed_url_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        # First call: signed URL generation, Second call: download
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_signed_url_response)
            ),
        )
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_file_manager = MagicMock()
        mock_file_manager.save_file_to_downloads = MagicMock()

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.file_manager.managers.local.LocalFileManager",
                return_value=mock_file_manager,
            ),
            patch(
                "unity.conversation_manager.domains.comms_utils.SETTINGS",
            ) as mock_settings,
        ):
            mock_settings.ORCHESTRA_URL = "http://localhost:8000"

            attachments = [
                {
                    "id": "att-1",
                    "filename": "report.pdf",
                    "gs_url": "gs://unify-message-attachments/12345/att-1_report.pdf",
                    # No "url" - should generate from gs_url
                },
            ]

            await comms_utils.add_unify_message_attachments(attachments)

            # Verify signed URL was requested from Orchestra
            mock_session.post.assert_called()
            post_call = mock_session.post.call_args
            assert "signed-url" in str(post_call)

    @pytest.mark.asyncio
    async def test_handles_unavailable_file_gracefully(self):
        """Gracefully handles errors when file is unavailable (deleted/quarantined).

        When a file download fails, the download is skipped without
        failing the entire attachment processing.
        """
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.read = AsyncMock(side_effect=Exception("Not Found"))

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_file_manager = MagicMock()
        mock_file_manager.save_file_to_downloads = MagicMock()

        with (
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "unity.file_manager.managers.local.LocalFileManager",
                return_value=mock_file_manager,
            ),
        ):
            attachments = [
                {
                    "id": "att-1",
                    "filename": "deleted_file.pdf",
                    "url": "https://storage.googleapis.com/signed-url",
                },
            ]

            # Should not raise - handles gracefully by logging and continuing
            await comms_utils.add_unify_message_attachments(attachments)


# =============================================================================
# Message Model Attachments Tests
# =============================================================================


class TestMessageModelAttachments:
    """Tests for Message model attachment field."""

    def test_message_has_attachments_field(self):
        """Message model has an attachments field."""
        from unity.transcript_manager.types.message import Message
        from unity.conversation_manager.types import Medium

        msg = Message(
            message_id=1,
            medium=Medium.UNIFY_MESSAGE,
            sender_id=1,
            receiver_ids=[2],
            timestamp=datetime.now(),
            content="Here's a file",
            exchange_id=1,
            attachments=[
                {
                    "id": "att-1",
                    "filename": "doc.pdf",
                    "gs_url": "gs://bucket/path",
                    "content_type": "application/pdf",
                    "size_bytes": 1024,
                },
            ],
        )

        assert hasattr(msg, "attachments")
        assert len(msg.attachments) == 1
        assert msg.attachments[0]["filename"] == "doc.pdf"

    def test_message_attachments_shorthand(self):
        """Message SHORTHAND_MAP includes attachments -> atts."""
        from unity.transcript_manager.types.message import Message

        assert "attachments" in Message.SHORTHAND_MAP
        assert Message.SHORTHAND_MAP["attachments"] == "atts"

    def test_message_to_post_json_includes_attachments(self):
        """to_post_json includes attachments in the payload."""
        from unity.transcript_manager.types.message import Message
        from unity.conversation_manager.types import Medium

        msg = Message(
            message_id=1,
            medium=Medium.UNIFY_MESSAGE,
            sender_id=1,
            receiver_ids=[2],
            timestamp=datetime.now(),
            content="File attached",
            exchange_id=1,
            attachments=[{"id": "att-1", "filename": "test.txt"}],
        )

        payload = msg.to_post_json()
        assert "attachments" in payload
        assert len(payload["attachments"]) == 1


# =============================================================================
# Upload Attachment Tests (Enhanced)
# =============================================================================


class TestUploadUnifyAttachmentEnhanced:
    """Enhanced tests for upload_unify_attachment with new metadata."""

    @pytest.mark.asyncio
    async def test_upload_returns_enhanced_metadata(self):
        """Upload response includes all metadata from server (including gs_url).

        When the communication adapter returns enhanced metadata (gs_url, content_type,
        size_bytes), this is passed through by upload_unify_attachment.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "test-uuid-123",
                "filename": "document.pdf",
                "url": "https://storage.googleapis.com/signed-url",
                "gs_url": "gs://unify-message-attachments/12345/test-uuid_document.pdf",
                "content_type": "application/pdf",
                "size_bytes": 2048,
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
                file_content=b"PDF content here",
                filename="document.pdf",
            )

            # Verify enhanced fields are passed through from server response
            assert "gs_url" in result
            assert result["gs_url"].startswith("gs://")
            assert "content_type" in result
            assert "size_bytes" in result


# =============================================================================
# Transcript Logging Tests
# =============================================================================


class TestTranscriptLoggingWithAttachments:
    """Tests for logging messages with attachments to transcripts."""

    def test_attachments_passed_to_transcript(self):
        """Attachments from events are passed directly to transcript logging."""
        # With Option C, attachments are always list[dict] - no normalization needed
        attachments = [
            {
                "id": "att-1",
                "filename": "report.pdf",
                "gs_url": "gs://bucket/path",
                "content_type": "application/pdf",
                "size_bytes": 1024,
            },
        ]

        # Attachments are passed through directly
        assert len(attachments) == 1
        assert attachments[0]["id"] == "att-1"
        assert attachments[0]["gs_url"] == "gs://bucket/path"
        assert attachments[0]["content_type"] == "application/pdf"
        assert attachments[0]["size_bytes"] == 1024

    def test_empty_attachments_list(self):
        """Empty attachments list is handled correctly."""
        attachments = []
        assert len(attachments) == 0
