"""
tests/conversation_manager/core/unit/test_attachment_ingestion.py
================================================================

Tests verifying that inbound attachment downloads are properly indexed
in the FileManager so they can be accessed via primitives.files.* methods.

RUNNING THESE TESTS:
    These are isolated unit tests that use a real LocalFileManager backed
    by a tmp_path, with only the network layer mocked.

    .venv/bin/python -m pytest tests/conversation_manager/core/test_attachment_ingestion.py \
        --confcutdir=tests/conversation_manager/core -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains import comms_utils
from unity.file_manager.managers.local import LocalFileManager


@pytest.fixture()
def real_file_manager(tmp_path):
    """Create a real LocalFileManager rooted at a temporary directory.

    Uses enable_sync=False to avoid any VM sync configuration.
    Patches the singleton so that ``LocalFileManager()`` inside
    ``comms_utils`` returns this same instance.
    """
    fm = LocalFileManager(root=str(tmp_path), enable_sync=False)
    with patch(
        "unity.file_manager.managers.local.LocalFileManager",
        return_value=fm,
    ):
        yield fm


class TestAttachmentIngestion:
    """After downloading an attachment, it must be indexed in FileManager."""

    @pytest.mark.asyncio
    async def test_downloaded_attachment_is_indexed(self, real_file_manager):
        """Attachment downloaded via add_unify_message_attachments should be
        indexed (describe().indexed_exists == True) so that the CodeActActor
        can access it via primitives.files.describe / ask_about_file.
        """
        fm = real_file_manager

        # Simulate a successful download from a direct URL.
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=b"PDF file content")
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
            ),
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            attachments = [
                {
                    "id": "att-1",
                    "filename": "report.pdf",
                    "url": "https://storage.googleapis.com/signed-url-here",
                },
            ]
            await comms_utils.add_unify_message_attachments(attachments)

        # The file should exist on the filesystem.
        display_name = "Downloads/report.pdf"
        assert fm.exists(display_name), (
            "File should exist on the filesystem after download"
        )

        # The file must also be indexed so that primitives.files.describe()
        # and primitives.files.ask_about_file() can find it.
        storage = fm.describe(file_path=display_name)
        assert storage.indexed_exists, (
            "Downloaded attachment must be indexed in FileManager so the "
            "CodeActActor can access it via primitives.files.describe() and "
            "primitives.files.ask_about_file(). Currently save_file_to_downloads "
            "writes bytes to disk but never registers the file in the index."
        )

    @pytest.mark.asyncio
    async def test_downloaded_email_attachment_is_indexed(self, real_file_manager):
        """Attachment downloaded via add_email_attachments should also be indexed."""
        fm = real_file_manager

        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=b"email attachment bytes")

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
            ),
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

            attachments = [
                {"id": "att-email-1", "filename": "invoice.pdf"},
            ]
            await comms_utils.add_email_attachments(
                attachments,
                receiver_email="assistant@example.com",
                gmail_message_id="msg-123",
            )

        display_name = "Downloads/invoice.pdf"
        assert fm.exists(display_name), (
            "File should exist on the filesystem after email attachment download"
        )

        storage = fm.describe(file_path=display_name)
        assert storage.indexed_exists, (
            "Downloaded email attachment must be indexed in FileManager so the "
            "CodeActActor can access it via primitives.files.* methods."
        )
