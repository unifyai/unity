"""
tests/conversation_manager/core/test_attachment_ingestion.py
================================================================

Tests verifying that inbound attachment downloads are automatically ingested
into the FileManager so they can be accessed via primitives.files.* methods.

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
from unity.settings import SETTINGS

# Valid plain-text content that the parser can handle.
SAMPLE_TEXT_CONTENT = (
    b"Quarterly revenue report\nTotal revenue: $1,234,567\nNet profit: $456,789\n"
)
SAMPLE_CSV_CONTENT = b"name,amount,date\nAlice,1000,2025-01-15\nBob,2500,2025-02-20\n"


@pytest.fixture()
def real_file_manager(tmp_path):
    """Create a real LocalFileManager rooted at a temporary directory.

    Uses enable_sync=False to avoid any VM sync configuration.
    Patches ManagerRegistry so ``comms_utils`` resolves this same instance.
    """
    fm = LocalFileManager(root=str(tmp_path), enable_sync=False)
    with patch(
        "unity.manager_registry.ManagerRegistry.get_file_manager",
        return_value=fm,
    ):
        yield fm


class TestAttachmentIngestion:
    """After downloading an attachment, it must be fully ingested in FileManager."""

    @pytest.mark.asyncio
    async def test_downloaded_unify_attachment_is_ingested(self, real_file_manager):
        """Attachment downloaded via add_unify_message_attachments should be
        fully ingested: indexed, parsed, and its content queryable via
        primitives.files.describe / ask_about_file.
        """
        fm = real_file_manager

        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=SAMPLE_TEXT_CONTENT)
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
                    "filename": "report.txt",
                    "url": "https://storage.googleapis.com/signed-url-here",
                },
            ]
            await comms_utils.add_unify_message_attachments(attachments)

        display_name = "Attachments/att-1_report.txt"
        assert fm.exists(display_name)

        storage = fm.describe(file_path=display_name)
        assert storage.indexed_exists, "Downloaded attachment must be indexed"
        assert storage.parsed_status == "success", "Attachment must be parsed"
        assert storage.has_document, "Parsed content must be available"

    @pytest.mark.asyncio
    async def test_downloaded_email_attachment_is_ingested(self, real_file_manager):
        """Attachment downloaded via add_email_attachments should be fully ingested."""
        fm = real_file_manager

        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=SAMPLE_CSV_CONTENT)

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
            patch.object(SETTINGS.file, "IMPLICIT_INGESTION", True),
        ):
            mock_settings.conversation.COMMS_URL = "http://localhost:8080"
            mock_settings.file.IMPLICIT_INGESTION = True

            attachments = [
                {"id": "att-email-1", "filename": "data.csv"},
            ]
            await comms_utils.add_email_attachments(
                attachments,
                receiver_email="assistant@example.com",
                message_id="msg-123",
            )

        display_name = "Attachments/att-email-1_data.csv"
        assert fm.exists(display_name)

        storage = fm.describe(file_path=display_name)
        assert storage.indexed_exists, "Downloaded email attachment must be indexed"
        assert storage.parsed_status == "success", "Email attachment must be parsed"
        assert storage.has_document, "Parsed content must be available"


class TestHydrationAttachmentMaterialization:
    """Verify that attachments collected during hydration are materialised
    when the post-init download step runs.
    """

    @pytest.mark.asyncio
    async def test_pending_unify_attachments_materialised(self, real_file_manager):
        """Unify message attachments stashed during hydration get downloaded
        and ingested when the post-init step fires.
        """
        fm = real_file_manager

        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=SAMPLE_TEXT_CONTENT)
        mock_response.raise_for_status = MagicMock()

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
            patch.object(
                SETTINGS.file,
                "IMPLICIT_INGESTION",
                True,
            ),
        ):
            await comms_utils.add_unify_message_attachments(
                [
                    {
                        "id": "hydrated-att-1",
                        "filename": "notes.txt",
                        "url": "https://storage.googleapis.com/signed-url",
                    },
                ],
            )

        display_name = "Attachments/hydrated-att-1_notes.txt"
        assert fm.exists(display_name), "Hydrated attachment must appear on disk"
        storage = fm.describe(file_path=display_name)
        assert storage.indexed_exists, "Hydrated attachment must be indexed"

    @pytest.mark.asyncio
    async def test_idempotent_with_existing_file(self, real_file_manager, tmp_path):
        """If a hydrated attachment already exists on disk (e.g. from a live
        message in the same session), the download is skipped.
        """
        att_dir = tmp_path / "Attachments"
        att_dir.mkdir(exist_ok=True)
        existing = att_dir / "hydrated-att-2_notes.txt"
        existing.write_bytes(b"already here")

        mock_session = MagicMock()
        mock_session.get = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await comms_utils.add_unify_message_attachments(
                [
                    {
                        "id": "hydrated-att-2",
                        "filename": "notes.txt",
                        "gs_url": "gs://bucket/hydrated-att-2_notes.txt",
                    },
                ],
            )

        mock_session.get.assert_not_called()
        assert existing.read_bytes() == b"already here"
