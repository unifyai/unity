"""
Tests for FileManager.save_attachment
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from tests.helpers import _handle_project
from unity.settings import SETTINGS


@pytest.mark.asyncio
@_handle_project
async def test_save_attachment_registers_and_persists_contents(
    file_manager,
    fm_root,
):
    fm = file_manager

    display_name = fm.save_attachment("att-1", "report.txt", b"hello world")

    # Registered in manager
    assert display_name in fm.list()
    assert fm.exists(display_name)

    # Path points into Attachments and contents are written
    att_path = Path(fm_root) / display_name
    assert att_path.exists()
    assert att_path.read_bytes() == b"hello world"


@pytest.mark.asyncio
@_handle_project
async def test_save_attachment_unique_ids_produce_distinct_files(
    file_manager,
    fm_root,
):
    fm = file_manager

    d1 = fm.save_attachment("att-1", "dup.txt", b"one")
    d2 = fm.save_attachment("att-2", "dup.txt", b"two")

    assert d1 != d2
    assert d1.startswith("Attachments/")
    assert d2.startswith("Attachments/")

    p1 = Path(fm_root) / d1
    p2 = Path(fm_root) / d2
    assert p1.exists() and p2.exists()
    assert p1.read_bytes() == b"one"
    assert p2.read_bytes() == b"two"


@pytest.mark.asyncio
@_handle_project
async def test_save_attachment_unparseable_file_still_indexed(
    file_manager,
    fm_root,
):
    """Files that fail to parse should still be indexed in FileRecords.

    When a downloaded attachment has an unrecognised or corrupt format (e.g.
    an .exe, a truncated PDF, or arbitrary binary), the parse stage inside
    ingest_files fails. The file must still appear in FileRecords with
    status='error' so that describe() returns indexed_exists=True and the
    CodeActActor can at least acknowledge the file's existence.
    """
    fm = file_manager

    with patch.object(SETTINGS.file, "IMPLICIT_INGESTION", True):
        display_name = fm.save_attachment(
            "att-corrupt",
            "corrupt.pdf",
            b"not-a-real-pdf-just-garbage-bytes",
        )

    # The file is on disk.
    assert fm.exists(display_name)

    # It must also be in the FileRecords index so that primitives.files.*
    # methods can acknowledge its existence.
    storage = fm.describe(file_path=display_name)
    assert storage.indexed_exists, (
        "Unparseable files must still be indexed in FileRecords so the "
        "CodeActActor can see them via primitives.files.describe(). "
        "Currently ingest_files silently drops files that fail to parse."
    )


@pytest.mark.asyncio
@_handle_project
async def test_save_attachment_can_skip_auto_ingest_even_when_enabled(
    file_manager,
):
    fm = file_manager

    with patch.object(SETTINGS.file, "IMPLICIT_INGESTION", True):
        display_name = fm.save_attachment(
            "att-manual",
            "notes.txt",
            b"queued later",
            auto_ingest=False,
        )

    assert fm.exists(display_name)
    storage = fm.describe(file_path=display_name)
    assert storage.indexed_exists is False
    assert storage.parsed_status is None
