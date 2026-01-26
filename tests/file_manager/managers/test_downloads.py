"""
Tests for FileManager.save_file_to_downloads
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_save_file_to_downloads_registers_and_persists_contents(
    file_manager,
    fm_root,
):
    fm = file_manager

    display_name = fm.save_file_to_downloads("report.txt", b"hello world")

    # Registered in manager
    assert display_name in fm.list()
    assert fm.exists(display_name)

    # Path points into Downloads and contents are written
    # Local adapter returns display name under Downloads/, so resolve on disk
    downloads_path = Path(fm_root) / display_name
    if not downloads_path.exists():
        downloads_path = Path(fm_root) / "Downloads" / Path(display_name).name
    assert downloads_path.exists()
    assert downloads_path.read_bytes() == b"hello world"


@pytest.mark.asyncio
@_handle_project
async def test_save_file_to_downloads_unique_names_per_downloads_dir(
    file_manager,
    fm_root,
):
    fm = file_manager

    d1 = fm.save_file_to_downloads("dup.txt", b"one")
    d2 = fm.save_file_to_downloads("dup.txt", b"two")

    assert d1 != d2
    # Both are namespaced under Downloads/
    assert d1.startswith("Downloads/")
    assert d2.startswith("Downloads/")

    # Underlying files exist and have correct content
    p1 = Path(fm_root) / d1
    if not p1.exists():
        p1 = Path(fm_root) / "Downloads" / Path(d1).name
    p2 = Path(fm_root) / d2
    if not p2.exists():
        p2 = Path(fm_root) / "Downloads" / Path(d2).name
    assert p1.exists() and p2.exists()
    assert p1.read_bytes() == b"one"
    assert p2.read_bytes() == b"two"
