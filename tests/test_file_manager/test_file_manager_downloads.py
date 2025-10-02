"""
Tests for FileManager.save_file_to_downloads
"""

from __future__ import annotations

from pathlib import Path

from unity.file_manager.file_manager import FileManager
from tests.helpers import _handle_project


@_handle_project
def test_save_file_to_downloads_registers_and_persists_contents():
    fm = FileManager()

    display_name = fm.save_file_to_downloads("report.txt", b"hello world")

    # Registered in manager
    assert display_name in fm.list()
    assert fm.exists(display_name)

    # Path points into Downloads and contents are written
    p = fm._display_to_path[display_name]
    assert isinstance(p, Path)
    assert p.exists()
    assert "Downloads" in str(p)
    assert p.read_bytes() == b"hello world"


@_handle_project
def test_save_file_to_downloads_unique_names_per_downloads_dir():
    fm = FileManager()

    d1 = fm.save_file_to_downloads("dup.txt", b"one")
    d2 = fm.save_file_to_downloads("dup.txt", b"two")

    assert d1 != d2
    # Both are namespaced under downloads/
    assert d1.startswith("downloads/")
    assert d2.startswith("downloads/")

    # Underlying files exist and have correct content
    p1 = fm._display_to_path[d1]
    p2 = fm._display_to_path[d2]
    assert p1.exists() and p2.exists()
    assert p1.read_bytes() == b"one"
    assert p2.read_bytes() == b"two"
