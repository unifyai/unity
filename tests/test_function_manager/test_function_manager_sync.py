from __future__ import annotations

import os
from pathlib import Path
import pytest
from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.file_manager.file_manager import FileManager


@_handle_project
@pytest.mark.unit
def test_filesystem_mirror_is_registered_on_add():
    src = (
        "def double(x):\n"
        "    y = 0\n"
        "    for _ in range(2):\n"
        "        y = y + x\n"
        "    return y\n"
    )
    fm = FunctionManager()
    result = fm.add_functions(implementations=src)
    assert result == {"double": "added"}
    # Filesystem mirror exists and is registered
    path = fm.get_function_file_path("double")
    assert path and os.path.exists(path)
    display = f"functions/{os.path.basename(path)}"
    assert fm._fm.exists(display)  # type: ignore[attr-defined]


@_handle_project
@pytest.mark.unit
def test_function_files_are_protected_and_visible_via_file_manager():
    fm_files = FileManager()
    fm = FunctionManager(file_manager=fm_files)

    src = "def hello():\n    return 'world'\n"
    fm.add_functions(implementations=src)
    path = fm.get_function_file_path("hello")
    assert path and os.path.exists(path)

    # File is registered under functions/<name>.py and should be present in FileManager list
    display = f"functions/{os.path.basename(path)}"
    assert display in fm_files.list()


@_handle_project
@pytest.mark.unit
def test_sync_from_disk_updates_unify_record():
    fm_files = FileManager()
    fm = FunctionManager(file_manager=fm_files)

    src = (
        "def compute(x: int) -> int:\n" '    """Double value"""\n' "    return x * 2\n"
    )
    fm.add_functions(implementations=src)

    # Modify file on disk
    p = Path(fm.get_function_file_path("compute"))
    p.write_text("def compute(x: int) -> int:\n    return x * 3\n", encoding="utf-8")

    updated = fm.sync_from_disk()
    assert "compute" in updated

    # Confirm implementation changed in listing
    listing = fm.list_functions(include_implementations=True)
    assert "return x * 3" in listing["compute"]["implementation"]
