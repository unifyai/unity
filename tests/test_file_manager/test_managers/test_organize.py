"""
FileManager organize functionality tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.eval
from tests.helpers import _handle_project
from tests.test_file_manager.helpers import ask_judge


@pytest.mark.asyncio
@_handle_project
async def test_rename(file_manager, tmp_path: Path):
    """Test renaming a file using the organize method."""
    fm = file_manager
    fm.clear()
    # Create test file under fm_root (no import needed)
    display_name = "rename_test.txt"
    (Path(fm._adapter._root) / display_name).write_text("This is a file to be renamed.")  # type: ignore[attr-defined]

    assert fm.exists(display_name)
    assert not fm.exists("renamed_file.txt")

    # Parse the file to add it to Unify logs before organize
    fm.ingest_files(display_name)

    instruction = f"Rename the file {display_name} to renamed_file.txt."
    before_state = {"files": fm.list()}
    handle = await fm.organize(instruction)
    response = await handle.result()
    after_state = {"files": fm.list()}

    assert not fm.exists("rename_test.txt")
    assert fm.exists("renamed_file.txt")

    verdict = await ask_judge(
        instruction,
        response,
        before_state=before_state,
        after_state=after_state,
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed rename incorrect. Verdict: {verdict}"


@pytest.mark.asyncio
@_handle_project
async def test_move(file_manager, fm_root: Path, tmp_path: Path):
    """Test moving a file using the organize method."""
    fm = file_manager
    fm.clear()
    target_dir = Path(fm_root) / "move_destination"
    target_dir.mkdir()

    # Create test file under fm_root (no import needed)
    display_name = "move_test.txt"
    (Path(fm._adapter._root) / display_name).write_text("This is a file to be moved.")  # type: ignore[attr-defined]

    assert fm.exists(display_name)

    # Parse the file to add it to Unify logs before organize
    fm.ingest_files(display_name)

    instruction = f"Move the file {display_name} into the 'move_destination' folder."
    before_state = {"files": fm.list()}
    handle = await fm.organize(instruction)
    response = await handle.result()
    after_state = {"files": fm.list()}

    # In the local adapter, moving changes the conceptual path stored in Unify
    assert not fm.exists("move_test.txt")
    assert fm.exists("move_destination/move_test.txt")

    verdict = await ask_judge(
        instruction,
        response,
        before_state=before_state,
        after_state=after_state,
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed move incorrect. Verdict: {verdict}"


@pytest.mark.asyncio
@_handle_project
async def test_delete(file_manager, tmp_path: Path):
    """Test deleting a file using the organize method."""
    fm = file_manager
    fm.clear()
    # Create test file under fm_root (no import needed)
    display_name = "delete_test.txt"
    (Path(fm._adapter._root) / display_name).write_text("This is a file to be deleted.")  # type: ignore[attr-defined]

    assert fm.exists(display_name)

    # Parse the file to add it to Unify logs so we can query for file_id
    fm.ingest_files(display_name)

    rows = fm.filter_files(filter=f"file_path == '{display_name}'")
    file_id = rows[0].get("file_id")

    instruction = f"Delete the file with ID {file_id}."
    before_state = {"files": fm.list()}
    handle = await fm.organize(instruction)
    response = await handle.result()
    after_state = {"files": fm.list()}

    assert not fm.exists(display_name)

    verdict = await ask_judge(
        instruction,
        response,
        before_state=before_state,
        after_state=after_state,
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed delete incorrect. Verdict: {verdict}"
