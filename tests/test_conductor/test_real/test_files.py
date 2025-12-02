from __future__ import annotations

import asyncio
import pytest

pytestmark = pytest.mark.eval

from unity.conductor.simulated import SimulatedConductor
from unity.file_manager.global_file_manager import GlobalFileManager
from unity.file_manager.managers.local import LocalFileManager

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "GlobalFileManager"


# ---------------------------------------------------------------------------
#  Real Conductor → GlobalFileManager.ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_global_manager(tmp_path):
    # Seed a local filesystem with at least one file
    fm_root = tmp_path / "fmroot"
    fm_root.mkdir(parents=True, exist_ok=True)
    (fm_root / "a.txt").write_text("alpha", encoding="utf-8")

    local = LocalFileManager(str(fm_root))
    # Parse to ensure records exist for retrieval (root-relative)
    local.parse("a.txt")

    gfm = GlobalFileManager([local])
    cond = SimulatedConductor(global_file_manager=gfm)

    handle = await cond.ask(
        "List available filesystems and provide a brief inventory overview.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, MANAGER)
    requested_list = assistant_requested_tool_names(messages, MANAGER)
    assert executed_list, "Expected at least one tool call"
    assert "GlobalFileManager_ask" in set(
        executed_list,
    ) or "GlobalFileManager_list_filesystems" in set(
        executed_list,
    ), f"Expected GlobalFileManager ask-side tool, saw: {sorted(set(executed_list))}"
    assert (
        requested_list
    ), "Assistant should request at least one GlobalFileManager tool"


# ---------------------------------------------------------------------------
#  Real Conductor → GlobalFileManager.organize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_organize_calls_global_manager(tmp_path):
    # Seed a local filesystem with some files
    fm_root = tmp_path / "fmroot"
    (fm_root / "docs").mkdir(parents=True, exist_ok=True)
    (fm_root / "invoices").mkdir(parents=True, exist_ok=True)
    (fm_root / "archive").mkdir(parents=True, exist_ok=True)
    (fm_root / "tmp").mkdir(parents=True, exist_ok=True)

    (fm_root / "docs" / "notes.txt").write_text("notes", encoding="utf-8")
    (fm_root / "invoices" / "jan.xlsx").write_text("xlsx placeholder", encoding="utf-8")
    (fm_root / "tmp" / "old.log").write_text("old", encoding="utf-8")

    local = LocalFileManager(str(fm_root))
    # Parse to create index rows (root-relative identifiers)
    local.parse(
        ["docs/notes.txt", "invoices/jan.xlsx", "tmp/old.log"],
    )  # relative to root
    gfm = GlobalFileManager([local])
    cond = SimulatedConductor(global_file_manager=gfm)

    handle = await cond.request(
        "Rename /docs/notes.txt to notes-2024.txt; move /invoices/jan.xlsx to /archive/; delete /tmp/old.log.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, MANAGER)
    requested_list = assistant_requested_tool_names(messages, MANAGER)
    assert executed_list, "Expected at least one tool call"
    assert "GlobalFileManager_organize" in set(
        executed_list,
    ), f"Expected GlobalFileManager_organize to run, saw: {sorted(set(executed_list))}"
    assert "GlobalFileManager_organize" in set(
        requested_list,
    ), f"Assistant should request GlobalFileManager_organize, saw: {sorted(set(requested_list))}"
