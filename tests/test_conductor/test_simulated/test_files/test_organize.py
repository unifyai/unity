from __future__ import annotations

import asyncio
import pytest

pytestmark = pytest.mark.eval

from unity.conductor.simulated import SimulatedConductor
from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager.global_file_manager import GlobalFileManager
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "GlobalFileManager"


@pytest.mark.asyncio
@_handle_project
async def test_organize_runs_on_request(tmp_path):
    # Seed a local filesystem with some files
    fm_root = tmp_path / "fmroot"
    (fm_root / "docs").mkdir(parents=True, exist_ok=True)
    (fm_root / "reports").mkdir(parents=True, exist_ok=True)
    (fm_root / "archive").mkdir(parents=True, exist_ok=True)

    (fm_root / "docs" / "notes.txt").write_text("meeting notes 2024", encoding="utf-8")
    (fm_root / "reports" / "q1.pdf").write_text(
        "q1 report placeholder",
        encoding="utf-8",
    )

    local = LocalFileManager(str(fm_root))
    # Parse to create index rows before organizing (root-relative identifiers)
    local.parse(["docs/notes.txt", "reports/q1.pdf"])  # relative to root
    gfm = GlobalFileManager([local])

    cond = SimulatedConductor(
        description=(
            "Assistant that can organise files across filesystems in addition to other domains."
        ),
        global_file_manager=gfm,
    )

    handle = await cond.request(
        "Rename /docs/notes.txt to notes-2024.txt and move /reports/q1.pdf to /archive/.",
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert (
        "GlobalFileManager_organize" in executed
    ), f"Expected GlobalFileManager_organize to run, saw: {sorted(executed)}"

    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert (
        "GlobalFileManager_organize" in requested
    ), f"Assistant should request GlobalFileManager_organize, saw: {sorted(requested)}"
