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
async def test_ask_then_organize_separate(tmp_path):
    # Seed a local filesystem with some files
    fm_root = tmp_path / "fmroot"
    (fm_root / "docs").mkdir(parents=True, exist_ok=True)
    (fm_root / "reports").mkdir(parents=True, exist_ok=True)
    (fm_root / "archive").mkdir(parents=True, exist_ok=True)
    (fm_root / "tmp").mkdir(parents=True, exist_ok=True)

    (fm_root / "docs" / "notes.txt").write_text("meeting notes 2024", encoding="utf-8")
    (fm_root / "reports" / "q1.pdf").write_text(
        "q1 report placeholder",
        encoding="utf-8",
    )
    (fm_root / "tmp" / "log.txt").write_text("old logs", encoding="utf-8")

    local = LocalFileManager(str(fm_root))
    # Parse to create index rows (use root-relative identifiers)
    local.ingest_files(
        ["docs/notes.txt", "reports/q1.pdf", "tmp/log.txt"],
    )  # relative to root

    gfm = GlobalFileManager([local])

    cond = SimulatedConductor(
        description=(
            "Assistant that can reason across contacts, knowledge, tasks, and filesystems."
        ),
        global_file_manager=gfm,
    )

    # First: query across filesystems
    ask_handle = await cond.request(
        "List available filesystems and provide a brief inventory overview.",
        _return_reasoning_steps=True,
    )
    ask_answer, ask_messages = await asyncio.wait_for(ask_handle.result(), timeout=300)
    assert (
        isinstance(ask_answer, str) and ask_answer.strip()
    ), "Ask answer should be non-empty"

    ask_executed = set(tool_names_from_messages(ask_messages, MANAGER))
    ask_requested = set(assistant_requested_tool_names(ask_messages, MANAGER))
    assert ask_executed, "Expected at least one GlobalFileManager tool call on ask()"
    assert (
        "GlobalFileManager_ask" in ask_executed
        or "GlobalFileManager_list_filesystems" in ask_executed
    ), f"Expected GlobalFileManager ask-side tool, saw: {sorted(ask_executed)}"
    assert (
        ask_requested
    ), "Assistant should have requested GlobalFileManager ask-side tool"

    # Then: propose an organisation plan
    req_handle = await cond.request(
        "Move /reports/q1.pdf to /archive/ and delete /tmp/log.txt.",
        _return_reasoning_steps=True,
    )
    req_answer, req_messages = await asyncio.wait_for(req_handle.result(), timeout=300)
    assert (
        isinstance(req_answer, str) and req_answer.strip()
    ), "Request answer should be non-empty"

    req_executed = set(tool_names_from_messages(req_messages, MANAGER))
    req_requested = set(assistant_requested_tool_names(req_messages, MANAGER))
    assert (
        req_executed
    ), "Expected at least one GlobalFileManager tool call on request()"
    assert (
        "GlobalFileManager_organize" in req_executed
    ), f"Expected GlobalFileManager_organize to run, saw: {sorted(req_executed)}"
    assert req_requested and (
        "GlobalFileManager_organize" in req_requested
    ), f"Assistant should request GlobalFileManager_organize, saw: {sorted(req_requested)}"
