"""Tests for SimulatedFileManager — pure simulation, no real files."""

from __future__ import annotations

import pytest

from unity.file_manager.simulated import (
    SimulatedFileManager,
)

from tests.helpers import (
    _handle_project,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    from unity.file_manager.base import BaseFileManager
    from unity.file_manager.simulated import SimulatedFileManager

    assert (
        BaseFileManager.ask_about_file.__doc__.strip()
        in SimulatedFileManager.ask_about_file.__doc__.strip()
    ), ".ask_about_file doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic ask_about_file                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_about_file():
    fm = SimulatedFileManager("Demo file storage for unit-tests.")
    handle = await fm.ask_about_file("report.pdf", "Summarize this document.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Synchronous methods                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_synchronous_operations():
    fm = SimulatedFileManager()

    assert fm.list() == []
    assert not fm.exists("any.txt")

    fm.add_simulated_file(
        "doc.txt",
        records=[{"content": "Document"}],
    )

    assert fm.list() == ["doc.txt"]
    assert fm.exists("doc.txt")
    assert not fm.exists("missing.txt")


# ────────────────────────────────────────────────────────────────────────────
# 11. Reasoning steps toggle                                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_reasoning_steps_toggle():
    fm = SimulatedFileManager()

    handle = await fm.ask_about_file(
        "analysis.txt",
        "What are the key insights?",
        _return_reasoning_steps=True,
    )
    result = await handle.result()
    assert isinstance(result, tuple) and len(result) == 2
    answer, messages = result
    assert isinstance(answer, str) and answer.strip()
    assert isinstance(messages, list) and len(messages) >= 1

    handle2 = await fm.ask_about_file(
        "analysis.txt",
        "What are the key insights?",
        _return_reasoning_steps=False,
    )
    result2 = await handle2.result()
    assert isinstance(result2, str) and result2.strip()


# ────────────────────────────────────────────────────────────────────────────
# 12. Reduce shapes                                                          #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_simulated_file_manager_reduce_shapes():
    fm = SimulatedFileManager()

    scalar = fm.reduce(metric="sum", columns="file_id")
    assert isinstance(scalar, (int, float))

    multi = fm.reduce(metric="max", columns=["file_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"file_id"}

    grouped_str = fm.reduce(metric="sum", columns="file_id", group_by="status")
    assert isinstance(grouped_str, dict)

    grouped_list = fm.reduce(
        metric="sum",
        columns=["file_id"],
        group_by=["status", "file_id"],
    )
    assert isinstance(grouped_list, dict)
