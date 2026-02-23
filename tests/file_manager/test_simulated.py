"""Tests for SimulatedFileManager — pure simulation, no real files."""

from __future__ import annotations

import asyncio
import functools
import pytest

from unity.file_manager.simulated import (
    SimulatedFileManager,
    _SimulatedFileHandle,
)

from tests.helpers import (
    _handle_project,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
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
# 4.  Interject                                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedFileHandle.interject

    @functools.wraps(orig)
    async def wrapped(self, msg, **kwargs):
        calls["interject"] += 1
        return await orig(self, msg, **kwargs)

    monkeypatch.setattr(_SimulatedFileHandle, "interject", wrapped, raising=True)

    fm = SimulatedFileManager()
    handle = await fm.ask_about_file("report.txt", "Summarize the key points.")
    await asyncio.sleep(0.05)
    await handle.interject("Focus on financial metrics.")
    await handle.result()
    assert calls["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stop                                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop():
    fm = SimulatedFileManager()
    handle = await fm.ask_about_file("large_export.csv", "Generate a full summary.")
    await asyncio.sleep(0.05)
    await handle.stop()
    await handle.result()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Clarification handshake                                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_requests_clarification():
    fm = SimulatedFileManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await fm.ask_about_file(
        "data.txt",
        "Analyze this file thoroughly.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    await down_q.put("Focus on statistical trends.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip()


# ────────────────────────────────────────────────────────────────────────────
# 7.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume(monkeypatch):
    call_counts = {"pause": 0, "resume": 0}

    original_pause = _SimulatedFileHandle.pause

    @functools.wraps(original_pause)
    def _patched_pause(self):
        call_counts["pause"] += 1
        return original_pause(self)

    monkeypatch.setattr(_SimulatedFileHandle, "pause", _patched_pause, raising=True)

    original_resume = _SimulatedFileHandle.resume

    @functools.wraps(original_resume)
    def _patched_resume(self):
        call_counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(_SimulatedFileHandle, "resume", _patched_resume, raising=True)

    fm = SimulatedFileManager()
    handle = await fm.ask_about_file("notes.md", "List the action items.")

    pause_msg = await handle.pause()
    assert "pause" in pause_msg.lower()

    res_task = await _assert_blocks_while_paused(handle.result())

    resume_msg = await handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    assert call_counts == {"pause": 1, "resume": 1}


# ────────────────────────────────────────────────────────────────────────────
# 8.  Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    fm = SimulatedFileManager()
    handle = await fm.ask_about_file("plan.docx", "Summarize the business plan.")

    await handle.interject("Focus on European market opportunities.")

    nested = await handle.ask("What is the key opportunity mentioned?")

    nested_answer = await nested.result()
    assert (
        isinstance(nested_answer, str) and nested_answer.strip()
    ), "Nested ask() should yield a non-empty string answer"
    assert any(substr in nested_answer.lower() for substr in ("europe", "eu"))

    handle_answer = await handle.result()
    assert (
        isinstance(handle_answer, str) and handle_answer.strip()
    ), "Handle should still yield a non-empty answer after nested ask"


# ────────────────────────────────────────────────────────────────────────────
# 9.  Stop while paused                                                      #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused():
    fm = SimulatedFileManager()
    h = await fm.ask_about_file("big_file.csv", "Produce a long analysis.")
    await h.pause()
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    await h.stop("cancelled by user")
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 10. Stop while waiting for clarification                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_clarification():
    fm = SimulatedFileManager()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await fm.ask_about_file(
        "inbox.json",
        "Parse this file.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )
    q = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in q.lower()
    await h.stop("no longer needed")
    out = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 11. Reasoning steps toggle                                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
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
