from __future__ import annotations

import asyncio
import pytest
import functools

from unity.file_manager import simulated as sim_mod

# helper that wraps each test in its own Unify project / trace context
from tests.helpers import (
    _handle_project,
    _ack_ok,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
)
from tests.test_file_manager.helpers import ask_judge


@pytest.fixture
def simulated_file_manager():
    """Fixture for a clean SimulatedFileManager singleton instance."""
    # The class is a singleton, so this will always return the same instance
    fm = sim_mod.SimulatedFileManager()
    # We must clear its state before each test
    fm.clear_simulated_files()
    yield fm
    # And clear it after, to prevent state leakage
    fm.clear_simulated_files()


# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """
    Public methods in SimulatedFileManager should copy the real
    BaseFileManager doc-strings one-for-one (via functools.wraps).
    """
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
async def test_ask_about_file(simulated_file_manager):
    fm = simulated_file_manager
    file_content = "Notes about Mars missions and space exploration"
    fm.add_simulated_file(
        "topic.txt",
        records=[{"content": "Space exploration notes"}],
        metadata={"mime_type": "text/plain"},
        full_text=file_content,
        description="Space notes",
    )
    instruction = "Summarize the key theme of topic.txt"
    handle = await fm.ask_about_file("topic.txt", instruction)
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip()

    verdict = await ask_judge(instruction, answer, file_content=file_content)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed 'ask_about_file' incorrect. Verdict: {verdict}"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Basic synchronous methods                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_synchronous_operations(simulated_file_manager):
    """Test the synchronous methods like list, exists, parse."""
    fm = simulated_file_manager

    # Initially empty
    assert fm.list() == []
    assert not fm.exists("any.txt")

    # Add files
    fm.add_simulated_file(
        "doc1.txt",
        records=[{"content": "Document 1"}],
        metadata={"mime_type": "text/plain"},
        full_text="Document 1 content",
        description="First test document",
    )
    fm.add_simulated_file(
        "doc2.txt",
        records=[{"content": "Document 2"}],
        metadata={"mime_type": "text/plain"},
        full_text="Document 2 content",
        description="Second test document",
    )

    # Check list and exists
    files = fm.list()
    assert len(files) == 2
    assert "doc1.txt" in files
    assert "doc2.txt" in files
    assert fm.exists("doc1.txt")
    assert not fm.exists("missing.txt")

    # Parse single file
    result = fm.ingest_files("doc1.txt")
    assert result["doc1.txt"].status == "success"
    assert result["doc1.txt"].content_ref.record_count == 1

    # Parse multiple files
    results = fm.ingest_files(["doc1.txt", "doc2.txt"])
    assert len(results.files) == 2
    assert all(r.status == "success" for r in results.files.values())

    # Parse non-existent file
    result = fm.ingest_files("missing.txt")
    assert result["missing.txt"].status == "error"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Steerable handle tests                                                 #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 5.  Interject                                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject(monkeypatch, simulated_file_manager):
    calls = {"interject": 0}
    orig = sim_mod._SimulatedFileHandle.interject

    @functools.wraps(orig)
    def wrapped(self, msg: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return orig(self, msg)

    monkeypatch.setattr(
        sim_mod._SimulatedFileHandle,
        "interject",
        wrapped,
        raising=True,
    )

    fm = simulated_file_manager
    fm.add_simulated_file(
        "report.txt",
        records=[{"content": "Annual report"}],
        full_text="Annual report with financial details",
        description="Company annual report",
    )

    instruction = "Summarize the key points of the report.txt file."
    handle = await fm.ask_about_file("report.txt", instruction)
    await asyncio.sleep(0.05)
    reply = handle.interject("Focus on financial metrics.")
    assert _ack_ok(reply)
    final_answer = await handle.result()
    assert calls["interject"] == 1, ".interject should be called exactly once"

    # Judge the final answer based on the initial instruction and the interjection
    full_instruction = (
        instruction + " Follow-up instruction: Focus on financial metrics."
    )
    verdict = await ask_judge(
        full_instruction,
        final_answer,
        file_content="Annual report with financial details",
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed interjected ask incorrect. Verdict: {verdict}"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop(simulated_file_manager):
    fm = simulated_file_manager
    fm.add_simulated_file(
        "large.txt",
        records=[{"content": "Very large document"}],
        full_text="Very large document with extensive content",
        description="Large document for testing",
    )

    handle = await fm.ask_about_file(
        "large.txt",
        "Generate a detailed analysis of the large.txt file.",
    )
    await asyncio.sleep(0.05)
    handle.stop()
    await handle.result()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_requests_clarification(simulated_file_manager):
    fm = simulated_file_manager
    fm.add_simulated_file(
        "data.txt",
        records=[{"content": "Research data"}],
        full_text="Research data with statistical analysis",
        description="Research dataset",
    )

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await fm.ask_about_file(
        "data.txt",
        "Please analyze 'data.txt' thoroughly.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    await down_q.put("Focus on statistical trends.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip()

    # Judge the final answer
    instruction = "Please analyze this file thoroughly. Clarification: Focus on statistical trends."
    file_content = "Research data with statistical analysis"
    verdict = await ask_judge(instruction, answer, file_content=file_content)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed clarification flow incorrect. Verdict: {verdict}"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume(monkeypatch, simulated_file_manager):
    """
    Ensure a `_SimulatedFileHandle` can be paused and resumed.
    """
    counts = {"pause": 0, "resume": 0}

    # --- patch pause -------------------------------------------------------
    orig_pause = sim_mod._SimulatedFileHandle.pause

    @functools.wraps(orig_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    monkeypatch.setattr(
        sim_mod._SimulatedFileHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- patch resume ------------------------------------------------------
    orig_resume = sim_mod._SimulatedFileHandle.resume

    @functools.wraps(orig_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(
        sim_mod._SimulatedFileHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    fm = simulated_file_manager
    fm.add_simulated_file(
        "complex.txt",
        records=[{"content": "Complex document"}],
        full_text="Complex document with detailed analysis",
        description="Complex test document",
    )

    handle = await fm.ask_about_file(
        "complex.txt",
        "Perform a comprehensive analysis of complex.txt.",
    )

    # Pause the handle
    pause_msg = await handle.pause()
    assert "pause" in pause_msg.lower() or "paused" in pause_msg.lower()

    # Start result() while still paused – it should await
    res_task = asyncio.create_task(handle.result())
    await _assert_blocks_while_paused(res_task)

    # Resume execution
    resume_msg = await handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    # Now result() should finish
    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()

    # Each steering method must have been invoked exactly once
    assert counts == {"pause": 1, "resume": 1}, "pause/resume must each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 9. Nested ask on handle                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask(simulated_file_manager):
    """
    The internal handle returned by SimulatedFileManager.ask_about_file exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    fm = simulated_file_manager
    fm.add_simulated_file(
        "business.txt",
        records=[{"content": "Business plan for European expansion"}],
        full_text="Business plan for European expansion with market analysis",
        description="European expansion business plan",
    )

    # Start an initial ask to obtain the live handle
    instruction1 = "Summarize this business document about European expansion."
    handle = await fm.ask_about_file("business.txt", instruction1)

    # Add extra context to ensure nested prompt includes it
    handle.interject("Focus on European market opportunities.")

    # Invoke the dynamic ask on the running handle
    instruction2 = "What is the key opportunity mentioned?"
    nested = await handle.ask(instruction2)

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip(), (
        "Nested ask() should yield a non-empty string answer",
    )

    # Judge the nested answer
    full_instruction_nested = (
        f"Initial summary task: '{instruction1}'. Follow-up question: '{instruction2}'"
    )
    file_content = "Business plan for European expansion with market analysis"
    verdict_nested = await ask_judge(
        full_instruction_nested,
        nested_answer,
        file_content=file_content,
    )
    assert (
        verdict_nested.lower().strip().startswith("correct")
    ), f"Judge deemed nested ask incorrect. Verdict: {verdict_nested}"

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip(), (
        "Handle should still yield a non-empty answer after nested ask",
    )

    # Judge the final answer of the main handle
    verdict_main = await ask_judge(
        instruction1,
        handle_answer,
        file_content=file_content,
    )
    assert (
        verdict_main.lower().strip().startswith("correct")
    ), f"Judge deemed main ask incorrect after nested call. Verdict: {verdict_main}"


# ────────────────────────────────────────────────────────────────────────────
# 10. Reasoning steps toggle                                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_reasoning_steps_toggle(simulated_file_manager):
    """Test that _return_reasoning_steps works correctly."""
    fm = simulated_file_manager
    fm.add_simulated_file(
        "analysis.txt",
        records=[{"content": "Market analysis document"}],
        full_text="Market analysis document with industry trends",
        description="Market analysis report",
    )

    # Ask with reasoning steps
    instruction = "What are the key insights from analysis.txt?"
    handle = await fm.ask_about_file(
        "analysis.txt",
        instruction,
        _return_reasoning_steps=True,
    )
    result = await handle.result()
    assert isinstance(result, tuple) and len(result) == 2
    answer, messages = result
    assert isinstance(answer, str) and answer.strip()
    assert isinstance(messages, list) and len(messages) >= 1

    # Judge the answer part of the result
    file_content = "Market analysis document with industry trends"
    verdict = await ask_judge(instruction, answer, file_content=file_content)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed ask with reasoning incorrect. Verdict: {verdict}"

    # Ask without reasoning steps
    handle2 = await fm.ask_about_file(
        "analysis.txt",
        instruction,
        _return_reasoning_steps=False,
    )
    result2 = await handle2.result()
    assert isinstance(result2, str) and result2.strip()


@_handle_project
def test_simulated_file_manager_reduce_shapes(simulated_file_manager):
    fm = simulated_file_manager

    scalar = fm.reduce(metric="sum", keys="file_id")
    assert isinstance(scalar, (int, float))

    multi = fm.reduce(metric="max", keys=["file_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"file_id"}

    grouped_str = fm.reduce(metric="sum", keys="file_id", group_by="status")
    assert isinstance(grouped_str, dict)

    grouped_list = fm.reduce(
        metric="sum",
        keys=["file_id"],
        group_by=["status", "file_id"],
    )
    assert isinstance(grouped_list, dict)


# ────────────────────────────────────────────────────────────────────────────
# 11. Simulated GlobalFileManager                                            #
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simulated_global_file_manager():
    """Fixture for a SimulatedGlobalFileManager with two underlying managers."""
    fm1 = sim_mod.SimulatedFileManager()
    fm2 = sim_mod.SimulatedFileManager()
    fm1.clear_simulated_files()
    fm2.clear_simulated_files()
    fm1.add_simulated_file(
        "greeting.txt",
        records=[{"content": "Hello world"}],
        full_text="Hello world from manager one",
        description="Greeting file",
    )
    fm2.add_simulated_file(
        "plan.md",
        records=[{"content": "Strategy notes"}],
        full_text="Strategy notes from manager two",
        description="Plan file",
    )
    gfm = sim_mod.SimulatedGlobalFileManager([fm1, fm2])
    yield gfm


def test_global_list_filesystems(simulated_global_file_manager):
    """Test that list_filesystems returns the expected manager class names."""
    gfm = simulated_global_file_manager
    filesystems = gfm.list_filesystems()
    assert isinstance(filesystems, list)
    assert "SimulatedFileManager" in filesystems
