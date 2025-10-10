from __future__ import annotations

import asyncio
import pytest
import functools

from unity.file_manager import simulated as sim_mod

# helper that wraps each test in its own Unify project / trace context
from tests.helpers import _handle_project
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
def test_simulated_fm_docstrings_match_base():
    """
    Public methods in SimulatedFileManager should copy the real
    BaseFileManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.file_manager.base import BaseFileManager
    from unity.file_manager.simulated import SimulatedFileManager

    assert (
        BaseFileManager.ask.__doc__.strip() in SimulatedFileManager.ask.__doc__.strip()
    ), ".ask doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_fm(simulated_file_manager):
    fm = simulated_file_manager
    # Add a sample file
    file_content = "Sample document content for testing"
    fm.add_simulated_file(
        "sample.txt",
        records=[{"content": "Sample document content"}],
        metadata={"file_type": "text/plain"},
        full_text=file_content,
        description="A sample document for testing purposes",
    )

    instruction = "What is the main topic of this file?"
    handle = await fm.ask("sample.txt", instruction)
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    verdict = await ask_judge(instruction, answer, file_content=file_content)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed 'ask' incorrect. Verdict: {verdict}"


@pytest.mark.asyncio
@_handle_project
async def test_ask_about_file_simulated_fm(simulated_file_manager):
    fm = simulated_file_manager
    file_content = "Notes about Mars missions and space exploration"
    fm.add_simulated_file(
        "topic.txt",
        records=[{"content": "Space exploration notes"}],
        metadata={"file_type": "text/plain"},
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


@pytest.mark.asyncio
@_handle_project
async def test_organize_simulated_fm(simulated_file_manager):
    fm = simulated_file_manager
    fm.add_simulated_file(
        "docA.txt",
        records=[{"content": "alpha"}],
        full_text="alpha",
        description="A",
    )
    fm.add_simulated_file(
        "docB.txt",
        records=[{"content": "beta"}],
        full_text="beta",
        description="B",
    )
    instruction = "Rename docA.txt to alpha_document.txt and delete docB.txt."
    handle = await fm.organize(instruction)
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip()

    # Ask LLM judge to verify the plausibility of the simulated response
    verdict = await ask_judge(instruction, answer)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed the operation incorrect. Verdict: {verdict}"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                          #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_fm_stateful_serial_asks(simulated_file_manager):
    """
    Two consecutive .ask() calls should share context.
    """
    fm = simulated_file_manager

    # Add a file
    file_content = "Project Alpha documentation with detailed specifications"
    fm.add_simulated_file(
        "project.txt",
        records=[{"content": "Project Alpha documentation"}],
        metadata={"project": "Alpha"},
        full_text=file_content,
        description="Documentation for Project Alpha",
    )

    # first question – ask for a single‐word theme of the file
    instruction1 = (
        "Using one word only, how would you describe the main theme of this file?"
    )
    h1 = await fm.ask(
        "project.txt",
        instruction1,
    )
    theme = (await h1.result()).strip()
    assert theme, "Theme word should not be empty"

    # Verify with judge
    verdict1 = await ask_judge(instruction1, theme, file_content=file_content)
    assert (
        verdict1.lower().strip().startswith("correct")
    ), f"Judge deemed the first ask incorrect. Verdict: {verdict1}"

    # follow-up question
    instruction2 = "What single word did you just use to describe this file?"
    h2 = await fm.ask(
        "project.txt",
        instruction2,
    )
    ans2 = (await h2.result()).lower()

    # Let the judge decide if the second answer is consistent with the first, given the context
    verdict2 = await ask_judge(
        instruction2,
        ans2,
        file_content=f"The previous response was: '{theme}'",
    )
    assert (
        verdict2.lower().strip().startswith("correct")
    ), f"LLM should recall the theme it produced earlier. Theme: '{theme}', Answer: '{ans2}', Verdict: {verdict2}"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Basic synchronous methods                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_simulated_fm_synchronous_operations(simulated_file_manager):
    """Test the synchronous methods like list, exists, parse."""
    fm = simulated_file_manager

    # Initially empty
    assert fm.list() == []
    assert not fm.exists("any.txt")

    # Add files
    fm.add_simulated_file(
        "doc1.txt",
        records=[{"content": "Document 1"}],
        metadata={"file_type": "text/plain"},
        full_text="Document 1 content",
        description="First test document",
    )
    fm.add_simulated_file(
        "doc2.txt",
        records=[{"content": "Document 2"}],
        metadata={"file_type": "text/plain"},
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
    result = fm.parse("doc1.txt")
    assert result["doc1.txt"]["status"] == "success"
    assert len(result["doc1.txt"]["records"]) == 1

    # Parse multiple files
    results = fm.parse(["doc1.txt", "doc2.txt"])
    assert len(results) == 2
    assert all(r["status"] == "success" for r in results.values())

    # Parse non-existent file
    result = fm.parse("missing.txt")
    assert result["missing.txt"]["status"] == "error"


# Remaining tests intentionally create isolated instances or operate on handles.


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 5.  Interject                                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_fm(monkeypatch, simulated_file_manager):
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
    handle = await fm.ask("report.txt", instruction)
    await asyncio.sleep(0.05)
    reply = handle.interject("Focus on financial metrics.")
    assert "ack" in reply.lower() or "noted" in reply.lower()
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
async def test_stop_simulated_fm(simulated_file_manager):
    fm = simulated_file_manager
    fm.add_simulated_file(
        "large.txt",
        records=[{"content": "Very large document"}],
        full_text="Very large document with extensive content",
        description="Large document for testing",
    )

    handle = await fm.ask("large.txt", "Generate a detailed analysis of this file.")
    await asyncio.sleep(0.05)
    handle.stop()
    await handle.result()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_fm_requests_clarification(simulated_file_manager):
    fm = simulated_file_manager
    fm.add_simulated_file(
        "data.txt",
        records=[{"content": "Research data"}],
        full_text="Research data with statistical analysis",
        description="Research dataset",
    )

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await fm.ask(
        "data.txt",
        "Please analyze this file thoroughly.",
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
async def test_pause_and_resume_simulated_fm(monkeypatch, simulated_file_manager):
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

    handle = await fm.ask("complex.txt", "Perform a comprehensive analysis.")

    # Pause the handle
    pause_msg = handle.pause()
    assert "pause" in pause_msg.lower() or "paused" in pause_msg.lower()

    # Start result() while still paused – it should await
    res_task = asyncio.create_task(handle.result())
    await _assert_blocks_while_paused(res_task)

    # Resume execution
    resume_msg = handle.resume()
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
    The internal handle returned by SimulatedFileManager.ask exposes a
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
    handle = await fm.ask("business.txt", instruction1)

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
    handle = await fm.ask(
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
    handle2 = await fm.ask(
        "analysis.txt",
        instruction,
        _return_reasoning_steps=False,
    )
    result2 = await handle2.result()
    assert isinstance(result2, str) and result2.strip()

    verdict2 = await ask_judge(instruction, result2, file_content=file_content)
    assert (
        verdict2.lower().strip().startswith("correct")
    ), f"Judge deemed ask without reasoning incorrect. Verdict: {verdict2}"
