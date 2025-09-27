from __future__ import annotations

import asyncio
import pytest
import functools

from unity.file_manager.simulated import (
    SimulatedFileManager,
    _SimulatedFileHandle,
)

# helper that wraps each test in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
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
async def test_start_and_ask_simulated_fm():
    fm = SimulatedFileManager("Demo file storage for unit-tests.")

    # Add a sample file
    fm.add_simulated_file(
        "sample.txt",
        records=[{"content": "Sample document content"}],
        metadata={"file_type": "text/plain"},
        full_text="Sample document content for testing",
        description="A sample document for testing purposes",
    )

    handle = await fm.ask("sample.txt", "What is the main topic of this file?")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                          #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_fm_stateful_serial_asks():
    """
    Two consecutive .ask() calls should share context.
    """
    fm = SimulatedFileManager()

    # Add a file
    fm.add_simulated_file(
        "project.txt",
        records=[{"content": "Project Alpha documentation"}],
        metadata={"project": "Alpha"},
        full_text="Project Alpha documentation with detailed specifications",
        description="Documentation for Project Alpha",
    )

    # first question – ask for a single‐word theme of the file
    h1 = await fm.ask(
        "project.txt",
        "Using one word only, how would you describe the main theme of this file?",
    )
    theme = (await h1.result()).strip()
    assert theme, "Theme word should not be empty"

    # follow-up question
    h2 = await fm.ask(
        "project.txt",
        "What single word did you just use to describe this file?",
    )
    ans2 = (await h2.result()).lower()
    # Check if the LLM's response is consistent with the theme
    # Extract the key word from both responses for comparison
    import re

    # Look for quoted words or key terms
    theme_words = re.findall(r'"([^"]*)"', theme.lower())
    ans2_words = re.findall(r'"([^"]*)"', ans2.lower())

    # If no quoted words, try to extract the last meaningful word from theme
    if not theme_words:
        theme_parts = theme.lower().strip('."').split()
        if theme_parts:
            theme_words = [theme_parts[-1]]  # Take the last word as the theme

    if not ans2_words:
        ans2_parts = ans2.lower().strip('."').split()
        if ans2_parts:
            ans2_words = [ans2_parts[-1]]  # Take the last word as answer

    # Check if there's overlap in the key words
    has_overlap = (
        any(word in ans2.lower() for word in theme_words) if theme_words else False
    )
    has_overlap = (
        has_overlap or any(word in theme.lower() for word in ans2_words)
        if ans2_words
        else has_overlap
    )

    assert (
        has_overlap
    ), f"LLM should recall the theme it produced earlier. Theme: '{theme}', Answer: '{ans2}'"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Basic synchronous methods                                              #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_fm_synchronous_operations():
    """Test the synchronous methods like list, exists, parse."""
    fm = SimulatedFileManager()

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


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 5.  Interject                                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_fm(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedFileHandle.interject

    @functools.wraps(orig)
    def wrapped(self, msg: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return orig(self, msg)

    monkeypatch.setattr(_SimulatedFileHandle, "interject", wrapped, raising=True)

    fm = SimulatedFileManager()
    fm.add_simulated_file(
        "report.txt",
        records=[{"content": "Annual report"}],
        full_text="Annual report with financial details",
        description="Company annual report",
    )

    handle = await fm.ask("report.txt", "Summarize the key points.")
    await asyncio.sleep(0.05)
    reply = handle.interject("Focus on financial metrics.")
    assert "ack" in reply.lower() or "noted" in reply.lower()
    await handle.result()
    assert calls["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_fm():
    fm = SimulatedFileManager()
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
async def test_fm_requests_clarification():
    fm = SimulatedFileManager()
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
        clarification_up_q=up_q,
        clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()

    await down_q.put("Focus on statistical trends.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip()
    assert "statistic" in answer.lower() or "trend" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_fm(monkeypatch):
    """
    Ensure a `_SimulatedFileHandle` can be paused and resumed.
    """
    counts = {"pause": 0, "resume": 0}

    # --- patch pause -------------------------------------------------------
    orig_pause = _SimulatedFileHandle.pause

    @functools.wraps(orig_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    monkeypatch.setattr(
        _SimulatedFileHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- patch resume ------------------------------------------------------
    orig_resume = _SimulatedFileHandle.resume

    @functools.wraps(orig_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(
        _SimulatedFileHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    fm = SimulatedFileManager()
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
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() must block while paused"

    # Resume execution
    resume_msg = handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    # Now result() should finish
    answer = await asyncio.wait_for(res_task, timeout=60)
    assert isinstance(answer, str) and answer.strip()

    # Each steering method must have been invoked exactly once
    assert counts == {"pause": 1, "resume": 1}, "pause/resume must each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 9. Nested ask on handle                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    """
    The internal handle returned by SimulatedFileManager.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    fm = SimulatedFileManager()
    fm.add_simulated_file(
        "business.txt",
        records=[{"content": "Business plan for European expansion"}],
        full_text="Business plan for European expansion with market analysis",
        description="European expansion business plan",
    )

    # Start an initial ask to obtain the live handle
    handle = await fm.ask("business.txt", "Summarize this business document.")

    # Add extra context to ensure nested prompt includes it
    handle.interject("Focus on European market opportunities.")

    # Invoke the dynamic ask on the running handle
    nested = await handle.ask("What is the key opportunity mentioned?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip(), (
        "Nested ask() should yield a non-empty string answer",
    )
    assert "europe" in nested_answer.lower()

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip(), (
        "Handle should still yield a non-empty answer after nested ask",
    )


# ────────────────────────────────────────────────────────────────────────────
# 10. Reasoning steps toggle                                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_reasoning_steps_toggle():
    """Test that _return_reasoning_steps works correctly."""
    fm = SimulatedFileManager()
    fm.add_simulated_file(
        "analysis.txt",
        records=[{"content": "Market analysis document"}],
        full_text="Market analysis document with industry trends",
        description="Market analysis report",
    )

    # Ask with reasoning steps
    handle = await fm.ask(
        "analysis.txt",
        "What are the key insights?",
        _return_reasoning_steps=True,
    )
    result = await handle.result()
    assert isinstance(result, tuple) and len(result) == 2
    answer, messages = result
    assert isinstance(answer, str) and answer.strip()
    assert isinstance(messages, list) and len(messages) >= 1

    # Ask without reasoning steps
    handle2 = await fm.ask(
        "analysis.txt",
        "What are the key insights?",
        _return_reasoning_steps=False,
    )
    result2 = await handle2.result()
    assert isinstance(result2, str) and result2.strip()


# ────────────────────────────────────────────────────────────────────────────
# 11. New FileManager functionality tests                                      #
# ────────────────────────────────────────────────────────────────────────────


def test_import_file_functionality():
    """Test the import_file method."""
    fm = SimulatedFileManager()

    # Test importing a file
    filename = fm.import_file("/path/to/document.pdf")
    assert filename == "document.pdf"
    assert fm.exists(filename)

    # Test duplicate filename handling
    filename2 = fm.import_file("/path/to/document.pdf")
    assert filename2 == "document (1).pdf"
    assert fm.exists(filename2)


def test_import_directory_functionality():
    """Test the import_directory method."""
    fm = SimulatedFileManager()

    # Test importing a directory
    added_files = fm.import_directory("/path/to/directory")
    assert len(added_files) >= 2

    # All files should exist
    for filename in added_files:
        assert fm.exists(filename)


def test_search_files_functionality():
    """Test the _search_files method."""
    fm = SimulatedFileManager()

    # Add test files with different content
    fm.add_simulated_file(
        "ai_research.pdf",
        records=[{"content": "AI research"}],
        full_text="Artificial intelligence and machine learning research",
        description="Research about AI",
    )
    fm.add_simulated_file(
        "cooking.txt",
        records=[{"content": "Cooking recipes"}],
        full_text="Chocolate chip cookie recipes",
        description="Cooking guide",
    )

    # Test semantic search
    results = fm._search_files(references={"full_text": "artificial intelligence"}, k=2)
    assert len(results) >= 1
    assert results[0]["filename"] == "ai_research.pdf"

    # Test search by description
    results = fm._search_files(references={"description": "cooking guide"}, k=2)
    assert len(results) >= 1
    assert results[0]["filename"] == "cooking.txt"

    # Test search without references (recent files)
    results = fm._search_files(k=5)
    assert len(results) == 2


def test_filter_files_functionality():
    """Test the _filter_files method."""
    fm = SimulatedFileManager()

    # Add test files
    fm.add_simulated_file(
        "success.pdf",
        records=[{"content": "Success doc"}],
        status="success",
    )
    fm.add_simulated_file(
        "error.txt",
        records=[],
        status="error",
    )

    # Test filtering
    results = fm._filter_files(filter="status == 'success'")
    assert len(results) == 1
    assert results[0]["filename"] == "success.pdf"

    # Test filename filtering
    results = fm._filter_files(filter="endswith('.pdf')")
    assert len(results) == 1
    assert results[0]["filename"] == "success.pdf"

    # Test no filter
    results = fm._filter_files()
    assert len(results) == 2


def test_list_columns_functionality():
    """Test the _list_columns method."""
    fm = SimulatedFileManager()

    # Test with types
    columns = fm._list_columns(include_types=True)
    assert isinstance(columns, dict)
    expected_columns = [
        "file_id",
        "filename",
        "status",
        "error",
        "records",
        "full_text",
        "metadata",
        "description",
        "imported_at",
    ]
    for col in expected_columns:
        assert col in columns
        assert isinstance(columns[col], str)  # Type should be a string

    # Test without types
    column_list = fm._list_columns(include_types=False)
    assert isinstance(column_list, list)
    for col in expected_columns:
        assert col in column_list


def test_delete_file_functionality():
    """Test the _delete_file method."""
    fm = SimulatedFileManager()

    # Add a test file
    fm.add_simulated_file(
        "test.txt",
        records=[{"content": "Test"}],
    )

    # Get the file to find its ID
    files = fm._filter_files()
    assert len(files) == 1
    file_id = files[0]["file_id"]

    # Delete the file
    result = fm._delete_file(file_id=file_id)
    assert result["outcome"] == "file deleted"
    assert result["details"]["file_id"] == file_id

    # File should no longer exist
    files = fm._filter_files()
    assert len(files) == 0

    # Deleting non-existent file should raise error
    import pytest

    with pytest.raises(ValueError, match="No file found with file_id"):
        fm._delete_file(file_id=999)


def test_enhanced_parse_functionality():
    """Test the enhanced parse method with new fields."""
    fm = SimulatedFileManager()

    # Add test files
    fm.add_simulated_file(
        "doc.txt",
        records=[{"content": "Document content"}],
        full_text="Full document text",
        description="Test document",
    )

    # Test parsing
    results = fm.parse("doc.txt")
    assert "doc.txt" in results
    result = results["doc.txt"]

    # Check all expected fields
    assert result["status"] == "success"
    assert result["error"] is None
    assert result["records"] == [{"content": "Document content"}]
    assert result["full_text"] == "Full document text"
    assert result["description"] == "Test document"

    # Test parsing non-existent file
    results = fm.parse("missing.txt")
    assert "missing.txt" in results
    result = results["missing.txt"]
    assert result["status"] == "error"
    assert "not found" in result["error"]


def test_file_data_consistency():
    """Test that file data is consistent across different methods."""
    fm = SimulatedFileManager()

    # Add a test file
    fm.add_simulated_file(
        "consistency.pdf",
        records=[{"content": "Consistent data"}],
        metadata={"topic": "testing"},
        full_text="Full text for consistency testing",
        description="Consistency test file",
    )

    # Check data via different methods
    # 1. Via list and exists
    assert "consistency.pdf" in fm.list()
    assert fm.exists("consistency.pdf")

    # 2. Via parse
    parse_result = fm.parse("consistency.pdf")["consistency.pdf"]
    assert parse_result["full_text"] == "Full text for consistency testing"
    assert parse_result["description"] == "Consistency test file"

    # 3. Via search
    search_results = fm._search_files(references={"full_text": "consistency"}, k=1)
    assert len(search_results) == 1
    search_result = search_results[0]
    assert search_result["filename"] == "consistency.pdf"
    assert search_result["full_text"] == "Full text for consistency testing"

    # 4. Via filter
    filter_results = fm._filter_files()
    assert len(filter_results) == 1
    filter_result = filter_results[0]
    assert filter_result["filename"] == "consistency.pdf"
    assert filter_result["description"] == "Consistency test file"


def test_file_ids_are_unique():
    """Test that file IDs are unique and sequential."""
    fm = SimulatedFileManager()

    # Add multiple files
    fm.add_simulated_file("file1.txt", records=[{"content": "1"}])
    fm.add_simulated_file("file2.txt", records=[{"content": "2"}])
    fm.add_simulated_file("file3.txt", records=[{"content": "3"}])

    # Get all files
    files = fm._filter_files()
    assert len(files) == 3

    # Check IDs are unique and sequential
    ids = [f["file_id"] for f in files]
    assert len(set(ids)) == 3  # All unique
    assert min(ids) == 1
    assert max(ids) == 3


# ────────────────────────────────────────────────────────────────────────────
# 12. Update existing test helper calls                                        #
# ────────────────────────────────────────────────────────────────────────────


def test_simulated_fm_updated_helpers():
    """Test all the helper methods and test updated functionality."""
    fm = SimulatedFileManager()

    # Test list_columns is available
    columns = fm._list_columns()
    assert "filename" in columns
    assert "full_text" in columns
    assert "description" in columns

    # Test updated add_simulated_file parameters work
    fm.add_simulated_file(
        "updated_test.pdf",
        records=[{"content": "Updated test"}],
        metadata={"version": "2.0"},
        full_text="This is the full text content",
        description="An updated test file",
        status="success",
    )

    # Verify the file was added with all fields
    files = fm._filter_files()
    assert len(files) == 1
    file_data = files[0]
    assert file_data["filename"] == "updated_test.pdf"
    assert file_data["full_text"] == "This is the full text content"
    assert file_data["description"] == "An updated test file"
    assert file_data["status"] == "success"
    assert file_data["metadata"]["version"] == "2.0"
