"""
FileManager ask functionality tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.eval
from tests.helpers import _handle_project
from tests.test_file_manager.helpers import ask_judge


@pytest.mark.asyncio
async def test_ask_nonexistent(file_manager):
    """Test asking about non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        await file_manager.ask_about_file("nonexistent.txt", "What is this file about?")


@pytest.mark.asyncio
async def test_ask_with_mocked_llm(file_manager, supported_file_examples: dict):
    """Test basic ask functionality with mocked LLM."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path; no import needed

    file_manager.ingest_files(display_name)

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unillm.AsyncUnify",
    ) as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()

        # Mock the tool loop to return immediately
        with patch(
            "unity.file_manager.managers.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="This is a test document.")
            mock_loop.return_value = mock_handle

            # Call ask
            instruction = f"What is this file about? The file is named {display_name}."
            handle = await file_manager.ask(instruction)

            # Verify the handle was returned
            assert handle is not None

            # Verify system message was set
            mock_client.set_system_message.assert_called_once()

            # Verify tool loop was started
            mock_loop.assert_called_once()
            args, kwargs = mock_loop.call_args
            assert kwargs["loop_id"] == "LocalFileManager.ask"


@pytest.mark.asyncio
async def test_ask_with_reasoning_steps(file_manager, supported_file_examples: dict):
    """Test ask with reasoning steps enabled."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path

    file_manager.ingest_files(display_name)

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unillm.AsyncUnify",
    ) as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()
        mock_client.messages = [{"role": "assistant", "content": "Test reasoning"}]

        # Mock the tool loop
        with patch(
            "unity.file_manager.managers.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="Answer")
            mock_loop.return_value = mock_handle

            # Call ask with reasoning steps
            instruction = f"What is this file about? The file is named {display_name}."
            handle = await file_manager.ask(
                instruction,
                _return_reasoning_steps=True,
            )

            # The result should be wrapped to include messages
            assert hasattr(handle, "result")


@pytest.mark.asyncio
async def test_ask_with_clarification_queues(
    file_manager,
    supported_file_examples: dict,
):
    """Test ask with clarification support."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path

    file_manager.ingest_files(display_name)

    # Create clarification queues
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unillm.AsyncUnify",
    ) as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()

        # Mock the tool loop
        with patch(
            "unity.file_manager.managers.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="Answer")
            mock_loop.return_value = mock_handle

            # Call ask with clarification queues
            instruction = f"What is this file about? The file is named {display_name}."
            handle = await file_manager.ask(
                instruction,
                _clarification_up_q=up_q,
                _clarification_down_q=down_q,
            )

            # Verify tools included request_clarification
            args, kwargs = mock_loop.call_args
            tools = args[2]  # Third argument is tools
            assert "request_clarification" in tools


@pytest.mark.asyncio
async def test_ask_with_rolling_summary(file_manager, supported_file_examples: dict):
    """Test ask with rolling summary in prompts."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path

    file_manager.ingest_files(display_name)

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unillm.AsyncUnify",
    ) as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()

        # Mock the tool loop
        with patch(
            "unity.file_manager.managers.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="Answer")
            mock_loop.return_value = mock_handle

            # Call ask with rolling_summary_in_prompts=True
            instruction = f"What is this file about? The file is named {display_name}."
            handle = await file_manager.ask(
                instruction,
                rolling_summary_in_prompts=True,
            )

            # Verify system message is set
            mock_client.set_system_message.assert_called_once()


@pytest.mark.asyncio
@_handle_project
async def test_ask_about_integration(file_manager, fm_root, tmp_path: Path):
    """Integration test for the ask_about_file method."""
    fm = file_manager
    # Create test file and parse by absolute path (no import needed)
    file_path = tmp_path / "about_test.txt"
    file_content = "This file is about the history of artificial intelligence."
    file_path.write_text(file_content)
    # Parse by absolute path to add to Unify logs before ask_about_file
    fm.ingest_files(str(file_path))

    instruction = f"Summarize the file {file_path}."
    handle = await fm.ask_about_file(str(file_path), instruction)
    response = await handle.result()

    assert response and isinstance(response, str)
    verdict = await ask_judge(instruction, response, file_content=file_content)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed ask_about_file incorrect. Verdict: {verdict}"


@pytest.mark.asyncio
@_handle_project
async def test_ask_multiple_integration(file_manager, fm_root, tmp_path: Path):
    """Integration test for asking about multiple files."""
    fm = file_manager
    # Create test files (no import needed)
    file1_path = tmp_path / "multi_ask1.txt"
    file1_content = "The first document discusses renewable energy sources."
    file1_path.write_text(file1_content)
    name1 = str(file1_path)

    file2_path = tmp_path / "multi_ask2.txt"
    file2_content = "The second document is a report on climate change."
    file2_path.write_text(file2_content)
    name2 = str(file2_path)

    # Parse the files to add them to Unify logs before ask
    fm.ingest_files([name1, name2])

    instruction = f"Compare and contrast the documents {name1} and {name2}."
    handle = await fm.ask(instruction)
    response = await handle.result()

    assert response and isinstance(response, str)
    combined_content = {name1: file1_content, name2: file2_content}
    verdict = await ask_judge(instruction, response, file_content=str(combined_content))
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed multi-file ask incorrect. Verdict: {verdict}"


@pytest.mark.asyncio
@_handle_project
async def test_ask_about_triggers_filter_join_via_loop(
    file_manager,
    tmp_path: Path,
):
    """Ensure ask_about_file calls into filter-join when needed (simulated LLM loop)."""
    fm = file_manager
    p = tmp_path / "join_src.txt"
    p.write_text("seed to create file record")
    name = str(p)
    fm.ingest_files(name)

    # Track tool invocations
    calls = {"filter_join": 0}

    def _stub_loop(client, text, tools, **kwargs):  # type: ignore[override]
        # Wrap filter_join to record invocation and return a canned result
        orig_filter_join = tools.get("filter_join")

        def _wrapped_filter_join(**kw):
            calls["filter_join"] += 1
            # Return structure matching tool contract: mapping ctx->rows
            return {"Derived": [{"product": "Alpha", "total": 3}]}

        # Inject wrapper
        tools = dict(tools)
        if orig_filter_join is not None:
            tools["filter_join"] = _wrapped_filter_join

        class _Handle:
            async def result(self):
                # Simulate the LLM deciding to use filter_join
                _ = tools["filter_join"](
                    tables=["A", "B"],
                    join_expr="A.id == B.id",
                    select={"A.name": "product", "B.qty": "total"},
                )
                return "Answer: Alpha total 3"

        return _Handle()

    with patch(
        "unity.file_manager.managers.file_manager.start_async_tool_loop",
        side_effect=_stub_loop,
    ):
        handle = await fm.ask_about_file(
            name,
            "Compute total orders per product by joining tables",
        )
        ans = await handle.result()
        assert "Alpha" in ans and "3" in ans
        assert calls["filter_join"] >= 1


@pytest.mark.asyncio
@_handle_project
async def test_ask_about_triggers_search_multi_join_via_loop(
    file_manager,
    tmp_path: Path,
):
    """Ensure ask_about_file can chain multi-join search (simulated)."""
    fm = file_manager
    p = tmp_path / "multi_join_src.txt"
    p.write_text("seed")
    name = str(p)
    fm.ingest_files(name)

    calls = {"search_multi_join": 0}

    def _stub_loop(client, text, tools, **kwargs):  # type: ignore[override]
        def _wrapped_search_multi_join(**kw):
            calls["search_multi_join"] += 1
            # Return list of rows (per contract)
            return [{"product": "Beta", "score": 0.99}]

        tools = dict(tools)
        tools["search_multi_join"] = _wrapped_search_multi_join

        class _Handle:
            async def result(self):
                _ = tools["search_multi_join"](
                    joins=[
                        {
                            "tables": ["A", "B"],
                            "join_expr": "A.id == B.id",
                            "select": {"A.name": "product"},
                        },
                        {
                            "tables": ["$prev", "C"],
                            "join_expr": "product == C.name",
                            "select": {"product": "product"},
                        },
                    ],
                    references={"product": "Beta"},
                    k=1,
                )
                return "Answer: Beta"

        return _Handle()

    with patch(
        "unity.file_manager.managers.file_manager.start_async_tool_loop",
        side_effect=_stub_loop,
    ):
        handle = await fm.ask_about_file(
            name,
            "Find the best matching product via multi-join search",
        )
        ans = await handle.result()
        assert "Beta" in ans
        assert calls["search_multi_join"] >= 1
