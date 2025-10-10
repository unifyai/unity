"""
FileManager ask functionality tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.helpers import _handle_project
from tests.test_file_manager.helpers import ask_judge


@pytest.mark.asyncio
async def test_ask_nonexistent_file(file_manager):
    """Test asking about non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        await file_manager.ask_about_file("nonexistent.txt", "What is this file about?")


@pytest.mark.asyncio
async def test_ask_with_mocked_llm(file_manager, supported_file_examples: dict):
    """Test basic ask functionality with mocked LLM."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = file_manager.import_file(example_data["path"])  # new API

    file_manager.parse(display_name)

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unify.AsyncUnify",
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
            system_msg = mock_client.set_system_message.call_args[0][0]
            assert "parse" in system_msg

            # Verify tool loop was started
            mock_loop.assert_called_once()
            args, kwargs = mock_loop.call_args
            assert kwargs["loop_id"] == "LocalFileManager.ask"
            assert kwargs.get("preprocess_msgs") is not None


@pytest.mark.asyncio
async def test_ask_with_reasoning_steps(file_manager, supported_file_examples: dict):
    """Test ask with reasoning steps enabled."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = file_manager.import_file(example_data["path"])  # new API

    file_manager.parse(display_name)

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unify.AsyncUnify",
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
    display_name = file_manager.import_file(example_data["path"])  # new API

    file_manager.parse(display_name)

    # Create clarification queues
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unify.AsyncUnify",
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
    display_name = file_manager.import_file(example_data["path"])  # new API

    file_manager.parse(display_name)

    # Mock the async unify client
    with patch(
        "unity.file_manager.managers.file_manager.unify.AsyncUnify",
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
async def test_ask_about_file_integration(file_manager, fm_root, tmp_path: Path):
    """Integration test for the ask_about_file method."""
    fm = file_manager
    # Create test file OUTSIDE fm_root to avoid duplication on import
    file_path = tmp_path / "about_test.txt"
    file_content = "This file is about the history of artificial intelligence."
    file_path.write_text(file_content)
    display_name = fm.import_file(file_path)

    # Parse the file to add it to Unify logs before ask_about_file
    fm.parse(display_name)

    instruction = f"Summarize the file {display_name}."
    handle = await fm.ask_about_file(display_name, instruction)
    response = await handle.result()

    assert response and isinstance(response, str)
    verdict = await ask_judge(instruction, response, file_content=file_content)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed ask_about_file incorrect. Verdict: {verdict}"


@pytest.mark.asyncio
@_handle_project
async def test_ask_multiple_files_integration(file_manager, fm_root, tmp_path: Path):
    """Integration test for asking about multiple files."""
    fm = file_manager
    # Create test files OUTSIDE fm_root to avoid duplication on import
    file1_path = tmp_path / "multi_ask1.txt"
    file1_content = "The first document discusses renewable energy sources."
    file1_path.write_text(file1_content)
    name1 = fm.import_file(file1_path)

    file2_path = tmp_path / "multi_ask2.txt"
    file2_content = "The second document is a report on climate change."
    file2_path.write_text(file2_content)
    name2 = fm.import_file(file2_path)

    # Parse the files to add them to Unify logs before ask
    fm.parse([name1, name2])

    instruction = f"Compare and contrast the documents {name1} and {name2}."
    handle = await fm.ask(instruction)
    response = await handle.result()

    assert response and isinstance(response, str)
    combined_content = {name1: file1_content, name2: file2_content}
    verdict = await ask_judge(instruction, response, file_content=str(combined_content))
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed multi-file ask incorrect. Verdict: {verdict}"
