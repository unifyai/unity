"""
FileManager ask functionality tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unity.file_manager.file_manager import FileManager


@pytest.mark.asyncio
async def test_ask_nonexistent_file():
    file_manager = FileManager()
    """Test asking about non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        await file_manager.ask("nonexistent.txt", "What is this file about?")


@pytest.mark.asyncio
async def test_ask_with_mocked_llm(supported_file_examples: dict):
    """Test basic ask functionality with mocked LLM."""
    file_manager = FileManager()
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = file_manager._add_file(example_data["path"])

    # Mock the async unify client
    with patch("unity.file_manager.file_manager.unify.AsyncUnify") as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()

        # Mock the tool loop to return immediately
        with patch(
            "unity.file_manager.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="This is a test document.")
            mock_loop.return_value = mock_handle

            # Call ask
            handle = await file_manager.ask(display_name, "What is this file about?")

            # Verify the handle was returned
            assert handle is not None

            # Verify system message was set
            mock_client.set_system_message.assert_called_once()
            system_msg = mock_client.set_system_message.call_args[0][0]
            assert "parse(filenames=" in system_msg

            # Verify tool loop was started
            mock_loop.assert_called_once()
            args, kwargs = mock_loop.call_args
            assert kwargs["loop_id"] == "FileManager.ask"
            assert kwargs.get("preprocess_msgs") is not None


@pytest.mark.asyncio
async def test_ask_with_reasoning_steps(supported_file_examples: dict):
    """Test ask with reasoning steps enabled."""
    file_manager = FileManager()
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = file_manager._add_file(example_data["path"])

    # Mock the async unify client
    with patch("unity.file_manager.file_manager.unify.AsyncUnify") as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()
        mock_client.messages = [{"role": "assistant", "content": "Test reasoning"}]

        # Mock the tool loop
        with patch(
            "unity.file_manager.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="Answer")
            mock_loop.return_value = mock_handle

            # Call ask with reasoning steps
            handle = await file_manager.ask(
                display_name,
                "What is this file about?",
                _return_reasoning_steps=True,
            )

            # The result should be wrapped to include messages
            assert hasattr(handle, "result")


@pytest.mark.asyncio
async def test_ask_with_clarification_queues(supported_file_examples: dict):
    """Test ask with clarification support."""
    file_manager = FileManager()
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = file_manager._add_file(example_data["path"])

    # Create clarification queues
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    # Mock the async unify client
    with patch("unity.file_manager.file_manager.unify.AsyncUnify") as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()

        # Mock the tool loop
        with patch(
            "unity.file_manager.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="Answer")
            mock_loop.return_value = mock_handle

            # Call ask with clarification queues
            handle = await file_manager.ask(
                display_name,
                "What is this file about?",
                clarification_up_q=up_q,
                clarification_down_q=down_q,
            )

            # Verify tools included request_clarification
            args, kwargs = mock_loop.call_args
            tools = args[2]  # Third argument is tools
            assert "request_clarification" in tools


@pytest.mark.asyncio
async def test_ask_with_rolling_summary(supported_file_examples: dict):
    """Test ask with rolling summary in prompts."""
    file_manager = FileManager()
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = file_manager._add_file(example_data["path"])

    # Mock the async unify client
    with patch("unity.file_manager.file_manager.unify.AsyncUnify") as mock_unify:
        # Setup mock
        mock_client = AsyncMock()
        mock_unify.return_value = mock_client
        mock_client.set_system_message = MagicMock()

        # Mock the tool loop
        with patch(
            "unity.file_manager.file_manager.start_async_tool_loop",
        ) as mock_loop:
            # Create a mock handle
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="Answer")
            mock_loop.return_value = mock_handle

            # Call ask with rolling_summary_in_prompts=True
            handle = await file_manager.ask(
                display_name,
                "What is this file about?",
                rolling_summary_in_prompts=True,
            )

            # Verify system message includes broader_context placeholder
            mock_client.set_system_message.assert_called_once()
            system_msg = mock_client.set_system_message.call_args[0][0]
            assert "{broader_context}" in system_msg
