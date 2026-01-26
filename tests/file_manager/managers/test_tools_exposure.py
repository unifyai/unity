"""
Tool exposure tests for ask_about_file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_about_file_exposes_join_tools(file_manager, tmp_path: Path):
    fm = file_manager
    p = tmp_path / "exp2.txt"
    p.write_text("other content")
    name = str(p)
    fm.ingest_files(name)

    with patch(
        "unity.file_manager.managers.file_manager.start_async_tool_loop",
    ) as mock_loop:
        mock_handle = MagicMock()
        mock_handle.result = AsyncMock(return_value="ok")
        mock_loop.return_value = mock_handle

        handle = await fm.ask_about_file(name, "What is it about?")
        assert handle is not None

        args, kwargs = mock_loop.call_args
        tools = args[2]
        assert "filter_join" in tools
        assert "search_join" in tools
        assert "filter_multi_join" in tools
        assert "search_multi_join" in tools
