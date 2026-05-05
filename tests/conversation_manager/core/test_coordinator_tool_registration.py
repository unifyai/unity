from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.helpers import _handle_project
from unity.common.single_shot import SingleShotResult
from unity.conversation_manager.domains.coordinator_tools import CoordinatorTools
from unity.session_details import SESSION_DETAILS


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_registers_workspace_tools_only_for_coordinator(
    initialized_cm,
):
    """Coordinator-only tools are registered through the real slow-brain assembly."""

    cm = initialized_cm.cm
    cm.initialized = False
    captured_tool_names: list[set[str]] = []

    async def fake_single_shot_tool_decision(_client, _messages, tools, **_kwargs):
        captured_tool_names.append(set(tools))
        return SingleShotResult(
            tools=[],
            text_response=None,
            structured_output=None,
        )

    coordinator_tool_names = set(CoordinatorTools(cm=cm).as_tools())
    with patch(
        "unity.conversation_manager.conversation_manager.single_shot_tool_decision",
        fake_single_shot_tool_decision,
    ):
        try:
            SESSION_DETAILS.is_coordinator = True
            await cm._run_llm()
            SESSION_DETAILS.is_coordinator = False
            await cm._run_llm()
        finally:
            SESSION_DETAILS.is_coordinator = False

    assert coordinator_tool_names <= captured_tool_names[0]
    assert coordinator_tool_names.isdisjoint(captured_tool_names[1])
