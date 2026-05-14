from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tests.helpers import _handle_project
from unity.common.single_shot import SingleShotResult
from unity.common.context_registry import ContextRegistry
from unity.coordinator_manager.workspace_manager import CoordinatorWorkspaceManager
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_UNITY_TEST_INIT") == "1",
    reason="Requires full runtime context initialization.",
)


@pytest.fixture(autouse=True)
def _ensure_context_base():
    previous_impl = os.environ.get("UNITY_FUNCTION_IMPL")
    previous_base_context = getattr(ContextRegistry, "_base_context", None)
    os.environ["UNITY_FUNCTION_IMPL"] = "simulated"
    ManagerRegistry.clear()
    ContextRegistry.set_base_context("UnityTests/Coordinator")
    yield
    if previous_impl is None:
        os.environ.pop("UNITY_FUNCTION_IMPL", None)
    else:
        os.environ["UNITY_FUNCTION_IMPL"] = previous_impl
    ContextRegistry.clear()
    if previous_base_context:
        ContextRegistry.set_base_context(previous_base_context)
    ManagerRegistry.clear()


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_registers_workspace_tools_only_for_coordinator(
    initialized_cm,
):
    """Coordinator lifecycle primitives are not registered as direct slow-brain tools."""

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

    coordinator_tool_names = set(CoordinatorWorkspaceManager._PRIMITIVE_METHODS)
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

    assert "act" in captured_tool_names[0]
    assert "act" in captured_tool_names[1]
    assert coordinator_tool_names.isdisjoint(captured_tool_names[0])
    assert coordinator_tool_names.isdisjoint(captured_tool_names[1])
