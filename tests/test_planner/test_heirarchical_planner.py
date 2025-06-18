import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from unity.planner.hierarchical_planner import (
    HierarchicalPlanner,
    HierarchicalPlan,
    VerificationAssessment,
    _HierarchicalPlanState,
)
from unity.controller.controller import Controller
from unity.function_manager.function_manager import FunctionManager
from unity.planner.tool_loop_planner import ComsManager

# --- Mocks for Dependencies ---


@pytest.fixture
def mock_controller():
    """Provides a mock Controller instance."""
    controller = MagicMock(spec=Controller)
    controller.act = AsyncMock(return_value="Action completed.")
    controller.observe = AsyncMock(return_value="Observed the initial page state.")
    return controller


@pytest.fixture
def mock_function_manager():
    """Provides a mock FunctionManager instance."""
    fm = MagicMock(spec=FunctionManager)
    fm.list_functions = MagicMock(return_value={})
    fm.add_functions = MagicMock()
    return fm


@pytest.fixture
def mock_coms_manager():
    """Provides a mock ComsManager instance."""
    return MagicMock(spec=ComsManager)


@pytest.fixture
def planner(mock_function_manager, mock_controller, mock_coms_manager):
    """Provides a HierarchicalPlanner instance with mocked dependencies."""
    return HierarchicalPlanner(
        function_manager=mock_function_manager,
        controller=mock_controller,
        coms_manager=mock_coms_manager,
    )


# --- Test Suite ---


@pytest.mark.asyncio
async def test_dynamic_implementation(planner: HierarchicalPlanner, monkeypatch):
    """
    Objective: Verify that the planner can correctly identify, implement, and
    execute a stubbed-out function at runtime.
    """
    # --- Arrange ---
    initial_plan_code = """
@verify
async def sign_in():
    raise NotImplementedError

@verify
async def main_plan():
    '''Main plan to sign in.'''
    await sign_in()
    return "Signed in successfully."
"""

    implemented_code = """
@verify
async def sign_in():
    '''Signs the user in.'''
    await act("Click the sign-in button")
"""

    # The JSON response for a successful verification step.
    successful_verification_json = (
        '{"status": "ok", "reason": "Action completed successfully."}'
    )

    # Mock the LLM calls
    mock_llm = AsyncMock()
    # The full sequence of LLM calls required is:
    # 1. Generate the initial plan.
    # 2. Implement the stubbed 'sign_in' function.
    # 3. Verify the result of the implemented 'sign_in' function.
    # 4. Verify the result of the top-level 'main_plan' function.
    mock_llm.side_effect = [
        initial_plan_code,
        implemented_code,
        successful_verification_json,  # For sign_in verification
        successful_verification_json,  # For main_plan verification
    ]
    monkeypatch.setattr("unity.planner.hierarchical_planner.llm_call", mock_llm)

    # --- Act ---
    plan = planner.plan(
        "Sign in to the website. Once signed in, respond **only** with 'Signed in successfully.'",
    )
    final_result = await plan.result()

    # --- Assert ---
    # 1. The plan should complete successfully.
    assert plan._state == _HierarchicalPlanState.COMPLETED

    # 2. The LLM should have been called four times.
    assert mock_llm.call_count == 4

    # 3. The planner should have dynamically implemented 'sign_in'.
    assert "raise NotImplementedError" not in plan.plan_source_code
    assert "await act('Click the sign-in button')" in plan.plan_source_code

    # 4. Assert the action log reflects the dynamic implementation and verification.
    action_log_str = " ".join(plan.action_log)
    assert "Implemented function: sign_in" in action_log_str
    assert "Verification for sign_in: ok" in action_log_str
    assert "Verification for main_plan: ok" in action_log_str
