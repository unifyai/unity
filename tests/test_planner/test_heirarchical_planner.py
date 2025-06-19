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


@pytest.mark.asyncio
async def test_verification_and_tactical_replanning(
    planner: HierarchicalPlanner,
    monkeypatch,
):
    """
    Objective: Verify that the @verify decorator can trigger a "local reimplementation"
    when the LLM deems an action was tactically flawed.
    """
    # --- Arrange ---
    initial_find_email_code = """
@verify
async def find_email():
    '''Finds an email on the page.'''
    await act("Scroll to the footer") # Flawed initial attempt
    return await observe("Find the email address")
"""

    reimplemented_find_email_code = """
@verify
async def find_email():
    '''Finds an email on the page.'''
    await act("Click on the 'Contact Us' link") # Corrected attempt
    return await observe("Find the email address")
"""

    main_plan_code = f"""
{initial_find_email_code}

@verify
async def main_plan():
    '''Main plan to find an email.'''
    email = await find_email()
    return f"Found email: {{email}}"
"""

    # Mock the planner's internal LLM-based verification
    mock_check_state = AsyncMock()
    # The full, correct sequence of verification calls:
    mock_check_state.side_effect = [
        # 1. First verification of find_email fails, triggering reimplementation.
        VerificationAssessment(
            status="reimplement_local",
            reason="Did not click contact page first.",
        ),
        # 2. Second verification of the *new* find_email succeeds.
        VerificationAssessment(status="ok", reason="Successfully found email."),
        # 3. Final verification of the parent main_plan succeeds.
        VerificationAssessment(status="ok", reason="Parent plan also looks good."),
    ]
    monkeypatch.setattr(planner, "_check_state_against_goal", mock_check_state)

    # Mock the dynamic implementation LLM call
    mock_dynamic_implement_llm = AsyncMock(return_value=reimplemented_find_email_code)
    monkeypatch.setattr(planner, "_dynamic_implement", mock_dynamic_implement_llm)

    # Mock the initial plan generation
    mock_generate_plan_llm = AsyncMock(return_value=main_plan_code)
    monkeypatch.setattr(planner, "_generate_initial_plan", mock_generate_plan_llm)

    # --- Act ---
    plan = planner.plan("Find the company email.")
    await plan.result()

    # --- Assert ---
    # 1. The plan should complete successfully after the replan.
    assert plan._state == _HierarchicalPlanState.COMPLETED

    # 2. _check_state_against_goal was called three times.
    assert mock_check_state.call_count == 3

    # 3. _dynamic_implement was called once for the tactical replan.
    mock_dynamic_implement_llm.assert_called_once()
    assert mock_dynamic_implement_llm.call_args.kwargs["function_name"] == "find_email"

    # 4. The action log should reflect the failure and reimplementation.
    action_log_str = " ".join(plan.action_log)
    assert "reimplement_local" in action_log_str
    assert "Verification for find_email: ok" in action_log_str
    assert "Retrying 'find_email' after reimplementation" in action_log_str


@pytest.mark.asyncio
async def test_strategic_replanning_escalation(
    planner: HierarchicalPlanner,
    monkeypatch,
):
    """
    Objective: Verify that a strategic failure in a child function correctly bubbles up
    and triggers a replan of the parent function, leading to an escalation pause.
    """
    # --- Arrange ---
    child_task_code = """
@verify
async def child_task():
    '''A child task that will fail strategically.'''
    await act("Perform an impossible action.")
    return "This should not be reached."
"""

    main_plan_code = f"""
{child_task_code}

@verify
async def main_plan():
    '''Calls a child task.'''
    await child_task()
    return "Completed."
"""

    # Mock verification to always fail strategically for the child task
    async def mock_check_state_against_goal(function_name: str, *args, **kwargs):
        if function_name == "child_task":
            return VerificationAssessment(
                status="replan_parent",
                reason="The child task is conceptually flawed.",
            )
        # Let the parent succeed if it's ever re-verified
        return VerificationAssessment(status="ok", reason="Parent is ok.")

    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        mock_check_state_against_goal,
    )

    # Mock the initial plan generation
    mock_generate_plan = AsyncMock(return_value=main_plan_code)
    monkeypatch.setattr(planner, "_generate_initial_plan", mock_generate_plan)

    # We will also mock the replan of the parent to see it gets called
    mock_handle_dynamic_implementation = AsyncMock()
    monkeypatch.setattr(
        HierarchicalPlan,
        "_handle_dynamic_implementation",
        mock_handle_dynamic_implementation,
    )

    # --- Act ---
    plan = planner.plan("Execute a plan with a flawed child task.")

    # --- Assert ---
    # 1. Wait for the escalation message. This is the correct way to sync.
    # It proves the escalation logic was reached. We use a timeout to prevent hangs.
    escalation_message = await asyncio.wait_for(
        plan.clarification_up_q.get(),
        timeout=10,
    )

    # 2. Now that we have the message, the state MUST be correct.
    assert plan._state == _HierarchicalPlanState.PAUSED_FOR_ESCALATION
    assert "ESCALATION" in escalation_message
    assert "child_task" in escalation_message

    # 3. The parent function ('main_plan') should have been triggered for a strategic replan
    # multiple times, reaching the escalation limit.
    assert mock_handle_dynamic_implementation.call_count == plan.MAX_ESCALATIONS

    # Check one of the calls to ensure it was for the right function and reason.
    last_call = mock_handle_dynamic_implementation.call_args_list[-1]
    assert last_call.args[0] == "main_plan"
    assert last_call.kwargs["is_strategic_replan"] is True


@pytest.mark.asyncio
async def test_full_plan_modification_and_correction(
    planner: HierarchicalPlanner,
    mock_controller,
    monkeypatch,
):
    """
    Objective: Test the end-to-end modify_plan workflow, including surgery
    and course correction.
    """
    # --- Arrange ---
    initial_code = """
@verify
async def go_to_site_a():
    await act("Navigate to site A")

@verify
async def click_button_b():
    await act("Click button B")

@verify
async def main_plan():
    await go_to_site_a()
    # Plan will be modified after this point
    await click_button_b()
    return "Finished at site A."
"""

    modified_code = """
@verify
async def go_to_site_c():
    await act("Navigate to site C")

@verify
async def click_button_d():
    await act("Click button D")

@verify
async def main_plan():
    await go_to_site_c()
    await click_button_d()
    return "Finished at site C."
"""

    correction_script = """
async def course_correction_main():
    '''Navigates to the correct starting site for the new plan.'''
    await act("Navigate to site C")
"""

    # Create an event to deterministically pause the plan's execution.
    plan_is_paused_event = asyncio.Event()

    # Configure a side effect for the mock controller to pause the plan.
    async def act_side_effect(instruction: str):
        if "Navigate to site A" in instruction:
            return "Navigated to site A."
        elif "Click button B" in instruction:
            # When the plan tries to click button B, pause it by waiting on our event.
            await plan_is_paused_event.wait()
            return "Clicked button B."
        # Default behavior for any other action (e.g., in the modified plan).
        return f"Action '{instruction}' completed."

    mock_controller.act.side_effect = act_side_effect

    # Mock the LLM calls
    monkeypatch.setattr(
        planner, "_generate_initial_plan", AsyncMock(return_value=initial_code)
    )
    monkeypatch.setattr(
        planner, "_perform_plan_surgery", AsyncMock(return_value=modified_code)
    )
    monkeypatch.setattr(
        planner,
        "_generate_course_correction_script",
        AsyncMock(return_value=correction_script),
    )
    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    plan = planner.plan("Go to site A and click B.")

    # Wait until the first 'act' call completes. This ensures the plan is
    # now running and paused inside the second 'act' call, waiting on our event.
    while mock_controller.act.call_count < 1:
        await asyncio.sleep(0.01)

    # Now that the plan is paused in the RUNNING state, modify it.
    modification_result = await plan.modify_plan(
        "Change the goal to go to site C and click D instead.",
    )
    await plan.result()

    # --- Assert ---
    # 1. The modification process should report success.
    assert "modified and resumed successfully" in modification_result

    # 2. The course correction script should have been called.
    planner._generate_course_correction_script.assert_called_once()
    mock_controller.act.assert_any_call("Navigate to site C")

    # 3. The final plan execution should reflect the new goal.
    mock_controller.act.assert_any_call("Click button D")

    # 4. The final result should be from the modified plan.
    assert plan._state == _HierarchicalPlanState.COMPLETED


@pytest.mark.asyncio
async def test_failed_plan_modification_rollback(
    planner: HierarchicalPlanner,
    mock_controller,
    monkeypatch,
):
    """
    Objective: Ensure that if plan surgery fails, the plan rolls back to its
    original state and continues execution.
    """
    # --- Arrange ---
    original_code = """
@verify
async def main_plan():
    await act("Do original task")
    return "Original task done."
"""
    # Mock initial generation
    monkeypatch.setattr(
        planner,
        "_generate_initial_plan",
        AsyncMock(return_value=original_code),
    )

    # Mock the plan surgery to fail
    monkeypatch.setattr(
        planner,
        "_perform_plan_surgery",
        AsyncMock(side_effect=Exception("LLM failed to generate new code.")),
    )

    # Mock verification to always succeed
    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    plan = planner.plan("Do the original task.")
    await asyncio.sleep(0.5)  # Let it start

    modification_result = await plan.modify_plan("This modification will fail.")
    await plan.result()

    # --- Assert ---
    # 1. The modification result string should indicate failure and rollback.
    assert "Failed to modify the plan. Rolled back" in modification_result

    # 2. The plan's source code should be the original code.
    assert plan.plan_source_code == original_code

    # 3. The plan should have continued and completed the original task.
    mock_controller.act.assert_called_with("Do original task")
    assert plan._state == _HierarchicalPlanState.COMPLETED

    # 4. Check the action log for the rollback message.
    action_log_str = " ".join(plan.action_log)
    assert "ERROR: Failed to modify plan, rolling back" in action_log_str
