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
import unity.planner.hierarchical_planner as hierarchical_planner_module
from unity.common.llm_helpers import AsyncToolUseLoopHandle

# --- Mocks for Dependencies ---


@pytest.fixture
def mock_controller():
    """Provides a mock Controller instance."""
    controller = MagicMock(spec=Controller)

    async def act_func(instruction: str):
        return f"Action completed: {instruction}"

    controller.act = act_func

    async def observe_func(query: str):
        return f"Observed: {query}"

    controller.observe = observe_func
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
    plan = planner.execute(
        "Sign in to the website. Once signed in, respond **only** with 'Signed in successfully.'",
        exploratory_mode=False,
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
    plan = planner.execute("Find the company email.", exploratory_mode=False)
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
    plan = planner.execute(
        "Execute a plan with a flawed child task.",
        exploratory_mode=False,
    )

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
    mock_act = AsyncMock()
    monkeypatch.setattr(mock_controller, "act", mock_act)

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
        planner,
        "_generate_initial_plan",
        AsyncMock(return_value=initial_code),
    )
    monkeypatch.setattr(
        planner,
        "_perform_plan_surgery",
        AsyncMock(return_value=modified_code),
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
    plan = planner.execute("Go to site A and click B.", exploratory_mode=False)

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
    mock_act = AsyncMock()
    monkeypatch.setattr(mock_controller, "act", mock_act)
    plan = planner.execute("Do the original task.", exploratory_mode=False)
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


@pytest.mark.asyncio
async def test_fatal_error_in_verification(planner: HierarchicalPlanner, monkeypatch):
    """
    Objective: Verify that a 'fatal_error' from the verifier stops the plan
    and sets its state to ERROR.
    """
    # --- Arrange ---
    plan_code = '@verify\nasync def main_plan(): await act("Do something")'
    monkeypatch.setattr(
        planner,
        "_generate_initial_plan",
        AsyncMock(return_value=plan_code),
    )
    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        AsyncMock(
            return_value=VerificationAssessment(
                status="fatal_error",
                reason="Unrecoverable error.",
            ),
        ),
    )

    # --- Act ---
    plan = planner.execute("Test fatal error handling.", exploratory_mode=False)
    await plan.result()

    # --- Assert ---
    assert plan._state == _HierarchicalPlanState.ERROR
    assert "fatal_error" in " ".join(plan.action_log)
    assert "Unrecoverable error" in " ".join(plan.action_log)


@pytest.mark.asyncio
async def test_retry_exhaustion_leads_to_escalation(
    planner: HierarchicalPlanner,
    monkeypatch,
):
    """
    Objective: Verify that a function failing repeatedly with a generic exception
    exhausts its local retries and escalates to replan the parent.
    """
    # --- Arrange ---
    failing_code = """
@verify
async def failing_task():
    raise ValueError("This will always fail")

@verify
async def main_plan():
    await failing_task()
"""
    monkeypatch.setattr(
        planner,
        "_generate_initial_plan",
        AsyncMock(return_value=failing_code),
    )
    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )
    mock_handle_dynamic_implementation = AsyncMock()
    monkeypatch.setattr(
        HierarchicalPlan,
        "_handle_dynamic_implementation",
        mock_handle_dynamic_implementation,
    )

    # --- Act ---
    plan = planner.execute(
        "A task that will fail and escalate.",
        exploratory_mode=False,
    )
    # The plan will escalate and pause, so we get the message. This waits for the *entire*
    # escalation process to finish.
    await asyncio.wait_for(plan.clarification_up_q.get(), timeout=20)

    # --- Assert ---
    log_str = "".join(plan.action_log)

    # It should have tried the failing task (MAX_ESCALATIONS + MAX_LOCAL_RETRIES) times.
    expected_failure_count = plan.MAX_ESCALATIONS + plan.MAX_LOCAL_RETRIES
    assert log_str.count("Function 'failing_task' failed") == expected_failure_count

    # The parent function ('main_plan') should have been replanned MAX_ESCALATIONS times.
    assert mock_handle_dynamic_implementation.call_count == plan.MAX_ESCALATIONS

    # Check the last replan call to ensure it was for the right function and reason.
    last_call = mock_handle_dynamic_implementation.call_args_list[-1]
    assert last_call.args[0] == "main_plan"
    assert last_call.kwargs["is_strategic_replan"] is True


@pytest.mark.asyncio
async def test_exploratory_mode_with_clarification(
    planner: HierarchicalPlanner,
    monkeypatch,
):
    """
    Objective: Verify that the enhanced exploratory mode can run an interactive
    tool loop, use the clarification queues to get user input, and then use
    the resulting summary to generate the main plan.
    """
    # --- Arrange ---
    goal = "Find the contact email for ExampleCorp."
    final_plan_code = """
@verify
async def main_plan():
    '''A plan generated from exploration.'''
    await act("Navigate to example.com based on user input.")
    return "Plan complete."
    """
    expected_summary = "Based on user input, the target website is example.com."

    queue_holder = {}

    async def mock_exploration_result(*args, **kwargs):
        question_to_ask = "What is the URL of ExampleCorp's website?"
        await queue_holder["up_q"].put(question_to_ask)
        plan.action_log.append(
            f"Exploration: Asking for clarification: '{question_to_ask}'",
        )
        answer = await queue_holder["down_q"].get()
        assert "example.com" in answer
        return expected_summary

    # 1. Create the mock handle specifically for the exploration phase
    mock_exploration_handle = MagicMock(spec=AsyncToolUseLoopHandle)
    mock_exploration_handle.result = AsyncMock(side_effect=mock_exploration_result)

    # 2. Store the original function so we can call it for the main loop
    original_start_loop = hierarchical_planner_module.start_async_tool_use_loop

    def smart_mock_start_loop(*args, **kwargs):
        """
        This mock distinguishes which loop is being started.
        - If it's the exploration loop, it returns our mock handle.
        - Otherwise, it calls the original, real function.
        """
        # The exploration loop is identifiable by the 'request_clarification' tool
        tools = kwargs.get("tools", {})
        if "request_clarification" in tools:
            return mock_exploration_handle
        else:
            return original_start_loop(*args, **kwargs)

    # 3. Apply the new, smarter mock function
    monkeypatch.setattr(
        hierarchical_planner_module,
        "start_async_tool_use_loop",
        smart_mock_start_loop,
    )

    # 4. Mock the plan generation and final verification as before.
    mock_generate_plan = AsyncMock(return_value=final_plan_code)
    monkeypatch.setattr(planner, "_generate_initial_plan", mock_generate_plan)
    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    plan = planner.execute(goal, exploratory_mode=True)
    queue_holder["up_q"] = plan.clarification_up_q
    queue_holder["down_q"] = plan.clarification_down_q

    async def clarification_handler():
        question = await asyncio.wait_for(plan.clarification_up_q.get(), timeout=5)
        assert "What is the URL" in question
        await plan.clarification_down_q.put("The URL is example.com")

    await asyncio.gather(plan.result(), clarification_handler())

    # --- Assert ---
    assert plan._state == _HierarchicalPlanState.COMPLETED
    mock_generate_plan.assert_called_once()
    call_args_tuple = mock_generate_plan.call_args.args
    assert len(call_args_tuple) == 2
    actual_summary = call_args_tuple[1]
    assert actual_summary == expected_summary
    action_log_str = " ".join(plan.action_log)
    assert "Starting interactive exploratory phase" in action_log_str
    assert "Exploration: Asking for clarification" in action_log_str
    assert f"Exploration Summary: {expected_summary}" in action_log_str


@pytest.mark.asyncio
async def test_user_initiated_stop(planner: HierarchicalPlanner, mock_controller):
    """
    Objective: Test that a user can cleanly stop a running plan.
    """
    # --- Arrange ---
    pause_event = asyncio.Event()
    mock_controller.act.side_effect = pause_event.wait  # This will wait forever

    plan_code = '@verify\nasync def main_plan(): await act("long running action")'
    planner._generate_initial_plan = AsyncMock(return_value=plan_code)
    planner._check_state_against_goal = AsyncMock(
        return_value=VerificationAssessment(status="ok", reason="OK"),
    )

    # --- Act ---
    plan = planner.execute("A long running plan to stop.", exploratory_mode=False)
    await asyncio.sleep(0.1)  # Ensure the plan has started and is waiting
    assert not plan.done()

    stop_result = await plan.stop()

    # --- Assert ---
    assert "Plan was stopped" in stop_result
    assert plan.done()
    assert plan._state == _HierarchicalPlanState.STOPPED


@pytest.mark.asyncio
async def test_user_initiated_pause_and_resume(
    planner: HierarchicalPlanner,
    mock_controller,
):
    """
    Objective: Test that a user can pause and then resume a running plan.
    """
    # --- Arrange ---
    pause_event = asyncio.Event()
    mock_controller.act.side_effect = pause_event.wait

    plan_code = '@verify\nasync def main_plan():\n    await act("long running action")\n    return "Done"'
    planner._generate_initial_plan = AsyncMock(return_value=plan_code)
    planner._check_state_against_goal = AsyncMock(
        return_value=VerificationAssessment(status="ok", reason="OK"),
    )

    # --- Act ---
    plan = planner.execute("A long running plan to pause.", exploratory_mode=False)
    await asyncio.sleep(0.1)
    assert plan._state == _HierarchicalPlanState.RUNNING

    # Pause the plan
    pause_result = await plan.pause()
    assert "Plan paused" in pause_result
    assert plan._state == _HierarchicalPlanState.PAUSED

    # Ensure it's actually paused by trying to await the result with a timeout
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(plan.result(), timeout=0.2)

    # Resume the plan
    resume_result = await plan.resume()
    assert "Plan resumed" in resume_result
    assert plan._state == _HierarchicalPlanState.RUNNING

    # Unblock the paused action and get the final result
    pause_event.set()
    await plan.result()

    # --- Assert ---
    assert plan._state == _HierarchicalPlanState.COMPLETED


@pytest.mark.asyncio
async def test_nested_dynamic_implementation(planner: HierarchicalPlanner, monkeypatch):
    """
    Objective: Verify the planner can handle a chain of dynamic implementations,
    where a newly implemented function itself calls another stubbed function.
    """
    # --- Arrange ---
    initial_code = """
@verify
async def child_task():
    raise NotImplementedError

@verify
async def parent_task():
    '''This task depends on a child task.'''
    raise NotImplementedError

@verify
async def main_plan():
    await parent_task()
    return "Nested implementation complete."
"""

    implemented_parent = """
@verify
async def parent_task():
    '''This task depends on a child task.'''
    await child_task()
"""
    implemented_child = '@verify\nasync def child_task():\n    await act("Perform the final child action.")'

    mock_llm = AsyncMock()
    mock_llm.side_effect = [
        initial_code,
        implemented_parent,
        implemented_child,
    ]
    monkeypatch.setattr("unity.planner.hierarchical_planner.llm_call", mock_llm)

    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    plan = planner.execute("Execute a plan with nested stubs.", exploratory_mode=False)
    final_result = await plan.result()

    # --- Assert ---
    # 1. The plan should complete successfully.
    assert plan._state == _HierarchicalPlanState.COMPLETED
    assert "Nested implementation complete" in final_result

    # 2. Check the sequence of implementations in the action log.
    action_log_str = " ".join(plan.action_log)
    parent_impl_index = action_log_str.find("Implemented function: parent_task")
    child_impl_index = action_log_str.find("Implemented function: child_task")

    assert parent_impl_index != -1 and child_impl_index != -1
    assert parent_impl_index < child_impl_index

    # 3. Both functions should now be fully implemented in the final source code.
    assert "raise NotImplementedError" not in plan.plan_source_code

    assert "await act('Perform the final child action.')" in plan.plan_source_code


@pytest.mark.asyncio
async def test_modify_plan_while_paused(
    planner: HierarchicalPlanner,
    mock_controller,
    monkeypatch,
):
    """
    Objective: Ensure a plan can be successfully modified while it is in an
    explicitly PAUSED state.
    """
    # --- Arrange ---
    pause_event = asyncio.Event()
    mock_controller.act.side_effect = pause_event.wait  # Will hang until event is set

    initial_code = '@verify\nasync def main_plan(): await act("long running action")'
    modified_code = (
        '@verify\nasync def main_plan(): await act("new action after pause")'
    )

    monkeypatch.setattr(
        planner,
        "_generate_initial_plan",
        AsyncMock(return_value=initial_code),
    )
    monkeypatch.setattr(
        planner,
        "_perform_plan_surgery",
        AsyncMock(return_value=modified_code),
    )
    # No course correction needed for this simple change
    monkeypatch.setattr(
        planner,
        "_generate_course_correction_script",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        planner,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    mock_act = AsyncMock()
    monkeypatch.setattr(mock_controller, "act", mock_act)
    plan = planner.execute(
        "A plan to be modified while paused.",
        exploratory_mode=False,
    )
    await asyncio.sleep(0.1)  # Let the plan start and hit the waiting act()

    # Pause the running plan
    await plan.pause()
    assert plan._state == _HierarchicalPlanState.PAUSED

    # Modify the plan while it's paused.
    modification_result = await plan.modify_plan("Change the action.")

    # Unblock the original action (which is now irrelevant but needs to terminate)
    pause_event.set()
    await plan.result()  # Await the result of the *new* plan execution

    # --- Assert ---
    # 1. The modification process should succeed.
    assert "modified and resumed successfully" in modification_result

    # 2. The plan should complete successfully with the new action.
    assert plan._state == _HierarchicalPlanState.COMPLETED
    mock_controller.act.assert_any_call("new action after pause")

    # 3. The action log should show the pause before the modification.
    log_str = " ".join(plan.action_log)
    pause_index = log_str.find("Plan paused by user.")
    modify_index = log_str.find("Modification requested:")
    assert pause_index != -1 and modify_index != -1
    assert pause_index < modify_index


@pytest.mark.asyncio
async def test_invalid_code_generation_handling(
    planner: HierarchicalPlanner,
    monkeypatch,
):
    """
    Objective: Verify that the system handles a SyntaxError from LLM-generated
    code gracefully and enters an ERROR state.
    """
    # --- Arrange ---
    initial_code = "@verify\nasync def main_plan(): raise NotImplementedError"
    invalid_code = "async def main_plan(:\n    pass # Invalid syntax"

    # Mock the LLM to return valid code initially, then invalid code for the implementation
    mock_llm = AsyncMock()
    mock_llm.side_effect = [initial_code, invalid_code]
    monkeypatch.setattr("unity.planner.hierarchical_planner.llm_call", mock_llm)

    # --- Act ---
    plan = planner.execute(
        "A plan that will fail code generation.",
        exploratory_mode=False,
    )
    result = await plan.result()

    # --- Assert ---
    # 1. The plan should be in an ERROR state.
    assert plan._state == _HierarchicalPlanState.ERROR

    # 2. The final result string should contain the error message.
    assert "ERROR:" in result
    assert "invalid syntax" in result


@pytest.mark.asyncio
async def test_failed_course_correction_triggers_rollback(
    planner: HierarchicalPlanner,
    mock_controller,
    monkeypatch,
):
    """
    Objective: Verify that if the course correction script fails its own
    verification, the entire plan modification is rolled back.
    """
    # --- Arrange ---
    initial_code = '@verify\nasync def main_plan():\n    await act("original action")\n    return "Original done."'
    modified_code = '@verify\nasync def main_plan():\n    await act("modified action")'
    correction_script = '@verify\nasync def course_correction_main():\n    await act("correction action")'

    # Mock the various generation steps
    monkeypatch.setattr(
        planner,
        "_generate_initial_plan",
        AsyncMock(return_value=initial_code),
    )
    monkeypatch.setattr(
        planner,
        "_perform_plan_surgery",
        AsyncMock(return_value=modified_code),
    )
    monkeypatch.setattr(
        planner,
        "_generate_course_correction_script",
        AsyncMock(return_value=correction_script),
    )

    # 1. Mock verification: Succeed for the original plan, but FAIL for the course correction.
    async def mock_check_state(function_name: str, *args, **kwargs):
        if function_name == "course_correction":
            return VerificationAssessment(
                status="fatal_error",
                reason="Correction script failed.",
            )
        return VerificationAssessment(status="ok", reason="OK")

    monkeypatch.setattr(planner, "_check_state_against_goal", mock_check_state)

    # --- Act ---
    mock_act = AsyncMock()
    monkeypatch.setattr(mock_controller, "act", mock_act)
    plan = planner.execute(
        "A plan with a failing course correction.",
        exploratory_mode=False,
    )
    await asyncio.sleep(0.1)  # Let the plan start

    modification_result = await plan.modify_plan(
        "A modification that will fail correction.",
    )
    await plan.result()

    # --- Assert ---
    # 1. The modification result should indicate failure and rollback.
    assert "Failed to modify the plan. Rolled back" in modification_result

    # 2. The course correction should have been attempted and failed.
    log_str = " ".join(plan.action_log)
    assert "Executing course correction script" in log_str
    assert "ERROR: Course correction failed" in log_str

    # 3. The plan should have rolled back and completed the ORIGINAL task.
    assert plan.plan_source_code == initial_code
    mock_controller.act.assert_called_with("original action")
    assert plan._state == _HierarchicalPlanState.COMPLETED
