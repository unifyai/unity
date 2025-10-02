import textwrap
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, call

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalPlan,
    VerificationAssessment,
    _HierarchicalPlanState,
)
from unity.actor.action_provider import ActionProvider
from unity.controller.browser import Browser
from unity.function_manager.function_manager import FunctionManager
import unity.actor.hierarchical_actor as hierarchical_actor_module
from unity.common.async_tool_loop import AsyncToolLoopHandle


# --- Mocks for Dependencies ---
@pytest.fixture
def mock_function_manager():
    """Provides a mock FunctionManager instance."""
    fm = MagicMock(spec=FunctionManager)
    fm.list_functions = MagicMock(return_value={})
    fm.add_functions = MagicMock()
    fm.search_functions_by_similarity = MagicMock(return_value=[])
    return fm


@pytest.fixture
def mock_browser():
    """Provides a mock Browser instance with async methods."""
    browser = MagicMock(spec=Browser)
    browser.act = AsyncMock(return_value="Action completed.")
    browser.observe = AsyncMock(return_value="Observation complete.")
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_action_provider(mock_browser):
    """Provides a mock ActionProvider which holds our mock_browser."""
    provider = MagicMock(spec=ActionProvider)
    provider.browser = mock_browser
    # Add aliases for browser methods to match the real ActionProvider
    provider.browser_act = mock_browser.act
    provider.browser_observe = mock_browser.observe
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def actor(mock_function_manager, mock_action_provider, monkeypatch):
    """
    Provides a HierarchicalActor instance where the real ActionProvider
    has been replaced by our mock *before* initialization.
    """
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ActionProvider",
        lambda *args, **kwargs: mock_action_provider,
    )

    p = HierarchicalActor(function_manager=mock_function_manager, headless=True)
    return p


# --- Test Suite ---


@pytest.mark.asyncio
async def test_dynamic_implementation(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify that the actor can correctly identify, implement, and
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
    await action_provider.browser_act("Click the sign-in button")
"""

    successful_verification_json = (
        '{"status": "ok", "reason": "Action completed successfully."}'
    )

    # Mock the LLM calls sequence.
    mock_llm = AsyncMock()
    mock_llm.side_effect = [
        initial_plan_code,
        implemented_code,
        successful_verification_json,  # For sign_in verification
        successful_verification_json,  # For main_plan verification
    ]
    monkeypatch.setattr("unity.actor.hierarchical_actor.llm_call", mock_llm)
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    # --- Act ---
    plan = await actor.act(
        "Sign in to the website. Once signed in, respond **only** with 'Signed in successfully.'",
    )
    await plan.result()

    # --- Assert ---
    # 1. The plan should complete successfully.
    assert plan._state == _HierarchicalPlanState.COMPLETED

    # 2. The LLM should have been called the expected number of times.
    assert mock_llm.call_count == 4

    # 3. The assertion now checks that the 'act' method was called
    #    on the MOCK BROWSER HANDLE, not the controller.
    mock_action_provider.browser_act.assert_called_once_with("Click the sign-in button")

    # 4. The source code check is updated to look for the new pattern.
    assert "raise NotImplementedError" not in plan.plan_source_code
    assert "action_provider.browser_act" in plan.plan_source_code

    # 5. The action log reflects the successful execution flow.
    action_log_str = " ".join(plan.action_log)
    assert "Implemented function: sign_in" in action_log_str
    assert "Verification for sign_in: ok" in action_log_str
    assert "Verification for main_plan: ok" in action_log_str


@pytest.mark.asyncio
async def test_verification_and_tactical_replanning(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
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
    await action_provider.browser_act("Scroll to the footer") # Flawed initial attempt
    return await action_provider.browser_observe("Find the email address")
"""

    reimplemented_find_email_code = """
@verify
async def find_email():
    '''Finds an email on the page.'''
    await action_provider.browser_act("Click on the 'Contact Us' link") # Corrected attempt
    return await action_provider.browser_observe("Find the email address")
"""

    main_plan_code = f"""
{initial_find_email_code}

@verify
async def main_plan():
    '''Main plan to find an email.'''
    email = await find_email()
    return f"Found email: {{email}}"
"""

    # Mock the actor's internal LLM-based verification
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
    monkeypatch.setattr(actor, "_check_state_against_goal", mock_check_state)

    # Mock the dynamic implementation LLM call
    mock_dynamic_implement = AsyncMock(return_value=reimplemented_find_email_code)
    monkeypatch.setattr(actor, "_dynamic_implement", mock_dynamic_implement)

    # Mock the initial plan generation
    mock_generate_plan_llm = AsyncMock(return_value=main_plan_code)
    monkeypatch.setattr(actor, "_generate_initial_plan", mock_generate_plan_llm)

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("Find the company email.")
    await plan.result()

    # --- Assert ---
    # 1. The plan should complete successfully after the replan.
    assert plan._state == _HierarchicalPlanState.COMPLETED

    # 2. _check_state_against_goal was called three times.
    assert mock_check_state.call_count == 3

    # 3. _dynamic_implement was called once for the tactical replan.
    mock_dynamic_implement.assert_called_once()
    assert mock_dynamic_implement.call_args.kwargs["function_name"] == "find_email"

    # 4. The action log should reflect the failure and reimplementation.
    action_log_str = " ".join(plan.action_log)
    assert "reimplement_local" in action_log_str
    assert "Verification for find_email: ok" in action_log_str
    assert "Retrying 'find_email' after reimplementation" in action_log_str


@pytest.mark.asyncio
async def test_strategic_replanning_escalation(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
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
    await action_provider.browser_act("Perform an impossible action.")
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
    async def mock_check_state_against_goal(plan, function_name: str, *args, **kwargs):
        if function_name == "child_task":
            return VerificationAssessment(
                status="replan_parent",
                reason="The child task is conceptually flawed.",
            )
        # Let the parent succeed if it's ever re-verified
        return VerificationAssessment(status="ok", reason="Parent is ok.")

    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        mock_check_state_against_goal,
    )

    # Mock the initial plan generation
    mock_generate_plan = AsyncMock(return_value=main_plan_code)
    monkeypatch.setattr(actor, "_generate_initial_plan", mock_generate_plan)

    # We will also mock the replan of the parent to see it gets called
    mock_handle_dynamic_implementation = AsyncMock()
    monkeypatch.setattr(
        HierarchicalPlan,
        "_handle_dynamic_implementation",
        mock_handle_dynamic_implementation,
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act(
        "Execute a plan with a flawed child task.",
    )

    # --- Assert ---
    # 1. Wait for the escalation message. This is the correct way to sync.
    # It proves the escalation logic was reached. We use a timeout to prevent hangs.
    escalation_message = await asyncio.wait_for(
        plan.clarification_up_q.get(),
        timeout=20,
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
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Test the end-to-end modify_plan workflow, including surgery
    and course correction, using the new handle-based API.
    """
    # --- Arrange ---
    initial_code = """
@verify
async def main_plan():
    '''Initial plan to go to site A and click B.'''
    await action_provider.browser_act("Navigate to site A")
    # Plan will be paused and modified after this point
    await action_provider.browser_act("Click button B")
    return "Finished at site A."
"""

    modified_code = """
@verify
async def main_plan():
    '''Modified plan to go to site C and click D.'''
    await action_provider.browser_act("Navigate to site C")
    await action_provider.browser_act("Click button D")
    return "Finished at site C."
"""

    correction_script = """
@verify
async def course_correction_main():
    '''Navigates to the correct starting site for the new plan.'''
    await action_provider.browser_act("Navigate to site C")
"""

    # Create an event to deterministically pause the plan's execution.
    plan_is_paused_event = asyncio.Event()
    mock_act = AsyncMock()
    monkeypatch.setattr(mock_action_provider, "browser_act", mock_act)

    # Configure a side effect for the mock handle to pause the plan.
    async def act_side_effect(instruction: str):
        if "Navigate to site A" in instruction:
            return "Navigated to site A."
        elif "Click button B" in instruction:
            # When the plan tries to click button B, pause it by waiting on our event.
            await plan_is_paused_event.wait()
            return "Clicked button B."
        # Default behavior for any other action (e.g., in the modified plan).
        return f"Action '{instruction}' completed."

    mock_action_provider.browser_act.side_effect = act_side_effect

    # Mock the LLM calls
    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=initial_code),
    )
    monkeypatch.setattr(
        actor,
        "_perform_plan_surgery",
        AsyncMock(return_value=modified_code),
    )
    monkeypatch.setattr(
        actor,
        "_generate_course_correction_script",
        AsyncMock(return_value=correction_script),
    )
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("Go to site A and click B.")

    # Wait until the first 'act' call completes. This ensures the plan is
    # now running and paused inside the second 'act' call, waiting on our event.
    while mock_action_provider.browser_act.call_count < 1:
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
    actor._generate_course_correction_script.assert_called_once()
    mock_action_provider.browser_act.assert_any_call("Navigate to site C")

    # 3. The final plan execution should reflect the new goal.
    mock_action_provider.browser_act.assert_any_call("Click button D")

    # 4. The final result should be from the modified plan.
    assert plan._state == _HierarchicalPlanState.COMPLETED


@pytest.mark.asyncio
async def test_failed_plan_modification_rollback(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
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
    await action_provider.browser_act("Do original task")
    return "Original task done."
"""
    # Mock initial generation
    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=original_code),
    )

    # Mock the plan surgery to fail
    monkeypatch.setattr(
        actor,
        "_perform_plan_surgery",
        AsyncMock(side_effect=Exception("LLM failed to generate new code.")),
    )

    # Mock verification to always succeed
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("Do the original task.")
    await asyncio.sleep(0.5)  # Let it start

    modification_result = await plan.modify_plan("This modification will fail.")
    await plan.result()

    # --- Assert ---
    # 1. The modification result string should indicate failure and rollback.
    assert "Failed to modify the plan. Rolled back" in modification_result

    # 2. The plan's source code should be the original code.
    assert "Do original task" in plan.plan_source_code

    # 3. The plan should have continued and completed the original task.
    mock_action_provider.browser_act.assert_called_with("Do original task")
    assert plan._state == _HierarchicalPlanState.COMPLETED

    # 4. Check the action log for the rollback message.
    action_log_str = " ".join(plan.action_log)
    assert "ERROR: Failed to modify plan, rolling back" in action_log_str


@pytest.mark.asyncio
async def test_fatal_error_in_verification(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify that a 'fatal_error' from the verifier stops the plan
    and sets its state to ERROR.
    """
    # --- Arrange ---
    plan_code = """
@verify
async def main_plan():
    await action_provider.browser_act("Do something")
"""
    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=plan_code),
    )
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(
            return_value=VerificationAssessment(
                status="fatal_error",
                reason="Unrecoverable error.",
            ),
        ),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("Test fatal error handling.")
    await plan.result()

    # --- Assert ---
    mock_action_provider.browser_act.assert_called_once_with("Do something")
    assert plan._state == _HierarchicalPlanState.ERROR
    assert "fatal_error" in " ".join(plan.action_log)
    assert "Unrecoverable error" in " ".join(plan.action_log)


@pytest.mark.asyncio
async def test_retry_exhaustion_leads_to_escalation(
    actor: HierarchicalActor,
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
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=failing_code),
    )
    monkeypatch.setattr(
        actor,
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
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act(
        "A task that will fail and escalate.",
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
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
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
    await action_provider.browser_act("Navigate to example.com based on user input.")
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
    mock_exploration_handle = MagicMock(spec=AsyncToolLoopHandle)
    mock_exploration_handle.result = AsyncMock(side_effect=mock_exploration_result)

    # 2. Store the original function so we can call it for the main loop
    original_start_loop = hierarchical_actor_module.start_async_tool_loop

    def smart_mock_start_loop(*args, **kwargs):
        """
        This mock distinguishes which loop is being started.
        - If it's the exploration loop, it returns our special mock handle.
        - Otherwise, it calls the original, real function for the main plan execution.
        """
        if kwargs.get("loop_id") == "ExploratoryPhase":
            return mock_exploration_handle
        else:
            return original_start_loop(*args, **kwargs)

    # 3. Apply the new, smarter mock function
    monkeypatch.setattr(
        hierarchical_actor_module,
        "start_async_tool_loop",
        smart_mock_start_loop,
    )

    # 4. Mock the plan generation and final verification as before.
    mock_generate_plan = AsyncMock(return_value=final_plan_code)
    monkeypatch.setattr(actor, "_generate_initial_plan", mock_generate_plan)
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=True))
    plan = await actor.act(goal)
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
    mock_action_provider.browser_act.assert_called_once_with(
        "Navigate to example.com based on user input.",
    )


@pytest.mark.asyncio
async def test_user_initiated_stop(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Test that a user can cleanly stop a running plan.
    """
    # --- Arrange ---
    pause_event = asyncio.Event()
    mock_action_provider.browser_act.side_effect = (
        pause_event.wait
    )  # This will wait forever

    plan_code = """
@verify
async def main_plan():
    await action_provider.browser_act("long running action")
"""
    actor._generate_initial_plan = AsyncMock(return_value=plan_code)
    actor._check_state_against_goal = AsyncMock(
        return_value=VerificationAssessment(status="ok", reason="OK"),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("A long running plan to stop.")
    await asyncio.sleep(0.1)  # Ensure the plan has started and is waiting
    assert not plan.done()

    stop_result = await plan.stop()

    # --- Assert ---
    assert "Plan was stopped" in stop_result
    assert plan.done()
    assert plan._state == _HierarchicalPlanState.STOPPED


@pytest.mark.asyncio
async def test_user_initiated_pause_and_resume(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Test that a user can pause and then resume a running plan.
    """
    # --- Arrange ---
    pause_event = asyncio.Event()

    async def act_side_effect(*args, **kwargs):
        await pause_event.wait()
        return "Action completed after resume."

    mock_action_provider.browser_act.side_effect = act_side_effect

    plan_code = """
@verify
async def main_plan():
    await action_provider.browser_act("long running action")
    return "Done"
"""
    actor._generate_initial_plan = AsyncMock(return_value=plan_code)
    actor._check_state_against_goal = AsyncMock(
        return_value=VerificationAssessment(status="ok", reason="OK"),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("A long running plan to pause.")
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
async def test_nested_dynamic_implementation(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify the actor can handle a chain of dynamic implementations,
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

    implemented_child = """
@verify
async def child_task():
    await action_provider.browser_act("Perform the final child action.")
"""

    # Mock the LLM calls to provide the sequence of implementations
    mock_llm = AsyncMock()
    mock_llm.side_effect = [
        initial_code,
        implemented_parent,
        implemented_child,
    ]
    monkeypatch.setattr("unity.actor.hierarchical_actor.llm_call", mock_llm)

    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("Execute a plan with nested stubs.")
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

    # 3. The final child action should have been called on the handle.
    mock_action_provider.browser_act.assert_called_once_with(
        "Perform the final child action.",
    )


@pytest.mark.asyncio
async def test_modify_plan_while_paused(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Ensure a plan can be successfully modified while it is in an
    explicitly PAUSED state.
    """
    # --- Arrange ---
    pause_event = asyncio.Event()

    async def act_side_effect(*args, **kwargs):
        # This side effect is only for the *initial* plan's act call.
        if "long running action" in args:
            await pause_event.wait()
            return "This result is for the old plan and will be discarded."
        # The new plan's action will pass through here without waiting.
        return "New action completed."

    mock_action_provider.browser_act.side_effect = act_side_effect

    initial_code = """
@verify
async def main_plan():
    await action_provider.browser_act("long running action")
"""
    modified_code = """
@verify
async def main_plan():
    await action_provider.browser_act("new action after pause")
"""

    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=initial_code),
    )
    monkeypatch.setattr(
        actor,
        "_perform_plan_surgery",
        AsyncMock(return_value=modified_code),
    )
    # No course correction needed for this simple change
    monkeypatch.setattr(
        actor,
        "_generate_course_correction_script",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("A plan to be modified while paused.")
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
    mock_action_provider.browser_act.assert_any_call("new action after pause")

    # 3. The action log should show the pause before the modification.
    log_str = " ".join(plan.action_log)
    pause_index = log_str.find("Plan paused by user.")
    modify_index = log_str.find("Modification requested:")
    assert pause_index != -1 and modify_index != -1
    assert pause_index < modify_index


@pytest.mark.asyncio
async def test_invalid_code_generation_handling(
    actor: HierarchicalActor,
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
    monkeypatch.setattr("unity.actor.hierarchical_actor.llm_call", mock_llm)

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("A plan that will fail code generation.")
    result = await plan.result()

    # --- Assert ---
    # 1. The plan should be in an ERROR state.
    assert plan._state == _HierarchicalPlanState.ERROR

    # 2. The final result string should contain the error message.
    assert "ERROR:" in result
    assert "invalid syntax" in result


@pytest.mark.asyncio
async def test_failed_course_correction_triggers_rollback(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify that if the course correction script fails its own
    verification, the entire plan modification is rolled back.
    """
    # --- Arrange ---
    initial_code = """
@verify
async def main_plan():
    await action_provider.browser_act("original action")
    return "Original done."
"""
    modified_code = """
@verify
async def main_plan():
    await action_provider.browser_act("modified action")
"""
    correction_script = """
@verify
async def course_correction_main():
    await action_provider.browser_act("correction action")
"""

    # Mock the various generation steps
    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=initial_code),
    )
    monkeypatch.setattr(
        actor,
        "_perform_plan_surgery",
        AsyncMock(return_value=modified_code),
    )
    monkeypatch.setattr(
        actor,
        "_generate_course_correction_script",
        AsyncMock(return_value=correction_script),
    )

    # 1. Mock verification: Succeed for the original plan, but FAIL for the course correction.
    async def mock_check_state(plan, function_name: str, *args, **kwargs):
        if function_name == "course_correction":
            return VerificationAssessment(
                status="fatal_error",
                reason="Correction script failed.",
            )
        return VerificationAssessment(status="ok", reason="OK")

    monkeypatch.setattr(actor, "_check_state_against_goal", mock_check_state)

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("A plan with a failing course correction.")
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
    assert "original action" in plan.plan_source_code
    mock_action_provider.browser_act.assert_called_with("original action")
    assert plan._state == _HierarchicalPlanState.COMPLETED


@pytest.mark.asyncio
async def test_dynamic_implementation_relies_on_cache(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify that after a dynamic implementation causes the plan to
    restart, previously completed steps are fetched from the cache and are
    not re-executed.
    """
    # --- Arrange ---
    initial_code = """
@verify
async def step_A():
    '''This step will run and be cached on the first pass.'''
    await action_provider.browser_act("Performing Step A")
    return "Step A is done."

@verify
async def step_B_stub():
    '''This step will trigger the dynamic implementation and restart.'''
    raise NotImplementedError

@verify
async def main_plan():
    '''The main plan that will be restarted.'''
    await step_A()
    await step_B_stub()
    return "Plan fully complete."
"""

    implemented_step_b = """
@verify
async def step_B_stub():
    '''This is the new implementation for Step B.'''
    await action_provider.browser_act("Performing NEWLY IMPLEMENTED Step B")
    return "Step B is now done."
"""

    # Mock the LLM calls to provide the initial plan, then the implementation.
    mock_llm = AsyncMock()
    mock_llm.side_effect = [initial_code, implemented_step_b]
    monkeypatch.setattr("unity.actor.hierarchical_actor.llm_call", mock_llm)

    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("A plan that tests the cache hit on restart.")
    await plan.result()

    # --- Assert ---
    # 1. The plan should complete successfully.
    assert plan._state == _HierarchicalPlanState.COMPLETED

    # 2. **CRUCIAL ASSERTION:** The action from `step_A` should have been
    #    executed exactly once, checking the mock handle.
    step_a_calls = [
        c
        for c in mock_action_provider.browser_act.call_args_list
        if c == call("Performing Step A")
    ]
    assert (
        len(step_a_calls) == 1
    ), "The action for step_A should only have been called once."

    # 3. Ensure the newly implemented step was also called.
    mock_action_provider.browser_act.assert_any_call(
        "Performing NEWLY IMPLEMENTED Step B",
    )
    assert mock_action_provider.browser_act.call_count == 2


@pytest.mark.asyncio
async def test_skip_cache_lifecycle(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify that the skip cache correctly prevents re-execution
    of completed functions and that it is correctly invalidated upon modification.
    """
    # --- Arrange ---
    plan_is_paused_event = asyncio.Event()
    initial_code = """
@verify
async def repeatable_task(param: str):
    '''A task that can be called multiple times.'''
    await action_provider.browser_act(f"Performing task for {param}")

@verify
async def main_plan():
    await repeatable_task(param="A")
    await repeatable_task(param="B")
    await repeatable_task(param="A") # This call should be cached
    return "Completed repeatable tasks."
"""
    modified_code = """
@verify
async def repeatable_task(param: str):
    '''A modified task that does something new.'''
    await action_provider.browser_act(f"Performing NEW task for {param}")

@verify
async def main_plan():
    await repeatable_task(param="A") # This should now execute the new version
    return "Completed modified task."
"""

    async def act_side_effect(instruction: str):
        if "Performing task for A" in instruction:
            await asyncio.sleep(0.01)
            return f"Action '{instruction}' completed."
        if "Performing task for B" in instruction:
            plan_is_paused_event.set()
            await asyncio.sleep(60)
        return f"Action '{instruction}' completed."

    # The mock side effect is on the handle.
    mock_action_provider.browser_act.side_effect = act_side_effect

    # Mock the actor's dependencies
    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=initial_code),
    )
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )
    monkeypatch.setattr(
        actor,
        "_perform_plan_surgery",
        AsyncMock(return_value=modified_code),
    )
    monkeypatch.setattr(
        actor,
        "_generate_course_correction_script",
        AsyncMock(return_value=None),
    )

    # --- Act 1: Initial Run & Caching ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("Test caching.")
    await asyncio.wait_for(plan_is_paused_event.wait(), timeout=5)
    await asyncio.sleep(0.1)

    # --- Assert 1: State Before Modification ---
    assert mock_action_provider.browser_act.call_count == 2
    assert len(plan.completed_functions) == 1
    key_a = ("repeatable_task", frozenset([("param", "A")]))
    assert key_a in plan.completed_functions

    # --- Act 2: Modification & Final Result ---
    modification_result = await plan.modify_plan(
        "Change the repeatable_task implementation.",
    )
    await plan.result()

    # --- Assert 2: State After Modification ---
    assert "modified and resumed successfully" in modification_result
    assert plan._state == _HierarchicalPlanState.COMPLETED
    assert mock_action_provider.browser_act.call_count == 3
    mock_action_provider.browser_act.assert_called_with("Performing NEW task for A")
    assert len(plan.completed_functions) == 2


@pytest.mark.asyncio
async def test_actor_reuses_skills_from_function_manager(
    actor: HierarchicalActor,
    mock_function_manager: MagicMock,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify that the actor retrieves an existing, relevant skill
    from the FunctionManager and uses it in the generated plan.
    """
    # --- Arrange ---
    goal = "Go to LinkedIn and check for new messages."
    prompt_capture = {}

    navigate_skill_code = """
@verify
async def navigate_to_linkedin():
    '''Navigates the browser to the main LinkedIn homepage.'''
    await action_provider.browser_act("Go to linkedin.com")
"""
    navigate_skill_code = textwrap.dedent(navigate_skill_code).strip()
    navigate_skill_dict = {
        "name": "navigate_to_linkedin",
        "argspec": "()",
        "docstring": "Navigates the browser to the main LinkedIn homepage.",
        "implementation": navigate_skill_code,
    }
    mock_function_manager.search_functions_by_similarity.return_value = [
        navigate_skill_dict,
    ]
    expected_plan_code = f"""
{navigate_skill_code}

@verify
async def check_messages():
    '''Checks for new messages after navigation.'''
    raise NotImplementedError

@verify
async def main_plan():
    '''Main plan to check LinkedIn messages.'''
    await navigate_to_linkedin()
    await check_messages()
    return "Finished checking messages."
"""

    async def mock_llm_call(client, prompt: str):
        if "### Existing Functions Library" in prompt and "main_plan" in prompt:
            prompt_capture["initial_plan_prompt"] = prompt
            return expected_plan_code
        return '{"status": "ok", "reason": "OK"}'

    monkeypatch.setattr(
        "unity.actor.hierarchical_actor.llm_call",
        AsyncMock(side_effect=mock_llm_call),
    )
    monkeypatch.setattr(
        actor,
        "_dynamic_implement",
        AsyncMock(
            return_value="@verify\nasync def check_messages(): return 'Test complete.'",
        ),
    )
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act(goal)
    await plan.result()

    # --- Assert ---
    # 1. The actor should have searched for relevant skills.
    mock_function_manager.search_functions_by_similarity.assert_called_once_with(
        query=goal,
        n=5,
    )

    # 2. The retrieved skill's code MUST have been included in the prompt to the LLM.
    assert "prompt_capture" in locals() and "initial_plan_prompt" in prompt_capture
    assert navigate_skill_code in prompt_capture["initial_plan_prompt"]

    # 3. The action from within the retrieved skill should have been executed.
    mock_action_provider.browser_act.assert_any_call("Go to linkedin.com")

    # 4. The plan should complete.
    assert plan._state == _HierarchicalPlanState.COMPLETED


@pytest.mark.asyncio
async def test_sandbox_supports_pydantic_classes(
    actor: HierarchicalActor,
    mock_action_provider: MagicMock,
    monkeypatch,
):
    """
    Objective: Verify that the sandbox environment properly supports creating
    and using Pydantic classes, regular classes, inheritance, and other
    class-related Python constructs.
    """
    # --- Arrange ---
    plan_code = """
# Test 1: Basic Pydantic model
@verify
async def test_basic_pydantic():
    '''Test creating and using a basic Pydantic model.'''

    class UserInfo(BaseModel):
        name: str = Field(..., description="User's name")
        age: int = Field(..., description="User's age")
        email: str = Field(default="", description="User's email")

    user = UserInfo(name="John Doe", age=30)
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == ""

    # Test validation
    try:
        invalid_user = UserInfo(name="Jane", age="not a number")
    except ValueError:
        pass  # Expected

    return user.model_dump()

# Test 2: Nested Pydantic models
@verify
async def test_nested_pydantic():
    '''Test nested Pydantic models and inheritance.'''

    class Address(BaseModel):
        street: str
        city: str
        country: str = "USA"

    class Person(BaseModel):
        name: str
        address: Address
        hobbies: list[str] = Field(default_factory=list)

    addr = Address(street="123 Main St", city="New York")
    person = Person(
        name="Alice",
        address=addr,
        hobbies=["reading", "coding"]
    )

    assert person.address.city == "New York"
    assert len(person.hobbies) == 2

    return person.model_dump_json()

# Test 3: Regular Python classes
@verify
async def test_regular_classes():
    '''Test regular Python classes with inheritance.'''

    class Animal:
        def __init__(self, name):
            self.name = name

        def speak(self):
            return f"{self.name} makes a sound"

    class Dog(Animal):
        def __init__(self, name, breed):
            super().__init__(name)
            self.breed = breed

        def speak(self):
            return f"{self.name} barks"

        @property
        def description(self):
            return f"{self.name} is a {self.breed}"

        @classmethod
        def create_puppy(cls, breed):
            return cls("Puppy", breed)

        @staticmethod
        def dog_fact():
            return "Dogs are loyal"

    dog = Dog("Rex", "Golden Retriever")
    assert dog.speak() == "Rex barks"
    assert dog.description == "Rex is a Golden Retriever"

    puppy = Dog.create_puppy("Beagle")
    assert puppy.name == "Puppy"
    assert Dog.dog_fact() == "Dogs are loyal"

    return "Classes work correctly"

# Test 4: Using Pydantic for browser observation
@verify
async def test_pydantic_with_browser():
    '''Test using Pydantic models with browser observations.'''

    class PageElements(BaseModel):
        has_login_button: bool = Field(description="Whether login button exists")
        button_text: str = Field(default="", description="Text on the button")
        page_title: str = Field(description="Title of the page")

    # Simulate observing page structure
    observation = await action_provider.browser_observe(
        "Check for login elements on the page",
        response_format=PageElements
    )

    # The mock will return a string, but in real usage it would be structured
    return f"Observation completed: {observation}"

# Test 5: Complex class operations
@verify
async def test_class_introspection():
    '''Test class introspection and dynamic attribute access.'''

    class DynamicClass:
        class_var = "shared"

        def __init__(self):
            self.instance_var = "unique"

    obj = DynamicClass()

    # Test hasattr/getattr/setattr
    assert hasattr(obj, "instance_var")
    assert getattr(obj, "instance_var") == "unique"
    setattr(obj, "new_attr", "dynamic")
    assert obj.new_attr == "dynamic"

    # Test type checking
    assert type(obj).__name__ == "DynamicClass"
    assert isinstance(obj, DynamicClass)
    assert callable(obj.__init__)

    # Test dir() and vars()
    attrs = dir(obj)
    assert "instance_var" in attrs
    assert "new_attr" in vars(obj)

    return "Introspection complete"

@verify
async def main_plan():
    '''Main plan to test all class-related functionality.'''
    basic_result = await test_basic_pydantic()
    nested_result = await test_nested_pydantic()
    regular_result = await test_regular_classes()
    browser_result = await test_pydantic_with_browser()
    introspection_result = await test_class_introspection()

    return {
        "basic_pydantic": basic_result,
        "nested_pydantic": nested_result,
        "regular_classes": regular_result,
        "browser_observation": browser_result,
        "introspection": introspection_result
    }
"""

    # Mock the LLM to return our test plan
    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=plan_code),
    )

    # Mock verification to always succeed
    monkeypatch.setattr(
        actor,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    # Mock browser observation to return a simple string
    mock_action_provider.browser_observe.return_value = "Mock observation result"

    # --- Act ---
    monkeypatch.setattr(actor, "_should_explore", AsyncMock(return_value=False))
    plan = await actor.act("Test that sandbox supports all class constructs")
    result = await plan.result()

    # --- Assert ---
    # 1. The plan should complete successfully
    assert plan._state == _HierarchicalPlanState.COMPLETED
    assert "Plan completed" in result

    # 2. All test functions should have been executed
    action_log_str = " ".join(plan.action_log)
    assert "Verification for test_basic_pydantic: ok" in action_log_str
    assert "Verification for test_nested_pydantic: ok" in action_log_str
    assert "Verification for test_regular_classes: ok" in action_log_str
    assert "Verification for test_pydantic_with_browser: ok" in action_log_str
    assert "Verification for test_class_introspection: ok" in action_log_str
    assert "Verification for main_plan: ok" in action_log_str

    # 3. Browser observation with Pydantic should have been called
    mock_action_provider.browser_observe.assert_called_once()
    call_args = mock_action_provider.browser_observe.call_args
    assert "login elements" in call_args[0][0]
    assert "response_format" in call_args[1]
