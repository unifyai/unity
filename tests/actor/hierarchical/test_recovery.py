"""Recovery & course-correction tests for HierarchicalActor."""

import asyncio
import contextlib
import textwrap
from unittest.mock import AsyncMock

import pytest

from unity.actor.hierarchical_actor import (
    CacheInvalidateSpec,
    CacheStepRange,
    FunctionPatch,
    HierarchicalActor,
    HierarchicalActorHandle,
    ImplementationDecision,
    InterjectionDecision,
)
from unity.function_manager.computer_backends import (
    MockComputerBackend,
    VALID_MOCK_SCREENSHOT_PNG,
)
from unity.function_manager.function_manager import FunctionManager

from tests.actor.hierarchical.helpers import (
    ConfigurableMockVerificationClient,
    wait_for_log_entry,
)

CANNED_PLAN_FOR_VERIFICATION_FAILURE_TEST_ADVANCE_COURSE_CORRECTION = textwrap.dedent(
    """
    async def _step_1_navigate_and_search():
        '''Navigates to a dummy site and searches for an item.'''
        print("EXEC: Running Step 1: Navigate and Search")
        await computer_primitives.navigate("https://www.allrecipes.com/search?q=pasta")

    async def _step_2_deviate_state():
        '''This function intentionally navigates away, creating a state deviation.'''
        print("EXEC: Running Step 2: Intentionally Deviating State")
        await computer_primitives.navigate("https://www.allrecipes.com/about-us-6648102")

    async def _step_3_attempt_action_on_wrong_page():
        '''This action is expected to fail verification because the popup is in the way.'''
        print("EXEC: Running Step 3: Attempting Action on Wrong Page")
        await computer_primitives.act("Click the first recipe link to go to the details page.")

    async def main_plan():
        await _step_1_navigate_and_search()
        await _step_2_deviate_state()
        await _step_3_attempt_action_on_wrong_page()
        return "Plan completed successfully."
    """,
)


CANNED_PLAN_FOR_INTERJECTION_TEST_ADVANCE_COURSE_CORRECTION = textwrap.dedent(
    """
    async def _multi_step_function():
        '''A function with multiple, distinct, state-changing actions.'''
        print("EXEC: Multi-step function, action 1/3 (Navigate to search page).")
        await computer_primitives.navigate("https://www.allrecipes.com/search?q=cookies")

        print("EXEC: Multi-step function, action 2/3 (Navigate to 'About Us').")
        await computer_primitives.navigate("https://www.allrecipes.com/about-us-6648102")

        print("EXEC: Multi-step function, pausing for interjection...")
        await asyncio.sleep(5)

        print("EXEC: Multi-step function, action 3/3 (This should be skipped).")
        await computer_primitives.act("Click a link on the About Us page.")

    async def main_plan():
        await _multi_step_function()
        return "Original plan finished."
    """,
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_recovery_agent_launches_on_user_interjection():
    """
    Tests intra-function recovery triggered by a user interjection.
    """
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    try:

        async def mock_recovery_agent(plan, target_screenshot, trajectory):
            assert target_screenshot is not None
            assert "about-us" in trajectory[0]
            active_task.action_log.append(
                "COURSE CORRECTION: Mock agent for interjection is running.",
            )

        actor._run_course_correction_agent = mock_recovery_agent

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test interjection recovery.",
            persist=True,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        mock_decision = InterjectionDecision(
            action="modify_task",
            reason="User wants to change the logic after the first navigation.",
            patches=[
                FunctionPatch(
                    function_name="_multi_step_function",
                    new_code=textwrap.dedent(
                        """
                    async def _multi_step_function():
                        print("EXEC: Multi-step function, action 1/3 (Navigate to search page).")
                        await computer_primitives.navigate("https://www.allrecipes.com/search?q=cookies")
                        print("EXEC: Running new, modified action after interjection.")
                        print("EXEC: Multi-step function, action 2/3 (Search for 'brownies').")
                        await computer_primitives.act("Search for 'brownies' instead.")
                """,
                    ),
                ),
            ],
            cache=CacheInvalidateSpec(
                invalidate_steps=[
                    CacheStepRange(
                        function_name="_multi_step_function",
                        from_step_inclusive=2,
                    ),
                ],
            ),
        )
        active_task.modification_client.generate = AsyncMock(
            return_value=mock_decision.model_dump_json(),
        )

        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_INTERJECTION_TEST_ADVANCE_COURSE_CORRECTION,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(active_task, "about-us-6648102")
        while len(active_task.idempotency_cache) != 2:
            await asyncio.sleep(0.1)

        await active_task.interject("Change the plan after the first search.")
        await wait_for_log_entry(
            active_task,
            "COURSE CORRECTION: Mock agent for interjection is running.",
        )
        await wait_for_log_entry(active_task, "Search for 'brownies' instead.")

        await active_task.stop("Test complete.")
        final_log = "\n".join(active_task.action_log)
        assert "COURSE CORRECTION: Mock agent for interjection is running." in final_log
        assert "CACHE HIT" in final_log

    finally:
        if active_task and not active_task.done():
            await active_task.stop()
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_recovery_agent_launches_on_verification_failure_and_restores_state():
    """
    Tests recovery triggered by verification failure; recovery agent restores state,
    then the actor re-implements and replays from the correct point.
    """
    fm = FunctionManager()
    fm.clear()

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=fm,
    )
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    try:
        mock_v_client = ConfigurableMockVerificationClient()
        mock_v_client.set_behavior(
            "_step_1_navigate_and_search",
            [("ok", "Navigated successfully.")],
        )
        mock_v_client.set_behavior(
            "_step_2_deviate_state",
            [("ok", "State deviated as planned.")],
        )
        mock_v_client.set_behavior(
            "_step_3_attempt_action_on_wrong_page",
            [
                (
                    "reimplement_local",
                    "Action failed, element not found on the 'About Us' page.",
                ),
                ("ok", "Action succeeded after state recovery."),
            ],
        )

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test recovery from verification failure.",
            persist=False,
            can_store=False,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = mock_v_client

        async def mock_dynamic_implement(*args, **kwargs):
            _ = (args, kwargs)
            return ImplementationDecision(
                action="implement_function",
                reason="Re-implementing after course correction.",
                code="async def _step_3_attempt_action_on_wrong_page(): await computer_primitives.act('Click any recipe.')",
            )

        actor._dynamic_implement = mock_dynamic_implement
        active_task.implementation_client.generate = AsyncMock(
            return_value=ImplementationDecision(
                action="implement_function",
                reason="Re-implementing after course correction.",
                code="async def _step_3_attempt_action_on_wrong_page(): await computer_primitives.act('Click any recipe.')",
            ).model_dump_json(),
        )

        async def mock_recovery_agent(plan, target_screenshot, trajectory):
            assert target_screenshot is not None
            assert len(trajectory) > 0
            assert "Click the first recipe link" in trajectory[0]
            active_task.action_log.append("COURSE CORRECTION: Mock agent is running.")

        actor._run_course_correction_agent = mock_recovery_agent

        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_VERIFICATION_FAILURE_TEST_ADVANCE_COURSE_CORRECTION,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        final_result = await asyncio.wait_for(active_task.result(), timeout=60)
        _ = final_result
        final_log = "\n".join(active_task.action_log)
        assert "COURSE CORRECTION: Mock agent is running." in final_log

    finally:
        if active_task and not active_task.done():
            await active_task.stop()
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_course_correction_orchestrator():
    """Deprecated orchestrator (kept for compatibility)."""
    pytest.skip("Orchestrator is redundant; run individual recovery tests instead.")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_modify_task_skips_course_correction_when_disabled():
    """
    When course correction is disabled, invalidations should not spawn the recovery sub-agent.
    """
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        enable_course_correction=False,
    )
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test course correction gating.",
            persist=True,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_task._execution_task
        active_task._execution_task = None

        await actor._prepare_execution_environment(active_task)
        active_task._restart_execution_loop = lambda *args, **kwargs: None  # type: ignore[method-assign]
        active_task._clear_browser_queue_for_run = AsyncMock(return_value=None)

        key_valid = (("main_plan",), (), (), 1, "computer_primitives.act", "valid")
        key_invalid = (
            ("main_plan",),
            (),
            (),
            2,
            "computer_primitives.observe",
            "invalid",
        )
        active_task.idempotency_cache = {
            key_valid: {
                "meta": {
                    "function": "main_plan",
                    "step": 1,
                    "post_state_screenshot": VALID_MOCK_SCREENSHOT_PNG,
                },
                "interaction_log": ["", "computer_primitives.act(...)", ""],
                "result": None,
            },
            key_invalid: {
                "meta": {"function": "main_plan", "step": 2, "impure": True},
                "interaction_log": [
                    "",
                    "computer_primitives.observe(...)",
                    "Returned: ok",
                ],
                "result": None,
            },
        }

        actor._run_course_correction_agent = AsyncMock(return_value=None)  # type: ignore[method-assign]

        decision = InterjectionDecision(
            action="modify_task",
            reason="Update plan; invalidate step 2.",
            patches=[
                FunctionPatch(
                    function_name="main_plan",
                    new_code=textwrap.dedent(
                        """
                        async def main_plan():
                            return "ok"
                        """,
                    ).strip(),
                ),
            ],
            cache=CacheInvalidateSpec(
                invalidate_steps=[
                    CacheStepRange(function_name="main_plan", from_step_inclusive=2),
                ],
            ),
        )

        _ = await active_task._execute_interjection_decision(decision)
        actor._run_course_correction_agent.assert_not_awaited()

    finally:
        if active_task and not active_task.done():
            with contextlib.suppress(Exception):
                await active_task.stop()
        with contextlib.suppress(Exception):
            await actor.close()
