"""Verification-related tests for HierarchicalActor (async verification, preemption, skip-verify)."""

import asyncio
import contextlib
import textwrap
import traceback
from unittest.mock import AsyncMock

import pytest

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalActorHandle,
    _HierarchicalHandleState,
)
from unity.function_manager.computer_backends import (
    MockComputerBackend,
    VALID_MOCK_SCREENSHOT_PNG,
)
from unity.function_manager.function_manager import FunctionManager

from tests.actor.hierarchical.helpers import (
    ConfigurableMockVerificationClient,
    MockImplementationClient,
    SimpleMockVerificationClient,
    wait_for_log_entry,
    wait_for_state,
)

# --- Canned Plans for predictable tests ---

CANNED_PLAN_SUCCESS_ASYNC_VERIFICATION = textwrap.dedent(
    """
async def step_A_navigate():
    '''Navigates to the site.'''
    await computer_primitives.navigate("https://www.google.com")

async def step_B_search():
    '''Searches for a term.'''
    await computer_primitives.act("Search for 'asynchronous programming'")

async def main_plan():
    await step_A_navigate()
    await step_B_search()
    return "Execution complete."
""",
)

CANNED_PLAN_FAIL_B_ASYNC_VERIFICATION = textwrap.dedent(
    """
async def step_A_navigate():
    '''Navigates to the site. This step will succeed.'''
    await computer_primitives.navigate("https://www.google.com")

async def step_B_fail_verification():
    '''This step executes correctly but is designed to fail verification.'''
    # The action is simple, but we will mock the verifier to return failure.
    await computer_primitives.act("Search for 'test'")

async def step_C_will_be_cancelled():
    '''This step should never run, and its verification should be cancelled.'''
    await computer_primitives.act("This should not be executed.")

async def main_plan():
    await step_A_navigate()
    await step_B_fail_verification()
    await step_C_will_be_cancelled()
    return "This should be unreachable on the first run."
""",
)

CANNED_PLAN_PREEMPTION_ASYNC_VERIFICATION = textwrap.dedent(
    """
async def step_A_ok():
    '''A successful step.'''
    await computer_primitives.navigate("https://www.google.com")

async def step_B_fails_slowly():
    '''A step whose verification will fail after a delay.'''
    await computer_primitives.act("Search for 'B'")

async def step_C_fails_fast():
    '''A step whose verification will fail immediately.'''
    await computer_primitives.act("Search for 'C'")

async def main_plan():
    await step_A_ok()
    await step_B_fails_slowly()
    await step_C_fails_fast()
    return "Execution complete."
""",
)


async def _test_non_blocking_and_success(actor):
    """Non-blocking verification + successful completion."""
    active_task = None
    try:

        async def tiny_delay(*_a, **_k):
            await asyncio.sleep(0.01)

        actor.computer_primitives.navigate = AsyncMock(side_effect=tiny_delay)
        actor.computer_primitives.act = AsyncMock(side_effect=tiny_delay)

        mock_client = ConfigurableMockVerificationClient()
        mock_client.set_behavior("step_A_navigate", 2, status="ok", reason="Mock OK")
        mock_client.set_behavior("step_B_search", 2, status="ok", reason="Mock OK")

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test non-blocking success.",
            persist=True,
            can_store=False,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = mock_client
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_SUCCESS_ASYNC_VERIFICATION,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
        )

        final_log = "\n".join(active_task.action_log)
        assert "step_A_navigate" in final_log
        assert "step_B_search" in final_log

        await active_task.stop()

    finally:
        if active_task:
            await active_task.stop()


async def _test_failure_and_cancellation(actor):
    """Verification failure triggers recovery; later verifications are cancelled."""
    active_task = None
    try:

        async def tiny_delay(*_a, **_k):
            await asyncio.sleep(0.01)

        actor.computer_primitives.navigate = AsyncMock(side_effect=tiny_delay)
        actor.computer_primitives.act = AsyncMock(side_effect=tiny_delay)

        mock_client = ConfigurableMockVerificationClient()
        mock_client.set_behavior(
            "step_A_navigate",
            0.1,
            status="ok",
            reason="Mock success",
        )
        mock_client.set_behavior(
            "step_B_fail_verification",
            0.1,
            status="ok",
            reason="Recovered",
            sequence=[
                ("reimplement_local", "Mocked tactical failure"),
                ("ok", "Recovered after fix"),
            ],
        )
        mock_client.set_behavior(
            "step_C_will_be_cancelled",
            10,
            status="ok",
            reason="This should not be seen",
        )

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test failure and cancellation.",
            persist=False,
            can_store=False,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        actor._run_course_correction_agent = AsyncMock(return_value=None)
        active_task.verification_client = mock_client
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FAIL_B_ASYNC_VERIFICATION,
            active_task,
        )

        active_task.implementation_client.generate = AsyncMock(
            return_value=textwrap.dedent(
                """
            {
                "action": "implement_function",
                "reason": "Fixing the function after mock verification failure.",
                "code": "async def step_B_fail_verification(): await computer_primitives.act(\\"Search for 'fixed test'\\")"
            }
        """,
            ),
        )
        active_task.course_correction_client = mock_client
        active_task.summarization_client = mock_client

        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )
        _ = await asyncio.wait_for(active_task.result(), timeout=60)

        final_log = "\n".join(active_task.action_log)
        assert "step_A_navigate" in final_log
        assert "step_B_fail_verification" in final_log

    finally:
        if active_task:
            await active_task.stop()


async def _test_preemption(actor):
    """Earlier failure preempts recovery of later failure."""
    active_task = None
    try:

        async def tiny_delay(*_a, **_k):
            await asyncio.sleep(0.01)

        actor.computer_primitives.navigate = AsyncMock(side_effect=tiny_delay)
        actor.computer_primitives.act = AsyncMock(side_effect=tiny_delay)

        mock_client = ConfigurableMockVerificationClient()
        mock_client.set_behavior("step_A_ok", 0.1, status="ok", reason="Mock success")
        mock_client.set_behavior(
            "step_B_fails_slowly",
            0.5,
            status="ok",
            reason="Recovered",
            sequence=[
                ("reimplement_local", "The earlier, more critical failure"),
                ("ok", "Recovered after fix"),
            ],
        )
        mock_client.set_behavior(
            "step_C_fails_fast",
            0.1,
            status="ok",
            reason="Recovered",
            sequence=[
                ("reimplement_local", "The later, less critical failure"),
                ("ok", "Recovered after fix"),
            ],
        )

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test preemption.",
            persist=False,
        )
        active_task.can_store = False
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        actor._run_course_correction_agent = AsyncMock(return_value=None)
        active_task.verification_client = mock_client
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_PREEMPTION_ASYNC_VERIFICATION,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        async def slow_generate(*_a, **_k):
            await asyncio.sleep(1.0)
            return textwrap.dedent(
                """
            {
                "action": "implement_function",
                "reason": "Fixing the function.",
                "code": "async def step_C_fails_fast(): pass"
            }
            """,
            )

        active_task.implementation_client.generate = AsyncMock(
            side_effect=slow_generate,
        )
        active_task.course_correction_client = mock_client
        active_task.summarization_client = mock_client

        _ = await asyncio.wait_for(active_task.result(), timeout=60)
        final_log = "\n".join(active_task.action_log)
        assert "step_A_ok" in final_log
        assert "step_B_fails_slowly" in final_log
        assert "step_C_fails_fast" in final_log

    finally:
        if active_task:
            await active_task.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_verification_runs_async_and_handles_failures_and_preemption():
    """End-to-end wrapper that exercises success, failure/cancellation, and preemption."""
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    try:
        await _test_non_blocking_and_success(actor)
        await _test_failure_and_cancellation(actor)
        await _test_preemption(actor)
    except Exception as e:
        print(f"\n\n❌❌❌ A TEST FAILED: {e} ❌❌❌")
        traceback.print_exc()
        raise
    finally:
        await actor.close()


# --- Nested function failure robustness ---

CANNED_PLAN_WITH_NESTED_FAILURE_ROBUSTNESS_FIXES = textwrap.dedent(
    """
    async def parent_skill():
        '''A top-level skill that can be saved to FunctionManager.'''

        async def _nested_child_fails_verification():
            '''A nested helper. It executes fine but its verification will fail.'''
            print("Executing nested child function...")
            await computer_primitives.act("Perform an action that will fail verification.")
            return "Nested child finished."

        print("Executing parent skill...")
        result = await _nested_child_fails_verification()
        print(f"Parent skill received: {result}")
        return "Parent skill finished successfully."

    async def main_plan():
        '''Main entry point.'''
        return await parent_skill()
    """,
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_nested_verification_failure_does_not_corrupt_parent_execution():
    """
    Verifies a delayed nested verification failure doesn't corrupt parent execution.
    """
    actor = None
    active_task = None
    try:
        fm = FunctionManager()
        fm.clear()

        actor = HierarchicalActor(
            function_manager=fm,
            headless=True,
            computer_mode="mock",
            connect_now=False,
            enable_course_correction=False,
        )
        actor.computer_primitives._computer = MockComputerBackend(
            url="https://mock-url.com",
            screenshot=VALID_MOCK_SCREENSHOT_PNG,
        )
        actor.computer_primitives.act = AsyncMock(return_value="Mock action completed.")
        actor.computer_primitives.navigate = AsyncMock(return_value=None)

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Run a plan designed to test recovery from a delayed verification failure.",
            persist=False,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_WITH_NESTED_FAILURE_ROBUSTNESS_FIXES,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        mock_v_client = ConfigurableMockVerificationClient()
        mock_v_client.set_behavior(
            "parent_skill",
            0.1,
            status="ok",
            reason="Parent skill looks fine.",
        )
        mock_v_client.set_behavior(
            "_nested_child_fails_verification",
            2.0,
            status="replan_parent",
            reason="Mocked strategic failure in nested child.",
        )
        active_task.verification_client = mock_v_client

        new_parent_code = textwrap.dedent(
            """
            async def parent_skill():
                print("Executing FIXED parent skill...")
                return "Fixed parent skill finished successfully."
        """,
        )
        active_task.implementation_client = MockImplementationClient(
            new_code=new_parent_code,
        )
        active_task.course_correction_client = mock_v_client
        active_task.summarization_client = mock_v_client

        _ = await asyncio.wait_for(active_task.result(), timeout=60)
        action_log_str = "\n".join(active_task.action_log)
        assert "parent_skill" in action_log_str
        assert "_nested_child_fails_verification" in action_log_str
        assert "TypeError: None is not a callable object" not in action_log_str
        assert "Could not add function" not in action_log_str

    finally:
        if active_task and not active_task.done():
            with contextlib.suppress(Exception):
                await active_task.stop()
        if actor:
            await actor.close()


# --- Skip-verify flag ---

FUNCTION_WITHOUT_VERIFY = textwrap.dedent(
    """
async def simple_navigation(url: str):
    '''
    Navigate to a URL without verification.
    This is a simple, low-risk action that doesn't need verification.
    '''
    print(f"NAVIGATING: Going to {url}")
    await computer_primitives.act(f"Navigate to {url}")
    return f"Navigated to {url}"
""",
)

FUNCTION_WITH_VERIFY = textwrap.dedent(
    """
async def complex_data_entry(field_name: str, value: str):
    '''
    Enter data into a form field with verification.
    This is a critical action that needs verification.
    '''
    print(f"DATA_ENTRY: Entering {value} into {field_name}")
    await computer_primitives.act(f"Enter {value} into the {field_name} field")
    return f"Entered {value} into {field_name}"
""",
)

CANNED_PLAN_WITH_FUNCTIONS_SKIP_VERIFY_FLAG = textwrap.dedent(
    """
async def simple_navigation(url: str):
    '''
    Navigate to a URL without verification.
    This is a simple, low-risk action that doesn't need verification.
    '''
    print(f"NAVIGATING: Going to {url}")
    await computer_primitives.act(f"Navigate to {url}")
    return f"Navigated to {url}"

async def complex_data_entry(field_name: str, value: str):
    '''
    Enter data into a form field with verification.
    This is a critical action that needs verification.
    '''
    print(f"DATA_ENTRY: Entering {value} into {field_name}")
    await computer_primitives.act(f"Enter {value} into the {field_name} field")
    return f"Entered {value} into {field_name}"

async def main_plan():
    '''Execute navigation and data entry.'''
    await simple_navigation("https://example.com")
    await complex_data_entry("username", "test_value")
    return "Plan completed with both navigation and data entry."
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_functions_with_skip_verify_flag_bypass_verification():
    actor = None
    active_task = None
    try:
        fm = FunctionManager()
        fm.clear()

        fm.add_functions(
            implementations=[FUNCTION_WITHOUT_VERIFY],
            verify={"simple_navigation": False},
        )
        fm.add_functions(
            implementations=[FUNCTION_WITH_VERIFY],
            verify={"complex_data_entry": True},
        )

        actor = HierarchicalActor(
            function_manager=fm,
            headless=True,
            computer_mode="mock",
            connect_now=False,
        )
        actor.computer_primitives._computer = MockComputerBackend(
            url="https://mock-url.com",
            screenshot=VALID_MOCK_SCREENSHOT_PNG,
        )
        actor.computer_primitives.act = AsyncMock(return_value="Mock action completed.")
        actor.computer_primitives.navigate = AsyncMock(return_value=None)

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Use simple_navigation then complex_data_entry.",
            persist=False,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()
        active_task.functions_skip_verify.add("simple_navigation")
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_WITH_FUNCTIONS_SKIP_VERIFY_FLAG,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(active_task, "main_plan", timeout=30)

        if not active_task.done():
            await active_task.stop("Test complete")

        assert "simple_navigation" in active_task.functions_skip_verify
        assert "complex_data_entry" not in active_task.functions_skip_verify
        final_plan_code = active_task.plan_source_code or ""
        assert "simple_navigation" in final_plan_code
        assert "complex_data_entry" in final_plan_code

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            await actor.close()
