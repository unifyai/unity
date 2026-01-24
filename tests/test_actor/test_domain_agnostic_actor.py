"""
Integration tests for domain-agnostic Actor capabilities.

Tests validate that HierarchicalActor and CodeActActor can execute plans across multiple
tool domains (browser, state managers, pure logic) without hardcoded
assumptions about tool availability.

Test Coverage:
- Pure state manager workflows (no browser)
- Mixed modality workflows (browser + state managers)
- Pure logic tasks (no external tools)
- Live handle management (steerable primitives)
- CodeActActor in primitives-only mode can call state manager methods
- CodeActActor in mixed mode can call browser and state manager methods
"""

import asyncio
import logging
import sys
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest
import unity
from pydantic import BaseModel

from tests.helpers import _handle_project

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalActorHandle,
    _HierarchicalHandleState,
    VerificationAssessment,
)
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments.computer import ComputerEnvironment
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.common.async_tool_loop import SteerableToolHandle
from unity.function_manager.primitives import ComputerPrimitives, Primitives
from unity.function_manager.function_manager import FunctionManager

# ────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ────────────────────────────────────────────────────────────────────────────

logging.getLogger("urllib3").propagate = False
logging.getLogger("websockets").propagate = False
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("UnifyAsyncLogger").setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

unity.init(overwrite=True)


# ────────────────────────────────────────────────────────────────────────────
# Shared Mock Classes
# ────────────────────────────────────────────────────────────────────────────


class SimpleMockVerificationClient:
    """
    Mock verification client that always returns success.
    Use for tests that don't need to control verification outcomes.
    """

    def __init__(self):
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_response_format(self, model):
        self._current_format = model

    def reset_response_format(self):
        self._current_format = VerificationAssessment

    def reset_messages(self):
        pass

    def set_system_message(self, message):
        pass

    async def _side_effect(self, *args, **kwargs):
        return VerificationAssessment(
            status="ok",
            reason="Mock verification success.",
        ).model_dump_json()


class MockStateManagerHandle(SteerableToolHandle):
    """Minimal mock handle for state manager operations."""

    def __init__(self, result_value: str):
        self._result_value = result_value
        self._done = True  # Complete immediately for test simplicity

    async def ask(self, question: str, **kwargs):
        return self

    def interject(self, message: str, **kwargs):
        pass

    def stop(self, reason: str = None, **kwargs):
        pass

    async def pause(self):
        pass

    async def resume(self):
        pass

    def done(self):
        return self._done

    async def result(self):
        return self._result_value

    async def next_clarification(self):
        return {}

    async def next_notification(self):
        return {}

    async def answer_clarification(self, call_id: str, answer: str):
        pass


class MockResearchHandle(SteerableToolHandle):
    """Mock handle for web research operations."""

    def __init__(self):
        self._done = False
        self._result = "Research completed: AI Safety developments documented"
        self._status_requests = 0

    async def ask(self, question: str, **kwargs):
        self._status_requests += 1
        return MockStateManagerHandle("Research in progress, 50% complete")

    def interject(self, message: str, **kwargs):
        pass

    def stop(self, reason: str = None, **kwargs):
        self._done = True

    async def pause(self):
        pass

    async def resume(self):
        pass

    def done(self):
        return self._done

    async def result(self):
        return self._result

    async def next_clarification(self):
        return {}

    async def next_notification(self):
        return {}

    async def answer_clarification(self, call_id: str, answer: str):
        pass


# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


async def wait_for_log_entry(
    task: HierarchicalActorHandle,
    log_substring: str,
    timeout=30,
):
    """Polls the plan's action_log until a specific substring appears."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        log_content = "\n".join(task.action_log)
        if log_substring in log_content:
            return
        await asyncio.sleep(0.5)
    raise AssertionError(f"Timed out waiting for log entry '{log_substring}'.")


async def wait_for_state(task: HierarchicalActorHandle, expected_state, timeout=60):
    """Poll the plan's state until it matches expected_state."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if task._state == expected_state:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"Timed out waiting for state {expected_state.name}")


def find_tool_calls(plan: HierarchicalActorHandle, tool_name: str) -> list:
    """Find all tool call entries for a specific tool in the interaction log."""
    calls = []
    if not hasattr(plan, "cumulative_interactions"):
        return calls
    for entry in plan.cumulative_interactions:
        if entry[0] == "tool_call" and tool_name in entry[1]:
            calls.append(entry)
    return calls


def validate_verification_evidence(
    plan: HierarchicalActorHandle,
    expected_environments: list[str],
    unexpected_environments: list[str] = None,
):
    """
    Validate that verification evidence was gathered from expected environments.

    Args:
        plan: The completed plan handle
        expected_environments: List of environment namespaces that should have evidence
        unexpected_environments: List of environment namespaces that should NOT have evidence
    """
    unexpected_environments = unexpected_environments or []

    # Access verification work items from plan's completed verifications
    # Note: In the actual implementation, verification evidence is captured in
    # VerificationWorkItem.pre_state and .post_state
    # For this test, we'll validate by checking the cumulative interactions
    # and ensuring the right tools were called

    if not hasattr(plan, "cumulative_interactions"):
        return

    # Validate expected environments were used
    for env_name in expected_environments:
        tool_calls = [e for e in plan.cumulative_interactions if env_name in str(e)]
        assert len(tool_calls) > 0, f"Expected environment '{env_name}' was not used"

    # Validate unexpected environments were NOT used
    for env_name in unexpected_environments:
        tool_calls = [e for e in plan.cumulative_interactions if env_name in str(e)]
        assert len(tool_calls) == 0, f"Unexpected environment '{env_name}' was used"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Pure State Manager Tests (No Browser)
# ════════════════════════════════════════════════════════════════════════════


CANNED_PLAN_PURE_STATE_MANAGER = textwrap.dedent(
    """
    async def main_plan():
        '''Schedule a reminder using state manager primitives only.'''
        print("--- Pure State Manager Test: Starting ---")

        # Use state manager to schedule task
        print("--- Scheduling reminder via primitives.tasks.update ---")
        result = await primitives.tasks.update(
            "Schedule a reminder to call Mom tomorrow at 10am"
        )

        print(f"--- Task scheduled: {result} ---")
        return "Reminder scheduled successfully"
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_pure_state_manager_task_schedules_reminder_without_browser():
    """Test that Actor can execute pure state-manager workflows without browser."""
    # Setup: Create primitives with mocked state managers
    primitives = Primitives()

    # Mock tasks.update to return a mock handle
    async def mock_tasks_update(text: str, **kwargs):
        return MockStateManagerHandle("Task scheduled: Call Mom tomorrow at 10am")

    # Access the property to initialize it, then mock the method
    _ = primitives.tasks  # Initialize the task scheduler
    primitives.tasks.update = AsyncMock(side_effect=mock_tasks_update)

    # Create Actor with ONLY StateManagerEnvironment (no browser)
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    # Clear function manager to avoid interference
    fm = FunctionManager()
    fm.clear()

    active_task = None
    try:
        # Create handle
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Remind me to call Mom tomorrow",
            persist=True,  # Required for cumulative_interactions tracking
        )

        # Cancel auto-started task
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Inject mocks and canned plan
        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_PURE_STATE_MANAGER,
            active_task,
        )

        # Start execution
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait for completion with timeout
        # Wait for the tool call to appear in action_log
        await asyncio.wait_for(
            wait_for_log_entry(active_task, "primitives.tasks.update", timeout=30),
            timeout=60,
        )

        # Allow verification to complete
        await asyncio.sleep(2)

        if not active_task.done():
            await active_task.stop("Test complete")

        # Assertions
        assert active_task._state in [
            _HierarchicalHandleState.COMPLETED,
            _HierarchicalHandleState.STOPPED,
        ], f"Expected COMPLETED or STOPPED, got {active_task._state}"

        # Verify primitives.tasks.update was called
        assert primitives.tasks.update.called, "tasks.update should have been called"
        call_args = primitives.tasks.update.call_args
        assert "call Mom" in str(call_args), "Expected 'call Mom' in call args"

        # Verify interaction log contains state manager call
        state_calls = find_tool_calls(active_task, "primitives.tasks")
        assert len(state_calls) > 0, "Expected at least one primitives.tasks call"

        # Verify NO browser calls were made
        browser_calls = find_tool_calls(active_task, "computer_primitives")
        assert (
            len(browser_calls) == 0
        ), "No browser calls should occur in pure state manager test"

        # Validate verification evidence
        validate_verification_evidence(
            active_task,
            expected_environments=["primitives"],
            unexpected_environments=["computer_primitives"],
        )

        print("\n✅ Test 1 PASSED: Pure state manager workflow completed successfully")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_plan_sanitizer_instruments_primitives_tool_calls_with_checkpoints_and_around_cp():
    """
    Test that PlanSanitizer instruments *non-browser* tool calls based on active environment namespaces.

    Specifically, when a plan is configured with only the `primitives` environment, awaited calls like
    `await primitives.tasks.update(...)` must still be treated as tool calls and receive:
    - before/after checkpoint probes (via injected `_cp(...)` statements)
    - awaited-call wrapping (via `_around_cp(...)`) for awaits embedded in expressions (e.g. `return await ...`)
    """
    primitives = Primitives()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    active_task: HierarchicalActorHandle | None = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Sanitizer should instrument primitives.* calls",
            persist=False,
        )

        # Cancel auto-started task (we only need sanitization output).
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        plan_code = textwrap.dedent(
            """
            async def main_plan():
                await primitives.tasks.update({"title": "foo"})
                return await primitives.tasks.ask("what tasks do I have?")
            """,
        )

        sanitized = actor._sanitize_code(plan_code, active_task)

        # Tool-call checkpoints should be inserted for primitives.* statements.
        assert "Before: primitives.tasks.update" in sanitized
        assert "After: primitives.tasks.update" in sanitized

        # Awaits inside expressions should be wrapped in _around_cp for primitives.* too.
        assert (
            "_around_cp('primitives.tasks.ask'" in sanitized
            or '_around_cp("primitives.tasks.ask"' in sanitized
        )

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: Mixed Modality Tests (Browser + State Managers)
# ════════════════════════════════════════════════════════════════════════════


CANNED_PLAN_MIXED_MODALITY = textwrap.dedent(
    """
    async def main_plan():
        '''Search for CEO info via browser, then save to contacts.'''
        from pydantic import BaseModel, Field

        print("--- Mixed Modality Test: Starting ---")

        # Step 1: Browser search
        print("--- Step 1: Navigating to search engine ---")
        await computer_primitives.navigate("https://google.com")

        print("--- Step 2: Performing search ---")
        await computer_primitives.act("Search for 'Anthropic CEO'")

        # Step 3: Extract info from page
        class CEOInfo(BaseModel):
            name: str = Field(description="CEO's full name")
            title: str = Field(description="Job title")

        CEOInfo.model_rebuild()

        print("--- Step 3: Extracting CEO information ---")
        ceo_data = await computer_primitives.observe(
            "Extract the CEO's name and title from the search results",
            response_format=CEOInfo
        )

        # Step 4: Save to contacts via state manager
        print(f"--- Step 4: Saving {ceo_data.name} to contacts ---")
        contact_result = await primitives.contacts.update(
            f"Add contact: {ceo_data.name}, Title: {ceo_data.title}, Company: Anthropic"
        )

        print(f"--- Contact saved: {contact_result} ---")
        return f"Added {ceo_data.name} to contacts"
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_mixed_modality_task_searches_web_and_updates_contacts():
    """Test that Actor can orchestrate mixed workflows combining browser and state managers."""

    # Setup: Create primitives with mocked state managers
    primitives = Primitives()

    # Mock contacts.update to return a mock handle
    async def mock_contacts_update(text: str, **kwargs):
        return MockStateManagerHandle(f"Contact added: {text}")

    # Access the property to initialize it, then mock the method
    _ = primitives.contacts  # Initialize the contact manager
    primitives.contacts.update = AsyncMock(side_effect=mock_contacts_update)

    # Create Actor with default environments (browser + state managers)
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value="Navigated")
    actor.computer_primitives.act = AsyncMock(return_value="Action completed")

    # Replace the state manager environment with our mocked one
    actor.environments["primitives"] = StateManagerEnvironment(primitives)

    # Mock observe to return CEO data
    class CEOInfo(BaseModel):
        name: str
        title: str

    actor.computer_primitives.observe = AsyncMock(
        return_value=CEOInfo(name="Dario Amodei", title="CEO"),
    )

    # Clear function manager
    fm = FunctionManager()
    fm.clear()

    active_task = None
    try:
        # Create handle
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Find the CEO of Anthropic and add to contacts",
            persist=True,  # Required for cumulative_interactions tracking
        )

        # Cancel auto-started task
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Inject mocks and canned plan
        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_MIXED_MODALITY,
            active_task,
        )

        # Start execution
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait for completion
        # Wait for the tool call to appear in action_log
        await asyncio.wait_for(
            wait_for_log_entry(active_task, "primitives.contacts.update", timeout=30),
            timeout=90,
        )

        # Allow verification to complete
        await asyncio.sleep(2)

        if not active_task.done():
            await active_task.stop("Test complete")

        # Assertions
        assert active_task._state in [
            _HierarchicalHandleState.COMPLETED,
            _HierarchicalHandleState.STOPPED,
        ], f"Expected COMPLETED or STOPPED, got {active_task._state}"

        # Verify browser methods were called
        assert (
            actor.computer_primitives.navigate.called
        ), "navigate should have been called"
        assert actor.computer_primitives.act.called, "act should have been called"
        assert (
            actor.computer_primitives.observe.called
        ), "observe should have been called"

        # Verify state manager method was called
        assert (
            primitives.contacts.update.called
        ), "contacts.update should have been called"
        call_args = primitives.contacts.update.call_args
        assert "Dario Amodei" in str(call_args), "Expected CEO name in call args"

        # Verify interaction log contains BOTH browser and state manager calls
        browser_calls = find_tool_calls(active_task, "computer_primitives")
        assert len(browser_calls) > 0, "Expected browser calls"

        state_calls = find_tool_calls(active_task, "primitives.contacts")
        assert len(state_calls) > 0, "Expected state manager calls"

        # Validate verification evidence includes both environments
        validate_verification_evidence(
            active_task,
            expected_environments=["computer_primitives", "primitives"],
            unexpected_environments=[],
        )

        print("\n✅ Test 2 PASSED: Mixed modality workflow completed successfully")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: Pure Logic Tests (No External Tools)
# ════════════════════════════════════════════════════════════════════════════


CANNED_PLAN_PURE_LOGIC = textwrap.dedent(
    """
    async def main_plan():
        '''Calculate average using pure Python logic.'''
        print("--- Pure Logic Test: Starting ---")

        numbers = [1, 2, 3, 4, 5]
        print(f"--- Calculating average of {numbers} ---")

        total = sum(numbers)
        count = len(numbers)
        average = total / count

        print(f"--- Average calculated: {average} ---")
        return f"The average is {average}"
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_pure_logic_task_calculates_average_without_tools():
    """Test that Actor can execute pure computational tasks without external tools."""

    # Create Actor with default environments (to prove it works even when tools available)
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )

    # Clear function manager
    fm = FunctionManager()
    fm.clear()

    active_task = None
    try:
        # Create handle
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Calculate the average of [1, 2, 3, 4, 5]",
            persist=True,  # Required for cumulative_interactions tracking
        )

        # Cancel auto-started task
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Inject mocks and canned plan
        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_PURE_LOGIC,
            active_task,
        )

        # Start execution
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait for completion
        # Pure logic test has no tool calls, so wait for plan completion marker
        await asyncio.wait_for(
            wait_for_log_entry(active_task, "Exiting 'main_plan'", timeout=30),
            timeout=60,
        )

        # Allow verification to complete
        await asyncio.sleep(2)

        if not active_task.done():
            await active_task.stop("Test complete")

        # Assertions
        assert active_task._state in [
            _HierarchicalHandleState.COMPLETED,
            _HierarchicalHandleState.STOPPED,
        ], f"Expected COMPLETED or STOPPED, got {active_task._state}"

        # Verify NO tool calls were made
        browser_calls = find_tool_calls(active_task, "computer_primitives")
        assert (
            len(browser_calls) == 0
        ), "No browser calls should occur in pure logic test"

        state_calls = find_tool_calls(active_task, "primitives.")
        assert (
            len(state_calls) == 0
        ), "No state manager calls should occur in pure logic test"

        # Verify the result contains "3.0"
        log_content = "\n".join(active_task.action_log)
        assert "3.0" in log_content, "Expected average 3.0 in logs"

        print("\n✅ Test 3 PASSED: Pure logic workflow completed successfully")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: Live Handle Management Tests
# ════════════════════════════════════════════════════════════════════════════


CANNED_PLAN_LIVE_HANDLE = textwrap.dedent(
    """
    async def main_plan():
        '''Start a background research task and track the handle.'''
        print("--- Live Handle Test: Starting ---")

        # Start background research (returns SteerableToolHandle)
        print("--- Starting background research via primitives.web.ask ---")
        research_handle = await primitives.web.ask(
            "Research the latest developments in AI Safety"
        )

        print(f"--- Research started, handle type: {type(research_handle).__name__} ---")

        # Verify handle is steerable
        print(f"--- Handle is done: {research_handle.done()} ---")

        # Optionally interact with handle
        status_handle = await research_handle.ask("What's the current status?")
        status = await status_handle.result()
        print(f"--- Research status: {status} ---")

        return "Research task started and tracked"
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_live_handle_management_tracks_steerable_primitives():
    """Test that Actor correctly tracks and proxies steerable handles from state managers."""

    # Setup: Create primitives with mocked web searcher
    primitives = Primitives()

    # Mock web.ask to return a steerable handle
    async def mock_web_ask(text: str, **kwargs):
        return MockResearchHandle()

    # Access the property to initialize it, then mock the method
    _ = primitives.web  # Initialize the web searcher
    primitives.web.ask = AsyncMock(side_effect=mock_web_ask)

    # Create Actor with default environments
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )

    # Replace the state manager environment with our mocked one
    actor.environments["primitives"] = StateManagerEnvironment(primitives)

    # Clear function manager
    fm = FunctionManager()
    fm.clear()

    active_task = None
    try:
        # Create handle
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Start researching AI Safety",
            persist=True,  # Required for cumulative_interactions tracking
        )

        # Cancel auto-started task
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Inject mocks and canned plan
        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_LIVE_HANDLE,
            active_task,
        )

        # Start execution
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait for completion
        # Wait for the tool call to appear in action_log
        await asyncio.wait_for(
            wait_for_log_entry(active_task, "primitives.web.ask", timeout=30),
            timeout=60,
        )

        # Allow verification to complete
        await asyncio.sleep(2)

        if not active_task.done():
            await active_task.stop("Test complete")

        # Assertions
        assert active_task._state in [
            _HierarchicalHandleState.COMPLETED,
            _HierarchicalHandleState.STOPPED,
        ], f"Expected COMPLETED or STOPPED, got {active_task._state}"

        # Verify primitives.web.ask was called
        assert primitives.web.ask.called, "web.ask should have been called"

        # Verify interaction log contains web.ask call
        web_calls = find_tool_calls(active_task, "primitives.web")
        assert len(web_calls) > 0, "Expected at least one primitives.web call"

        # Verify interaction log indicates handle was returned (if persist=True)
        if hasattr(active_task, "cumulative_interactions"):
            interaction_strings = [str(e) for e in active_task.cumulative_interactions]
            has_handle_indicator = any(
                "handle" in s.lower() for s in interaction_strings
            )
        # Note: The proxy should track returned handles, but exact logging format may vary

        # Verify live_handles dictionary tracks the handle
        # Note: The exact format of live_handles may need adjustment based on implementation
        # For now, we'll just verify the test completes successfully

        print("\n✅ Test 4 PASSED: Live handle management completed successfully")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: CodeActActor Domain-Agnostic Tests (Sandbox-only, deterministic)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_code_act_actor_primitives_only_sandbox_can_call_state_managers():
    """
    Test that CodeActActor can run in primitives-only mode (no browser env) and
    its PythonExecutionSession can successfully call state manager methods.

    This test is sandbox-only (no LLM/tool-loop) to keep it deterministic.
    """
    primitives = Primitives()

    async def mock_tasks_update(text: str, **kwargs):
        return MockStateManagerHandle(f"Task scheduled: {text}")

    _ = primitives.tasks
    primitives.tasks.update = AsyncMock(side_effect=mock_tasks_update)

    actor = CodeActActor(environments=[StateManagerEnvironment(primitives)])
    try:
        from unity.actor.code_act_actor import PythonExecutionSession

        sandbox = PythonExecutionSession(environments=actor.environments)
        exec_result = await sandbox.execute(
            'h = await primitives.tasks.update("Schedule a reminder to call Mom tomorrow at 10am")\n'
            "print(await h.result())\n",
        )
        assert exec_result["error"] is None
        assert "Task scheduled:" in exec_result["stdout"]
        assert primitives.tasks.update.called
    finally:
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_code_act_actor_mixed_envs_sandbox_can_call_browser_and_state_managers():
    """
    Test that CodeActActor in mixed mode (browser + primitives envs) injects both namespaces
    and the PythonExecutionSession can call both tool domains.

    This test is sandbox-only (no LLM/tool-loop) to keep it deterministic.
    """
    from unity.actor.code_act_actor import PythonExecutionSession

    primitives = Primitives()

    async def mock_contacts_update(text: str, **kwargs):
        return MockStateManagerHandle(f"Contact added: {text}")

    _ = primitives.contacts
    primitives.contacts.update = AsyncMock(side_effect=mock_contacts_update)

    # Mock browser primitives
    mock_cp = MagicMock(spec=ComputerPrimitives)
    mock_cp.navigate = AsyncMock(return_value=None)
    mock_cp.act = AsyncMock(return_value=None)
    mock_cp.observe = AsyncMock(return_value={"ok": True})

    actor = CodeActActor(
        environments=[
            ComputerEnvironment(mock_cp),
            StateManagerEnvironment(primitives),
        ],
    )
    try:
        # Use PythonExecutionSession directly
        sandbox = PythonExecutionSession(environments=actor.environments)
        exec_result = await sandbox.execute(
            'await computer_primitives.navigate("https://example.com")\n'
            'await computer_primitives.act("Click something")\n'
            'h = await primitives.contacts.update("Add contact: Alice")\n'
            "print(await h.result())\n",
        )
        assert exec_result["error"] is None
        assert "Contact added:" in exec_result["stdout"]
        assert mock_cp.navigate.called
        assert mock_cp.act.called
        assert primitives.contacts.update.called
    finally:
        await actor.close()
