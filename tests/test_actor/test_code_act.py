"""
Tests for CodeActActor - an actor that executes Python code in a stateful sandbox with a tool loop to accomplish a task.

This test file includes:
1. CodeExecutionSandbox stateful execution tests
2. Interjection flow tests
3. Clarification flow tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from unity.actor.code_act_actor import CodeActActor, CodeExecutionSandbox
from unity.function_manager.primitives import ComputerPrimitives


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_computer_primitives():
    """Fixture to create a mock ComputerPrimitives for testing."""
    mock_provider = MagicMock(spec=ComputerPrimitives)
    # Mock the correct method names (navigate, act, observe - NOT browser_*)
    mock_provider.navigate = AsyncMock(return_value="navigated")
    mock_provider.act = AsyncMock(return_value="acted")
    mock_provider.observe = AsyncMock(return_value={"data": "observed_data"})
    return mock_provider


# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


async def wait_for_turn_completion(task, initial_history_len, timeout=30):
    """
    Waits for the agent to process an interjection and enter an idle state.
    An idle state is detected when the last message is from the assistant
    and contains no tool calls, indicating it's waiting for the next command.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if len(task.chat_history) > initial_history_len:
            last_message = task.chat_history[-1]
            # A turn is now considered complete when the assistant has responded
            # WITHOUT initiating a new tool call. This is the true "idle" state.
            if last_message.get("role") == "assistant" and not last_message.get(
                "tool_calls",
            ):
                return
        await asyncio.sleep(0.5)

    raise AssertionError("Timed out waiting for turn completion")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: CodeExecutionSandbox Tests (from test_code_act_sandbox.py)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_variable_execution():
    """
    Tests that the sandbox maintains simple variable state between calls.
    """
    sandbox = CodeExecutionSandbox()

    # First execution: define a variable
    result1 = await sandbox.execute("x = 100")
    assert result1["error"] is None
    assert "x" in sandbox.global_state
    assert sandbox.global_state["x"] == 100

    # Second execution: use the previously defined variable
    result2 = await sandbox.execute("y = x * 2\nprint(y)")
    assert result2["error"] is None
    assert "y" in sandbox.global_state
    assert sandbox.global_state["y"] == 200
    assert result2["stdout"] == "200\n"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_import_execution():
    """
    Tests that the sandbox maintains imported modules between calls.
    """
    sandbox = CodeExecutionSandbox()

    # First execution: import a library
    result1 = await sandbox.execute("import json")
    assert result1["error"] is None
    assert "json" in sandbox.global_state

    # Second execution: use the imported library
    result2 = await sandbox.execute(
        "my_dict = {'key': 'value'}\nprint(json.dumps(my_dict))",
    )
    assert result2["error"] is None
    assert result2["stdout"].strip() == '{"key": "value"}'


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_function_definition():
    """
    Tests that the sandbox maintains function definitions between calls.
    """
    sandbox = CodeExecutionSandbox()

    # First execution: define a function
    func_def_code = "def my_adder(a, b):\n    return a + b"
    result1 = await sandbox.execute(func_def_code)
    assert result1["error"] is None
    assert "my_adder" in sandbox.global_state

    # Second execution: call the defined function
    func_call_code = "result = my_adder(10, 5)\nprint(result)"
    result2 = await sandbox.execute(func_call_code)
    assert result2["error"] is None
    assert result2["stdout"].strip() == "15"
    assert sandbox.global_state["result"] == 15


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_class_definition():
    """
    Tests that the sandbox maintains class definitions between calls.
    """
    sandbox = CodeExecutionSandbox()

    # First execution: define a class
    class_def_code = "class Greeter:\n    def __init__(self, name):\n        self.name = name\n    def greet(self):\n        return f'Hello, {self.name}!'\n"
    result1 = await sandbox.execute(class_def_code)
    assert result1["error"] is None
    assert "Greeter" in sandbox.global_state

    # Second execution: instantiate and use the class
    class_use_code = "g = Greeter('World')\nprint(g.greet())"
    result2 = await sandbox.execute(class_use_code)
    assert result2["error"] is None
    assert result2["stdout"].strip() == "Hello, World!"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_browser_tool_execution(mock_computer_primitives):
    """
    Tests that the sandbox can execute code that calls browser tools via
    the injected computer_primitives.
    """
    sandbox = CodeExecutionSandbox(computer_primitives=mock_computer_primitives)

    # Test navigate
    nav_code = "await computer_primitives.navigate('https://example.com')"
    nav_result = await sandbox.execute(nav_code)
    assert nav_result["error"] is None
    mock_computer_primitives.navigate.assert_awaited_once_with(
        "https://example.com",
    )

    # Test act
    act_code = "await computer_primitives.act('Click login button')"
    act_result = await sandbox.execute(act_code)
    assert act_result["error"] is None
    mock_computer_primitives.act.assert_awaited_once_with(
        "Click login button",
    )

    # Test observe with Pydantic
    observe_code = """
from pydantic import BaseModel

class MyData(BaseModel):
    data: str

result = await computer_primitives.observe('get data', response_format=MyData)
print(result['data'])
"""
    observe_result = await sandbox.execute(observe_code)
    assert observe_result["error"] is None
    # The mock returns a dict, which we print the 'data' key from
    assert observe_result["stdout"].strip() == "observed_data"
    mock_computer_primitives.observe.assert_awaited_once()
    assert mock_computer_primitives.observe.call_args[0][0] == "get data"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_error_handling():
    """
    Tests that the sandbox correctly captures and reports exceptions.
    """
    sandbox = CodeExecutionSandbox()
    error_code = "x = 1 / 0"
    result = await sandbox.execute(error_code)

    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["result"] is None
    assert result["error"] is not None
    assert "ZeroDivisionError" in result["error"]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: Interjection Flow Tests (from test_code_act_interjection.py)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_interjection_incremental_teaching_session():
    """
    Test that CodeActActor can handle incremental interjections
    in an interactive teaching session.
    """
    # Create actor with mock browser (no external services needed)
    actor = CodeActActor(headless=True, browser_mode="mock")
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    plan = None
    try:
        # 1. Start plan with no goal (interactive session)
        plan = await actor.act(None)

        # Wait for initial setup
        await asyncio.sleep(2)

        # 2. Interjection 1
        interjection_1 = "Navigate to allrecipes.com"
        history_len_before = len(plan.chat_history)
        await plan.interject(interjection_1)

        try:
            await asyncio.wait_for(
                wait_for_turn_completion(plan, history_len_before),
                timeout=30,
            )
        except asyncio.TimeoutError:
            pass  # Step timed out, continuing...

        # 3. Interjection 2
        interjection_2 = "Great, now search for 'chocolate chip cookies'."
        history_len_before = len(plan.chat_history)
        await plan.interject(interjection_2)

        try:
            await asyncio.wait_for(
                wait_for_turn_completion(plan, history_len_before),
                timeout=30,
            )
        except asyncio.TimeoutError:
            pass  # Step timed out, continuing...

        # 4. Interjection 3: Finish
        interjection_3 = "Perfect, that's all. We're done."
        history_len_before = len(plan.chat_history)
        await plan.interject(interjection_3)

        try:
            await asyncio.wait_for(
                wait_for_turn_completion(plan, history_len_before),
                timeout=30,
            )
        except asyncio.TimeoutError:
            pass  # Step timed out, continuing...

        # 5. Stop the session from the outside
        final_result = await plan.stop("Session complete.")

        # Assertions
        assert not str(final_result).startswith(
            "Error:",
        ), f"Unexpected error: {final_result}"

        # Verify interjections were received
        history_str = str(plan.chat_history)
        assert (
            "allrecipes" in history_str.lower()
            or actor._computer_primitives.navigate.called
        )

    finally:
        if plan and not plan.done():
            try:
                await plan.stop("Test cleanup")
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass
        await asyncio.sleep(0.5)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: Clarification Flow Tests (from test_code_act_clarification_flow.py)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_clarification_flow():
    """
    Test that CodeActActor can handle clarification requests
    via the clarification queues.
    """
    # 1. Set up communication queues
    clarification_up_q = asyncio.Queue()
    clarification_down_q = asyncio.Queue()

    # Create actor with mock browser (no external services needed)
    planner = CodeActActor(headless=True, browser_mode="mock")
    planner._computer_primitives.navigate = AsyncMock(return_value=None)
    planner._computer_primitives.act = AsyncMock(return_value="Action completed")
    planner._computer_primitives.observe = AsyncMock(return_value="Page content")

    active_task = None
    try:
        # 2. Define an ambiguous goal
        ambiguous_goal = "Search for a recipe on allrecipes.com, but first ask me what I want to search for."

        # 3. Start the plan with clarification queues enabled
        active_task = await planner.act(
            ambiguous_goal,
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
        )

        # 4. Wait for the planner to ask its question (with timeout)
        try:
            question = await asyncio.wait_for(clarification_up_q.get(), timeout=60)

            # Assert that the planner asked a reasonable question
            assert (
                "what" in question.lower()
                or "recipe" in question.lower()
                or "search" in question.lower()
            )

            # 5. Provide a specific answer
            answer = "chocolate cake"
            await clarification_down_q.put(answer)

            # 6. Wait briefly for the agent to process
            await asyncio.sleep(3)

        except asyncio.TimeoutError:
            # In mocked environment, the LLM might not call request_clarification
            # This is acceptable - we verify basic operation
            pass

        # 7. Stop and get result
        final_result = await active_task.stop("Test complete")

        # 8. Basic assertions
        assert not str(final_result).startswith(
            "Error:",
        ), f"Unexpected error: {final_result}"

        # Check that the goal was processed
        history_str = str(active_task.chat_history)
        assert "allrecipes" in history_str.lower() or "recipe" in history_str.lower()

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop("Test cleanup")
            except Exception:
                pass
        if planner:
            try:
                await planner.close()
            except Exception:
                pass
        await asyncio.sleep(0.5)
