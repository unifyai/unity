"""
Tests for the can_compose and can_store flags on HierarchicalActor and HierarchicalActorHandle.

These flags control:
- can_compose: Whether the actor can generate new code on the fly (plan generation,
  dynamic implementation, verification failure recovery)
- can_store: Whether verified functions are persisted to the FunctionManager
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalActorHandle,
    VerificationAssessment,
    _HierarchicalHandleState,
)
from unity.function_manager.primitives import ComputerPrimitives
from unity.function_manager.browser import Browser
from unity.function_manager.function_manager import FunctionManager
import unity.actor.hierarchical_actor as hierarchical_actor_module


# --- Fixtures ---


@pytest.fixture
def mock_function_manager():
    """Provides a mock FunctionManager instance."""
    fm = MagicMock(spec=FunctionManager)
    fm.list_functions = MagicMock(return_value={})
    fm.add_functions = MagicMock()
    fm.search_functions_by_similarity = MagicMock(return_value=[])
    fm.search_functions = MagicMock(return_value=[])
    return fm


@pytest.fixture
def mock_browser():
    """Provides a mock Browser instance with async methods."""
    browser = MagicMock(spec=Browser)
    browser.act = AsyncMock(return_value="Action completed.")
    browser.observe = AsyncMock(return_value="Observation complete.")
    browser.get_screenshot = AsyncMock(return_value=None)
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_computer_primitives(mock_browser):
    """Provides a mock ComputerPrimitives which holds our mock_browser."""
    provider = MagicMock(spec=ComputerPrimitives)
    provider.browser = mock_browser
    provider.browser_act = mock_browser.act
    provider.browser_observe = mock_browser.observe
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def actor(mock_function_manager, mock_computer_primitives, monkeypatch):
    """Provides a HierarchicalActor with default can_compose=True, can_store=True."""
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ComputerPrimitives",
        lambda *args, **kwargs: mock_computer_primitives,
    )
    return HierarchicalActor(function_manager=mock_function_manager, headless=True)


@pytest.fixture
def actor_no_compose(mock_function_manager, mock_computer_primitives, monkeypatch):
    """Provides a HierarchicalActor with can_compose=False."""
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ComputerPrimitives",
        lambda *args, **kwargs: mock_computer_primitives,
    )
    return HierarchicalActor(
        function_manager=mock_function_manager,
        headless=True,
        can_compose=False,
    )


@pytest.fixture
def actor_no_store(mock_function_manager, mock_computer_primitives, monkeypatch):
    """Provides a HierarchicalActor with can_store=False."""
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ComputerPrimitives",
        lambda *args, **kwargs: mock_computer_primitives,
    )
    return HierarchicalActor(
        function_manager=mock_function_manager,
        headless=True,
        can_store=False,
    )


# --- Constructor Tests ---


def test_actor_default_can_compose_and_store(actor):
    """Test that default constructor has can_compose=True and can_store=True."""
    assert actor.can_compose is True
    assert actor.can_store is True


def test_actor_can_compose_false(actor_no_compose):
    """Test that can_compose=False is preserved on the actor."""
    assert actor_no_compose.can_compose is False
    assert actor_no_compose.can_store is True


def test_actor_can_store_false(actor_no_store):
    """Test that can_store=False is preserved on the actor."""
    assert actor_no_store.can_compose is True
    assert actor_no_store.can_store is False


# --- Semantic Search When can_compose=False ---


@pytest.mark.asyncio
async def test_semantic_search_when_can_compose_false_no_entrypoint(
    mock_function_manager,
    mock_computer_primitives,
    monkeypatch,
):
    """
    Test that when can_compose=False and no entrypoint is provided, the actor
    uses semantic search to find the best matching function.
    """
    # Set up a function that matches the goal
    matching_code = '''
async def send_email_task():
    """Sends an email to someone."""
    return "Email sent"
'''
    mock_function_manager.search_functions_by_similarity.return_value = [
        {
            "function_id": 42,
            "name": "send_email_task",
            "implementation": matching_code,
            "verify": False,
            "calls": [],
        },
    ]
    mock_function_manager.search_functions.return_value = [
        {
            "function_id": 42,
            "name": "send_email_task",
            "implementation": matching_code,
            "verify": False,
            "calls": [],
        },
    ]
    mock_function_manager.list_functions.return_value = {
        "send_email_task": {
            "function_id": 42,
            "name": "send_email_task",
            "implementation": matching_code,
            "calls": [],
            "verify": False,
        },
    }

    # Create actor with can_compose=False
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ComputerPrimitives",
        lambda *args, **kwargs: mock_computer_primitives,
    )
    actor = HierarchicalActor(
        function_manager=mock_function_manager,
        headless=True,
        can_compose=False,
    )

    # Ensure _generate_initial_plan is NOT called
    generate_plan_mock = AsyncMock(side_effect=ValueError("Should not be called"))
    monkeypatch.setattr(actor, "_generate_initial_plan", generate_plan_mock)

    # Act without providing an entrypoint - should use semantic search
    plan = await actor.act(
        "Send an email to John",  # Goal that should match send_email_task
        persist=False,
    )

    # Wait for initialization
    await asyncio.sleep(0.1)

    # Verify semantic search was called with the goal
    mock_function_manager.search_functions_by_similarity.assert_called_once()
    call_args = mock_function_manager.search_functions_by_similarity.call_args
    assert call_args.kwargs["query"] == "Send an email to John"

    # Verify plan generation was NOT called
    generate_plan_mock.assert_not_called()

    # Verify the action log shows semantic search was used
    assert any("semantic search" in log.lower() for log in plan.action_log)
    assert any("send_email_task" in log for log in plan.action_log)

    # Verify the plan source contains the selected function
    assert plan.plan_source_code is not None
    assert "send_email_task" in plan.plan_source_code


@pytest.mark.asyncio
async def test_semantic_search_fails_when_no_functions_match(
    mock_function_manager,
    mock_computer_primitives,
    monkeypatch,
):
    """
    Test that when can_compose=False and semantic search finds no matching
    functions, an appropriate error is raised.
    """
    # Return empty results from semantic search
    mock_function_manager.search_functions_by_similarity.return_value = []

    # Create actor with can_compose=False
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ComputerPrimitives",
        lambda *args, **kwargs: mock_computer_primitives,
    )
    actor = HierarchicalActor(
        function_manager=mock_function_manager,
        headless=True,
        can_compose=False,
    )

    # Act without entrypoint - should fail since no functions match
    plan = await actor.act(
        "Do something completely unique",
        persist=False,
    )

    # Should fail with error about no matching functions
    with pytest.raises(RuntimeError):
        await plan.result()

    assert plan._state == _HierarchicalHandleState.ERROR
    # Check that the error message mentions no functions found
    assert any(
        "no functions found" in log.lower() for log in plan.action_log
    ) or "No functions found" in (plan._final_result_str or "")


@pytest.mark.asyncio
async def test_entrypoint_works_when_can_compose_false(
    mock_function_manager,
    mock_computer_primitives,
    monkeypatch,
):
    """
    Test that using an entrypoint (pre-existing function) works when can_compose=False.
    This test verifies that the entrypoint path does not require code generation.
    """
    # Set up a pre-existing function in the FunctionManager
    entrypoint_code = '''
async def my_task():
    """A simple pre-existing task."""
    return "Task completed"
'''
    mock_function_manager.search_functions.return_value = [
        {
            "function_id": 1,
            "name": "my_task",
            "implementation": entrypoint_code,
            "verify": False,  # Skip verification to avoid needing browser
        },
    ]
    mock_function_manager.list_functions.return_value = {
        "my_task": {
            "function_id": 1,
            "name": "my_task",
            "implementation": entrypoint_code,
            "calls": [],
            "verify": False,
        },
    }

    # Create actor with can_compose=False
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ComputerPrimitives",
        lambda *args, **kwargs: mock_computer_primitives,
    )
    actor = HierarchicalActor(
        function_manager=mock_function_manager,
        headless=True,
        can_compose=False,
    )

    # Verify that _generate_initial_plan is NOT called when using entrypoint
    generate_plan_mock = AsyncMock(side_effect=ValueError("Should not be called"))
    monkeypatch.setattr(actor, "_generate_initial_plan", generate_plan_mock)

    plan = await actor.act(
        "Execute my task",
        entrypoint=1,
        persist=False,
    )

    # Verify the handle has can_compose=False
    assert plan.can_compose is False

    # Verify that plan generation was bypassed (entrypoint path was used)
    generate_plan_mock.assert_not_called()

    # Wait a short time for the plan to initialize enough to populate action_log
    await asyncio.sleep(0.1)

    # Verify the entrypoint was used (check action_log OR plan_source_code)
    # The plan source code should contain my_task since it was injected from entrypoint
    assert plan.plan_source_code is not None
    assert "my_task" in plan.plan_source_code


# --- can_compose Override on act() ---


@pytest.mark.asyncio
async def test_act_can_override_can_compose_to_false(actor):
    """Test that act(can_compose=False) overrides the actor's default True."""
    # Default actor has can_compose=True
    assert actor.can_compose is True

    plan = await actor.act("Do something", can_compose=False, persist=False)

    # The handle should have can_compose=False
    assert plan.can_compose is False

    # And it should fail to generate a plan
    with pytest.raises(RuntimeError):
        await plan.result()

    assert plan._state == _HierarchicalHandleState.ERROR


@pytest.mark.asyncio
async def test_act_can_override_can_compose_to_true(
    actor_no_compose,
    monkeypatch,
):
    """Test that act(can_compose=True) overrides the actor's default False."""
    assert actor_no_compose.can_compose is False

    # Mock plan generation to succeed
    simple_plan = '''
async def main_plan():
    """Simple plan."""
    return "Done"
'''
    monkeypatch.setattr(
        actor_no_compose,
        "_generate_initial_plan",
        AsyncMock(return_value=simple_plan),
    )
    monkeypatch.setattr(
        actor_no_compose,
        "_check_state_against_goal",
        AsyncMock(return_value=VerificationAssessment(status="ok", reason="OK")),
    )

    plan = await actor_no_compose.act(
        "Do something",
        can_compose=True,  # Override
        persist=False,
    )

    # The handle should have can_compose=True
    assert plan.can_compose is True


# --- can_store Override on act() ---


@pytest.mark.asyncio
async def test_act_can_override_can_store_to_false(actor, monkeypatch):
    """Test that act(can_store=False) overrides the actor's default True."""
    assert actor.can_store is True

    simple_plan = '''
async def main_plan():
    """Simple plan."""
    return "Done"
'''
    monkeypatch.setattr(
        actor,
        "_generate_initial_plan",
        AsyncMock(return_value=simple_plan),
    )

    plan = await actor.act("Do something", can_store=False, persist=False)

    # The handle should have can_store=False
    assert plan.can_store is False


@pytest.mark.asyncio
async def test_act_can_override_can_store_to_true(actor_no_store, monkeypatch):
    """Test that act(can_store=True) overrides the actor's default False."""
    assert actor_no_store.can_store is False

    simple_plan = '''
async def main_plan():
    """Simple plan."""
    return "Done"
'''
    monkeypatch.setattr(
        actor_no_store,
        "_generate_initial_plan",
        AsyncMock(return_value=simple_plan),
    )

    plan = await actor_no_store.act(
        "Do something",
        can_store=True,  # Override
        persist=False,
    )

    # The handle should have can_store=True
    assert plan.can_store is True


# --- Dynamic Implementation Blocked When can_compose=False ---


@pytest.mark.asyncio
async def test_dynamic_implementation_blocked_when_can_compose_false(
    actor,
    mock_function_manager,
    monkeypatch,
):
    """
    Test that dynamic implementation (stub functions) fails when can_compose=False.
    This test uses an entrypoint with a stub to test dynamic implementation blocking.
    """
    # Function with a stub that would need dynamic implementation
    entrypoint_code = '''
async def my_entrypoint():
    """Entry function that calls a stub."""
    raise NotImplementedError("This is a stub that needs implementation")
'''

    mock_function_manager.search_functions.return_value = [
        {
            "function_id": 1,
            "name": "my_entrypoint",
            "implementation": entrypoint_code,
            "verify": False,
            "calls": [],
        },
    ]
    mock_function_manager.list_functions.return_value = {
        "my_entrypoint": {
            "function_id": 1,
            "name": "my_entrypoint",
            "implementation": entrypoint_code,
            "calls": [],
            "verify": False,
        },
    }

    # Use can_compose=False via act() with entrypoint
    plan = await actor.act(
        "Do something",
        entrypoint=1,
        can_compose=False,
        persist=False,
    )

    # Verify handle has can_compose=False
    assert plan.can_compose is False

    # The plan will execute but since the function raises NotImplementedError,
    # it will try to call _handle_dynamic_implementation which should raise
    # RuntimeError due to can_compose=False
    with pytest.raises(RuntimeError):
        await plan.result()

    assert plan._state == _HierarchicalHandleState.ERROR


# --- Verification Failure Blocked When can_compose=False ---


@pytest.mark.asyncio
async def test_verification_failure_raises_when_can_compose_false(
    actor,
    mock_function_manager,
    monkeypatch,
):
    """
    Test that verification failures raise RuntimeError when can_compose=False
    (since recovery requires re-implementation).
    """
    # Set up a simple entrypoint function that will "pass" initially
    entrypoint_code = '''
async def simple_task():
    """A simple task that gets verified."""
    return "Done"
'''
    mock_function_manager.search_functions.return_value = [
        {
            "function_id": 1,
            "name": "simple_task",
            "implementation": entrypoint_code,
            "verify": True,
        },
    ]

    # Create plan with can_compose=False using entrypoint
    plan = await actor.act(
        "Run simple task",
        entrypoint=1,
        can_compose=False,
        persist=False,
    )

    # Create a mock VerificationWorkItem
    from unity.actor.hierarchical_actor import VerificationWorkItem

    mock_item = VerificationWorkItem(
        ordinal=1,
        function_name="simple_task",
        parent_stack=(),
        func_source="async def simple_task(): pass",
        docstring="A simple task",
        func_sig_str="()",
        pre_state={"computer_primitives": {"screenshot": None, "url": ""}},
        post_state={"computer_primitives": {"screenshot": None, "url": ""}},
        interactions=[],
        return_value_repr="None",
        cache_miss_counter=0,
        exit_seq=1,
    )

    failed_assessment = VerificationAssessment(
        status="reimplement_local",
        reason="Something went wrong",
    )

    # Call _on_verification_failure directly and expect RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        await plan._on_verification_failure(mock_item, failed_assessment)

    assert "can_compose=False" in str(exc_info.value)
    assert "Cannot recover" in str(exc_info.value)


# --- can_store=False Skips Function Persistence ---


@pytest.mark.asyncio
async def test_on_verification_success_skips_storage_when_can_store_false(
    actor,
    mock_function_manager,
):
    """
    Test that _on_verification_success does not call add_functions when can_store=False.
    """
    from unity.actor.hierarchical_actor import VerificationWorkItem

    # Create a plan with can_store=False
    simple_plan = '''
async def helper_func():
    """A helper function."""
    return "Helper done"

async def main_plan():
    """Main plan."""
    await helper_func()
    return "Done"
'''

    # We need to manually create a handle with can_store=False
    plan = HierarchicalActorHandle(
        actor=actor,
        goal="Test goal",
        can_compose=True,
        can_store=False,  # Key setting
    )

    # Set up state that would normally trigger storage
    plan.top_level_function_names = {"helper_func"}
    plan.clean_function_source_map = {
        "helper_func": 'async def helper_func():\n    """A helper."""\n    return "Done"',
    }

    mock_item = VerificationWorkItem(
        ordinal=1,
        function_name="helper_func",
        parent_stack=(),
        func_source='async def helper_func(): return "Done"',
        docstring="A helper function",
        func_sig_str="()",
        pre_state={"computer_primitives": {"screenshot": None, "url": ""}},
        post_state={"computer_primitives": {"screenshot": None, "url": ""}},
        interactions=[],
        return_value_repr='"Done"',
        cache_miss_counter=0,
        exit_seq=1,
    )

    success_assessment = VerificationAssessment(
        status="ok",
        reason="Success",
    )

    # Call _on_verification_success
    await plan._on_verification_success(mock_item, success_assessment)

    # add_functions should NOT have been called
    mock_function_manager.add_functions.assert_not_called()


@pytest.mark.asyncio
async def test_on_verification_success_stores_function_when_can_store_true(
    actor,
    mock_function_manager,
    monkeypatch,
):
    """
    Test that _on_verification_success does call add_functions when can_store=True.
    """
    from unity.actor.hierarchical_actor import (
        VerificationWorkItem,
        PreconditionDecision,
    )

    # Create a plan with can_store=True
    plan = HierarchicalActorHandle(
        actor=actor,
        goal="Test goal",
        can_compose=True,
        can_store=True,  # Key setting
    )

    # Set up state that would trigger storage
    plan.top_level_function_names = {"helper_func"}
    plan.clean_function_source_map = {
        "helper_func": 'async def helper_func():\n    """A helper."""\n    return "Done"',
    }

    # Mock the summarization client
    mock_summarization_client = MagicMock()
    mock_summarization_client.set_response_format = MagicMock()
    mock_summarization_client.reset_response_format = MagicMock()
    plan.summarization_client = mock_summarization_client

    # Mock llm_call to return a valid PreconditionDecision
    async def mock_llm_call(*args, **kwargs):
        return PreconditionDecision(
            status="ok",
            url=None,
            description=None,
        ).model_dump_json()

    monkeypatch.setattr(
        "unity.actor.hierarchical_actor.llm_call",
        mock_llm_call,
    )

    mock_item = VerificationWorkItem(
        ordinal=1,
        function_name="helper_func",
        parent_stack=(),
        func_source='async def helper_func(): return "Done"',
        docstring="A helper function",
        func_sig_str="()",
        pre_state={"computer_primitives": {"screenshot": None, "url": ""}},
        post_state={"computer_primitives": {"screenshot": None, "url": ""}},
        interactions=[],
        return_value_repr='"Done"',
        cache_miss_counter=0,
        exit_seq=1,
    )

    success_assessment = VerificationAssessment(
        status="ok",
        reason="Success",
    )

    # Call _on_verification_success
    await plan._on_verification_success(mock_item, success_assessment)

    # add_functions SHOULD have been called
    mock_function_manager.add_functions.assert_called_once()


# --- Combination Tests ---


def test_both_flags_false_on_constructor(
    mock_function_manager,
    mock_computer_primitives,
    monkeypatch,
):
    """Test that both flags can be set to False on the constructor."""
    monkeypatch.setattr(
        hierarchical_actor_module,
        "ComputerPrimitives",
        lambda *args, **kwargs: mock_computer_primitives,
    )

    actor = HierarchicalActor(
        function_manager=mock_function_manager,
        headless=True,
        can_compose=False,
        can_store=False,
    )

    assert actor.can_compose is False
    assert actor.can_store is False


@pytest.mark.asyncio
async def test_both_flags_false_on_act(actor):
    """Test that both flags can be set to False on act()."""
    assert actor.can_compose is True
    assert actor.can_store is True

    plan = await actor.act(
        "Do something",
        can_compose=False,
        can_store=False,
        persist=False,
    )

    assert plan.can_compose is False
    assert plan.can_store is False
