"""Real GuidanceManager tests for Actor.

Tests that Actor correctly calls real GuidanceManager methods and verifies
actual state mutations.
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_memoized_function_used,
    assert_tool_called,
    get_state_manager_tools,
    make_hierarchical_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager(mock_verification):
    """Test that Actor calls GuidanceManager.ask for guidance queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real GuidanceManager and seed data
        gm = ManagerRegistry.get_guidance_manager()
        gm._add_guidance(
            title="Onboarding Overview",
            content="We walk through onboarding steps for new users.",
        )

        # Call actor with natural language query
        handle = await actor.act(
            "What does the Guidance entry titled 'Onboarding Overview' say?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result contains expected content
        assert "onboarding" in result.lower()

        # Assert correct tool was called
        assert_tool_called(handle, "primitives.guidance.ask")

        # Assert only guidance tools were used
        state_manager_tools = get_state_manager_tools(handle)
        assert all("guidance" in tool for tool in state_manager_tools)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for guidance queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real GuidanceManager and seed data
        gm = ManagerRegistry.get_guidance_manager()
        gm._add_guidance(
            title="Onboarding Overview",
            content="We walk through onboarding steps for new users.",
        )

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def ask_guidance_question(question: str, response_format=None) -> str:
    """Query internal guidance/policies/runbooks via the guidance manager (read-only).

    **Use when** the question is about internal operating guidance, runbooks, incident
    response procedures, best practices, or other curated guidance content.

    **Do NOT use when**:
    - the user wants to create/update guidance entries (use `primitives.guidance.update`)
    - the user is asking about their message history/transcripts (use `primitives.transcripts.ask`)
    - the user needs current external facts (use `primitives.web.ask`)
    - the user is asking about contacts or tasks (use the appropriate manager)

    Args:
        question: The guidance-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the guidance manager as a string.
    """
    handle = await primitives.guidance.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with natural language query
        handle = await actor.act(
            "What does the Guidance entry titled 'Onboarding Overview' say? Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result contains expected content
        assert "onboarding" in result.lower()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "ask_guidance_question")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.guidance.ask")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager(mock_verification):
    """Test that Actor calls GuidanceManager.update for guidance mutations."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real GuidanceManager
        gm = ManagerRegistry.get_guidance_manager()

        # Call actor with create instruction
        handle = await actor.act(
            "Create a new guidance entry titled 'Incident Response' with the content 'Escalate sev-1 to on-call within 5 minutes.'",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert correct tool was called
        assert_tool_called(handle, "primitives.guidance.update")

        # Assert ask was NOT called (mutation only)
        state_manager_tools = get_state_manager_tools(handle)
        assert "primitives.guidance.ask" not in state_manager_tools

        # Verify mutation occurred in GuidanceManager
        rows = gm._filter(filter="title == 'Incident Response'")
        assert len(rows) > 0, "Expected 'Incident Response' entry to be created"

        # Check content contains expected keywords
        content = rows[0].content.lower()
        assert (
            "sev-1" in content or "on-call" in content
        ), f"Expected content to contain 'sev-1' or 'on-call', got: {content}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for guidance mutations."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real GuidanceManager
        gm = ManagerRegistry.get_guidance_manager()

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def create_guidance_entry(title: str, content: str) -> str:
    """Create/update/delete guidance entries via the guidance manager (mutation).

    **Use when** the user requests changes to internal guidance content: add a runbook,
    update an existing entry, or correct/replace guidance text.

    **Do NOT use when**:
    - the user is asking a read-only question about existing guidance (use `primitives.guidance.ask`)
    - the user is asking about transcripts, contacts, tasks, or current web facts

    Args:
        title: The title of the guidance entry to create.
        content: The content of the guidance entry.

    Returns:
        The result from the guidance manager update operation as a string.
    """
    handle = await primitives.guidance.update(
        f"Create a new guidance entry titled '{title}' with the content '{content}'."
    )
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with create instruction
        handle = await actor.act(
            "Create a new guidance entry titled 'Incident Response' with the content 'Escalate sev-1 to on-call within 5 minutes.' Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "create_guidance_entry")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.guidance.update")

        # Verify mutation occurred in GuidanceManager
        rows = gm._filter(filter="title == 'Incident Response'")
        assert len(rows) > 0, "Expected 'Incident Response' entry to be created"

        # Check content contains expected keywords
        content = rows[0].content.lower()
        assert (
            "sev-1" in content or "on-call" in content
        ), f"Expected content to contain 'sev-1' or 'on-call', got: {content}"
