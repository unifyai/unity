"""Real KnowledgeManager tests for Actor.

Tests that Actor correctly calls real KnowledgeManager methods and verifies
actual state mutations.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    assert_memoized_function_used,
    assert_tool_called,
    get_state_manager_tools,
    make_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.knowledge_manager.types import ColumnType
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager(mock_verification):
    """Test that Actor calls KnowledgeManager.ask for knowledge queries."""
    async with make_actor(impl="real") as actor:

        # Access real KnowledgeManager and seed data
        km = ManagerRegistry.get_knowledge_manager()
        km._create_table(
            name="Policies",
            description="Company policies and procedures",
            columns={"title": ColumnType.str, "content": ColumnType.str},
        )
        km._add_rows(
            table="Policies",
            rows=[{"title": "Office Hours", "content": "Office hours are 9–5 PT."}],
        )

        # Call actor with natural language query
        handle = await actor.act(
            "What are our office hours?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result is non-empty
        assert result and len(result) > 0

        # Assert correct tool was called
        assert_tool_called(handle, "primitives.knowledge.ask")

        # Assert only knowledge tools were used
        state_manager_tools = get_state_manager_tools(handle)
        assert all("knowledge" in tool for tool in state_manager_tools)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for knowledge queries."""
    async with make_actor(impl="real") as actor:

        # Access real KnowledgeManager and seed data
        km = ManagerRegistry.get_knowledge_manager()
        km._create_table(
            name="Policies",
            description="Company policies and procedures",
            columns={"title": ColumnType.str, "content": ColumnType.str},
        )
        km._add_rows(
            table="Policies",
            rows=[{"title": "Office Hours", "content": "Office hours are 9–5 PT."}],
        )

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def ask_knowledge_question(question: str, response_format=None) -> str:
    """Query internal structured knowledge via the knowledge manager (read-only).

    **Use when** the question should be answered from stored organizational knowledge:
    policies, facts, reference material, and previously recorded information.

    **How it works**: calls:
    - `await primitives.knowledge.ask(question, response_format=response_format)`

    **Do NOT use when**:
    - the user needs current external facts (use `primitives.web.ask`)
    - the user is asking about message history/transcripts (use `primitives.transcripts.ask`)
    - the user is asking about contact records (use `primitives.contacts.ask`)
    - the user is requesting a knowledge mutation (use `primitives.knowledge.update`)

    Args:
        question: The knowledge-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the knowledge manager as a string.
    """
    handle = await primitives.knowledge.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with natural language query
        handle = await actor.act(
            "What are our office hours? Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result is non-empty
        assert result and len(result) > 0

        # Assert memoized function was used
        assert_memoized_function_used(handle, "ask_knowledge_question")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.knowledge.ask")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager(mock_verification):
    """Test that Actor calls KnowledgeManager.update for mutations."""
    async with make_actor(impl="real") as actor:

        # Access real KnowledgeManager
        km = ManagerRegistry.get_knowledge_manager()

        # Call actor with update request
        handle = await actor.act(
            "Store: Office hours are 9–5 PT.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert update tool was called (NOT ask)
        assert_tool_called(handle, "primitives.knowledge.update")

        # Verify the tool selection - may call ask for read-before-write/verification,
        # but must call update for a mutation request.
        state_manager_tools = get_state_manager_tools(handle)
        assert "primitives.knowledge.update" in state_manager_tools

        # Verify mutation: fact was actually persisted
        km = ManagerRegistry.get_knowledge_manager()
        # Query all tables to find the persisted fact.
        #
        # NOTE: KnowledgeManager does not expose `_list_tables()` / `_filter_rows()`.
        # Use `_tables_overview()` + `_filter()` which are the current internal helpers.
        found = False
        tables = list(km._tables_overview(include_column_info=False).keys())
        for table_name in tables:
            try:
                rows_by_table = km._filter(tables=table_name, filter=None, limit=1000)
                rows = (
                    rows_by_table.get(table_name, [])
                    if isinstance(rows_by_table, dict)
                    else []
                )
                for row in rows:
                    # Check if any field contains the office hours fact (case-insensitive)
                    row_str = str(row).lower()
                    if (
                        "office hours" in row_str
                        and "9" in row_str
                        and "5" in row_str
                        and "pt" in row_str
                    ):
                        found = True
                        break
                if found:
                    break
            except Exception:
                # Some tables may be unreadable in certain backends/configurations, skip them
                continue

        assert (
            found
        ), "Expected fact 'Office hours are 9–5 PT.' was not found in any knowledge table"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for knowledge updates."""
    async with make_actor(impl="real") as actor:

        # Access real KnowledgeManager
        km = ManagerRegistry.get_knowledge_manager()

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def store_knowledge(fact: str) -> str:
    """Mutate internal knowledge via the knowledge manager (create/update facts).

    **Use when** the user requests to store new knowledge, update an existing policy/fact,
    or otherwise change the knowledge base.

    **How it works**: calls:
    - `await primitives.knowledge.update(instruction, response_format=response_format)`

    **Do NOT use when**:
    - the request is read-only (use `primitives.knowledge.ask`)
    - the user is asking about transcripts, contacts, tasks, guidance, or web facts

    Args:
        fact: The fact or knowledge to store.

    Returns:
        The result from the knowledge manager update operation as a string.
    """
    handle = await primitives.knowledge.update(f"Store: {fact}")
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with update request
        handle = await actor.act(
            "Store: Office hours are 9–5 PT. Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "store_knowledge")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.knowledge.update")

        # Verify mutation: fact was actually persisted
        km = ManagerRegistry.get_knowledge_manager()
        # Query all tables to find the persisted fact.
        #
        # NOTE: KnowledgeManager does not expose `_list_tables()` / `_filter_rows()`.
        # Use `_tables_overview()` + `_filter()` which are the current internal helpers.
        found = False
        tables = list(km._tables_overview(include_column_info=False).keys())
        for table_name in tables:
            try:
                rows_by_table = km._filter(tables=table_name, filter=None, limit=1000)
                rows = (
                    rows_by_table.get(table_name, [])
                    if isinstance(rows_by_table, dict)
                    else []
                )
                for row in rows:
                    # Check if any field contains the office hours fact (case-insensitive)
                    row_str = str(row).lower()
                    if (
                        "office hours" in row_str
                        and "9" in row_str
                        and "5" in row_str
                        and "pt" in row_str
                    ):
                        found = True
                        break
                if found:
                    break
            except Exception:
                # Some tables may be unreadable in certain backends/configurations, skip them
                continue

        assert (
            found
        ), "Expected fact 'Office hours are 9–5 PT.' was not found in any knowledge table"
