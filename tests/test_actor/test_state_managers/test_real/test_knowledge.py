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
    make_hierarchical_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.knowledge_manager.types import ColumnType
from unity.manager_registry import ManagerRegistry


def _office_hours_fact_present() -> bool:
    """Best-effort check that office hours were persisted by KnowledgeManager.update.

    The real KnowledgeManager may normalize "Office hours are 9–5 PT" into a structured
    schedule table (e.g. `Business_Hours`) rather than storing the exact input string.
    This helper intentionally checks for semantic presence (9–5 + PT/Pacific) rather than
    an exact phrase match.
    """
    km = ManagerRegistry.get_knowledge_manager()

    def _row_matches(row: object) -> bool:
        s = str(row).lower()
        # Accept either explicit "pt" or a spelled-out pacific time reference.
        tz_ok = ("pt" in s) or ("pacific" in s and "time" in s)
        # Accept either literal 9/5, or normalized 09:00/17:00 style.
        time_ok = (
            ("9" in s and "5" in s)
            or ("09:00" in s and "17:00" in s)
            or ("09:00" in s and "5:00" in s)
        )
        return tz_ok and time_ok and ("hour" in s or "business" in s or "office" in s)

    # Fast-path: common normalized tables.
    for table in ("Business_Hours", "Office_Hours", "Policies"):
        try:
            rows_by_table = km._filter(tables=table, filter=None, limit=1000)
            rows = (
                rows_by_table.get(table, []) if isinstance(rows_by_table, dict) else []
            )
            if any(_row_matches(r) for r in rows):
                return True
        except Exception:
            continue

    # Slow-path: scan all knowledge tables.
    try:
        tables = list(km._tables_overview(include_column_info=False).keys())
    except Exception:
        return False

    for table_name in tables:
        try:
            rows_by_table = km._filter(tables=table_name, filter=None, limit=1000)
            rows = (
                rows_by_table.get(table_name, [])
                if isinstance(rows_by_table, dict)
                else []
            )
            if any(_row_matches(r) for r in rows):
                return True
        except Exception:
            continue
    return False


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager(mock_verification):
    """Test that Actor calls KnowledgeManager.ask for knowledge queries."""
    async with make_hierarchical_actor(impl="real") as actor:

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
    async with make_hierarchical_actor(impl="real") as actor:

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
    async with make_hierarchical_actor(impl="real") as actor:

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
        assert _office_hours_fact_present(), (
            "Expected office hours to be persisted (9–5 PT / Pacific). "
            "No matching rows found in knowledge tables."
        )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for knowledge updates."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real KnowledgeManager
        km = ManagerRegistry.get_knowledge_manager()

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def store_knowledge(fact: str) -> str:
    """Mutate internal knowledge via the knowledge manager (create/update facts).

    **Use when** the user requests to store new knowledge, update an existing policy/fact,
    or otherwise change the knowledge base.

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
        assert _office_hours_fact_present(), (
            "Expected office hours to be persisted (9–5 PT / Pacific). "
            "No matching rows found in knowledge tables."
        )
