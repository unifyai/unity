"""Real KnowledgeManager routing tests for CodeActActor.

These mirror `test_knowledge.py` but use CodeActActor (code-first tool loop).
"""

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_python_code_snippets,
    make_code_act_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.knowledge_manager.types import ColumnType
from unity.manager_registry import ManagerRegistry


def _office_hours_fact_present() -> bool:
    km = ManagerRegistry.get_knowledge_manager()
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
            continue
    return found


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager():
    """CodeAct routes read-only knowledge question → primitives.knowledge.ask."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
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

        handle = await actor.act(
            "What are our office hours?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert "primitives.knowledge.ask" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized():
    """CodeAct uses FunctionManager (when available) for knowledge queries."""
    fm = FunctionManager()
    implementation = """
async def ask_knowledge_question(question: str, response_format=None) -> str:
    \"\"\"Query internal knowledge via the knowledge manager (read-only).\"\"\"
    handle = await primitives.knowledge.ask(question, response_format=response_format)
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
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

        handle = await actor.act(
            "What are our office hours?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_python_code_snippets(handle))
        assert "ask_knowledge_question" in snippets

        assert "primitives.knowledge.ask" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager():
    """CodeAct routes knowledge mutation → primitives.knowledge.update."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        _km = ManagerRegistry.get_knowledge_manager()

        handle = await actor.act(
            "Store: Office hours are 9–5 PT.",
            clarification_enabled=False,
        )
        await handle.result()

        assert "primitives.knowledge.update" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)

        assert (
            _office_hours_fact_present()
        ), "Expected fact 'Office hours are 9–5 PT.' was not found in any knowledge table"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_memoized():
    """CodeAct uses FunctionManager (when available) for knowledge mutations."""
    fm = FunctionManager()
    implementation = """
async def store_knowledge(fact: str) -> str:
    \"\"\"Store knowledge via the knowledge manager.\"\"\"
    handle = await primitives.knowledge.update(f"Store: {fact}")
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        _km = ManagerRegistry.get_knowledge_manager()

        handle = await actor.act(
            "Store: Office hours are 9–5 PT.",
            clarification_enabled=False,
        )
        await handle.result()

        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_python_code_snippets(handle))
        assert "store_knowledge" in snippets

        assert "primitives.knowledge.update" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)

        assert (
            _office_hours_fact_present()
        ), "Expected fact 'Office hours are 9–5 PT.' was not found in any knowledge table"
